#!/usr/bin/env python3
"""Retarget streamed 21-point hands to simulated Sharpa Wave hands and record an episode."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=Path(__file__).with_name("trex_isaac.yaml"))
parser.add_argument("--output-dir", type=Path, default=None)
parser.add_argument("--max-steps", type=int, default=None)
parser.add_argument("--no-images", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json
import io
import time

import numpy as np
import torch
import yaml
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils.math import matrix_from_quat

from hand_keypoint_retarget import (
    KeypointFilter,
    SimilarityTransform,
    TRACKED_INDICES,
    UdpKeypointReceiver,
    fit_similarity,
    palm_normalize,
    tracked_points,
)


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def exact_ids(robot: Articulation, names: list[str], bodies: bool = False) -> list[int]:
    ids = []
    finder = robot.find_bodies if bodies else robot.find_joints
    for name in names:
        found, matched = finder(f"^{name}$")
        if len(found) != 1:
            kind = "body" if bodies else "joint"
            raise RuntimeError(f"Expected one {kind} named {name!r}; matched {matched}")
        ids.append(found[0])
    return ids


def set_named_positions(robot: Articulation, mapping: dict[str, float]) -> None:
    positions = robot.data.default_joint_pos.torch.clone()
    for name, value in mapping.items():
        positions[:, exact_ids(robot, [name])[0]] = value
    robot.write_joint_position_to_sim_index(position=positions)
    robot.set_joint_position_target_index(target=positions)


def make_camera(name: str, cfg: dict, camera_cfg: dict, robot_prim: str) -> Camera:
    return Camera(
        CameraCfg(
            prim_path=f"{robot_prim}/{cfg['parent']}/{name}",
            update_period=0.0,
            width=camera_cfg["width"],
            height=camera_cfg["height"],
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=camera_cfg["focal_length"],
                horizontal_aperture=camera_cfg["horizontal_aperture"],
                clipping_range=(0.02, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(cfg["position"]), rot=tuple(cfg["quaternion_wxyz"]), convention="ros"
            ),
        )
    )


def body_points_in_palm(robot: Articulation, palm_id: int, marker_ids: list[int]) -> np.ndarray:
    palm = robot.data.body_pose_w.torch[0, palm_id]
    rotation = matrix_from_quat(palm[3:7].unsqueeze(0))[0]
    world_delta = robot.data.body_pose_w.torch[0, marker_ids, :3] - palm[:3]
    return (world_delta @ rotation).detach().cpu().numpy()


class HandRetargeter:
    def __init__(
        self,
        side: str,
        robot: Articulation,
        joint_ids: list[int],
        palm_id: int,
        marker_ids: list[int],
        cfg: dict,
    ):
        self.side = side
        self.robot = robot
        self.joint_ids = joint_ids
        self.palm_id = palm_id
        self.marker_ids = marker_ids
        self.cfg = cfg
        self.calibration_samples: list[np.ndarray] = []
        self.transform: SimilarityTransform | None = None
        self.target = robot.data.joint_pos.torch[:, joint_ids].clone()
        base_weights = torch.tensor(cfg["marker_weights"], device=robot.device)
        self.weights = base_weights.repeat(5).repeat_interleave(1)

    @property
    def calibrated(self) -> bool:
        return self.transform is not None

    def observe_calibration(self, local: np.ndarray) -> bool:
        if self.transform is not None:
            return True
        self.calibration_samples.append(tracked_points(local))
        if len(self.calibration_samples) < int(self.cfg["calibration_frames"]):
            return False
        source = np.mean(self.calibration_samples, axis=0)
        target = body_points_in_palm(self.robot, self.palm_id, self.marker_ids)
        self.transform = fit_similarity(source, target)
        residual = np.linalg.norm(self.transform.apply(source) - target, axis=1)
        print(
            f"{self.side} calibration complete: scale={self.transform.scale:.4f} m/palm-width, "
            f"marker RMSE={np.sqrt(np.mean(residual**2)) * 1000:.1f} mm"
        )
        return True

    def solve(self, local: np.ndarray, confidence: np.ndarray) -> torch.Tensor:
        assert self.transform is not None
        desired_local = torch.as_tensor(
            self.transform.apply(tracked_points(local)), dtype=torch.float32, device=self.robot.device
        )
        palm = self.robot.data.body_pose_w.torch[:, self.palm_id]
        palm_rotation = matrix_from_quat(palm[:, 3:7])[0]
        desired_world = desired_local @ palm_rotation.T + palm[0, :3]
        actual_world = self.robot.data.body_pose_w.torch[0, self.marker_ids, :3]
        error = desired_world - actual_world

        marker_conf = torch.as_tensor(
            confidence[TRACKED_INDICES], dtype=torch.float32, device=self.robot.device
        )
        marker_conf = torch.where(
            marker_conf >= float(self.cfg["min_confidence"]), marker_conf, torch.zeros_like(marker_conf)
        )
        weights = self.weights * marker_conf
        jacobians = []
        for body in self.marker_ids:
            jac_body = body - 1 if self.robot.is_fixed_base else body
            columns = [joint + self.robot.num_base_dofs for joint in self.joint_ids]
            jacobians.append(self.robot.data.body_link_jacobian_w.torch[0, jac_body, :3, columns])
        jacobian = torch.stack(jacobians)
        sqrt_w = torch.sqrt(weights).view(-1, 1, 1)
        weighted_j = (jacobian * sqrt_w).reshape(-1, len(self.joint_ids))
        weighted_e = (error * sqrt_w.squeeze(-1)).reshape(-1)

        damping = float(self.cfg["ik_damping"])
        regularization = float(self.cfg["neutral_regularization"])
        identity = torch.eye(len(self.joint_ids), device=self.robot.device)
        lhs = weighted_j.T @ weighted_j + (damping**2 + regularization) * identity
        neutral = torch.zeros_like(self.target[0])
        rhs = weighted_j.T @ weighted_e + regularization * (neutral - self.target[0])
        delta = torch.linalg.solve(lhs, rhs)
        delta = torch.clamp(delta, -float(self.cfg["max_joint_step"]), float(self.cfg["max_joint_step"]))
        candidate = self.target + delta.unsqueeze(0)
        limits = self.robot.data.soft_joint_pos_limits.torch[:, self.joint_ids]
        candidate = torch.maximum(torch.minimum(candidate, limits[..., 1]), limits[..., 0])
        alpha = float(self.cfg["target_smoothing"])
        self.target = alpha * candidate + (1.0 - alpha) * self.target
        return self.target

    def calibration_dict(self) -> dict | None:
        if self.transform is None:
            return None
        return {
            "scale": self.transform.scale,
            "rotation": self.transform.rotation.tolist(),
            "translation": self.transform.translation.tolist(),
        }


def camera_rgb(camera: Camera) -> np.ndarray:
    return camera.data.output["rgb"].torch[0, ..., :3].detach().cpu().numpy().astype(np.uint8)


def camera_jpeg(camera: Camera) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(camera_rgb(camera), mode="RGB").save(stream, format="JPEG", quality=90)
    return stream.getvalue()


def save_episode(output_dir: Path, records: dict[str, list], metadata: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = output_dir / time.strftime("episode_%Y%m%d_%H%M%S")
    suffix = 1
    base = episode_dir
    while episode_dir.exists():
        episode_dir = Path(f"{base}_{suffix:02d}")
        suffix += 1
    episode_dir.mkdir()
    arrays = {}
    for key, values in records.items():
        if not values:
            continue
        if isinstance(values[0], bytes):
            lengths = np.asarray([len(value) for value in values], dtype=np.int64)
            arrays[f"{key}_offsets"] = np.concatenate(([0], np.cumsum(lengths)))
            arrays[f"{key}_jpeg"] = np.frombuffer(b"".join(values), dtype=np.uint8)
        else:
            arrays[key] = np.asarray(values)
    np.savez_compressed(episode_dir / "trajectory.npz", **arrays)
    (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return episode_dir


def main() -> None:
    config_path = args_cli.config.resolve()
    cfg = yaml.safe_load(config_path.read_text())
    sim_cfg = cfg["simulation"]
    robot_cfg = cfg["robot"]
    teleop_cfg = cfg["keypoint_teleop"]
    required_sides = teleop_cfg.get("required_sides", ["left", "right"])
    if not required_sides or any(side not in ("left", "right") for side in required_sides):
        raise ValueError("keypoint_teleop.required_sides must contain left and/or right")
    usd_path = resolve_path(config_path, cfg["asset"]["usd_path"])
    if not usd_path.exists():
        raise FileNotFoundError(f"Missing {usd_path}; run prepare_asset.sh first")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=sim_cfg["dt"], render_interval=sim_cfg["render_interval"], device=args_cli.device)
    )
    sim.set_camera_view((2.5, 2.5, 2.0), (0.5, 0.0, 0.8))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    light.func("/World/Light", light)
    table = sim_utils.CuboidCfg(
        size=(1.2, 1.6, 0.04), collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.22)),
    )
    table.func("/World/Table", table, translation=(0.8, 0.0, sim_cfg["table_height"] - 0.02))

    object_cfg = RigidObjectCfg(
        prim_path="/World/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.12), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.75, 0.0, sim_cfg["table_height"] + 0.06)),
    )
    object_asset = RigidObject(object_cfg)
    robot = Articulation(
        ArticulationCfg(
            prim_path=cfg["asset"]["prim_path"], spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
            actuators={"all": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=400.0, damping=40.0, effort_limit_sim=200.0
            )},
        )
    )
    cameras = {
        key: make_camera(f"{key}_camera", cfg["cameras"][key], cfg["cameras"], cfg["asset"]["prim_path"])
        for key in ("head", "left_wrist", "right_wrist")
    }
    sim.reset()

    initial = dict(zip(robot_cfg["left_arm_joints"], robot_cfg["left_arm_initial"]))
    initial.update(dict(zip(robot_cfg["right_arm_joints"], robot_cfg["right_arm_initial"])))
    initial.update(dict(zip(["torso_j1", "torso_j2", "torso_j3"], robot_cfg["torso_initial"])))
    initial.update(dict(zip(["head_j1", "head_j2", "head_j3"], robot_cfg["head_initial"])))
    initial.update({name: 0.0 for name in robot_cfg["left_hand_joints"] + robot_cfg["right_hand_joints"]})
    set_named_positions(robot, initial)

    for _ in range(sim_cfg["settle_steps"]):
        robot.write_data_to_sim()
        object_asset.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg["dt"])
        object_asset.update(sim_cfg["dt"])
        for camera in cameras.values():
            camera.update(sim_cfg["dt"])

    retargeters = {}
    for side in ("left", "right"):
        joint_ids = exact_ids(robot, robot_cfg[f"{side}_hand_joints"])
        palm_id = exact_ids(robot, [robot_cfg["hand_palm_bodies"][side]], bodies=True)[0]
        marker_ids = exact_ids(robot, robot_cfg["hand_tracking_bodies"][side], bodies=True)
        retargeters[side] = HandRetargeter(side, robot, joint_ids, palm_id, marker_ids, teleop_cfg)

    receiver = UdpKeypointReceiver(
        teleop_cfg["bind_host"], int(teleop_cfg["port"]), float(teleop_cfg["timeout_s"])
    )
    receiver.start()
    keypoint_filter = KeypointFilter(
        float(teleop_cfg["filter_alpha"]), float(teleop_cfg["min_confidence"]),
        float(teleop_cfg["max_keypoint_jump_m"]),
    )
    print(f"Listening for 21-point hand JSON on udp://{teleop_cfg['bind_host']}:{teleop_cfg['port']}")
    print("Hold each visible hand open and steady during calibration; recording starts automatically afterward.")

    records: dict[str, list] = {
        "time": [], "tracker_timestamp": [],
        "left_keypoints_palm_normalized": [], "right_keypoints_palm_normalized": [],
        "left_target_joint_pos": [], "right_target_joint_pos": [],
        "left_joint_pos": [], "right_joint_pos": [], "object_pose": [], "object_velocity": [],
        "image_head": [], "image_wrist_left": [], "image_wrist_right": [],
    }
    start_time = time.monotonic()
    recording = False
    step = 0
    max_steps = args_cli.max_steps or int(teleop_cfg["max_steps"])
    output_dir = args_cli.output_dir or resolve_path(config_path, teleop_cfg["output_dir"])
    last_packet_timestamp = np.nan

    try:
        while simulation_app.is_running() and step < max_steps:
            packet, age, error = receiver.latest()
            if error and step % 120 == 0:
                print(f"Ignoring malformed keypoint packet: {error}")
            normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            if packet is not None:
                last_packet_timestamp = packet.timestamp
                for side, sample in packet.hands.items():
                    filtered = keypoint_filter.update(side, sample)
                    if filtered is None:
                        continue
                    try:
                        local, _, _, _ = palm_normalize(filtered)
                    except ValueError:
                        continue
                    normalized[side] = (local, filtered.confidence)
                    retargeters[side].observe_calibration(local)

            if not recording and all(retargeters[side].calibrated for side in required_sides):
                recording = True
                start_time = time.monotonic()
                print(f"Required hands calibrated ({', '.join(required_sides)}); episode recording started.")

            for side, (local, confidence) in normalized.items():
                retargeter = retargeters[side]
                if retargeter.calibrated:
                    target = retargeter.solve(local, confidence)
                    robot.set_joint_position_target_index(target=target, joint_ids=retargeter.joint_ids)

            robot.write_data_to_sim()
            object_asset.write_data_to_sim()
            sim.step()
            robot.update(sim_cfg["dt"])
            object_asset.update(sim_cfg["dt"])
            for camera in cameras.values():
                camera.update(sim_cfg["dt"])

            if recording and step % int(teleop_cfg["record_stride"]) == 0:
                nan_keypoints = np.full((21, 3), np.nan, dtype=np.float32)
                records["time"].append(time.monotonic() - start_time)
                records["tracker_timestamp"].append(last_packet_timestamp)
                records["left_keypoints_palm_normalized"].append(
                    normalized.get("left", (nan_keypoints, None))[0]
                )
                records["right_keypoints_palm_normalized"].append(
                    normalized.get("right", (nan_keypoints, None))[0]
                )
                for side in ("left", "right"):
                    r = retargeters[side]
                    records[f"{side}_target_joint_pos"].append(r.target[0].detach().cpu().numpy())
                    records[f"{side}_joint_pos"].append(
                        robot.data.joint_pos.torch[0, r.joint_ids].detach().cpu().numpy()
                    )
                records["object_pose"].append(object_asset.data.root_pose_w.torch[0].detach().cpu().numpy())
                records["object_velocity"].append(object_asset.data.root_vel_w.torch[0].detach().cpu().numpy())
                if not args_cli.no_images:
                    records["image_head"].append(camera_jpeg(cameras["head"]))
                    records["image_wrist_left"].append(camera_jpeg(cameras["left_wrist"]))
                    records["image_wrist_right"].append(camera_jpeg(cameras["right_wrist"]))
            step += 1
    finally:
        receiver.stop()

    if records["time"]:
        metadata = {
            "format": "trex_isaac_keypoint_demo_v1",
            "simulation_dt": sim_cfg["dt"],
            "record_stride": teleop_cfg["record_stride"],
            "joint_order": {side: robot_cfg[f"{side}_hand_joints"] for side in ("left", "right")},
            "tracked_keypoint_indices": TRACKED_INDICES.tolist(),
            "calibration": {side: retargeters[side].calibration_dict() for side in ("left", "right")},
        }
        episode_dir = save_episode(Path(output_dir), records, metadata)
        print(f"Saved {len(records['time'])} frames to {episode_dir}")
    else:
        print("No episode saved: required hands were not calibrated before exit.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
