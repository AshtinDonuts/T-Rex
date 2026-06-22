#!/usr/bin/env python3
"""Retarget streamed 21-point hands to simulated Sharpa Wave hands and record an episode."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=Path(__file__).with_name("trex_isaac.yaml"))
parser.add_argument("--output-dir", type=Path, default=None)
parser.add_argument("--max-steps", type=int, default=None)
parser.add_argument("--no-images", action="store_true")
parser.add_argument("--diagnostics", action="store_true", help="Print per-finger retargeting errors.")
side_group = parser.add_mutually_exclusive_group()
side_group.add_argument(
    "--left-only", action="store_true",
    help="Calibrate and retarget only the left hand, overriding required_sides in YAML.",
)
side_group.add_argument(
    "--right-only", action="store_true",
    help="Calibrate and retarget only the right hand, overriding required_sides in YAML.",
)
parser.add_argument(
    "--hand-only", action="store_true", help="Retarget Wave fingers while holding DexMate arms fixed."
)
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
from isaaclab.utils.math import compute_pose_error, matrix_from_quat, quat_from_matrix

from hand_keypoint_retarget import (
    KeypointFilter,
    PalmPose,
    PalmPoseFilter,
    SimilarityTransform,
    TRACKED_INDICES,
    UdpKeypointReceiver,
    assemble_trex_vector,
    average_rotation,
    fit_similarity,
    palm_pose,
    palm_normalize,
    quaternion_wxyz_to_matrix,
    relative_pose_target,
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
        self.marker_error_mm = np.full(15, np.nan)
        base_weights = torch.tensor(cfg["marker_weights"], device=robot.device)
        self.weights = base_weights.repeat(5).repeat_interleave(1)

    @property
    def calibrated(self) -> bool:
        return self.transform is not None

    def observe_calibration(self, local: np.ndarray, confidence: np.ndarray) -> bool:
        if self.transform is not None:
            return True
        if np.any(confidence[TRACKED_INDICES] < float(self.cfg["min_confidence"])):
            return False
        self.calibration_samples.append(tracked_points(local))
        if len(self.calibration_samples) < int(self.cfg["calibration_frames"]):
            return False
        source = np.mean(self.calibration_samples, axis=0)
        target = body_points_in_palm(self.robot, self.palm_id, self.marker_ids)
        self.transform = fit_similarity(source, target)
        residual = np.linalg.norm(self.transform.apply(source) - target, axis=1)
        per_finger = np.sqrt(np.mean(residual.reshape(5, 3) ** 2, axis=1)) * 1000
        print(
            f"{self.side} calibration complete: scale={self.transform.scale:.4f} m/palm-width, "
            f"marker RMSE={np.sqrt(np.mean(residual**2)) * 1000:.1f} mm; "
            f"thumb/index/middle/ring/pinky={np.round(per_finger, 1).tolist()} mm"
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
        self.marker_error_mm = torch.linalg.norm(error, dim=1).detach().cpu().numpy() * 1000.0

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

    def diagnostic_summary(self) -> str:
        per_finger = np.sqrt(np.mean(self.marker_error_mm.reshape(5, 3) ** 2, axis=1))
        total = float(np.sqrt(np.mean(self.marker_error_mm ** 2)))
        return (
            f"{self.side} marker RMSE={total:.1f} mm; "
            f"thumb/index/middle/ring/pinky={np.round(per_finger, 1).tolist()} mm"
        )

    def calibration_dict(self) -> dict | None:
        if self.transform is None:
            return None
        return {
            "scale": self.transform.scale,
            "rotation": self.transform.rotation.tolist(),
            "translation": self.transform.translation.tolist(),
        }


class ArmRetargeter:
    """Map relative vision palm poses to one DexMate EEF and solve its 7-DoF IK."""

    def __init__(
        self,
        side: str,
        robot: Articulation,
        joint_ids: list[int],
        ee_body_id: int,
        cfg: dict,
        camera_to_robot_rotation: np.ndarray,
    ):
        self.side = side
        self.robot = robot
        self.joint_ids = joint_ids
        self.ee_body_id = ee_body_id
        self.cfg = cfg
        self.camera_to_robot_rotation = camera_to_robot_rotation
        self.calibration_samples: list[PalmPose] = []
        self.initial_palm: PalmPose | None = None
        self.initial_ee: PalmPose | None = None
        self.target_joint_pos = robot.data.joint_pos.torch[:, joint_ids].clone()
        self.default_joint_pos = self.target_joint_pos.clone()
        self.target_pose: PalmPose | None = None
        self.position_error = np.full(3, np.nan)
        self.rotation_error = np.full(3, np.nan)

    @property
    def calibrated(self) -> bool:
        return self.initial_palm is not None

    def observe_calibration(self, pose: PalmPose) -> bool:
        if self.calibrated:
            return True
        self.calibration_samples.append(pose)
        if len(self.calibration_samples) < int(self.cfg["calibration_frames"]):
            return False
        initial_position = np.mean([sample.position for sample in self.calibration_samples], axis=0)
        initial_rotation = average_rotation([sample.rotation for sample in self.calibration_samples])
        ee_pose = self.robot.data.body_pose_w.torch[0, self.ee_body_id]
        ee_rotation = matrix_from_quat(ee_pose[3:7].unsqueeze(0))[0].detach().cpu().numpy()
        self.initial_palm = PalmPose(initial_position, initial_rotation)
        self.initial_ee = PalmPose(ee_pose[:3].detach().cpu().numpy(), ee_rotation)
        self.target_pose = self.initial_ee
        print(f"{self.side} arm calibration complete at EEF {self.initial_ee.position.round(3)}")
        return True

    def solve(self, pose: PalmPose) -> torch.Tensor:
        assert self.initial_palm is not None and self.initial_ee is not None
        desired = relative_pose_target(
            pose,
            self.initial_palm,
            self.initial_ee,
            self.camera_to_robot_rotation,
            float(self.cfg["arm_translation_scale"]),
        )
        workspace = self.cfg["arm_workspace"][self.side]
        desired_position = np.clip(
            desired.position,
            np.asarray(workspace["min"], dtype=np.float64),
            np.asarray(workspace["max"], dtype=np.float64),
        )
        self.target_pose = PalmPose(desired_position, desired.rotation)

        target_position = torch.as_tensor(
            desired_position, dtype=torch.float32, device=self.robot.device
        ).unsqueeze(0)
        target_rotation = torch.as_tensor(
            desired.rotation, dtype=torch.float32, device=self.robot.device
        ).unsqueeze(0)
        target_quaternion = quat_from_matrix(target_rotation)
        current = self.robot.data.body_pose_w.torch[:, self.ee_body_id]
        position_error, rotation_error = compute_pose_error(
            current[:, :3], current[:, 3:7], target_position, target_quaternion
        )
        self.position_error = position_error[0].detach().cpu().numpy()
        self.rotation_error = rotation_error[0].detach().cpu().numpy()
        error = torch.cat(
            (position_error, float(self.cfg["arm_orientation_weight"]) * rotation_error), dim=-1
        ).unsqueeze(-1)

        jac_body_id = self.ee_body_id - 1 if self.robot.is_fixed_base else self.ee_body_id
        columns = [joint + self.robot.num_base_dofs for joint in self.joint_ids]
        jacobian = self.robot.data.body_link_jacobian_w.torch[:, jac_body_id, :, columns].clone()
        jacobian[:, 3:6] *= float(self.cfg["arm_orientation_weight"])
        damping = float(self.cfg["arm_ik_damping"])
        posture = float(self.cfg["arm_posture_regularization"])
        identity = torch.eye(len(self.joint_ids), device=self.robot.device).unsqueeze(0)
        lhs = jacobian.transpose(1, 2) @ jacobian + (damping**2 + posture) * identity
        rhs = jacobian.transpose(1, 2) @ error
        rhs += posture * (self.default_joint_pos - self.target_joint_pos).unsqueeze(-1)
        delta = torch.linalg.solve(lhs, rhs).squeeze(-1)
        delta = torch.clamp(
            delta, -float(self.cfg["arm_max_joint_step"]), float(self.cfg["arm_max_joint_step"])
        )
        candidate = self.target_joint_pos + delta
        limits = self.robot.data.soft_joint_pos_limits.torch[:, self.joint_ids]
        candidate = torch.maximum(torch.minimum(candidate, limits[..., 1]), limits[..., 0])
        alpha = float(self.cfg["arm_target_smoothing"])
        self.target_joint_pos = alpha * candidate + (1.0 - alpha) * self.target_joint_pos
        return self.target_joint_pos

    def target_matrix(self) -> np.ndarray:
        matrix = np.full((4, 4), np.nan, dtype=np.float32)
        if self.target_pose is not None:
            matrix[:] = np.eye(4, dtype=np.float32)
            matrix[:3, :3] = self.target_pose.rotation
            matrix[:3, 3] = self.target_pose.position
        return matrix

    def calibration_dict(self) -> dict | None:
        if self.initial_palm is None or self.initial_ee is None:
            return None
        return {
            "initial_palm_position": self.initial_palm.position.tolist(),
            "initial_palm_rotation": self.initial_palm.rotation.tolist(),
            "initial_ee_position": self.initial_ee.position.tolist(),
            "initial_ee_rotation": self.initial_ee.rotation.tolist(),
        }


def camera_rgb(camera: Camera) -> np.ndarray:
    return camera.data.output["rgb"].torch[0, ..., :3].detach().cpu().numpy().astype(np.uint8)


def camera_jpeg(camera: Camera) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(camera_rgb(camera), mode="RGB").save(stream, format="JPEG", quality=90)
    return stream.getvalue()


def pose_matrix(pose: PalmPose | None) -> np.ndarray:
    matrix = np.full((4, 4), np.nan, dtype=np.float32)
    if pose is not None:
        matrix[:] = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = pose.rotation
        matrix[:3, 3] = pose.position
    return matrix


def body_pose_matrix(robot: Articulation, body_id: int) -> np.ndarray:
    value = robot.data.body_pose_w.torch[0, body_id]
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = matrix_from_quat(value[3:7].unsqueeze(0))[0].detach().cpu().numpy()
    matrix[:3, 3] = value[:3].detach().cpu().numpy()
    return matrix


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
    if args_cli.left_only:
        required_sides = ["left"]
    elif args_cli.right_only:
        required_sides = ["right"]
    else:
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
    object_asset: RigidObject | None = None
    if args_cli.diagnostics:
        print("Diagnostic scene: robot + ground only (table and rigid object omitted).")
    else:
        table = sim_utils.CuboidCfg(
            size=(1.2, 1.6, 0.04), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.22)),
        )
        table.func(
            "/World/Table", table,
            translation=(0.8, 0.0, sim_cfg["table_height"] - 0.02),
        )
        object_cfg = RigidObjectCfg(
            prim_path="/World/Object",
            spawn=sim_utils.CuboidCfg(
                size=(0.06, 0.06, 0.12), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.75, 0.0, sim_cfg["table_height"] + 0.06)
            ),
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
        if object_asset is not None:
            object_asset.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg["dt"])
        if object_asset is not None:
            object_asset.update(sim_cfg["dt"])
        for camera in cameras.values():
            camera.update(sim_cfg["dt"])

    hand_retargeters = {}
    arm_retargeters = {}
    camera_to_robot_rotation = quaternion_wxyz_to_matrix(
        np.asarray(teleop_cfg["camera_to_robot_quaternion_wxyz"], dtype=np.float64)
    )
    for side in ("left", "right"):
        hand_joint_ids = exact_ids(robot, robot_cfg[f"{side}_hand_joints"])
        palm_id = exact_ids(robot, [robot_cfg["hand_palm_bodies"][side]], bodies=True)[0]
        marker_ids = exact_ids(robot, robot_cfg["hand_tracking_bodies"][side], bodies=True)
        hand_retargeters[side] = HandRetargeter(
            side, robot, hand_joint_ids, palm_id, marker_ids, teleop_cfg
        )
        arm_joint_ids = exact_ids(robot, robot_cfg[f"{side}_arm_joints"])
        ee_body_id = exact_ids(robot, [robot_cfg[f"{side}_ee_body"]], bodies=True)[0]
        arm_retargeters[side] = ArmRetargeter(
            side, robot, arm_joint_ids, ee_body_id, teleop_cfg, camera_to_robot_rotation
        )

    receiver = UdpKeypointReceiver(
        teleop_cfg["bind_host"], int(teleop_cfg["port"]), float(teleop_cfg["timeout_s"])
    )
    receiver.start()
    keypoint_filter = KeypointFilter(
        float(teleop_cfg["filter_alpha"]), float(teleop_cfg["min_confidence"]),
        float(teleop_cfg["max_keypoint_jump_m"]),
    )
    palm_filter = PalmPoseFilter(
        float(teleop_cfg["palm_position_alpha"]),
        float(teleop_cfg["palm_rotation_alpha"]),
        float(teleop_cfg["max_palm_jump_m"]),
        float(teleop_cfg["max_palm_jump_rad"]),
    )
    print(f"Listening for 21-point hand JSON on udp://{teleop_cfg['bind_host']}:{teleop_cfg['port']}")
    print(f"Retarget mode: {'hand only (arms fixed)' if args_cli.hand_only else 'hand + arm'}")
    calibration_frames = int(teleop_cfg["calibration_frames"])
    calibration_countdown_s = float(teleop_cfg.get("calibration_countdown_s", 3.0))
    print(
        f"CALIBRATION WAITING: show {', '.join(required_sides)} hand(s). A "
        f"{calibration_countdown_s:g}s countdown starts after all are tracked."
    )
    print(
        f"Then hold every required hand OPEN and STILL for {calibration_frames} valid frames "
        f"(~{calibration_frames / 30.0:.1f}s at 30 Hz). Recording starts after completion."
    )

    records: dict[str, list] = {
        "time": [], "tracker_timestamp": [], "keypoint_age": [], "input_valid": [],
        "left_keypoints_palm_normalized": [], "right_keypoints_palm_normalized": [],
        "left_palm_pose_camera": [], "right_palm_pose_camera": [],
        "left_arm_target_pose": [], "right_arm_target_pose": [],
        "left_arm_actual_pose": [], "right_arm_actual_pose": [],
        "left_arm_ik_position_error": [], "right_arm_ik_position_error": [],
        "left_arm_ik_rotation_error": [], "right_arm_ik_rotation_error": [],
        "left_arm_target_joint_pos": [], "right_arm_target_joint_pos": [],
        "left_hand_target_joint_pos": [], "right_hand_target_joint_pos": [],
        "left_arm_joint_pos": [], "right_arm_joint_pos": [],
        "left_hand_joint_pos": [], "right_hand_joint_pos": [],
        "observation_state": [], "action": [],
        "object_pose": [], "object_velocity": [],
        "image_head": [], "image_wrist_left": [], "image_wrist_right": [],
    }
    start_time = time.monotonic()
    recording = False
    step = 0
    max_steps = args_cli.max_steps or int(teleop_cfg["max_steps"])
    output_dir = args_cli.output_dir or resolve_path(config_path, teleop_cfg["output_dir"])
    last_packet_timestamp = np.nan
    last_sequence = -1
    last_diagnostic_report = 0.0
    last_valid_input_time: dict[str, float | None] = {"left": None, "right": None}
    latest_observations: dict[str, tuple[np.ndarray, np.ndarray, PalmPose | None]] = {}
    calibration_ready_since: float | None = None
    calibration_capture_since: float | None = None
    calibration_countdown_value: int | None = None
    calibration_last_progress = 0
    calibration_paused = False

    try:
        while simulation_app.is_running() and step < max_steps:
            packet, sequence, _transport_age, error = receiver.latest()
            new_packet = packet is not None and sequence != last_sequence
            updated_observations: dict[str, tuple[np.ndarray, np.ndarray, PalmPose | None]] = {}
            if error and step % 120 == 0:
                print(f"Ignoring malformed keypoint packet: {error}")
            if new_packet:
                last_sequence = sequence
                last_packet_timestamp = packet.timestamp
                if packet.frame_id != teleop_cfg["keypoint_frame_id"]:
                    print(
                        f"Ignoring keypoint frame {packet.frame_id!r}; expected "
                        f"{teleop_cfg['keypoint_frame_id']!r}"
                    )
                    new_packet = False
                for side, sample in (packet.hands.items() if new_packet else ()):
                    filtered = keypoint_filter.update(side, sample)
                    if filtered is None:
                        continue
                    try:
                        local, _, _, _ = palm_normalize(filtered)
                    except ValueError:
                        continue
                    try:
                        filtered_palm = palm_filter.update(side, palm_pose(filtered))
                    except ValueError:
                        filtered_palm = None
                    if filtered_palm is None and not args_cli.hand_only:
                        continue
                    latest_observations[side] = (local, filtered.confidence, filtered_palm)
                    updated_observations[side] = (local, filtered.confidence, filtered_palm)
                update_time = time.monotonic()
                for side in updated_observations:
                    last_valid_input_time[side] = update_time

                if not recording:
                    all_required_visible = all(side in updated_observations for side in required_sides)
                    if all_required_visible and calibration_ready_since is None:
                        calibration_ready_since = update_time
                        calibration_countdown_value = None
                        print(
                            "CALIBRATION HANDS DETECTED: keep all required hands open and still; "
                            "countdown starting."
                        )

                    if all_required_visible and calibration_ready_since is not None:
                        countdown_elapsed = update_time - calibration_ready_since
                        if countdown_elapsed < calibration_countdown_s:
                            remaining = max(
                                1, int(math.ceil(calibration_countdown_s - countdown_elapsed))
                            )
                            if remaining != calibration_countdown_value:
                                print(f"CALIBRATION STARTS IN {remaining}...")
                                calibration_countdown_value = remaining
                        else:
                            if calibration_capture_since is None:
                                calibration_capture_since = update_time
                                print(
                                    "CALIBRATION CAPTURE STARTED: HOLD OPEN + STILL until completion."
                                )
                            if calibration_paused:
                                print("CALIBRATION RESUMED: all required hands tracked.")
                                calibration_paused = False
                            for side in required_sides:
                                local, confidence, filtered_palm = updated_observations[side]
                                hand_retargeters[side].observe_calibration(local, confidence)
                                if not args_cli.hand_only and filtered_palm is not None:
                                    arm_retargeters[side].observe_calibration(filtered_palm)
                            captured = min(
                                min(len(hand_retargeters[side].calibration_samples), calibration_frames)
                                for side in required_sides
                            )
                            if captured > calibration_last_progress and (
                                captured == 1
                                or captured % 5 == 0
                                or captured == calibration_frames
                            ):
                                elapsed = update_time - calibration_capture_since
                                print(
                                    f"CALIBRATION PROGRESS: {captured}/{calibration_frames} valid "
                                    f"frames ({elapsed:.1f}s elapsed)."
                                )
                                calibration_last_progress = captured
                    elif calibration_capture_since is None and calibration_ready_since is not None:
                        print("CALIBRATION COUNTDOWN RESET: a required hand was lost.")
                        calibration_ready_since = None
                        calibration_countdown_value = None
                    elif calibration_capture_since is not None and not calibration_paused:
                        print("CALIBRATION PAUSED: required hand lost; show it again to continue.")
                        calibration_paused = True

            now = time.monotonic()
            valid_ages = {
                side: now - timestamp if timestamp is not None else np.inf
                for side, timestamp in last_valid_input_time.items()
            }
            valid_age = max(valid_ages[side] for side in required_sides)
            if recording and valid_age > float(teleop_cfg["abort_timeout_s"]):
                print(
                    f"Valid keypoint stream stale for {valid_age:.2f}s; "
                    "ending episode while holding targets."
                )
                break

            if not recording and all(
                hand_retargeters[side].calibrated
                and (args_cli.hand_only or arm_retargeters[side].calibrated)
                for side in required_sides
            ):
                recording = True
                start_time = time.monotonic()
                calibration_elapsed = (
                    time.monotonic() - calibration_capture_since
                    if calibration_capture_since is not None else float("nan")
                )
                print(
                    f"CALIBRATION COMPLETE in {calibration_elapsed:.1f}s "
                    f"({', '.join(required_sides)}). EPISODE RECORDING STARTED."
                )

            if new_packet:
                for side, (local, confidence, filtered_palm) in updated_observations.items():
                    hand_retargeter = hand_retargeters[side]
                    arm_retargeter = arm_retargeters[side]
                    if hand_retargeter.calibrated:
                        hand_target = hand_retargeter.solve(local, confidence)
                        robot.set_joint_position_target_index(
                            target=hand_target, joint_ids=hand_retargeter.joint_ids
                        )
                    if (
                        not args_cli.hand_only
                        and arm_retargeter.calibrated
                        and filtered_palm is not None
                    ):
                        arm_target = arm_retargeter.solve(filtered_palm)
                        robot.set_joint_position_target_index(
                            target=arm_target, joint_ids=arm_retargeter.joint_ids
                        )
                diagnostic_now = time.monotonic()
                if args_cli.diagnostics and diagnostic_now - last_diagnostic_report >= 1.0:
                    for side in required_sides:
                        hand_retargeter = hand_retargeters[side]
                        if hand_retargeter.calibrated and np.isfinite(
                            hand_retargeter.marker_error_mm
                        ).any():
                            print(hand_retargeter.diagnostic_summary())
                    last_diagnostic_report = diagnostic_now

            robot.write_data_to_sim()
            if object_asset is not None:
                object_asset.write_data_to_sim()
            sim.step()
            robot.update(sim_cfg["dt"])
            if object_asset is not None:
                object_asset.update(sim_cfg["dt"])
            for camera in cameras.values():
                camera.update(sim_cfg["dt"])

            if recording and step % int(teleop_cfg["record_stride"]) == 0:
                nan_keypoints = np.full((21, 3), np.nan, dtype=np.float32)
                records["time"].append(time.monotonic() - start_time)
                records["tracker_timestamp"].append(last_packet_timestamp)
                records["keypoint_age"].append(valid_age)
                records["input_valid"].append(valid_age <= float(teleop_cfg["timeout_s"]))
                records["left_keypoints_palm_normalized"].append(
                    latest_observations.get("left", (nan_keypoints, None, None))[0]
                )
                records["right_keypoints_palm_normalized"].append(
                    latest_observations.get("right", (nan_keypoints, None, None))[0]
                )
                records["left_palm_pose_camera"].append(
                    pose_matrix(latest_observations.get("left", (None, None, None))[2])
                )
                records["right_palm_pose_camera"].append(
                    pose_matrix(latest_observations.get("right", (None, None, None))[2])
                )
                state_components = {}
                action_components = {}
                for side in ("left", "right"):
                    hand = hand_retargeters[side]
                    arm = arm_retargeters[side]
                    arm_actual = robot.data.joint_pos.torch[0, arm.joint_ids].detach().cpu().numpy()
                    hand_actual = robot.data.joint_pos.torch[0, hand.joint_ids].detach().cpu().numpy()
                    arm_target = arm.target_joint_pos[0].detach().cpu().numpy()
                    hand_target = hand.target[0].detach().cpu().numpy()
                    records[f"{side}_arm_target_joint_pos"].append(arm_target)
                    records[f"{side}_hand_target_joint_pos"].append(hand_target)
                    records[f"{side}_arm_joint_pos"].append(arm_actual)
                    records[f"{side}_hand_joint_pos"].append(hand_actual)
                    records[f"{side}_arm_target_pose"].append(arm.target_matrix())
                    records[f"{side}_arm_actual_pose"].append(
                        body_pose_matrix(robot, arm.ee_body_id)
                    )
                    records[f"{side}_arm_ik_position_error"].append(arm.position_error.copy())
                    records[f"{side}_arm_ik_rotation_error"].append(arm.rotation_error.copy())
                    state_components[f"{side}_arm"] = arm_actual
                    state_components[f"{side}_hand"] = hand_actual
                    action_components[f"{side}_arm"] = arm_target
                    action_components[f"{side}_hand"] = hand_target
                records["observation_state"].append(
                    assemble_trex_vector(
                        state_components["left_arm"], state_components["left_hand"],
                        state_components["right_arm"], state_components["right_hand"],
                    )
                )
                records["action"].append(
                    assemble_trex_vector(
                        action_components["left_arm"], action_components["left_hand"],
                        action_components["right_arm"], action_components["right_hand"],
                    )
                )
                if object_asset is not None:
                    records["object_pose"].append(
                        object_asset.data.root_pose_w.torch[0].detach().cpu().numpy()
                    )
                    records["object_velocity"].append(
                        object_asset.data.root_vel_w.torch[0].detach().cpu().numpy()
                    )
                if not args_cli.no_images:
                    records["image_head"].append(camera_jpeg(cameras["head"]))
                    records["image_wrist_left"].append(camera_jpeg(cameras["left_wrist"]))
                    records["image_wrist_right"].append(camera_jpeg(cameras["right_wrist"]))
            step += 1
    finally:
        receiver.stop()

    if records["time"]:
        metadata = {
            "format": "trex_isaac_keypoint_demo_v2",
            "retarget_mode": "hand_only" if args_cli.hand_only else "hand_and_arm",
            "diagnostic_scene": bool(args_cli.diagnostics),
            "required_sides": required_sides,
            "simulation_dt": sim_cfg["dt"],
            "record_stride": teleop_cfg["record_stride"],
            "fps": 1.0 / (sim_cfg["dt"] * teleop_cfg["record_stride"]),
            "keypoint_frame_id": teleop_cfg["keypoint_frame_id"],
            "state_action_order": [
                *robot_cfg["left_arm_joints"], *robot_cfg["left_hand_joints"],
                *robot_cfg["right_arm_joints"], *robot_cfg["right_hand_joints"],
            ],
            "tracked_keypoint_indices": TRACKED_INDICES.tolist(),
            "camera_to_robot_rotation": camera_to_robot_rotation.tolist(),
            "calibration": {
                side: {
                    "hand": hand_retargeters[side].calibration_dict(),
                    "arm": arm_retargeters[side].calibration_dict(),
                }
                for side in ("left", "right")
            },
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
