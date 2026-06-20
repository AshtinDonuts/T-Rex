#!/usr/bin/env python3
"""Run a T-Rex checkpoint closed-loop in a single Isaac Lab environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=Path(__file__).with_name("trex_isaac.yaml"))
parser.add_argument("--task-description", type=str, default=None)
parser.add_argument("--dry-run", action="store_true", help="Build and render the scene without contacting T-Rex.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import io
import pickle
import time

import numpy as np
import torch
import yaml
import zmq
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils.math import compute_pose_error, matrix_from_quat, quat_from_matrix


def resolve_path(config_path: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def jpeg_bytes(rgb: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(stream, format="JPEG", quality=90)
    return stream.getvalue()


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def rot6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    first = torch.nn.functional.normalize(value[..., 0:3], dim=-1)
    second_raw = value[..., 3:6]
    second = torch.nn.functional.normalize(second_raw - (first * second_raw).sum(-1, keepdim=True) * first, dim=-1)
    third = torch.linalg.cross(first, second)
    return torch.stack((first, second, third), dim=-1)


def make_camera(name: str, cfg: dict, camera_cfg: dict, robot_prim: str) -> Camera:
    parent = cfg["parent"]
    sensor_cfg = CameraCfg(
        prim_path=f"{robot_prim}/{parent}/{name}",
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
    return Camera(sensor_cfg)


def exact_ids(robot: Articulation, names: list[str]) -> list[int]:
    ids = []
    for name in names:
        found, matched = robot.find_joints(f"^{name}$")
        if len(found) != 1:
            raise RuntimeError(f"Expected one joint named {name!r}; matched {matched}")
        ids.append(found[0])
    return ids


def body_id(robot: Articulation, name: str) -> int:
    found, matched = robot.find_bodies(f"^{name}$")
    if len(found) != 1:
        raise RuntimeError(f"Expected one body named {name!r}; matched {matched}")
    return found[0]


def set_named_positions(robot: Articulation, mapping: dict[str, float]) -> None:
    positions = robot.data.default_joint_pos.torch.clone()
    for name, value in mapping.items():
        positions[:, exact_ids(robot, [name])[0]] = value
    robot.write_joint_position_to_sim_index(positions)
    robot.set_joint_position_target_index(positions)


def camera_rgb(camera: Camera) -> np.ndarray:
    image = camera.data.output["rgb"].torch[0].detach().cpu().numpy()
    return image[..., :3]


def dls_step(
    robot: Articulation,
    joint_ids: list[int],
    ee_body_id: int,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    damping: float,
    scale: float,
) -> torch.Tensor:
    current = robot.data.body_pose_w.torch[:, ee_body_id]
    pos_error, rot_error = compute_pose_error(current[:, :3], current[:, 3:7], target_pos, target_quat)
    error = torch.cat((pos_error, rot_error), dim=-1).unsqueeze(-1)
    jac_body_id = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
    jac_joint_ids = [joint_id + robot.num_base_dofs for joint_id in joint_ids]
    jacobian = robot.data.body_link_jacobian_w.torch[:, jac_body_id, :, jac_joint_ids]
    identity = torch.eye(6, device=robot.device).expand(jacobian.shape[0], -1, -1)
    delta = jacobian.transpose(1, 2) @ torch.linalg.solve(
        jacobian @ jacobian.transpose(1, 2) + damping**2 * identity, error
    )
    return robot.data.joint_pos.torch[:, joint_ids] + scale * delta.squeeze(-1)


def main() -> None:
    config_path = args_cli.config.resolve()
    cfg = yaml.safe_load(config_path.read_text())
    sim_cfg = cfg["simulation"]
    robot_cfg = cfg["robot"]
    inference_cfg = cfg["inference"]
    task = args_cli.task_description or inference_cfg["task_description"]
    usd_path = resolve_path(config_path, cfg["asset"]["usd_path"])
    if not usd_path.exists():
        raise FileNotFoundError(f"Missing {usd_path}; run prepare_asset.sh first")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=sim_cfg["dt"], render_interval=sim_cfg["render_interval"], device=args_cli.device)
    )
    sim.set_camera_view((2.5, 2.5, 2.0), (0.5, 0.0, 0.8))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)
    table_cfg = sim_utils.CuboidCfg(
        size=(1.2, 1.6, 0.04),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.22)),
    )
    table_cfg.func(
        "/World/Table", table_cfg, translation=(0.8, 0.0, sim_cfg["table_height"] - 0.02)
    )
    object_cfg = sim_utils.CuboidCfg(
        size=(0.06, 0.06, 0.12),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    )
    object_cfg.func(
        "/World/Object", object_cfg, translation=(0.75, 0.0, sim_cfg["table_height"] + 0.06)
    )

    articulation_cfg = ArticulationCfg(
        prim_path=cfg["asset"]["prim_path"],
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=400.0, damping=40.0, effort_limit_sim=200.0
            )
        },
    )
    robot = Articulation(articulation_cfg)
    cameras = {
        key: make_camera(f"{key}_camera", camera_cfg[key], camera_cfg, cfg["asset"]["prim_path"])
        for key in ("head", "left_wrist", "right_wrist")
        for camera_cfg in [cfg["cameras"]]
    }
    sim.reset()

    initial = {}
    initial.update(dict(zip(robot_cfg["left_arm_joints"], robot_cfg["left_arm_initial"])))
    initial.update(dict(zip(robot_cfg["right_arm_joints"], robot_cfg["right_arm_initial"])))
    initial.update(dict(zip(["torso_j1", "torso_j2", "torso_j3"], robot_cfg["torso_initial"])))
    initial.update(dict(zip(["head_j1", "head_j2", "head_j3"], robot_cfg["head_initial"])))
    set_named_positions(robot, initial)

    left_arm_ids = exact_ids(robot, robot_cfg["left_arm_joints"])
    right_arm_ids = exact_ids(robot, robot_cfg["right_arm_joints"])
    left_hand_ids = exact_ids(robot, robot_cfg["left_hand_joints"])
    right_hand_ids = exact_ids(robot, robot_cfg["right_hand_joints"])
    if len(left_hand_ids) != 22 or len(right_hand_ids) != 22:
        raise RuntimeError(f"Expected 22 Wave joints per hand, got {len(left_hand_ids)} and {len(right_hand_ids)}")
    left_ee_id = body_id(robot, robot_cfg["left_ee_body"])
    right_ee_id = body_id(robot, robot_cfg["right_ee_body"])

    for _ in range(sim_cfg["settle_steps"]):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg["dt"])
        for camera in cameras.values():
            camera.update(sim_cfg["dt"])

    if args_cli.dry_run:
        print("Scene and three RGB cameras initialized successfully.")
        return

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(inference_cfg["server_address"])
    socket.setsockopt(zmq.RCVTIMEO, inference_cfg["request_timeout_ms"])
    print(f"Connected to {inference_cfg['server_address']}; task={task!r}")

    step = 0
    while simulation_app.is_running() and step < inference_cfg["max_steps"]:
        left_pose = robot.data.body_pose_w.torch[:, left_ee_id]
        right_pose = robot.data.body_pose_w.torch[:, right_ee_id]
        state = torch.cat(
            (
                left_pose[:, :3], matrix_to_rot6d(matrix_from_quat(left_pose[:, 3:7])),
                robot.data.joint_pos.torch[:, left_hand_ids],
                right_pose[:, :3], matrix_to_rot6d(matrix_from_quat(right_pose[:, 3:7])),
                robot.data.joint_pos.torch[:, right_hand_ids],
            ), dim=-1,
        )[0].detach().cpu().numpy().astype(np.float32)
        images = {name: camera_rgb(camera) for name, camera in cameras.items()}
        crop = inference_cfg.get("head_crop_box")
        if crop:
            images["head"] = images["head"][crop[0]:crop[1], crop[2]:crop[3]]
        payload = {
            "mode": "slow",
            "task_description": task,
            "image_head": jpeg_bytes(images["head"]),
            "image_wrist_left": jpeg_bytes(images["left_wrist"]),
            "image_wrist_right": jpeg_bytes(images["right_wrist"]),
            "state_fast": state,
            "state_slow": state,
        }
        socket.send(pickle.dumps(payload))
        response = pickle.loads(socket.recv())
        if response.get("status") != "success":
            raise RuntimeError(f"T-Rex inference failed: {response}")
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.shape != (inference_cfg["chunk_size"], 62):
            raise RuntimeError(f"Expected ({inference_cfg['chunk_size']}, 62), got {actions.shape}")
        print(f"chunk at step {step}: inference={response.get('latency_ms', 0.0):.1f} ms")

        chunk_left = left_pose.clone()
        chunk_right = right_pose.clone()
        left_rotation = matrix_from_quat(chunk_left[:, 3:7])
        right_rotation = matrix_from_quat(chunk_right[:, 3:7])
        for row in actions[: inference_cfg["execute_steps_per_chunk"]]:
            action = torch.as_tensor(row, device=robot.device).unsqueeze(0)
            left_target_pos = chunk_left[:, :3] + torch.bmm(left_rotation, action[:, 0:3].unsqueeze(-1)).squeeze(-1)
            left_target_quat = quat_from_matrix(torch.bmm(left_rotation, rot6d_to_matrix(action[:, 3:9])))
            right_target_pos = chunk_right[:, :3] + torch.bmm(right_rotation, action[:, 31:34].unsqueeze(-1)).squeeze(-1)
            right_target_quat = quat_from_matrix(torch.bmm(right_rotation, rot6d_to_matrix(action[:, 34:40])))
            left_q = dls_step(robot, left_arm_ids, left_ee_id, left_target_pos, left_target_quat,
                              sim_cfg["ik_damping"], sim_cfg["ik_step_scale"])
            right_q = dls_step(robot, right_arm_ids, right_ee_id, right_target_pos, right_target_quat,
                               sim_cfg["ik_damping"], sim_cfg["ik_step_scale"])
            robot.set_joint_position_target_index(left_q, joint_ids=left_arm_ids)
            robot.set_joint_position_target_index(right_q, joint_ids=right_arm_ids)
            robot.set_joint_position_target_index(action[:, 9:31], joint_ids=left_hand_ids)
            robot.set_joint_position_target_index(action[:, 40:62], joint_ids=right_hand_ids)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_cfg["dt"])
            for camera in cameras.values():
                camera.update(sim_cfg["dt"])
            step += 1
            if step >= inference_cfg["max_steps"]:
                break

    socket.close(linger=0)
    context.term()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
