"""Hand-keypoint preprocessing shared by the Isaac Wave teleoperation client.

The wire format is deliberately tracker-agnostic. A UDP datagram is JSON with
``left`` and/or ``right`` entries, each either a 21x3/21x4 array or an object
containing ``keypoints``. Keypoints use the MediaPipe/OpenPose ordering and may
include confidence as the fourth value.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


WRIST = 0
MCP_INDICES = np.array([1, 5, 9, 13, 17])
PIP_INDICES = np.array([2, 6, 10, 14, 18])
DIP_INDICES = np.array([3, 7, 11, 15, 19])
TIP_INDICES = np.array([4, 8, 12, 16, 20])
# Target order matches the simulated body-name order: PIP, DIP, tip per finger.
TRACKED_INDICES = np.stack((PIP_INDICES, DIP_INDICES, TIP_INDICES), axis=1).reshape(-1)


@dataclass(frozen=True)
class HandSample:
    xyz: np.ndarray
    confidence: np.ndarray

    def __post_init__(self) -> None:
        if self.xyz.shape != (21, 3):
            raise ValueError(f"Expected keypoints with shape (21, 3), got {self.xyz.shape}")
        if self.confidence.shape != (21,):
            raise ValueError(f"Expected confidence with shape (21,), got {self.confidence.shape}")
        if not np.isfinite(self.xyz).all() or not np.isfinite(self.confidence).all():
            raise ValueError("Keypoints and confidence must be finite")


@dataclass(frozen=True)
class KeypointPacket:
    timestamp: float
    frame_id: str
    hands: dict[str, HandSample]


def _parse_hand(value: Any) -> HandSample:
    if isinstance(value, dict):
        value = value.get("keypoints")
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (21, 3):
        return HandSample(array.copy(), np.ones(21, dtype=np.float64))
    if array.shape == (21, 4):
        return HandSample(array[:, :3].copy(), array[:, 3].copy())
    raise ValueError(f"Expected a 21x3 or 21x4 keypoint array, got {array.shape}")


def parse_packet(payload: bytes | str | dict[str, Any]) -> KeypointPacket:
    """Parse one tracker-independent bimanual keypoint packet."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("Keypoint packet must be a JSON object")
    hands = {side: _parse_hand(payload[side]) for side in ("left", "right") if side in payload}
    if not hands:
        raise ValueError("Keypoint packet contains neither 'left' nor 'right'")
    return KeypointPacket(
        float(payload.get("timestamp", time.time())), str(payload.get("frame_id", "camera")), hands
    )


class UdpKeypointReceiver:
    """Background UDP receiver retaining only the newest valid packet."""

    def __init__(self, host: str, port: int, timeout_s: float):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._latest: KeypointPacket | None = None
        self._arrival_time = 0.0
        self._sequence = 0
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def start(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((self.host, self.port))
        self._socket.settimeout(0.2)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                payload, _ = self._socket.recvfrom(1 << 20)
                packet = parse_packet(payload)
                with self._lock:
                    self._latest = packet
                    self._arrival_time = time.monotonic()
                    self._sequence += 1
                    self._error = None
            except socket.timeout:
                continue
            except Exception as exc:  # malformed input must not kill simulation
                with self._lock:
                    self._error = str(exc)

    def latest(self) -> tuple[KeypointPacket | None, int, float, str | None]:
        with self._lock:
            age = time.monotonic() - self._arrival_time if self._latest is not None else np.inf
            packet = self._latest if age <= self.timeout_s else None
            return packet, self._sequence, float(age), self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._socket is not None:
            self._socket.close()


class KeypointFilter:
    """Confidence gate, bounded-jump rejection and exponential smoothing."""

    def __init__(self, alpha: float, min_confidence: float, max_jump_m: float):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.max_jump_m = max_jump_m
        self._state: dict[str, np.ndarray] = {}
        self._valid_state: dict[str, np.ndarray] = {}

    def update(self, side: str, sample: HandSample) -> HandSample | None:
        valid = sample.confidence >= self.min_confidence
        if valid.sum() < 16 or not valid[[WRIST, 5, 9, 13, 17]].all():
            return None
        previous = self._state.get(side)
        current = sample.xyz.copy()
        if previous is not None:
            previous_valid = self._valid_state[side]
            jumps = np.linalg.norm(current - previous, axis=1)
            newly_valid = valid & ~previous_valid
            accept = newly_valid | (valid & (jumps <= self.max_jump_m))
            current[~accept] = previous[~accept]
            smoothed = self.alpha * current + (1.0 - self.alpha) * previous
            smoothed[newly_valid] = current[newly_valid]
            current = smoothed
            valid = previous_valid | valid
        self._state[side] = current
        self._valid_state[side] = valid.copy()
        return HandSample(current.copy(), sample.confidence.copy())


def palm_normalize(sample: HandSample) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return keypoints in a scale-free palm frame plus frame pose and width.

    The frame is computed from wrist/index/middle/pinky MCPs. A later geometric
    calibration maps this convention into each simulated hand, so no camera or
    handedness-specific axis constants are needed here.
    """
    points = sample.xyz
    origin = points[WRIST]
    across = points[5] - points[17]
    forward = points[9] - origin
    width = float(np.linalg.norm(across))
    if width < 1e-4:
        raise ValueError("Degenerate palm width")
    x_axis = across / width
    forward = forward - np.dot(forward, x_axis) * x_axis
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-4:
        raise ValueError("Degenerate palm direction")
    y_axis = forward / forward_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    local = (points - origin) @ rotation / width
    return local, origin.copy(), rotation, width


@dataclass(frozen=True)
class PalmPose:
    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        if self.position.shape != (3,) or self.rotation.shape != (3, 3):
            raise ValueError("PalmPose expects position (3,) and rotation (3, 3)")
        if not np.isfinite(self.position).all() or not np.isfinite(self.rotation).all():
            raise ValueError("Palm pose must be finite")


def palm_pose(sample: HandSample) -> PalmPose:
    """Estimate a 6-DoF palm pose from confidence-gated palm landmarks."""
    _, wrist, rotation, _ = palm_normalize(sample)
    palm_ids = np.array([WRIST, 5, 9, 13, 17])
    weights = np.maximum(sample.confidence[palm_ids], 1e-6)
    # Bias toward the wrist while reducing single-landmark depth noise.
    points = sample.xyz[palm_ids]
    center = 0.5 * wrist + 0.5 * np.average(points, axis=0, weights=weights)
    return PalmPose(center, rotation)


def project_rotation(matrix: np.ndarray) -> np.ndarray:
    """Project a noisy matrix onto SO(3)."""
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def average_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError("At least one rotation is required")
    return project_rotation(np.mean(rotations, axis=0))


def _rotation_angle(rotation: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))


def _interpolate_rotation(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    relative = first.T @ second
    angle = _rotation_angle(relative)
    if angle < 1e-8:
        return first.copy()
    skew = (relative - relative.T) / (2.0 * np.sin(angle))
    scaled = alpha * angle
    incremental = np.eye(3) + np.sin(scaled) * skew + (1.0 - np.cos(scaled)) * (skew @ skew)
    return project_rotation(first @ incremental)


class PalmPoseFilter:
    """Bounded-jump SE(3) filter for vision-derived palm poses."""

    def __init__(self, position_alpha: float, rotation_alpha: float, max_jump_m: float, max_jump_rad: float):
        self.position_alpha = position_alpha
        self.rotation_alpha = rotation_alpha
        self.max_jump_m = max_jump_m
        self.max_jump_rad = max_jump_rad
        self._state: dict[str, PalmPose] = {}

    def update(self, side: str, pose: PalmPose) -> PalmPose | None:
        previous = self._state.get(side)
        if previous is None:
            self._state[side] = pose
            return pose
        if np.linalg.norm(pose.position - previous.position) > self.max_jump_m:
            return None
        if _rotation_angle(previous.rotation.T @ pose.rotation) > self.max_jump_rad:
            return None
        filtered = PalmPose(
            self.position_alpha * pose.position + (1.0 - self.position_alpha) * previous.position,
            _interpolate_rotation(previous.rotation, pose.rotation, self.rotation_alpha),
        )
        self._state[side] = filtered
        return filtered


def relative_pose_target(
    current: PalmPose,
    initial: PalmPose,
    initial_target: PalmPose,
    camera_to_robot_rotation: np.ndarray,
    translation_scale: float = 1.0,
) -> PalmPose:
    """Map relative camera-frame palm motion onto an initial robot-frame target."""
    camera_to_robot_rotation = project_rotation(np.asarray(camera_to_robot_rotation, dtype=np.float64))
    delta_position = camera_to_robot_rotation @ (current.position - initial.position)
    delta_rotation_camera = current.rotation @ initial.rotation.T
    delta_rotation_robot = (
        camera_to_robot_rotation @ delta_rotation_camera @ camera_to_robot_rotation.T
    )
    return PalmPose(
        initial_target.position + translation_scale * delta_position,
        project_rotation(delta_rotation_robot @ initial_target.rotation),
    )


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError("Quaternion must have shape (4,)")
    quaternion = quaternion / np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * points @ self.rotation.T + self.translation


def fit_similarity(source: np.ndarray, target: np.ndarray) -> SimilarityTransform:
    """Least-squares Umeyama similarity mapping row-vector source to target."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Similarity inputs must have matching (N, 3) shapes, got {source.shape}, {target.shape}")
    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src_centered = source - src_mean
    dst_centered = target - dst_mean
    covariance = dst_centered.T @ src_centered / source.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(src_centered**2, axis=1)))
    if variance < 1e-10:
        raise ValueError("Cannot calibrate from coincident source points")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    return SimilarityTransform(scale, rotation, translation)


def tracked_points(local_keypoints: np.ndarray) -> np.ndarray:
    return np.asarray(local_keypoints, dtype=np.float64)[TRACKED_INDICES]


def assemble_trex_vector(
    left_arm: np.ndarray,
    left_hand: np.ndarray,
    right_arm: np.ndarray,
    right_hand: np.ndarray,
) -> np.ndarray:
    """Assemble canonical T-Rex `[L arm|L hand|R arm|R hand]` 58-D data."""
    values = [
        np.asarray(left_arm),
        np.asarray(left_hand),
        np.asarray(right_arm),
        np.asarray(right_hand),
    ]
    expected = [(7,), (22,), (7,), (22,)]
    if [value.shape for value in values] != expected:
        raise ValueError(
            f"Expected component shapes {expected}, got {[value.shape for value in values]}"
        )
    return np.concatenate(values)
