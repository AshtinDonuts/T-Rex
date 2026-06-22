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
    return KeypointPacket(float(payload.get("timestamp", time.time())), hands)


class UdpKeypointReceiver:
    """Background UDP receiver retaining only the newest valid packet."""

    def __init__(self, host: str, port: int, timeout_s: float):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._latest: KeypointPacket | None = None
        self._arrival_time = 0.0
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
                    self._error = None
            except socket.timeout:
                continue
            except Exception as exc:  # malformed input must not kill simulation
                with self._lock:
                    self._error = str(exc)

    def latest(self) -> tuple[KeypointPacket | None, float, str | None]:
        with self._lock:
            age = time.monotonic() - self._arrival_time if self._latest is not None else np.inf
            packet = self._latest if age <= self.timeout_s else None
            return packet, float(age), self._error

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

    def update(self, side: str, sample: HandSample) -> HandSample | None:
        valid = sample.confidence >= self.min_confidence
        if valid.sum() < 16 or not valid[[WRIST, 5, 9, 13, 17]].all():
            return None
        previous = self._state.get(side)
        current = sample.xyz.copy()
        if previous is not None:
            jumps = np.linalg.norm(current - previous, axis=1)
            accept = valid & (jumps <= self.max_jump_m)
            current[~accept] = previous[~accept]
            current = self.alpha * current + (1.0 - self.alpha) * previous
        self._state[side] = current
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

