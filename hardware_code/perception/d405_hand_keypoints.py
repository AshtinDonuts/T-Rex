#!/usr/bin/env python3
"""Extract metric 21-point hand landmarks from a RealSense D405 and publish UDP JSON."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def robust_depth_m(
    depth_m: np.ndarray,
    u: int,
    v: int,
    radius: int,
    min_depth_m: float,
    max_depth_m: float,
    min_valid_pixels: int,
) -> float | None:
    """Median valid depth in a clipped square neighborhood."""
    height, width = depth_m.shape
    x0, x1 = max(0, u - radius), min(width, u + radius + 1)
    y0, y1 = max(0, v - radius), min(height, v + radius + 1)
    values = depth_m[y0:y1, x0:x1]
    valid = values[(values >= min_depth_m) & (values <= max_depth_m) & np.isfinite(values)]
    if valid.size < min_valid_pixels:
        return None
    return float(np.median(valid))


def lift_landmarks(
    normalized_xy: np.ndarray,
    depth_m: np.ndarray,
    deproject: Callable[[Sequence[float], float], Sequence[float]],
    detection_confidence: float,
    depth_radius: int,
    min_depth_m: float,
    max_depth_m: float,
    min_valid_pixels: int,
) -> np.ndarray:
    """Lift 21 normalized image landmarks to metric camera-frame XYZC."""
    normalized_xy = np.asarray(normalized_xy, dtype=np.float64)
    if normalized_xy.shape != (21, 2):
        raise ValueError(f"Expected normalized landmarks (21, 2), got {normalized_xy.shape}")
    height, width = depth_m.shape
    output = np.zeros((21, 4), dtype=np.float64)
    for index, (x, y) in enumerate(normalized_xy):
        u = int(np.clip(round(x * (width - 1)), 0, width - 1))
        v = int(np.clip(round(y * (height - 1)), 0, height - 1))
        depth = robust_depth_m(
            depth_m, u, v, depth_radius, min_depth_m, max_depth_m, min_valid_pixels
        )
        if depth is None:
            continue
        output[index, :3] = np.asarray(deproject((float(u), float(v)), depth), dtype=np.float64)
        output[index, 3] = detection_confidence
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="MediaPipe hand_landmarker.task")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7001)
    parser.add_argument("--frame-id", default="d405")
    parser.add_argument("--serial", default=None, help="Optional RealSense serial number")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth-radius", type=int, default=3)
    parser.add_argument("--min-valid-depth-pixels", type=int, default=5)
    parser.add_argument("--min-depth-m", type=float, default=0.07)
    parser.add_argument("--max-depth-m", type=float, default=0.50)
    parser.add_argument("--min-valid-landmarks", type=int, default=16)
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--presence-confidence", type=float, default=0.5)
    parser.add_argument("--tracking-confidence", type=float, default=0.5)
    parser.add_argument("--swap-handedness", action="store_true")
    parser.add_argument("--no-align", action="store_true", help="Skip depth-to-color alignment")
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Missing MediaPipe model: {args.model}")
    try:
        import cv2
        import mediapipe as mp
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "Missing D405 perception dependencies. Install requirements-d405.txt in a separate venv."
        ) from exc

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    profile = pipeline.start(config)
    device = profile.get_device()
    product = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = None if args.no_align else rs.align(rs.stream.color)
    print(f"RealSense: {product}, serial={serial}, depth_scale={depth_scale:g} m/unit")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model.resolve())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=args.detection_confidence,
        min_hand_presence_confidence=args.presence_confidence,
        min_tracking_confidence=args.tracking_confidence,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destination = (args.host, args.port)
    previous_timestamp_ms = -1
    sent = 0
    report_start = time.monotonic()

    try:
        # Let auto-exposure settle before inference/calibration packets begin.
        for _ in range(20):
            pipeline.wait_for_frames()
        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            color_rgb = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            if color_rgb.shape[:2] != depth_raw.shape:
                raise RuntimeError(
                    f"Color/depth shapes differ ({color_rgb.shape[:2]} vs {depth_raw.shape}); "
                    "enable alignment or choose matching D405 profiles."
                )
            depth_m = depth_raw.astype(np.float32) * depth_scale
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            timestamp_ms = max(previous_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
            previous_timestamp_ms = timestamp_ms
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_rgb)
            result = landmarker.detect_for_video(image, timestamp_ms)

            packet: dict[str, object] = {
                "timestamp": time.time(),
                "frame_id": args.frame_id,
            }
            selected_scores: dict[str, float] = {}
            preview = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR) if args.preview else None
            for hand_index, landmarks in enumerate(result.hand_landmarks):
                category = result.handedness[hand_index][0]
                side = category.category_name.lower()
                if args.swap_handedness:
                    side = "left" if side == "right" else "right"
                normalized_xy = np.asarray([[point.x, point.y] for point in landmarks])
                keypoints = lift_landmarks(
                    normalized_xy,
                    depth_m,
                    lambda pixel, depth: rs.rs2_deproject_pixel_to_point(
                        intrinsics, list(pixel), depth
                    ),
                    float(category.score),
                    args.depth_radius,
                    args.min_depth_m,
                    args.max_depth_m,
                    args.min_valid_depth_pixels,
                )
                valid_count = int(np.count_nonzero(keypoints[:, 3] > 0.0))
                if valid_count < args.min_valid_landmarks:
                    continue
                # If duplicate labels occur, retain the higher-confidence detection.
                if side not in selected_scores or category.score > selected_scores[side]:
                    packet[side] = {"keypoints": keypoints.tolist()}
                    selected_scores[side] = float(category.score)
                if preview is not None:
                    color = (80, 220, 80) if side == "right" else (220, 160, 40)
                    image_height, image_width = color_rgb.shape[:2]
                    for point, confidence in zip(normalized_xy, keypoints[:, 3]):
                        if confidence > 0:
                            pixel = (int(point[0] * image_width), int(point[1] * image_height))
                            cv2.circle(preview, pixel, 3, color, -1)
                    cv2.putText(
                        preview, f"{side} {valid_count}/21", (10, 30 + 30 * hand_index),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    )
            if "left" in packet or "right" in packet:
                udp.sendto(json.dumps(packet, separators=(",", ":")).encode("utf-8"), destination)
                sent += 1

            if preview is not None:
                cv2.imshow("D405 hand keypoints (q/esc to quit)", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            now = time.monotonic()
            if now - report_start >= 2.0:
                print(f"publishing {sent / (now - report_start):.1f} packets/s to udp://{args.host}:{args.port}")
                sent = 0
                report_start = now
    finally:
        landmarker.close()
        pipeline.stop()
        udp.close()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
