# T-Rex in Isaac Lab

This directory contains keypoint teleoperation/collection and T-Rex checkpoint
evaluation for the fixed-base DexMate Vega-1 with two 22-DoF Sharpa Wave hands.

## Keypoint retargeting quickstart

### 1. Prepare the asset once

Run this in any terminal, then wait for it to finish:

```bash
cd /home/khw/RoCoIROS26/T-Rex
hardware_code/eval/isaac_lab/prepare_asset.sh
```

Set `ISAACLAB_ROOT` first if Isaac Lab is not at `/home/khw/IsaacLab`.

### 2. Start the D405 keypoint sender

Create a small, separate environment so MediaPipe and RealSense do not alter
the Isaac Lab environment. Run this in **Terminal 1** and leave the sender
running during either retargeting test:

```bash
cd /home/khw/RoCoIROS26/T-Rex
python3 -m venv .venv-d405
source .venv-d405/bin/activate
# python -m pip install --upgrade pip
# python -m pip install -r hardware_code/perception/requirements-d405.txt
hardware_code/perception/download_hand_landmarker.sh
python hardware_code/perception/d405_hand_keypoints.py \
  --model hardware_code/perception/models/hand_landmarker.task \
  --diagnostics
```

Keep the hand within the D405 depth range and check that most points show in
the preview. Add `--swap-handedness` if left and right are reversed. The sender
aligns depth to RGB, lifts MediaPipe's 21 image landmarks into metric D405 XYZ,
and publishes `frame_id: "d405"` to UDP port `7001`.

Diagnostic mode shows RGB beside aligned depth. Fingertip labels are
`landmark_index:depth_mm`; red segments indicate an unusually long 3D bone or
large depth discontinuity, often caused by sampling the background.

For one-hand testing, pass `--right-only` or `--left-only` to the Isaac command;
these override `required_sides` in `trex_isaac.yaml` for that run. Without either
flag, the YAML setting applies. After all required hands are visible, Isaac
prints a three-second countdown. Keep them open and still during the countdown
and the following 30-frame capture; the console prints progress, pause/resume
events, completion, and recording start.

### 3A. Test hand-only retargeting

With the D405 sender still running in Terminal 1, open **Terminal 2** and run
the following. The Wave fingers move while both DexMate arms remain fixed.

```bash
cd /home/khw/RoCoIROS26/T-Rex
hardware_code/eval/isaac_lab/run_keypoint_teleop.sh \
  --hand-only --right-only --diagnostics --viz kit \
  --output-dir /tmp/trex_hand_only
```

Replace `--right-only` with `--left-only` for the left hand, or remove it for
bimanual calibration. `--hand-only` keeps both DexMate arms fixed; the side flag
selects which Sharpa hand is calibrated and retargeted.

Confirm that each simulated fingertip follows its corresponding human
fingertip before enabling arm movement. The console reports total and
per-finger marker errors once per second, which separates a bad camera lift
from a finger-specific Sharpa fit or IK problem. In Isaac, `--diagnostics`
also removes the table and rigid cube, leaving only the robot and ground plane.

### 3B. Test hand + arm retargeting

Stop the hand-only run in Terminal 2 before starting this test; do not run 3A
and 3B simultaneously. Keep the D405 sender running in Terminal 1.

First calibrate by holding each required hand near the matching DexMate flange
during the countdown; the sim learns a per-arm camera-to-robot rotation from the
paired wrist and EEF poses. Override with `arm_align_camera_at_calibration: false`
and set `camera_to_robot_quaternion_wxyz` manually if needed. Verify
`arm_workspace` in `trex_isaac.yaml`.

```bash
hardware_code/eval/isaac_lab/run_keypoint_teleop.sh \
  --viz kit \
  --output-dir /tmp/trex_hand_and_arm
```

Palm translation/orientation drives each 7-DoF arm; palm-normalized landmarks
drive the Wave fingers. Arm retargeting uses the tracked **wrist position** and
MCP-derived **palm orientation** in the D405 frame, mapped relatively onto
`left_hand_flange` / `right_hand_flange`.

### 4. Check both recordings

After each Isaac run has stopped, run this in either terminal:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

for root in (Path('/tmp/trex_hand_only'), Path('/tmp/trex_hand_and_arm')):
    episode = sorted(root.glob('episode_*'))[-1]
    data = np.load(episode / 'trajectory.npz')
    print(root.name, data['observation_state'].shape, data['action'].shape)
PY
```

Both arrays should be `(frames, 58)` in
`[L arm 7 | L hand 22 | R arm 7 | R hand 22]` order. Add `--no-images` for a
smaller smoke test or `--max-steps N` for a shorter run.

## Keypoint configuration

Important settings are under `keypoint_teleop` in `trex_isaac.yaml`:

- `required_sides`: hands required to complete calibration.
- `keypoint_frame_id`: accepted source coordinate frame.
- `camera_to_robot_quaternion_wxyz`: source-frame to robot-base rotation.
- `arm_workspace`: permitted EEF translation bounds.
- `calibration_frames`, filtering, IK, smoothing, and timeout parameters.

The default 60 Hz simulation is recorded every second step, producing 30 Hz
demonstrations. Short tracking gaps hold the previous targets. After
`stale_hold_timeout_s`, every robot joint is frozen at its current position and
the episode remains active. Retargeting resumes automatically when valid input
returns; warnings repeat every `stale_report_interval_s`. Commands remain
actuator targets during normal tracking, so PhysX contact is active.

## T-Rex checkpoint evaluation

In **Terminal 1**, start the inference server in the T-Rex environment and
leave it running:

```bash
python scripts/test.py \
  --checkpoint_path /path/to/checkpoint \
  --dataset_name YOUR_STATS_KEY \
  --action_dim 62 --action_chunk 16 \
  --use_robot_state 1 --disable_tactile 1 \
  --port 5678
```

In **Terminal 2**, run Isaac Lab:

```bash
# Asset/camera smoke test
hardware_code/eval/isaac_lab/run_eval.sh --dry-run --viz none

# Closed-loop policy evaluation
hardware_code/eval/isaac_lab/run_eval.sh \
  --task-description "Pick up the red object." --viz kit
```

The evaluator uses a 62-D interface per step: left EEF xyz + rot6d + 22 hand
joints, followed by the right side. Returned actions contain local EEF deltas
and absolute hand targets. This environment has RGB and proprioception but no
simulated Wave tactile/deformation observations, hence `--disable_tactile 1`.
