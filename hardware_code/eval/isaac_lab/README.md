# T-Rex in Isaac Lab

This directory contains keypoint teleoperation/collection and T-Rex checkpoint
evaluation for the fixed-base DexMate Vega-1 with two 22-DoF Sharpa Wave hands.

## Keypoint retargeting quickstart

### 1. Prepare the asset once

```bash
cd /home/khw/RoCoIROS26/T-Rex
hardware_code/eval/isaac_lab/prepare_asset.sh
```

Set `ISAACLAB_ROOT` first if Isaac Lab is not at `/home/khw/IsaacLab`.

### 2. Start the D405 keypoint sender

Create a small, separate environment so MediaPipe and RealSense do not alter
the Isaac Lab environment:

```bash
cd /home/khw/RoCoIROS26/T-Rex
python3 -m venv .venv-d405
source .venv-d405/bin/activate
# python -m pip install --upgrade pip
# python -m pip install -r hardware_code/perception/requirements-d405.txt
hardware_code/perception/download_hand_landmarker.sh
python hardware_code/perception/d405_hand_keypoints.py \
  --model hardware_code/perception/models/hand_landmarker.task \
  --preview
```

Keep the hand within the D405 depth range and check that most points show in
the preview. Add `--swap-handedness` if left and right are reversed. The sender
aligns depth to RGB, lifts MediaPipe's 21 image landmarks into metric D405 XYZ,
and publishes `frame_id: "d405"` to UDP port `7001`.

For one-hand testing, change `required_sides` in `trex_isaac.yaml` to `[right]`
or `[left]`. Keep `[left, right]` for bimanual testing. Hold each required hand
open and steady for the first 30 valid packets.

### 3A. Test hand-only retargeting

The Wave fingers move while both DexMate arms remain fixed.

```bash
cd /home/khw/RoCoIROS26/T-Rex
hardware_code/eval/isaac_lab/run_keypoint_teleop.sh \
  --hand-only --viz kit \
  --output-dir /tmp/trex_hand_only
```

Confirm that each simulated fingertip follows its corresponding human
fingertip before enabling arm movement.

### 3B. Test hand + arm retargeting

First calibrate `camera_to_robot_quaternion_wxyz` for the physical D405 mount
and verify `arm_workspace` in `trex_isaac.yaml`. The identity default is only a
safe starting point for hand-only testing; arm motion uses metric camera XYZ.

```bash
hardware_code/eval/isaac_lab/run_keypoint_teleop.sh \
  --viz kit \
  --output-dir /tmp/trex_hand_and_arm
```

Palm translation/orientation drives each 7-DoF arm; palm-normalized landmarks
drive the Wave fingers.

### 4. Check both recordings

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
demonstrations. Short tracking gaps hold the previous targets; a long gap ends
the episode. Commands remain actuator targets, so PhysX contact is active.

## T-Rex checkpoint evaluation

Start the inference server in the T-Rex environment:

```bash
python scripts/test.py \
  --checkpoint_path /path/to/checkpoint \
  --dataset_name YOUR_STATS_KEY \
  --action_dim 62 --action_chunk 16 \
  --use_robot_state 1 --disable_tactile 1 \
  --port 5678
```

Then run Isaac Lab in another terminal:

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