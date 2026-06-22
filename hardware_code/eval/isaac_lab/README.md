# T-Rex checkpoint evaluation in Isaac Lab

This client runs the bimanual T-Rex policy closed-loop in one Isaac Lab scene.
It assembles the bundled Vega-1 and **22-DoF Sharpa Wave** URDFs, renders head
and wrist RGB observations, sends the standard 62-D state to the existing ZMQ
inference server, and executes returned delta-EEF/absolute-hand actions.

This is a simulation smoke-test environment, not a reproduction of a paper
task. Camera extrinsics, objects, lighting, and dynamics should be tuned to the
checkpoint's post-training data before interpreting task success.

## 1. Prepare the robot asset

```bash
cd /home/khw/RoCoIROS26/T-Rex
hardware_code/eval/isaac_lab/prepare_asset.sh
```

Set `ISAACLAB_ROOT` if Isaac Lab is elsewhere. The script builds a combined
URDF with absolute mesh paths, swaps Vega `.glb` visuals for renderable `.obj`
meshes (Isaac Sim's URDF importer skips GLB visuals), checks in neither generated
meshes nor copied third-party assets, and converts it to a fixed-base USD.
Run it again after changing the hand mount transforms or source URDFs; it removes
the stale converted USD before importing the rebuilt asset.

## 2. Start T-Rex inference

Use the Python environment in which T-Rex is installed:

```bash
cd /home/khw/RoCoIROS26/T-Rex
python scripts/test.py \
  --checkpoint_path /path/to/checkpoint \
  --dataset_name YOUR_STATS_KEY \
  --action_dim 62 --action_chunk 16 \
  --use_robot_state 1 \
  --disable_tactile 1 \
  --port 5678
```

Keep the checkpoint's normal image-size and architecture arguments if it needs
explicit overrides. `--disable_tactile 1` is intentional: the first evaluator
version has RGB and proprioception but no simulated Wave deform maps. The
server therefore returns a full action-expert chunk for `mode="slow"`.

## 3. Run the simulation

Isaac Lab runs under Isaac Sim's bundled Python (3.12), not the T-Rex training
conda env. The helper scripts unset `CONDA_PREFIX` automatically so you can
keep the inference server running in `(trex)` in another terminal.

First verify the asset, joint mapping, and cameras without the checkpoint:

```bash
hardware_code/eval/isaac_lab/run_eval.sh --dry-run --viz none
```

Then run with a GUI:

```bash
hardware_code/eval/isaac_lab/run_eval.sh \
  --task-description "Pick up the red object." --viz kit
```

For headless evaluation, use `--viz none`. Edit `trex_isaac.yaml` to change
the endpoint, camera extrinsics, table/object geometry, initial pose, episode
length, or controller gains.

## Interface details

- State order: left EEF xyz + rot6d + 22 Wave joints, then right side (62-D).
- Action order: local delta xyz + delta rot6d + 22 absolute Wave targets per side.
- Every returned action is relative to the EEF poses at that chunk's start.
- The evaluator resolves the 22 joints per hand in explicit Sharpa SDK/policy
order (not PhysX topology order) and stops if any mapped joint is absent.
- Tactile refinement is deliberately not faked with zero inputs. Add a contact
  sensor-to-F6/deform model before enabling the cascaded fast requests.

## Hand-keypoint teleoperation and collection

`teleop_sharpa_keypoints.py` retargets tracker-independent 21-point hand
keypoints to the two simulated 22-DoF Wave hands. It does not use Manus or
VIVE. The first stable open-hand frames fit a similarity transform from each
human hand into the actual simulated Wave geometry. Each later frame solves a
weighted, damped differential IK problem over PIP, DIP, and fingertip markers,
with confidence weighting, joint limits, neutral-posture regularization, and
target smoothing. Commands are actuator position targets, so PhysX contacts
remain active rather than having joint state overwritten.

Prepare the asset as above, then run:

```bash
hardware_code/eval/isaac_lab/run_keypoint_teleop.sh --viz kit
```

The default receiver listens on UDP port 7001. Each datagram is UTF-8 JSON:

```json
{
  "timestamp": 1710000000.125,
  "left": {"keypoints": [[0.0, 0.0, 0.0, 0.99]]},
  "right": {"keypoints": [[0.0, 0.0, 0.0, 0.99]]}
}
```

Each `keypoints` array must contain 21 rows in MediaPipe/OpenPose hand order.
Rows are `[x, y, z]` or `[x, y, z, confidence]`; coordinates must be metric and
all hands in a packet must use one consistent camera/world frame. The abbreviated
arrays above illustrate the schema only and are not valid packets.

Hold each required hand open and steady for `calibration_frames`. Recording
starts automatically after calibration and stops at `max_steps` or when the
application closes. Episodes are saved under `keypoint_teleop.output_dir` as:

- `trajectory.npz`: palm-normalized keypoints, exact SDK/policy-order hand
  targets, measured hand joints, object pose/velocity, and optional RGB
  observations packed as concatenated JPEG bytes plus frame offsets.
- `metadata.json`: joint order, keypoint indices, timing, and fitted calibration.

Set `required_sides: [right]` for one-handed operation. `--no-images` reduces
recording size, while `--output-dir` and `--max-steps` override their YAML
counterparts.
