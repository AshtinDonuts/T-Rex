## Jun 22 1700

Dealing with:

1. Early exit from lost tracking (i.e. disable the exit early behavior)
2. Setting up a reference pose that corresponds to robot starting wrist pose (position + orientation).
* capture an image as reference pose


## Jun 22 1630

Fixed robot body from drooping in hand-only mode teleop.

In progress:
Debug why hand pose detection is worse when palm faces camera.
Calibrate -> natural robot teleop pose.


## Jun 22 1430

Implemented tracker-free D405 teleoperation for collecting Isaac demonstrations.

Pipeline:

1. RealSense D405 captures aligned RGB and metric depth.
2. MediaPipe extracts 21 hand landmarks from RGB.
3. Local median depth lifts each landmark into D405 camera-frame XYZ.
4. Keypoints are sent over UDP to Isaac Lab.
5. Palm-normalized landmarks control the 22-DoF Sharpa Wave hands; global palm motion optionally controls the 7-DoF DexMate arms through IK.

Hand-only and hand+arm modes are separate. Hand-only does not require camera extrinsic calibration. Hand+arm requires calibrating the D405-to-robot-base rotation before collecting demonstrations.

Dry-run results:

- D405, MediaPipe, Isaac USD, 70 robot joints, and three RGB cameras initialized successfully.
- Both retargeting modes completed calibration and produced valid 32-frame, 58-D recordings.
- Hand+arm IK had approximately 4.9 mm maximum position error in the synthetic test.

Remaining live checks are handedness, D405 tracking quality/occlusion, and camera-to-robot rotation. PhysX also reports invalid mass/inertia on `arm_center`; this should be fixed before relying on contact-dynamics accuracy.


## Jun 21 1700

Current Strategy:

1. Pilot with 50–100 related public episodes to validate conversion and training.
2. Collect 100–300 successful Isaac demonstrations of the exact task, including varied initial poses, colors, distractors, and grasp approaches.
3. Fine-tune from the released midtrain checkpoint on the combined dataset.
4. Evaluate on held-out object poses/scenes, not merely random frames.

#### How to collect the successful demos?

With limited time, we have 1 viable option.

Teleoperation: VR trackers/SpaceMouse for EEF control, with predefined hand re-targeting or glove-based finger retargeting.

Since we don't have gloves, let's do WILOR-like hand retargeting.


## Jun 21 1645

Beginning work to fine-tune on T-Rex.

Currently dealing with format mismatch between the author provided dataset and Dataloader expected format.

This mismatch is noted by the HF public dataset card
`https://huggingface.co/datasets/zekaiwang/trex_dataset`

#### Format mismatch 
Schema:
* 58D joint state and target
* Single-step absolute joint actions
* Original image / tactile field names

Post-training loader
* 62D EEF + hand state 
* [16, 62] local delta-EEF action chunks
* renamed camera/tactile field names
* meta/trex_norm_stats.json


High-level WorkFlow:

1. Inspect `meta/episodes/*.parquet` and select episodes by:
`motor_primitive`
`object`
`caption`
`target`

2. Convert selected episodes into T-Rex’s canonical post-training representation:
    
- use Vega/Sharpa forward kinematics to convert each 7-D arm configuration to 9-D EEF pose;
- build 16-step local delta-EEF chunks with `build_action_chunk`;
- retain the 22 hand targets per hand;
- rename/copy RGB, wrench and deform streams;
- calculate the required q01/q99 normalization sidecar.

The existing `convert_inlab_to_lerobot.py` contains most transformation logic, but it expects the authors’ raw HDF5 recording layout—not the released LeRobot dataset. A small public-dataset adapter is therefore needed.

3. Configure `train.sh`


Note:
The authors’ data can provide useful manipulation priors, but it probably will not produce a successful Isaac policy by itself because it contains real-camera Vega demonstrations, while your environment has different rendering, object geometry, control timing, and no simulated tactile input. Your own Isaac demonstrations should be the final post-training dataset; mixing a related authors’ subset with Isaac demonstrations may be useful.


## Jun 20 1700

#### Debugging embodiment gap

`use_robot_state=0` means the policy does not receive proprioception as an input token.

The midtrain checkpoint was trained this way: [training_args.json (line 5)](/home/khw/RoCoIROS26/T-Rex/checkpoints/trex_midtrain/training_args.json:5).

Incorrectly using `--use_robot_state 1.` creates a state embedder absent from the checkpoint, likely with randomly initialized weights.

#### Camera mismatch




#### Reference official scripts
Original real-robot evaluator is `eval_trex_async.py`
Inference server, `scripts/test.py`.


## Jun 20 16:45

Mid-Training checkpoint eval currently unsuccessful on "pick up red box" task.

#### Main Causes:
trex_midtrain is not task-specific. It is intended as the starting point for post-training, not direct evaluation on “Pick up the red object.”

The checkpoint was trained with "use_robot_state": 0 ([training_args.json (line 5)](/home/khw/RoCoIROS26/T-Rex/checkpoints/trex_midtrain/training_args.json:5)). Starting the server with --use_robot_state 1 creates an architecture mismatch and potentially leaves a new state embedder randomly initialized. Use --use_robot_state 0.

Isaac disables tactile refinement, while this midtrain checkpoint is tactile-reactive. It therefore runs only the action-expert ablation.

Camera poses, optics, lighting, textures, and object geometry are rough approximations. This visual domain gap is probably substantial.

Control differs: hardware executes at 30 Hz using Pink IK plus a 300 Hz smoothing/safety loop; Isaac executes at 60 Hz with one damped-least-squares update per action. This changes trajectory timing and tracking.

All 16 predictions are executed open-loop before acquiring another observation.


----

## Jun 20 16:15

Debugged Sharpa / DexMate Asset URDF structure in Isaac Sim.
The major issues were the orientation when attaching Sharpa Wave 22DoF hands to the DexMate arms.

----

## Jun 20 15:30

Created isaac-lab supported eval module to test mid-training T-Rex checkpoint.

----

## Jun 20 15:00

Forked Repo under AshtinDonuts.
