## Jun 21 1700

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


Current strategy:
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

