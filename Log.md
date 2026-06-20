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

