# Public-Trajectory Pickup Fine-Tuning

This folder is a self-contained one-node Slurm workflow for post-training T-Rex
on 64 selected public `grasp and lift` episodes. It uses four A100 GPUs through
Accelerate + DeepSpeed ZeRO-2. It does not modify or call `scripts/train.sh`.

Experiment aim: show a bare-minimum improvement in pickup-like reaching, hand closure, and upward motion over the raw midtrain checkpoint. It is not intended to establish robust pickup success.

## 1. Configure the cluster

```bash
cd cluster/pickup_finetune
cp cluster.env.example cluster.env
$EDITOR cluster.env
```

Also edit the commented `--partition` and `--account` lines in both `.sbatch`
files if your cluster requires them. The conda environment must already contain
T-Rex, LeRobot v3, Pinocchio, and the dependencies described in the root README.

Important paths in `cluster.env`:

- `ORIGIN_MODEL_PATH`: Qwen3-VL-2B-Instruct base directory.
- `MIDTRAIN_CHECKPOINT`: released T-Rex midtrain checkpoint directory.
- `PUBLIC_DATASET_CACHE`: scratch cache for selected source shards.
- `PICKUP_DATASET_ROOT` / `SMOKE_DATASET_ROOT`: converted LeRobot outputs.
- `OUTPUT_ROOT`: training logs and checkpoints.

Budget at least 300 GB of scratch space initially. The actual required shard
size is reported before download.

## 2. Select, review, and convert

The first run downloads metadata only and writes deterministic manifests:

```bash
bash prepare_dataset.sh
```

Review `PICKUP_MANIFEST_DIR/selection.csv` and `selection.json`. Selection uses
seed 42, the `grasp and lift` primitive, a deformable/tool-term denylist,
right-hand preference, and at most four examples per canonical object where
possible. `selection.json` includes the estimated source-shard size.

When satisfied, set `CONFIRM_DOWNLOAD=1` in `cluster.env` and rerun:

```bash
bash prepare_dataset.sh
```

This writes the 64-episode dataset and a four-episode smoke dataset. Every
episode is relabelled to `Pick up the object.`. Conversion is intentionally
non-overwriting; remove an incomplete output manually only after inspecting it.

## 3. Smoke test and train

```bash
sbatch smoke.sbatch
```

The smoke job performs ten optimizer steps and saves at steps 5 and 10. Before
submitting the full run, verify its final checkpoint contains:

```text
model.pt
training_args.json
stats_data.json
processor/
```

Then submit the three-epoch run:

```bash
sbatch train.sbatch
```

The fixed configuration is microbatch 1/GPU, accumulation 2 (global batch 8),
sample stride 4, LR `2e-5`, 5% warmup, cosine decay to 10%, validation every 500
optimizer steps, and checkpoints every 1,000 steps plus every epoch. Tactile and
robot-state conditioning are disabled; FLARE remains enabled.
If the 40 GB smoke job runs out of memory, set `USE_FLARE=0` in `cluster.env`
and rerun the smoke job before changing batch or GPU count.

## 4. Compare locally in Isaac

After copying a selected checkpoint back, set `FINETUNED_CHECKPOINT`,
`MIDTRAIN_CHECKPOINT`, and local paths in `cluster.env`. For each seed, start a
server in one terminal and evaluation in another:

```bash
# Terminal 1
bash compare_local.sh server baseline 0
# Terminal 2
bash compare_local.sh eval baseline 0
```

Repeat for `finetuned` and seeds 0, 1, and 2. Each evaluation writes a
three-camera MP4 under `COMPARISON_DIR`. Restart the server for every seed so
the flow-noise sequence matches.

Blind-score videos as: 0=no directed approach, 1=directed reach, 2=closure at
the object, 3=contact followed by upward motion. Treat the pilot as improved if
fine-tuning gains at least one level in two of three matched seeds.

## Notes

- Four GPUs on one node still use distributed data parallelism; only multi-node
rendezvous is unnecessary.
- `TREX_CLUSTER_ENV=/other/path/env` can override the default `cluster.env`.
- W&B defaults to offline mode. Set `WANDB_MODE=online` only after configuring
credentials outside these scripts.

