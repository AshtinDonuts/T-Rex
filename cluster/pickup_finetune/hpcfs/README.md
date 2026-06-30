# Pickup Fine-Tuning — University of Adelaide hpcfs

This folder is a drop-in replacement for `cluster/pickup_finetune/` tuned for the
University of Adelaide **hpcfs** cluster. It is otherwise functionally identical to
the upstream workflow described in `cluster/pickup_finetune/README.md`.

## Differences from the upstream scripts

| Topic | Upstream (`pickup_finetune/`) | This folder (`hpcfs/`) |
|---|---|---|
| Conda init | `source "${CONDA_SH}"` (path var) | `module load Anaconda3/2025.06-1` then auto-derives conda base |
| `CONDA_SH` in `cluster.env` | Required | **Not used** — removed from required list |
| `set -euo pipefail` | Top of every script | **After** `conda activate` to avoid conda `-u` failures |
| Extra env vars | — | `PYTHONUNBUFFERED=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` |
| Partition | `##SBATCH --partition=EDIT_ME` | `#SBATCH --partition=a100` |
| GPU GRES | `--gres=gpu:a100:4` (typed) | `--gres=gpu:4` (generic, matches hpcfs `sinfo` naming) |
| Separate error log | No `--error` directive | `--error=%x-%j.err` |
| Email | Not set | `--mail-type=END,FAIL` + `a1865590@adelaide.edu.au` |
| `prepare_dataset.sh` | References `${SCRIPT_DIR}/select_episodes.py` | References `${SCRIPT_DIR}/../select_episodes.py` (parent folder) |

## 1. Configure

```bash
cd cluster/pickup_finetune/hpcfs
cp cluster.env.example cluster.env
$EDITOR cluster.env
```

All paths in `cluster.env.example` are pre-filled for `a1865590`. Verify that:

- `PROJECT_ROOT` points to your T-Rex checkout.
- `CONDA_ENV` is the name (or absolute path) of your conda environment that has T-Rex,
  LeRobot v3, Pinocchio, and their dependencies installed.
- All `scratch/` paths exist or will be created automatically by the scripts.

> **GPU GRES note:** The `.sbatch` files use `--gres=gpu:4`. If `sinfo -o "%P %G"` shows
> a typed GRES (e.g. `gpu:a100:4`) you must change the directive accordingly.

## 2. Select, review, and convert the dataset

```bash
bash prepare_dataset.sh
```

This calls `cluster/pickup_finetune/select_episodes.py` (no duplication). Review
`PICKUP_MANIFEST_DIR/selection.csv` and `selection.json`, then set `CONFIRM_DOWNLOAD=1`
in `cluster.env` and rerun to download and convert:

```bash
bash prepare_dataset.sh
```

## 3. Smoke test

```bash
sbatch smoke.sbatch
```

Ten optimizer steps, saves at steps 5 and 10. Confirm the checkpoint contains
`model.pt`, `training_args.json`, `stats_data.json`, and `processor/` before
proceeding.

If the smoke job runs out of memory, set `USE_FLARE=0` in `cluster.env` and resubmit.

## 4. Full training run

```bash
sbatch train.sbatch
```

Three epochs, microbatch 1/GPU, global batch 8, LR `2e-5`. Wall time limit is 48 h.
Email notifications are sent on job completion or failure.

## 5. Monitor

```bash
squeue -u a1865590
tail -f trex-pickup-<JOBID>.out
tail -f trex-pickup-<JOBID>.err
```

W&B is set to `offline` by default. To sync after the job completes:

```bash
wandb sync <OUTPUT_ROOT>/<EXPERIMENT_NAME>/wandb/offline-run-*
```

## Notes

- `TREX_CLUSTER_ENV=/other/path/env` overrides the default `cluster.env` lookup.
- `PRINT_COMMAND=1 bash train_pickup.sh full` dry-runs and prints the full accelerate
  command without executing it.
- Files not present in this folder (`select_episodes.py`, `test_helpers.py`,
  `compare_local.sh`) live in the parent `cluster/pickup_finetune/` directory and are
  referenced directly; they need no adaptation for hpcfs.
