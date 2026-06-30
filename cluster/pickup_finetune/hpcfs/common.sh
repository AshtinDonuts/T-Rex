#!/usr/bin/env bash
# hpcfs-specific common setup sourced by train_pickup.sh and prepare_dataset.sh.
# Key differences from the upstream cluster/pickup_finetune/common.sh:
#   - Uses `module load` to initialise conda (hpcfs provides Anaconda as a module).
#   - strict-mode (set -euo pipefail) is placed AFTER conda activation to avoid
#     conda's own init scripts tripping on -u (unset variable) checks.
#   - Adds PYTHONUNBUFFERED, OMP_NUM_THREADS, MKL_NUM_THREADS for HPC correctness.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${TREX_CLUSTER_ENV:-${SCRIPT_DIR}/cluster.env}

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy cluster.env.example to cluster.env and edit it." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

required=(PROJECT_ROOT CONDA_ENV HF_HOME PUBLIC_DATASET_SOURCE PUBLIC_DATASET_CACHE PICKUP_DATASET_ROOT SMOKE_DATASET_ROOT ORIGIN_MODEL_PATH MIDTRAIN_CHECKPOINT OUTPUT_ROOT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "${name} is not set in ${ENV_FILE}" >&2
    exit 2
  fi
done

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "PROJECT_ROOT does not exist: ${PROJECT_ROOT}" >&2
  exit 2
fi

# Load Anaconda and initialise the conda shell function (hpcfs module system).
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# hpcfs non-interactive shells may keep base python on PATH after conda activate.
if [[ -x "${CONDA_ENV}/bin/python" ]]; then
  export PATH="${CONDA_ENV}/bin:${PATH}"
fi

# Strict mode after conda activation (conda deactivate scripts don't handle -u cleanly).
set -euo pipefail

export HF_HOME
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}

mkdir -p "${HF_HOME}" "${OUTPUT_ROOT}" "${PUBLIC_DATASET_CACHE}"
