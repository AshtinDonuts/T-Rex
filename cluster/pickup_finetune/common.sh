#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${TREX_CLUSTER_ENV:-${SCRIPT_DIR}/cluster.env}

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy cluster.env.example to cluster.env and edit it." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

required=(PROJECT_ROOT CONDA_SH CONDA_ENV HF_HOME PUBLIC_DATASET_SOURCE PUBLIC_DATASET_CACHE PICKUP_DATASET_ROOT SMOKE_DATASET_ROOT ORIGIN_MODEL_PATH MIDTRAIN_CHECKPOINT OUTPUT_ROOT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "${name} is not set in ${ENV_FILE}" >&2
    exit 2
  fi
done

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "CONDA_SH does not exist: ${CONDA_SH}" >&2
  exit 2
fi
if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "PROJECT_ROOT does not exist: ${PROJECT_ROOT}" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export HF_HOME
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}

mkdir -p "${HF_HOME}" "${OUTPUT_ROOT}" "${PUBLIC_DATASET_CACHE}"
