#!/usr/bin/env bash
#SBATCH --job-name=trex-convert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=a100cpu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a1865590@adelaide.edu.au

# Usage:
#   sbatch convert_dataset.sbatch smoke
#   sbatch convert_dataset.sbatch full

set -euo pipefail

MODE=${1:-smoke}
if [[ -f "${SLURM_SUBMIT_DIR}/common.sh" ]]; then
  SCRIPT_DIR=${SLURM_SUBMIT_DIR}
else
  SCRIPT_DIR=${SLURM_SUBMIT_DIR}/cluster/pickup_finetune/hpcfs
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export AV_LOG_LEVEL=error
export IMAGEIO_FFMPEG_EXE=${IMAGEIO_FFMPEG_EXE:-$(command -v ffmpeg 2>/dev/null || true)}

case "${MODE}" in
  smoke)
    MANIFEST="${PICKUP_MANIFEST_DIR:-${OUTPUT_ROOT}/manifests}/smoke_selection.json"
    OUTPUT="${SMOKE_DATASET_ROOT}"
    REPO_ID="local/trex_pickup_smoke_4"
    ;;
  full)
    MANIFEST="${PICKUP_MANIFEST_DIR:-${OUTPUT_ROOT}/manifests}/selection.json"
    OUTPUT="${PICKUP_DATASET_ROOT}"
    REPO_ID="local/trex_pickup_64"
    ;;
  *)
    echo "Usage: sbatch convert_dataset.sbatch [smoke|full]" >&2
    exit 2
    ;;
esac

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 2
fi

if [[ -e "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  if [[ ! -f "${OUTPUT}/meta/trex_norm_stats.json" ]]; then
    echo "Removing incomplete dataset at ${OUTPUT}" >&2
    rm -rf "${OUTPUT}"
  else
    echo "Dataset already complete: ${OUTPUT}" >&2
    exit 0
  fi
fi

python "${PROJECT_ROOT}/utils/convert_public_trex_to_lerobot.py" \
  --source "${PUBLIC_DATASET_SOURCE}" \
  --cache_dir "${PUBLIC_DATASET_CACHE}" \
  --episode_manifest "${MANIFEST}" \
  --instruction_override "Pick up the object." \
  --output_root "${OUTPUT}" \
  --repo_id "${REPO_ID}" \
  --max_download_gb "${MAX_DOWNLOAD_GB:-300}"

test -f "${OUTPUT}/meta/info.json"
test -f "${OUTPUT}/meta/trex_norm_stats.json"
echo "Conversion complete: ${OUTPUT}"
