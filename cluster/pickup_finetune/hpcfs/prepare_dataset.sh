#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

# select_episodes.py lives in the parent pickup_finetune folder alongside this hpcfs/ subdirectory.
SELECT_SCRIPT="${SCRIPT_DIR}/../select_episodes.py"
if [[ ! -f "${SELECT_SCRIPT}" ]]; then
  echo "select_episodes.py not found at ${SELECT_SCRIPT}" >&2
  exit 2
fi

MANIFEST_DIR=${PICKUP_MANIFEST_DIR:-${OUTPUT_ROOT}/manifests}
python "${SELECT_SCRIPT}" \
  --source "${PUBLIC_DATASET_SOURCE}" \
  --cache_dir "${PUBLIC_DATASET_CACHE}" \
  --output_dir "${MANIFEST_DIR}" \
  --count 64 --seed 42 --max_per_object 4

if [[ "${CONFIRM_DOWNLOAD:-0}" != "1" ]]; then
  echo "Selection complete. Review ${MANIFEST_DIR}/selection.csv and selection.json." >&2
  echo "Set CONFIRM_DOWNLOAD=1 in cluster.env, then rerun to download and convert." >&2
  exit 0
fi

convert() {
  local manifest=$1
  local output=$2
  local repo_id=$3
  if [[ -e "${output}" && -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to overwrite non-empty dataset: ${output}" >&2
    exit 2
  fi
  python "${PROJECT_ROOT}/utils/convert_public_trex_to_lerobot.py" \
    --source "${PUBLIC_DATASET_SOURCE}" \
    --cache_dir "${PUBLIC_DATASET_CACHE}" \
    --episode_manifest "${manifest}" \
    --instruction_override "Pick up the object." \
    --output_root "${output}" \
    --repo_id "${repo_id}" \
    --max_download_gb "${MAX_DOWNLOAD_GB:-300}"
}

convert "${MANIFEST_DIR}/selection.json" "${PICKUP_DATASET_ROOT}" "local/trex_pickup_64"
convert "${MANIFEST_DIR}/smoke_selection.json" "${SMOKE_DATASET_ROOT}" "local/trex_pickup_smoke_4"
