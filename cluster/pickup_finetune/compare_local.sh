#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

COMMAND=${1:-}
KIND=${2:-}
SEED=${3:-0}
case "${KIND}" in
  baseline) CHECKPOINT=${MIDTRAIN_CHECKPOINT} ;;
  finetuned) CHECKPOINT=${FINETUNED_CHECKPOINT:-} ;;
  *) echo "Usage: $0 server|eval baseline|finetuned [seed]" >&2; exit 2 ;;
esac
if [[ "${COMMAND}" == "server" && -z "${CHECKPOINT}" ]]; then
  echo "Checkpoint path for ${KIND} is empty in ${ENV_FILE}" >&2
  exit 2
fi

mkdir -p "${COMPARISON_DIR:-/tmp/trex_pickup_comparison}"
case "${COMMAND}" in
  server)
    exec python "${PROJECT_ROOT}/scripts/test.py" \
      --checkpoint_path "${CHECKPOINT}" \
      --dataset_name rlbench \
      --action_dim 62 --action_chunk 16 \
      --use_robot_state 0 --disable_tactile 1 \
      --seed "${SEED}" --port 5678
    ;;
  eval)
    exec "${PROJECT_ROOT}/hardware_code/eval/isaac_lab/run_eval.sh" \
      --task-description "Pick up the object." \
      --record-video "${COMPARISON_DIR:-/tmp/trex_pickup_comparison}/${KIND}_seed${SEED}.mp4" \
      --seed "${SEED}" --viz none
    ;;
  *) echo "Usage: $0 server|eval baseline|finetuned [seed]" >&2; exit 2 ;;
esac
