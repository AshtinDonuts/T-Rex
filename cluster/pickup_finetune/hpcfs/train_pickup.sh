#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

MODE=${1:-full}
case "${MODE}" in
  smoke)
    DATASET_ROOT=${SMOKE_DATASET_ROOT}
    EXPERIMENT_NAME=trex_pickup_smoke
    RUN_NAME="smoke_${SLURM_JOB_ID:-local}"
    MAX_STEPS=10
    SAVE_STEPS=5
    VAL_RATIO=0.25
    ;;
  full)
    DATASET_ROOT=${PICKUP_DATASET_ROOT}
    EXPERIMENT_NAME=trex_pickup_public64
    RUN_NAME="pickup_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
    MAX_STEPS=0
    SAVE_STEPS=1000
    VAL_RATIO=0.125
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

for path in "${DATASET_ROOT}/meta/info.json" "${DATASET_ROOT}/meta/trex_norm_stats.json" \
            "${ORIGIN_MODEL_PATH}" "${MIDTRAIN_CHECKPOINT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path is missing: ${path}" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
cd "${PROJECT_ROOT}/scripts"

TRAIN_COMMAND=(accelerate launch \
  --config_file "${PROJECT_ROOT}/config/sft_qwen.yaml" \
  --num_processes "${GPUS_PER_NODE}" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${MASTER_PORT:-29500}" \
  train.py \
  --model_path "${ORIGIN_MODEL_PATH}" \
  --data_format lerobot \
  --lerobot_root "${DATASET_ROOT}" \
  --lerobot_repo_id "$(basename "${DATASET_ROOT}")" \
  --n_epochs 3 \
  --max_train_steps "${MAX_STEPS}" \
  --save_freq 1 \
  --save_steps "${SAVE_STEPS}" \
  --max_ckpts 10 \
  --action_dim 62 \
  --action_chunk 16 \
  --train_bsz_per_gpu 1 \
  --gradient_accumulation_steps 2 \
  --sample_stride 4 \
  --learning_rate 2e-5 \
  --warmup_rates 0.05 \
  --min_lr_ratio 0.1 \
  --weight_decay 0 \
  --output_dir "${OUTPUT_ROOT}" \
  --log_dir "${OUTPUT_ROOT}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --run_name "${RUN_NAME}" \
  --use_robot_state 0 \
  --use_tactile_vec 0 \
  --use_tactile_deform 0 \
  --training_stage 2 \
  --resume_checkpoint "${MIDTRAIN_CHECKPOINT}" \
  --resume_source midtrain \
  --use_flare "${USE_FLARE:-1}" \
  --n_flare_tokens_per_frame 4 \
  --n_flare_steps 8 \
  --flare_loss_weight 0.5 \
  --flare_frame_stride 4 \
  --flare_layer_index -1 \
  --image_size 384 288 \
  --val_ratio "${VAL_RATIO}" \
  --val_freq 500 \
  --max_val_batches 30)

if [[ "${PRINT_COMMAND:-0}" == "1" ]]; then
  printf '%q ' "${TRAIN_COMMAND[@]}"
  printf '\n'
  exit 0
fi
exec "${TRAIN_COMMAND[@]}"
