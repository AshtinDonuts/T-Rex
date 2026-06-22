#!/usr/bin/env bash
#SBATCH --job-name=trex-pickup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
##SBATCH --partition=EDIT_ME
##SBATCH --account=EDIT_ME

set -euo pipefail
if [[ -f "${SLURM_SUBMIT_DIR}/train_pickup.sh" ]]; then
  SCRIPT_DIR=${SLURM_SUBMIT_DIR}
else
  SCRIPT_DIR=${SLURM_SUBMIT_DIR}/cluster/pickup_finetune
fi
srun --ntasks=1 bash "${SCRIPT_DIR}/train_pickup.sh" full
