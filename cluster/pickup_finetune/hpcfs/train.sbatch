#!/usr/bin/env bash
#SBATCH --job-name=trex-pickup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4                    # 4 GPUs on the a100 partition
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=a100
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a1865590@adelaide.edu.au

set -euo pipefail

if [[ -f "${SLURM_SUBMIT_DIR}/train_pickup.sh" ]]; then
  SCRIPT_DIR=${SLURM_SUBMIT_DIR}
else
  SCRIPT_DIR=${SLURM_SUBMIT_DIR}/cluster/hpcfs
fi

srun --ntasks=1 bash "${SCRIPT_DIR}/train_pickup.sh" full
