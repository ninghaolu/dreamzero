#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-smoke
#SBATCH --account=torch_pr_147_courant
#SBATCH --partition=h200_public
#SBATCH --reservation=my3149-h200-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/dreamzero-abc-loader/train-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/dreamzero-abc-loader/train-%j.err
set -euo pipefail
ROOT=/projects/work/yang-lab/projects/policy-finetuning
source "${ROOT}/envs/dreamzero/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/dreamzero/outputs/abc_yam_smoke_${SLURM_JOB_ID}}"
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-2}"
exec bash scripts/train/abc_yam_v3_training.sh "$@"
