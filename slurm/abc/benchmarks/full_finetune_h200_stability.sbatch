#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-h48-full-100
#SBATCH --account=torch_pr_147_courant
#SBATCH --partition=h200_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/dreamzero/outputs/abc/h48/stats/full-stability-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/dreamzero/outputs/abc/h48/stats/full-stability-%j.err
set -euo pipefail
ROOT=/projects/work/yang-lab/projects/policy-finetuning
source "${ROOT}/envs/dreamzero/activate.sh"
export ABC_TEMPORAL=h48
export ABC_METADATA_ROOT="${ROOT}/dreamzero/data/abc/h48/train"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-2}"
export MAX_STEPS=100 SAVE_STEPS=50 NUM_WORKERS=2 BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1
export OUTPUT_DIR="${ROOT}/dreamzero/outputs/abc/h48/full_stability_${SLURM_JOB_ID}"
nvidia-smi
bash scripts/train/abc_yam_v3_training.sh "global_batch_size=${NUM_GPUS}" training_args.logging_steps=1 "$@"
python scripts/train/validate_abc_yam_run.py "${OUTPUT_DIR}" --steps 100 --ranks "${NUM_GPUS}"
