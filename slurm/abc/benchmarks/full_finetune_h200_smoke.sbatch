#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-h48-full-smoke
#SBATCH --account=torch_pr_147_courant
#SBATCH --partition=h200_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/dreamzero/outputs/abc/h48/stats/full-smoke-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/dreamzero/outputs/abc/h48/stats/full-smoke-%j.err
set -euo pipefail
ROOT=/projects/work/yang-lab/projects/policy-finetuning
source "${ROOT}/envs/dreamzero/activate.sh"
export ABC_TEMPORAL=h48
export ABC_METADATA_ROOT="${ROOT}/dreamzero/data/abc/h48/train"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-2}"
export MAX_STEPS=3 SAVE_STEPS=3 NUM_WORKERS=2 BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=1
export OUTPUT_DIR="${ROOT}/dreamzero/outputs/abc/h48/full_smoke_${SLURM_JOB_ID}"
nvidia-smi
mkdir -p "${OUTPUT_DIR}"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv --loop=5 > "${OUTPUT_DIR}/gpu_utilization.csv" 2> "${OUTPUT_DIR}/gpu_utilization.err" &
gpu_sampler_pid=$!
cleanup_sampler() {
  kill "${gpu_sampler_pid}" 2>/dev/null || true
  wait "${gpu_sampler_pid}" 2>/dev/null || true
}
trap cleanup_sampler EXIT
bash scripts/train/abc_yam_v3_training.sh "global_batch_size=$((NUM_GPUS * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))" training_args.logging_steps=1 "$@"
python scripts/train/validate_abc_yam_run.py "${OUTPUT_DIR}" --steps 3 --ranks "${NUM_GPUS}"
