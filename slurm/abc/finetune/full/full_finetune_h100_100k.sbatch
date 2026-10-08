#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-h48-full-100k
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1024G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/alex/policy-finetuning/dreamzero
#SBATCH --output=/home/alex/policy-finetuning/dreamzero/logs/full-h100-%j.out
#SBATCH --error=/home/alex/policy-finetuning/dreamzero/logs/full-h100-%j.err
set -euo pipefail
ROOT="${DREAMZERO_REPO_ROOT:-/home/alex/policy-finetuning/dreamzero}"
cd "${ROOT}"
source "${ROOT}/env.sh"
export ABC_TEMPORAL="${ABC_TEMPORAL:-h48}"
export ABC_METADATA_ROOT="${ABC_METADATA_ROOT:-${ROOT}/data/abc/${ABC_TEMPORAL}/train}"
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-8}"
export MAX_STEPS="${MAX_STEPS:-100000}" SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1000}"
export NUM_WORKERS="${NUM_WORKERS:-2}" BATCH_SIZE="${BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
# Reuse OUTPUT_DIR to resume an interrupted run from its latest complete checkpoint.
export OUTPUT_DIR="${OUTPUT_DIR:-/projects/data/cloud_user/checkpoints/dreamzero_abc130k/full_${ABC_TEMPORAL}_${SLURM_JOB_ID:?Submit with sbatch}}"
python - <<'PY'
import os
import torch
count = torch.cuda.device_count()
if count != 8 or int(os.environ['NUM_GPUS']) != 8:
    raise RuntimeError(f'This launcher requires 8 visible GPUs; found {count}')
for index in range(count):
    name = torch.cuda.get_device_name(index)
    if 'H100' not in name:
        raise RuntimeError(f'Expected H100, GPU {index} is {name}')
print('Validated eight H100 GPUs', flush=True)
PY
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
bash scripts/train/abc_yam_v3_training.sh \
  "global_batch_size=$((NUM_GPUS * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))" \
  "save_total_limit=${SAVE_TOTAL_LIMIT}" \
  "training_args.logging_steps=${LOGGING_STEPS:-10}" "$@"
if [[ "${ABC_VALIDATE_ONLY:-0}" != 1 && "${ABC_DRY_RUN:-0}" != 1 ]]; then
  python scripts/train/validate_abc_yam_run.py "${OUTPUT_DIR}" --steps "${MAX_STEPS}" --ranks "${NUM_GPUS}"
fi
