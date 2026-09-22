#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-h48-no-offload
#SBATCH --account=torch_pr_147_courant
#SBATCH --partition=h200_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1000G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/dreamzero/outputs/abc/h48/stats/full-no-offload-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/dreamzero/outputs/abc/h48/stats/full-no-offload-%j.err
set -euo pipefail
ROOT=/projects/work/yang-lab/projects/policy-finetuning
source "${ROOT}/envs/dreamzero/activate.sh"
export ABC_TEMPORAL=h48
export ABC_METADATA_ROOT="${ROOT}/dreamzero/data/abc/h48/train"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-8}"
if [[ "${NUM_GPUS}" != 8 ]]; then
  echo 'This comparison requires eight GPUs and global batch eight.' >&2
  exit 2
fi
export MAX_STEPS=100 SAVE_STEPS=100 NUM_WORKERS=2 BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1
export OUTPUT_DIR="${ROOT}/dreamzero/outputs/abc/h48/full_no_offload_${SLURM_JOB_ID}"
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
# Keep the original eight-GPU recipe, changing only optimizer offload and run length.
# The shared CUDA preflight still exercises CPUAdam; training uses GPU AdamW.
bash scripts/train/abc_yam_v3_training.sh global_batch_size=8 training_args.logging_steps=10 \
  training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json
python scripts/train/validate_abc_yam_run.py "${OUTPUT_DIR}" --steps "${MAX_STEPS}" --ranks "${NUM_GPUS}"
