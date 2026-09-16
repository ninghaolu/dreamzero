#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-eval
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/home/alex/policy-finetuning/policy-finetuning-logs/dreamzero-eval/eval-%j.out
#SBATCH --error=/home/alex/policy-finetuning/policy-finetuning-logs/dreamzero-eval/eval-%j.err
set -euo pipefail
ROOT="${DREAMZERO_ROOT:-/home/alex/policy-finetuning/dreamzero}"
cd "${ROOT}"
source env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec python scripts/eval_abc_yam.py \
  --metadata-root "${ABC_EVAL_METADATA_ROOT:-${ROOT}/data/abc_yam_v3_val}" \
  --anchors "${ABC_EVAL_ANCHORS:-${ROOT}/logs/abc-validation-baseline/anchors.json}" \
  --checkpoint "${ABC_CHECKPOINT_PATH:?Set ABC_CHECKPOINT_PATH to a complete saved checkpoint}" \
  --num-samples "${ABC_EVAL_SAMPLES:-32}" \
  --output-dir "${ABC_EVAL_OUTPUT_DIR:-${ROOT}/logs/abc-evaluation/job-${SLURM_JOB_ID}}"
