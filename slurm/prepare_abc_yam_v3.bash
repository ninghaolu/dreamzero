#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-index
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/alex/policy-finetuning/policy-finetuning-logs/dreamzero-data/index-%A_%a.out
#SBATCH --error=/home/alex/policy-finetuning/policy-finetuning-logs/dreamzero-data/index-%A_%a.err
set -euo pipefail
ROOT="${DREAMZERO_ROOT:-/home/alex/policy-finetuning/dreamzero}"
cd "${ROOT}"
source env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == 0 ]]; then SUFFIX=train; SPLIT=train; else SUFFIX=val; SPLIT=validation; fi
exec python scripts/data/prepare_abc_yam_v3.py \
  --source-root "${ABC_SOURCE_ROOT:-/projects/data/datasets/abc_130k_v3_${SUFFIX}}" \
  --output-root "${ABC_METADATA_ROOT:-${ROOT}/data/abc_yam_v3_${SUFFIX}}" \
  --normalizer-root "${ABC_NORMALIZER_ROOT:-${ROOT}/data/abc_yam_v3_norm_stats}" --split "${SPLIT}"
