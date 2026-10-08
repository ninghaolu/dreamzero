#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-index
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/alex/policy-finetuning/dreamzero/logs/prepare-%A_%a.out
#SBATCH --error=/home/alex/policy-finetuning/dreamzero/logs/prepare-%A_%a.err
set -euo pipefail
export ABC_TEMPORAL="${ABC_TEMPORAL:-h48}"
ROOT=/home/alex/policy-finetuning
source "${ROOT}/dreamzero/env.sh"
cd "${ROOT}/dreamzero"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == 0 ]]; then SUFFIX=train; SPLIT=train; else SUFFIX=val; SPLIT=validation; fi
exec python scripts/data/prepare_abc_yam_v3.py \
  --temporal "${ABC_TEMPORAL}" \
  --source-root "${ABC_SOURCE_ROOT:-/projects/data/datasets/abc_130k_v3_${SUFFIX}}" \
  --output-root "${ROOT}/dreamzero/data/abc/${ABC_TEMPORAL}/${SUFFIX}" \
  --normalizer-root "${ROOT}/dreamzero/data/abc/${ABC_TEMPORAL}/norm_stats" --split "${SPLIT}"
