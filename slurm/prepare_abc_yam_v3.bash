#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-index
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT=/projects/work/yang-lab/projects/policy-finetuning
source "${ROOT}/envs/dreamzero/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == 0 ]]; then SUFFIX=train; SPLIT=train; else SUFFIX=val; SPLIT=validation; fi
exec python scripts/data/prepare_abc_yam_v3.py \
  --source-root "/projects/work/yang-lab/projects/pretrain_world_model/abc_130k_v3_${SUFFIX}" \
  --output-root "${ROOT}/dreamzero/data/abc_yam_v3_${SUFFIX}" \
  --normalizer-root "${ROOT}/dreamzero/data/abc_yam_v3_norm_stats" --split "${SPLIT}"
