#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-stats
#SBATCH --chdir=/projects/work/yang-lab/projects/policy-finetuning
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --export=ALL

set -euo pipefail
ROOT=/projects/work/yang-lab/projects/policy-finetuning
source "${ROOT}/envs/dreamzero/activate.sh"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ARGS=(
    --stage "${ABC_DZ_STATS_STAGE:?stage required}"
    --dataset-root "${ABC_DZ_STATS_DATASET_ROOT:-/projects/work/yang-lab/projects/pretrain_world_model/abc_130k_v3_train}"
    --output-dir "${ABC_DZ_STATS_OUTPUT_DIR:-${ROOT}/dreamzero/data/abc_yam_v3_norm_stats}"
    --work-dir "${ABC_DZ_STATS_WORK_DIR:-${ROOT}/policy-finetuning-logs/dreamzero-abc-stats/work}"
    --num-shards "${ABC_DZ_STATS_NUM_SHARDS:?shard count required}"
    --num-bins "${ABC_DZ_STATS_NUM_BINS:-5000}"
    --batch-rows "${ABC_DZ_STATS_BATCH_ROWS:-8192}"
    --max-files "${ABC_DZ_STATS_MAX_FILES:-0}"
)
exec python "${ROOT}/dreamzero/scripts/data/compute_abc_yam_stats.py" "${ARGS[@]}"
