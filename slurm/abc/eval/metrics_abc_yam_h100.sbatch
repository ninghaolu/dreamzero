#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-metrics
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/alex/policy-finetuning/dreamzero
#SBATCH --output=/home/alex/policy-finetuning/dreamzero/logs/metrics-%j.out
#SBATCH --error=/home/alex/policy-finetuning/dreamzero/logs/metrics-%j.err
set -euo pipefail
ROOT="${DREAMZERO_REPO_ROOT:-/home/alex/policy-finetuning/dreamzero}"
cd "${ROOT}"
source "${ROOT}/env.sh"
source "${ROOT}/scripts/train/abc_cuda_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CHECKPOINT="${1:?Usage: sbatch metrics_abc_yam_h100.sbatch CHECKPOINT OUTPUT_DIR}"
OUTPUT="${2:?Provide a fresh output directory}"
ARGS=()
if [[ -n "${ABC_EVAL_ANCHORS:-}" ]]; then ARGS+=(--anchors "${ABC_EVAL_ANCHORS}"); fi
exec python scripts/eval_abc_yam.py \
  --metadata-root "${ABC_METADATA_ROOT:-${ROOT}/data/abc/h48/val}" --temporal h48 \
  --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT}" \
  --num-samples "${ABC_EVAL_SAMPLES:-256}" --seed "${ABC_EVAL_SEED:-42}" \
  --validation-loss --loss-noise-draws "${ABC_LOSS_NOISE_DRAWS:-2}" "${ARGS[@]}"
