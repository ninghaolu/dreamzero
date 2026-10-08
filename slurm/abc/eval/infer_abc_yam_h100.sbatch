#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-video
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/alex/policy-finetuning/dreamzero
#SBATCH --output=/home/alex/policy-finetuning/dreamzero/logs/infer-%j.out
#SBATCH --error=/home/alex/policy-finetuning/dreamzero/logs/infer-%j.err
set -euo pipefail
ROOT="${DREAMZERO_REPO_ROOT:-/home/alex/policy-finetuning/dreamzero}"
cd "${ROOT}"
source "${ROOT}/env.sh"
source "${ROOT}/scripts/train/abc_cuda_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ABC_TEMPORAL="${ABC_TEMPORAL:-h48}"
CHECKPOINT="${1:?Usage: sbatch infer_abc_yam_h100.sbatch CHECKPOINT [OUTPUT_DIR]}"
OUTPUT="${2:-${ROOT}/outputs/abc/${ABC_TEMPORAL}/inference-${SLURM_JOB_ID}}"
ARGS=()
if [[ -n "${ABC_EVAL_ANCHORS:-}" ]]; then ARGS+=(--anchors "${ABC_EVAL_ANCHORS}"); fi
if [[ -n "${ABC_SAVE_CHUNK_PREFIX:-}" ]]; then ARGS+=(--save-chunk-prefixes "${ABC_SAVE_CHUNK_PREFIX}"); fi
exec python scripts/eval_abc_yam.py \
  --metadata-root "${ABC_METADATA_ROOT:-${ROOT}/data/abc/${ABC_TEMPORAL}/val}" \
  --temporal "${ABC_TEMPORAL}" --checkpoint "${CHECKPOINT}" \
  --num-samples "${ABC_EVAL_SAMPLES:-4}" --seed "${ABC_EVAL_SEED:-42}" \
  --num-chunks "${ABC_INFERENCE_CHUNKS:-1}" \
  --save-videos --output-dir "${OUTPUT}" "${ARGS[@]}"
