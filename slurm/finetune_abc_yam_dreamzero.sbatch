#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-lora
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/home/alex/policy-finetuning/policy-finetuning-logs/dreamzero-train/train-%j.out
#SBATCH --error=/home/alex/policy-finetuning/policy-finetuning-logs/dreamzero-train/train-%j.err
#SBATCH --export=ALL
set -euo pipefail
ROOT="${DREAMZERO_ROOT:-/home/alex/policy-finetuning/dreamzero}"
cd "${ROOT}"
source env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}"
export EXP_NAME="${EXP_NAME:-dreamzero_abc130k_lora_lr1e-5_${SLURM_JOB_ID:-manual}}"
export OUTPUT_DIR="${OUTPUT_DIR:-/projects/data/cloud_user/checkpoints/dreamzero_abc130k/${EXP_NAME}}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export MAX_STEPS="${MAX_STEPS:-100000}" SAVE_STEPS="${SAVE_STEPS:-2000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}" NUM_WORKERS="${NUM_WORKERS:-2}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}" LOGGING_STEPS="${LOGGING_STEPS:-10}"
export REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_ENTITY="${WANDB_ENTITY:-oneworld-ai}"
export WANDB_PROJECT="${WANDB_PROJECT:-policy-finetuning}"
export WANDB_MODE="${WANDB_MODE:-online}" WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-dreamzero-abc130k}"
export WANDB_JOB_TYPE=finetune WANDB_LOG_MODEL=false WANDB_WATCH=false
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${ROOT}/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${ROOT}/.cache/wandb-config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${ROOT}/.cache/wandb-data}"
# BaseExperiment sets WANDB_DIR to OUTPUT_DIR and the run name to its basename.
export WANDB_DIR="${OUTPUT_DIR}" WANDB_NAME="$(basename -- "${OUTPUT_DIR}")"
if [[ "${DRY_RUN:-0}" == 1 ]]; then export ABC_DRY_RUN=1; fi
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then export ABC_PREFLIGHT_ONLY=1; fi
if [[ "${ABC_DRY_RUN:-0}" == 1 || "${ABC_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    exec bash scripts/train/abc_yam_v3_training.sh "$@"
fi
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch; use DRY_RUN=1 or PREFLIGHT_ONLY=1 on the login node.' >&2; exit 2; }
RUN_MODE="${RUN_MODE:-fresh}"
case "${RUN_MODE}" in
    fresh)
        if [[ -d "${OUTPUT_DIR}" && -n "$(ls -A "${OUTPUT_DIR}")" ]]; then
            echo "Output already exists: ${OUTPUT_DIR}. Use a new EXP_NAME or RUN_MODE=resume." >&2; exit 2
        fi
        ;;
    resume)
        [[ -s "${OUTPUT_DIR}/.wandb-run-id" ]] || { echo 'Missing saved W&B run ID for resume.' >&2; exit 2; }
        [[ ! -f "${OUTPUT_DIR}/config.json" ]] || { echo 'This run already has a final model. Use a new experiment for further adaptation.' >&2; exit 2; }
        compgen -G "${OUTPUT_DIR}/checkpoint-*/trainer_state.json" >/dev/null || { echo 'No saved training checkpoint to resume.' >&2; exit 2; }
        ;;
    *) echo 'RUN_MODE must be fresh or resume.' >&2; exit 2 ;;
esac
if [[ "${REPORT_TO}" == wandb && "${WANDB_MODE}" == online && -z "${WANDB_API_KEY:-}" ]]; then
    KEY_FILE="${WANDB_API_KEY_FILE:-${ROOT}/../yam-abc-reproduce/.secrets/wandb_alex_api_key}"
    [[ -r "${KEY_FILE}" ]] || { echo 'Set WANDB_API_KEY, WANDB_API_KEY_FILE, or WANDB_MODE=offline.' >&2; exit 2; }
    IFS= read -r WANDB_API_KEY < "${KEY_FILE}" || [[ -n "${WANDB_API_KEY:-}" ]]
    [[ -n "${WANDB_API_KEY}" && ! "${WANDB_API_KEY}" =~ [[:space:]] ]] || { echo 'W&B key file is empty or malformed.' >&2; exit 2; }
    export WANDB_API_KEY
fi
mkdir -p "${OUTPUT_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"
if [[ "${RUN_MODE}" == fresh ]]; then
    export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex[:12])')}"
    printf '%s\n' "${WANDB_RUN_ID}" > "${OUTPUT_DIR}/.wandb-run-id"
else
    export WANDB_RUN_ID="$(cat "${OUTPUT_DIR}/.wandb-run-id")"
fi
export WANDB_RESUME=allow
printf 'Output: %s\nW&B: https://wandb.ai/%s/%s/runs/%s (mode=%s)\n' \
    "${OUTPUT_DIR}" "${WANDB_ENTITY}" "${WANDB_PROJECT}" "${WANDB_RUN_ID}" "${WANDB_MODE}"
exec bash scripts/train/abc_yam_v3_training.sh "$@"
