#!/usr/bin/env bash
# Submit task-specific training through the existing ABC Slurm launcher.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MODE="${1:---help}"
case "${MODE}" in
    --dry-run|--preflight|--submit) shift ;;
    *) echo 'Usage: TASK_DATA_ROOT=/path/to/subset bash finetune.sh --dry-run|--preflight|--submit [Hydra overrides]'
       echo 'Optional: NUM_GPUS, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, MAX_STEPS, NUM_WORKERS, EXP_NAME, RUN_MODE'
       echo 'Slurm: TASK_CPUS, TASK_TIME, TASK_MEMORY, TASK_NODELIST, TASK_DEPENDENCY'
       [[ "${MODE}" == --help ]] && exit 0 || exit 2 ;;
esac
source "${ROOT}/env.sh"
: "${TASK_DATA_ROOT:?Set TASK_DATA_ROOT to the directory containing train/ and validation/}"
TASK_DATA_ROOT="$(cd -- "${TASK_DATA_ROOT}" && pwd)"
export DREAMZERO_ROOT="${ROOT}" ABC_METADATA_ROOT="${TASK_DATA_ROOT}/train"
# Dataset selection is controlled by TASK_DATA_ROOT, not conflicting Hydra overrides.
for arg in "$@"; do
    key="${arg%%=*}"
    key="${key#+}"
    key="${key#+}"
    case "${key}" in
        abc_metadata_root|yam_data_root|train_dataset|train_dataset.*|data|mixture_dataset_cls|single_dataset_cls)
            echo 'Select the dataset with TASK_DATA_ROOT; dataset overrides are not accepted here.' >&2; exit 2 ;;
        per_device_train_batch_size|gradient_accumulation_steps|global_batch_size|training_args.per_device_train_batch_size|training_args.gradient_accumulation_steps)
            echo 'Set BATCH_SIZE and GRADIENT_ACCUMULATION_STEPS in the environment for consistent batch settings.' >&2; exit 2 ;;
    esac
done
python "${SCRIPT_DIR}/validate.py" --metadata-root "${ABC_METADATA_ROOT}" --expected-split train
export NUM_GPUS="${NUM_GPUS:-8}" BATCH_SIZE="${BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
for value in "${NUM_GPUS}" "${BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "${TASK_CPUS:-1}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo 'GPU/batch/accumulation/CPU counts must be positive integers.' >&2; exit 2; }
done
export EXP_NAME="${EXP_NAME:-dreamzero_$(basename -- "${TASK_DATA_ROOT}")_lora_run1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-dreamzero-task-specific}"
GLOBAL_BATCH=$((NUM_GPUS * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
ARGS=("$@" "global_batch_size=${GLOBAL_BATCH}")
LAUNCHER="${ROOT}/slurm/finetune_abc_yam_dreamzero.sbatch"
case "${MODE}" in
    --dry-run) DRY_RUN=1 exec bash "${LAUNCHER}" "${ARGS[@]}" ;;
    --preflight) PREFLIGHT_ONLY=1 exec bash "${LAUNCHER}" "${ARGS[@]}" ;;
    --submit)
        # Check the resolved launch before consuming a GPU allocation.
        PREFLIGHT_ONLY=1 bash "${LAUNCHER}" "${ARGS[@]}"
        CMD=(sbatch --parsable --chdir="${ROOT}" --gres="gpu:${NUM_GPUS}"
             --cpus-per-task="${TASK_CPUS:-$((NUM_GPUS * 4))}"
             --mem="${TASK_MEMORY:-1024G}" --time="${TASK_TIME:-24:00:00}")
        if [[ -n "${TASK_NODELIST:-}" ]]; then CMD+=(--nodelist="${TASK_NODELIST}"); fi
        if [[ -n "${TASK_DEPENDENCY:-}" ]]; then CMD+=(--dependency="${TASK_DEPENDENCY}" --kill-on-invalid-dep=yes); fi
        exec "${CMD[@]}" "${LAUNCHER}" "${ARGS[@]}"
        ;;
esac
