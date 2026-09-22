#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"
export HYDRA_FULL_ERROR=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export ABC_TEMPORAL="${ABC_TEMPORAL:-h48}"
ABC_METADATA_ROOT="${ABC_METADATA_ROOT:-${ROOT}/data/abc/${ABC_TEMPORAL}/train}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-2}}"
if [[ ! "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then echo 'NUM_GPUS must be positive' >&2; exit 2; fi
mapfile -d '' -t DEFAULTS < <(python scripts/abc_yam_config.py "${ABC_METADATA_ROOT}")
if (( ${#DEFAULTS[@]} == 0 )); then echo 'Failed to build configuration' >&2; exit 2; fi
ARGS=("${DEFAULTS[@]}" "per_device_train_batch_size=${BATCH_SIZE:-1}" \
  "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-1}" "max_steps=${MAX_STEPS:-50}" \
  "save_steps=${SAVE_STEPS:-10}" "dataloader_num_workers=${NUM_WORKERS:-0}" "report_to=${REPORT_TO:-wandb}" "$@")
if [[ "${ABC_DRY_RUN:-0}" == 1 ]]; then
  exec python scripts/train/validate_abc_yam_launch.py --config-only "${ARGS[@]}"
fi
python scripts/train/validate_abc_yam_launch.py "${ARGS[@]}"
if [[ "${ABC_VALIDATE_ONLY:-0}" == 1 ]]; then exit 0; fi
# Respect the final Hydra reporting override, including explicit opt-outs.
REPORTING="${REPORT_TO:-wandb}"
for arg in "${ARGS[@]}"; do
  case "${arg}" in
    report_to=*|training_args.report_to=*) REPORTING="${arg#*=}" ;;
  esac
done
if [[ "${REPORTING}" == *wandb* || "${REPORTING}" == all ]]; then
  source "${SCRIPT_DIR}/abc_wandb_env.sh"
fi
source "${SCRIPT_DIR}/abc_cuda_env.sh"
python scripts/train/validate_abc_cuda.py
CMD=(torchrun --standalone --nproc_per_node="${NUM_GPUS}" groot/vla/experiment/experiment.py "${ARGS[@]}")
printf '%q ' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
