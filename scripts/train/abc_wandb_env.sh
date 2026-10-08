#!/usr/bin/env bash
# Source for the same W&B destination and identity as the pi0.5 ABC launcher.
export WANDB_ENTITY="${WANDB_ENTITY:-oneworld-ai}"
export WANDB_PROJECT="${WANDB_PROJECT:-policy-finetuning}"
export WANDB_MODE="${WANDB_MODE:-online}"
ABC_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${ABC_WORKSPACE}/yam-abc-reproduce/.secrets/wandb_alex_api_key}"
unset ABC_WORKSPACE
if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE}" == online ]]; then
  if [[ ! -r "${WANDB_API_KEY_FILE}" ]]; then
    echo "ERROR: W&B API key file is missing or unreadable: ${WANDB_API_KEY_FILE}" >&2
    exit 2
  fi
  IFS= read -r WANDB_API_KEY < "${WANDB_API_KEY_FILE}" || [[ -n "${WANDB_API_KEY}" ]]
  if [[ -z "${WANDB_API_KEY}" || "${WANDB_API_KEY}" =~ [[:space:]] ]]; then
    echo "ERROR: W&B API key file is empty or malformed: ${WANDB_API_KEY_FILE}" >&2
    exit 2
  fi
  export WANDB_API_KEY
fi
