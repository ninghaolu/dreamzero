#!/usr/bin/env bash
# Source from any directory; keep caches and Python packages inside the repo.
DREAMZERO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DREAMZERO_ROOT}/.venv/bin/activate"
export PYTHONPATH="${DREAMZERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${DREAMZERO_ROOT}/.cache/huggingface}"
export XDG_CACHE_HOME="${DREAMZERO_ROOT}/.cache"
export MPLCONFIGDIR="${DREAMZERO_ROOT}/.cache/matplotlib"
export WANDB_CONFIG_DIR="${DREAMZERO_ROOT}/.cache/wandb-config"
export WANDB_DATA_DIR="${DREAMZERO_ROOT}/.cache/wandb-data"
export WAN_CKPT_DIR="${WAN_CKPT_DIR:-/projects/data/cloud_user/checkpoints/Wan2.1-I2V-14B-480P}"
export AGIBOT_CKPT_DIR="${AGIBOT_CKPT_DIR:-/projects/data/cloud_user/checkpoints/DreamZero-AgiBot}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-/projects/data/cloud_user/checkpoints/umt5-xxl}"
export ABC_DATASET_ROOT="${ABC_DATASET_ROOT:-/projects/data/datasets/abc_130k_v3_train}"
export ABC_CUDA_HOME="${ABC_CUDA_HOME:-${DREAMZERO_ROOT}/.cache/cuda}"
export PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_LOG_MODEL=false WANDB_WATCH=false
unset DREAMZERO_ROOT
