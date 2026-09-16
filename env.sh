# Source this file from any directory: source /path/to/dreamzero/env.sh
_dreamzero_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_dreamzero_root}/.venv/bin/activate"
export UV_CACHE_DIR="${_dreamzero_root}/.cache/uv"
export UV_PYTHON_INSTALL_DIR="${_dreamzero_root}/.cache/python"
export HF_HOME="${HF_HOME:-${_dreamzero_root}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${_dreamzero_root}/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${_dreamzero_root}/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${_dreamzero_root}/.cache/matplotlib}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/dreamzero-${UID}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/dreamzero-${UID}/torch_extensions}"
export TMPDIR="${TMPDIR:-${_dreamzero_root}/.cache/tmp}"
export WAN_CKPT_DIR="${WAN_CKPT_DIR:-/projects/data/cloud_user/checkpoints/Wan2.1-I2V-14B-480P}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-/projects/data/cloud_user/checkpoints/umt5-xxl}"
export AGIBOT_CKPT_DIR="${AGIBOT_CKPT_DIR:-/projects/data/cloud_user/checkpoints/DreamZero-AgiBot}"
export NO_ALBUMENTATIONS_UPDATE=1
# Local compiler and headers match the CUDA runtime bundled with PyTorch.
if [[ -z "${CUDA_HOME:-}" && -x "${_dreamzero_root}/.cache/cuda/bin/nvcc" ]]; then
    export CUDA_HOME="${_dreamzero_root}/.cache/cuda"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
mkdir -p "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TMPDIR}"
unset _dreamzero_root
