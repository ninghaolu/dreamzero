#!/usr/bin/env bash
# Source after activating DreamZero. Keep build tools separate from Python packages.
ABC_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_HOME="${ABC_CUDA_HOME:-${ABC_REPO_ROOT}/.cache/cuda-12.9}"
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "Missing CUDA compiler: ${CUDA_HOME}/bin/nvcc. Install the toolkit matching torch.version.cuda or set ABC_CUDA_HOME." >&2
  return 1
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
if [[ -x "${CUDA_HOME}/bin/x86_64-conda-linux-gnu-c++" ]]; then
  export CXX="${CUDA_HOME}/bin/x86_64-conda-linux-gnu-c++"
  export CC="${CUDA_HOME}/bin/x86_64-conda-linux-gnu-cc"
fi
# Separate jobs and node CPU architectures; all local ranks reuse the preflight build.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${SLURM_TMPDIR:-/tmp}/dreamzero-extensions-${USER}-${SLURM_JOB_ID:-$(hostname)}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TORCH_EXTENSIONS_DIR}/triton}"
export MAX_JOBS="${MAX_JOBS:-4}"
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}"
# PyTorch's CUDA wheel supplies cuRAND, while the local compiler-only toolkit
# does not. CPUAdam links with -lcurand, which requires an unversioned name.
ABC_CURAND_LIB="$(python -c 'import sysconfig; from pathlib import Path; print(Path(sysconfig.get_path("purelib"))/"nvidia/curand/lib/libcurand.so.10")')"
if [[ -f "${ABC_CURAND_LIB}" ]]; then
  mkdir -p "${TORCH_EXTENSIONS_DIR}/cuda-libs"
  ln -sfn "${ABC_CURAND_LIB}" "${TORCH_EXTENSIONS_DIR}/cuda-libs/libcurand.so"
  ln -sfn "${ABC_CURAND_LIB}" "${TORCH_EXTENSIONS_DIR}/cuda-libs/libcurand.so.10"
  export LIBRARY_PATH="${TORCH_EXTENSIONS_DIR}/cuda-libs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${TORCH_EXTENSIONS_DIR}/cuda-libs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
unset ABC_CURAND_LIB
unset ABC_REPO_ROOT
