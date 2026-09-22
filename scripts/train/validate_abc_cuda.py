"""Build and exercise CPUAdam before loading the full DreamZero model."""
import json
import os
import re
import subprocess

import torch
from torch.utils.cpp_extension import CUDA_HOME


def main():
    if os.environ.get("DS_SKIP_CUDA_CHECK") == "1":
        raise RuntimeError("Remove DS_SKIP_CUDA_CHECK; this launch requires a matching CUDA toolkit")
    if not CUDA_HOME:
        raise RuntimeError("CUDA_HOME must point to the toolkit matching PyTorch")
    nvcc = subprocess.check_output([f"{CUDA_HOME}/bin/nvcc", "--version"], text=True)
    match = re.search(r"release (\d+\.\d+)", nvcc)
    if not match or match[1] != torch.version.cuda:
        raise RuntimeError(f"CUDA toolkit at {CUDA_HOME} does not match torch CUDA {torch.version.cuda}: {nvcc}")
    if not torch.cuda.is_available():
        raise RuntimeError("Run training preflight on an allocated GPU node")

    from deepspeed.ops.adam import DeepSpeedCPUAdam

    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = DeepSpeedCPUAdam([parameter], lr=1e-3, weight_decay=1e-5, adamw_mode=True)
    expected = torch.optim.AdamW([reference], lr=1e-3, weight_decay=1e-5)
    for _ in range(3):
        parameter.grad = torch.tensor([0.25, -0.5])
        reference.grad = parameter.grad.clone()
        optimizer.step()
        expected.step()
    torch.testing.assert_close(parameter, reference)
    value = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(value @ value, torch.full_like(value, 32))
    torch.cuda.synchronize()
    print(json.dumps(dict(cuda_preflight="passed", torch=torch.__version__,
                          torch_cuda=torch.version.cuda, cuda_home=CUDA_HOME,
                          deepspeed_cpu_adam=optimizer.ds_opt_adam.__file__,
                          gpu=torch.cuda.get_device_name()), indent=2), flush=True)


if __name__ == "__main__":
    main()
