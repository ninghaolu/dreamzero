"""Check native training dependencies and exercise CUDA kernels on an allocated GPU."""
import importlib
import json
import os
import sys
from pathlib import Path

import torch


def main():
    versions = {}
    for name in ('torch', 'torchvision', 'flash_attn', 'deepspeed', 'transformers',
                 'diffusers', 'hydra', 'av', 'cv2', 'decord', 'wandb'):
        module = importlib.import_module(name)
        versions[name] = getattr(module, '__version__', 'imported')
    if not torch.cuda.is_available():
        raise RuntimeError('Run this check on a Slurm GPU allocation')
    from flash_attn import flash_attn_func
    q = torch.randn(2, 128, 4, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    result = flash_attn_func(q, q, q, causal=True)
    result.float().square().mean().backward()
    if not torch.isfinite(result).all() or not torch.isfinite(q.grad).all():
        raise RuntimeError('FlashAttention produced nonfinite outputs/gradients')
    from torchvision.ops import nms
    boxes = torch.tensor([[0., 0., 1., 1.]], device='cuda')
    assert nms(boxes, torch.ones(1, device='cuda'), 0.5).tolist() == [0]
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'train'))
    from validate_abc_cuda import main as validate_cuda
    validate_cuda()
    for variable in ('WAN_CKPT_DIR', 'AGIBOT_CKPT_DIR', 'TOKENIZER_DIR', 'ABC_DATASET_ROOT'):
        path = Path(os.environ[variable])
        if not path.is_dir():
            raise FileNotFoundError(path)
    report = dict(passed=True, python=sys.version, versions=versions,
                  gpu=torch.cuda.get_device_name(), gpu_count=torch.cuda.device_count(),
                  flash_attention_forward_backward=True, torchvision_cuda_nms=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
