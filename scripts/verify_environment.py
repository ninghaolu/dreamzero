"""Smoke-check imports and FlashAttention on one allocated GPU."""
import torch
import torch.nn.functional as F
import flash_attn
import deepspeed
import groot.vla.experiment.experiment
from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
    WANPolicyHead,
)

print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}, "
      f"FlashAttention={flash_attn.__version__}, DeepSpeed={deepspeed.__version__}", flush=True)
assert torch.cuda.is_available(), "Run this check on an allocated GPU node"
print(torch.cuda.get_device_name(0), flush=True)
torch.manual_seed(42)
q, k, v = [torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16,
                      requires_grad=True) for _ in range(3)]
actual = flash_attn.flash_attn_func(q, k, v, causal=True)
expected = F.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
).transpose(1, 2)
torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
actual.float().square().mean().backward()
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v))
torch.cuda.synchronize()
print("PASS: model/training imports and FlashAttention CUDA forward/backward", flush=True)
