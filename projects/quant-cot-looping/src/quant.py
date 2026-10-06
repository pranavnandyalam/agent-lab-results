"""Simulated weight-only RTN group-wise asymmetric fake-quantization (in place).

For every nn.Linear except those whose qualified name matches `skip` (default: lm_head),
weights are split along the input dim into groups of `group_size`; each group gets
scale = (max - min) / (2^bits - 1), zero = round(-min / scale), and
w_hat = (clamp(round(w / scale) + zero, 0, 2^bits - 1) - zero) * scale.
Computed in fp32, written back in the weight's original dtype.
Note: Qwen3-0.6B ties lm_head to embed_tokens; neither is quantized.
"""
import torch
import torch.nn as nn


@torch.no_grad()
def fake_quant_tensor(w: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    out_f, in_f = w.shape
    assert in_f % group_size == 0, f"in_features {in_f} not divisible by group {group_size}"
    qmax = 2 ** bits - 1
    wg = w.float().reshape(out_f, in_f // group_size, group_size)
    wmin = wg.amin(dim=-1, keepdim=True)
    wmax = wg.amax(dim=-1, keepdim=True)
    scale = ((wmax - wmin) / qmax).clamp(min=1e-8)
    zero = torch.round(-wmin / scale)
    q = torch.clamp(torch.round(wg / scale) + zero, 0, qmax)
    return ((q - zero) * scale).reshape(out_f, in_f).to(w.dtype)


@torch.no_grad()
def fake_quantize_model(model: nn.Module, bits: int = 4, group_size: int = 64, skip=("lm_head",)) -> int:
    """Quantize in place; returns number of Linear layers quantized."""
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not any(s in name for s in skip):
            mod.weight.data.copy_(fake_quant_tensor(mod.weight.data, bits, group_size))
            n += 1
    return n
