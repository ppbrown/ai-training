
"""
Routines to allow "gradient sparsification".
For back propagation, only update gradients that are making the most difference.
This avoids knowledge loss in areas that we dont even need to touch.
"""

import torch
from typing import Sequence


def sparsify_sd15_gradients(unet, keep_frac):
    """
    This is the one routine outside should be calling.
    "keep_frac" is really a decimal percentage, 0.0 < frac <1.0
    """
    important_params = collect_sd15_important_params(unet)
    apply_topk_to_params(important_params, keep_frac)


######################################################################


def apply_topk_to_params(params: Sequence[torch.nn.Parameter], keep_frac: float) -> None:
    if not (0.0 < keep_frac < 1.0):
        return

    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.view(-1))

    if not grads:
        return

    flat = torch.cat(grads)
    total = flat.numel()
    if total <= 1:
        return

    k = max(1, min(total - 1, int(total * keep_frac)))
    abs_flat = flat.abs()
    thresh = abs_flat.kthvalue(total - k + 1).values

    # second pass over the SAME params
    for p in params:
        g = p.grad
        if g is None:
            continue
        g.masked_fill_(g.abs() < thresh, 0.0)




def collect_sd15_important_params(unet) -> list[torch.nn.Parameter]:
    """
    Optimizing filter for topk, targetting SD1.5 model
    Applying it to the entire model is REALLY SLOW, so
    identify just the layers that matter a lot:

    - attention weights (q/k/v/out) in attn1/attn2/transformer
    - optionally mid_block
    """
    important = []
    for name, p in unet.named_parameters():
        if p.grad is None:  # during collection this is usually None, but we
            pass            # just want params that *can* get grads
        # focus on attention-heavy parts
        if any(tok in name for tok in ["attn1", "attn2", "transformer_blocks"]):
            if p.dim() >= 2:   # skip biases / norms
                important.append(p)
            continue
        # optional: mid_block convs
        if "mid_block" in name and p.dim() >= 2:
            important.append(p)
    return important


# Keeping this for reference purposes
def apply_global_topk_gradients(model, keep_frac: float) -> None:
    """
    Use this only if you want to suffer a HUGE performance penalty.

    Apply global top-k gradient sparsification in-place on 
    ALL training gradients for the `model`.
    (typically model == unet)

    This function assumes `loss.backward()` has already been called.
    Purpose is to reduce "catastrophic forgetting" from overtraining,
    and potentially allow more knowledge to be stored

    Args:
        model: torch.nn.Module with gradients already computed (loss.backward()).
        keep_frac: Fraction of gradient entries (by absolute magnitude) to keep.
                   0 < keep_frac < 1. Values outside this range are a no-op.
    """
    if keep_frac <= 0.0 or keep_frac >= 1.0:
        return

    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))

    if not grads:
        return

    flat = torch.cat(grads)
    total = flat.numel()

    k = int(total * keep_frac)
    if k <= 0 or k >= total:
        return

    abs_flat = flat.abs()
    # Keep the largest-k entries by |g| threshold is (N - k)-th smallest
    thresh = abs_flat.kthvalue(total - k).values

    # Zero out small grads in-place
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        mask = g.abs() >= thresh
        g.mul_(mask)
