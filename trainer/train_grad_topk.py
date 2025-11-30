
import torch

def apply_global_topk_gradients(model, keep_frac: float) -> None:
    """
    Apply global top-k gradient sparsification in-place on `model`'s 
    training gradients. This function assumes `loss.backward()` 
    has already been called.
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
