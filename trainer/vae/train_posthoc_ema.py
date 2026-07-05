"""
train_posthoc_ema.py

Power-function EMA, adapted from Karras et al., "Analyzing and Improving
the Training Dynamics of Diffusion Models" (EDM2, arXiv 2312.02696).

This is the DIRECT-SAVE variant: instead of storing fp16 snapshots and
solving a least-squares post-hoc reconstruction after training, we keep
one full-precision online EMA accumulator per requested sigma_rel and
checkpoint each accumulator directly. The saved shadow IS the EMA model
for that sigma_rel -- no reconstruction, no interpolation between values.

Rationale for the change:
  - Post-hoc reconstruction combined fp16 snapshots with large,
    alternating-sign least-squares coefficients. At short sigma_rel that
    amplifies fp16 quantization into large perceptual (VGG) error while
    barely moving L1 -- reconstructions could score WORSE than raw.
  - It was also fragile to snapshot-set contamination (mixing runs /
    restarts, or reconstructing over a different set of t_i than was
    intended silently changed every coefficient).
  - We give up "reconstruct any sigma_rel in between the anchors after
    training" in exchange for correctness and space efficiency. You now
    pre-declare the sigma_rels you want and get exact online EMAs of each.

The power-EMA update at step t is
    shadow = beta * shadow + (1 - beta) * param
    beta   = (1 - 1/t) ** (gamma + 1)
which corresponds to an averaging profile w(tau) ~ tau ** gamma over the
run so far. At t=1, beta=0, so the accumulator self-initializes to the
model weights (no init bias).

sigma_rel is the relative std of that profile (roughly, the fraction of
the whole run being averaged over). Larger sigma_rel = longer averaging.
sigma_rel and gamma relate by
    sigma_rel = sqrt((gamma + 1) / (gamma + 3)) / (gamma + 2)

Storage: only the trainable (requires_grad) params are tracked and saved,
fp32, as one safetensors file per (t, sigma_rel). Frozen params and all
buffers are NOT duplicated -- they live once in the raw step_NNNNNN/
checkpoint and are merged back at load time (see load_ema_vae). Snapshots
are self-contained: any file can be deleted independently (keep a rolling
window around your best metric).

NOTE: accumulators are not checkpointed for resume. A saved shadow is a
valid EMA model on its own, but the running accumulator state is not
restored across a --skip_steps restart; treat each uninterrupted run's
snapshots as belonging to that run.
"""

import numpy as np
import torch
import safetensors.torch as st
from pathlib import Path


def sigma_rel_to_gamma(sigma_rel: float) -> float:
    """Invert sigma_rel(gamma) for gamma; solves the equivalent cubic."""
    t = sigma_rel ** -2
    roots = np.roots([1.0, 7.0, 16.0 - t, 12.0 - t])
    return float(roots[np.isreal(roots)].real.max())


class PowerEma:
    """
    Maintains one power-EMA shadow per sigma_rel, over the model's
    trainable (requires_grad) parameters only. Call update() after every
    optimizer step and save_snapshots() at every checkpoint.

    Each saved snapshot is a full-precision, standalone EMA of the
    trainable params for one sigma_rel. Reconstitute a complete VAE with
    load_ema_vae(raw_ckpt_dir, ema_file).
    """

    def __init__(self, model, sigma_rels):
        self.sigma_rels = list(sigma_rels)
        self.gammas = [sigma_rel_to_gamma(sr) for sr in self.sigma_rels]
        self.t = 0
        # Track only trainable params. Keyset is fixed at construction:
        # if you unfreeze params mid-run, they will NOT be picked up.
        self.shadows = [
            {n: p.detach().clone()
             for n, p in model.named_parameters() if p.requires_grad}
            for _ in self.gammas
        ]

    @torch.no_grad()
    def update(self, model):
        self.t += 1
        betas = [(1.0 - 1.0 / self.t) ** (g + 1.0) for g in self.gammas]
        for n, p in model.named_parameters():
            if n not in self.shadows[0]:
                continue
            pd = p.detach()
            for beta, shadow in zip(betas, self.shadows):
                shadow[n].lerp_(pd, 1.0 - beta)

    @torch.no_grad()
    def save_snapshots(self, dir_path):
        """
        Write one fp32 safetensors of EMA'd trainable params per
        accumulator. No frozen params, no buffers -- merged from the raw
        step checkpoint at load time. Filenames encode (t, sigma_rel);
        metadata carries them too. Each file is independently deletable.
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        for sr, shadow in zip(self.sigma_rels, self.shadows):
            tensors = {n: v.detach().float().cpu().contiguous()
                       for n, v in shadow.items()}
            fname = dir_path / f"ema_t{self.t:08d}_sr{sr:.4f}.safetensors"
            tmp = fname.with_suffix(".safetensors.tmp")
            st.save_file(
                tensors,
                str(tmp),
                metadata={"t": str(self.t), "sigma_rel": f"{sr:.4f}"},
            )
            tmp.replace(fname)  # atomic: no half-written snapshot on crash


def load_ema_vae(raw_ckpt_dir, ema_file, device="cpu", dtype=torch.float32):
    """
    Build a complete EMA VAE: the raw checkpoint's frozen params + buffers,
    with trainable params overwritten by a saved EMA shadow.

    raw_ckpt_dir : a step_NNNNNN/ dir (config + full weights) saved by the
                   trainer. Supplies architecture, buffers, and any frozen
                   params the EMA did not track.
    ema_file     : an ema_t*_sr*.safetensors written by save_snapshots.
    """
    from diffusers import AutoencoderKL
    try:
        vae = AutoencoderKL.from_pretrained(str(raw_ckpt_dir), torch_dtype=dtype)
    except EnvironmentError:
        vae = AutoencoderKL.from_pretrained(
            str(raw_ckpt_dir), subfolder="vae", torch_dtype=dtype)
    vae = vae.to(device)

    ema = st.load_file(str(ema_file))
    model_params = dict(vae.named_parameters())
    missing = [k for k in ema if k not in model_params]
    if missing:
        raise SystemExit(
            f"{len(missing)} EMA param(s) absent from model, e.g. "
            f"{missing[0]} -- wrong raw_ckpt_dir for this snapshot?")
    with torch.no_grad():
        for n, p in vae.named_parameters():
            if n in ema:
                p.copy_(ema[n].to(device=device, dtype=p.dtype))
    return vae
