"""
train_ema.py

Power-function EMA, adapted from Karras et al., "Analyzing and Improving
the Training Dynamics of Diffusion Models" (EDM2, arXiv 2312.02696).

This is the DIRECT-SAVE variant: instead of storing fp16 snapshots and
solving a least-squares post-hoc reconstruction after training, we keep
one full-precision online EMA accumulator per requested sigma_rel. At each
checkpoint the trainer swaps an accumulator into the live model and calls
vae.save_pretrained(), so every saved EMA dir is a complete, standalone
VAE for that sigma_rel -- no reconstruction, no interpolation, no merge.

Rationale for the change:
  - Post-hoc reconstruction combined fp16 snapshots with large,
    alternating-sign least-squares coefficients. At short sigma_rel that
    amplifies fp16 quantization into large perceptual (VGG) error while
    barely moving L1 -- reconstructions could score WORSE than raw.
  - It was also fragile to snapshot-set contamination (mixing runs /
    restarts, or reconstructing over a different set of t_i than was
    intended silently changed every coefficient).
  - We give up "reconstruct any sigma_rel in between the anchors after
    training" in exchange for correctness. You now pre-declare the
    sigma_rels you want and get an exact online EMA of each.

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

Usage in the trainer:
    ema = PowerEma(vae, sigma_rels=[...])
    ...
    opt.step()
    ema.update(vae)               # after every optimizer step
    ...
    # at each checkpoint, per sigma_rel index idx:
    ema.apply_to(vae, idx)        # swap shadow in, back up live weights
    vae.save_pretrained(ema_dir)  # a complete standalone EMA VAE
    ema.restore(vae)              # put the live weights back

Storage: each EMA is written as a full standalone VAE dir (config + all
weights + buffers), directly loadable with AutoencoderKL.from_pretrained.
The accumulators track only the trainable (requires_grad) params; frozen
params and buffers are taken from the live model at apply_to() time, so a
saved dir is always complete regardless of what was frozen.

Accumulator state (the shadows and step counter t) can be checkpointed via
state_dict()/load_state_dict() so a --continue_steps resume picks up the
EMA average exactly where it left off, rather than restarting it from the
resumed model weights.
"""

import numpy as np
import torch


def sigma_rel_to_gamma(sigma_rel: float) -> float:
    """Invert sigma_rel(gamma) for gamma; solves the equivalent cubic."""
    t = sigma_rel ** -2
    roots = np.roots([1.0, 7.0, 16.0 - t, 12.0 - t])
    return float(roots[np.isreal(roots)].real.max())


class PowerEma:
    """
    Maintains one power-EMA shadow per sigma_rel, over the model's
    trainable (requires_grad) parameters only. Call update() after every
    optimizer step.

    At each checkpoint, swap a shadow into the model with apply_to(idx),
    save it with vae.save_pretrained() (a complete standalone EMA VAE for
    that sigma_rel), then restore() the live weights.
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
        self.backup: dict = {}

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
    def apply_to(self, model, idx: int):
        """
        Swap the EMA shadow for sigma_rels[idx] into model, backing up the
        live weights. Call restore() afterward to put the live weights back.
        """
        self.backup = {}
        shadow = self.shadows[idx]
        for n, p in model.named_parameters():
            if n in shadow:
                self.backup[n] = p.detach().clone()
                p.copy_(shadow[n])

    @torch.no_grad()
    def restore(self, model):
        """Undo apply_to(): put the backed-up live weights back into model."""
        for n, p in model.named_parameters():
            if n in self.backup:
                p.copy_(self.backup[n])
        self.backup = {}

    def state_dict(self) -> dict:
        """Snapshot the step counter and all shadows for checkpointing."""
        return {
            "t": self.t,
            "shadows": [
                {n: p.detach().clone() for n, p in shadow.items()}
                for shadow in self.shadows
            ],
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        """Restore the step counter and shadows saved by state_dict()."""
        self.t = state["t"]
        for shadow, saved in zip(self.shadows, state["shadows"]):
            for n in shadow:
                shadow[n].copy_(saved[n])
