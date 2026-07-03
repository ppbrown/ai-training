"""
train_posthoc_ema.py

Power-function EMA with post-hoc reconstruction, from Karras et al.,
"Analyzing and Improving the Training Dynamics of Diffusion Models"
(EDM2, arXiv 2312.02696).

Instead of committing to one EMA decay before training, maintain two
power-EMA accumulators (sigma_rel 0.05 and 0.10 by default) and save
fp16 snapshots of both at every checkpoint. After training, an EMA of
any averaging length can be synthesized from the stored snapshots by
solving a small least-squares problem. See reconstruct_ph_ema.py.

The power-EMA update at step t is

    shadow = beta * shadow + (1 - beta) * param
    beta   = (1 - 1/t) ** (gamma + 1)

which corresponds to an averaging profile w(tau) proportional to
tau ** gamma over the run so far. At t=1 beta is 0, so the accumulator
self-initializes to the model weights (no init bias).

sigma_rel is the relative standard deviation of that profile:
roughly, the fraction of the total run being averaged over.
Larger sigma_rel = longer averaging. sigma_rel and gamma relate by

    sigma_rel = sqrt((gamma + 1) / (gamma + 3)) / (gamma + 2)

NOTE: accumulators are not checkpointed. Snapshots are only valid for
reconstruction within a single uninterrupted run; do not mix snapshots
from a run and its --skip_steps restart.
"""

import numpy as np
import torch


def sigma_rel_to_gamma(sigma_rel: float) -> float:
    """Invert sigma_rel(gamma) for gamma; solves the equivalent cubic."""
    t = sigma_rel ** -2
    roots = np.roots([1.0, 7.0, 16.0 - t, 12.0 - t])
    return float(roots[np.isreal(roots)].real.max())


def p_dot_p(t_a, gamma_a, t_b, gamma_b):
    """Inner product between two normalized power-EMA weight profiles."""
    t_ratio = t_a / t_b
    t_exp = np.where(t_a < t_b, gamma_b, -gamma_a)
    t_max = np.maximum(t_a, t_b)
    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio ** t_exp
    den = (gamma_a + gamma_b + 1) * t_max
    return num / den


def solve_weights(t_i, gamma_i, t_r, gamma_r):
    """
    Least-squares coefficients that combine stored snapshots
    (t_i, gamma_i) into the best approximation of a target power-EMA
    profile (t_r, gamma_r). Negative coefficients are normal.
    """
    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)
    a = p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i))
    b = p_dot_p(rv(t_i), rv(gamma_i), cv([t_r]), cv([gamma_r]))
    return np.linalg.solve(a, b).flatten()


class PowerEma:
    """
    Maintains one power-EMA shadow per sigma_rel, over the model's
    trainable parameters only. Call update() after every optimizer
    step and save_snapshots() at every checkpoint.
    """

    def __init__(self, model, sigma_rels):
        self.sigma_rels = list(sigma_rels)
        self.gammas = [sigma_rel_to_gamma(sr) for sr in self.sigma_rels]
        self.t = 0
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
            for beta, shadow in zip(betas, self.shadows):
                shadow[n].lerp_(p.detach(), 1.0 - beta)

    def save_snapshots(self, dir_path):
        """Write one fp16 snapshot per accumulator, named by (t, sigma_rel)."""
        dir_path.mkdir(parents=True, exist_ok=True)
        for sr, shadow in zip(self.sigma_rels, self.shadows):
            fname = f"ph_ema_t{self.t:08d}_sr{sr:.4f}.pt"
            torch.save(
                {
                    "t": self.t,
                    "sigma_rel": sr,
                    "state": {n: v.half().cpu() for n, v in shadow.items()},
                },
                dir_path / fname,
            )
