#!/usr/bin/env python3
"""
Compare raw checkpoints against post-hoc EMA reconstructions across a 2D
grid of (step, sigma_rel), on a single test image.

For each step, reports L1 / LPIPS / PSNR / SSIM against the original test
image for:
  - the raw (non-EMA) checkpoint at that step
  - a post-hoc EMA reconstruction at that step for each sigma_rel in the
    sweep, synthesized from ph_ema/ snapshots with t <= step
    (see reconstruct_ph_ema.py)

Both axes self-adjust from what's actually in ph_ema/ unless overridden:
  - --step is optional. Without it, the sweep covers the last 5 available
    steps. With it, the sweep covers that step plus the 2 available steps
    before and after it (by position in the available list, not raw step
    arithmetic), clamped so it never reaches past the last actual
    checkpoint - e.g. picking the newest available step still gives you
    the last 5, not a request for 2 steps that don't exist yet.
  - --sigma_rels defaults to 5 points spanning the sigma_rel anchors
    actually saved in ph_ema/

Run from inside the training --output_dir (contains step_NNNNNN/ dirs
and ph_ema/):

    python eval_ema_checkpoints.py --test_img /data/test/sample.jpg

Center the 5-step window on a specific step once you know which region
of training matters:

    python eval_ema_checkpoints.py --step 150000 --test_img /data/test/sample.jpg

Each (step, sigma_rel) reconstruction is cheap (a weighted tensor sum, not
a training pass), but loading a VAE from disk per candidate is not.
Test image is always evaluated at 512x512.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import lpips

from train_datatools import load_image_tensor
from reconstruct_ph_ema import reconstruct_state, load_autoencoder, apply_state, scan_snapshots


def available_steps(ph_ema_dir: Path) -> list:
    """Every distinct step with a ph_ema snapshot in ph_ema_dir, sorted ascending."""
    return sorted({t for t, _, _ in scan_snapshots(ph_ema_dir)})


def step_window(available: list, step: int = None, width: int = 5) -> list:
    """
    5-wide (by default) slice of `available` steps: last `width` if step is
    None, else `width` entries centered on `step`'s position, clamped to
    stay in range (shifts instead of running off either end).
    """
    n = len(available)
    if step is None:
        return available[-width:]
    if step not in available:
        raise SystemExit(f"No ph_ema snapshot at step {step}; available: {available}")
    idx = available.index(step)
    w = min(width, n)
    start = max(0, min(idx - width // 2, n - w))
    return available[start:start + w]


def auto_sigma_rels(ph_ema_dir: Path, num: int = 5) -> list:
    """
    Evenly spaced sigma_rel sweep spanning the anchors actually saved in
    ph_ema_dir (e.g. two anchors 0.05/0.15 -> 5 points from 0.05 to 0.15).
    """
    anchors = sorted({sr for _, sr, _ in scan_snapshots(ph_ema_dir)})
    if len(anchors) == 1:
        return anchors
    lo, hi = anchors[0], anchors[-1]
    return [round(v, 4) for v in np.linspace(lo, hi, num)]


def _gaussian_window(window_size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def ssim(a: torch.Tensor, b: torch.Tensor, window_size: int = 11) -> float:
    """Single-scale SSIM. a, b: (1, C, H, W) in [0, 1]."""
    device = a.device
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    g1d = _gaussian_window(window_size, 1.5, device)
    window = (g1d[:, None] @ g1d[None, :]).unsqueeze(0).unsqueeze(0)
    channels = a.shape[1]
    window = window.expand(channels, 1, window_size, window_size)
    pad = window_size // 2

    mu_a = F.conv2d(a, window, padding=pad, groups=channels)
    mu_b = F.conv2d(b, window, padding=pad, groups=channels)
    mu_a_sq, mu_b_sq, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a_sq = F.conv2d(a * a, window, padding=pad, groups=channels) - mu_a_sq
    sigma_b_sq = F.conv2d(b * b, window, padding=pad, groups=channels) - mu_b_sq
    sigma_ab = F.conv2d(a * b, window, padding=pad, groups=channels) - mu_ab

    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    )
    return ssim_map.mean().item()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


@torch.no_grad()
def evaluate(vae, x_pm1: torch.Tensor, lpips_fn) -> dict:
    """x_pm1: (1, 3, H, W) in [-1, 1]. Runs encode/decode and scores against x_pm1."""
    was_training = vae.training
    vae.eval()
    dec = vae.decode(vae.encode(x_pm1).latent_dist.mean).sample
    if was_training:
        vae.train()

    x_01 = (x_pm1 / 2 + 0.5).clamp(0, 1)
    dec_01 = (dec / 2 + 0.5).clamp(0, 1)
    return {
        "l1": F.l1_loss(dec, x_pm1).item(),
        "lpips": lpips_fn(dec, x_pm1).mean().item(),
        "psnr": psnr(dec_01, x_01),
        "ssim": ssim(dec_01, x_01),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", default=".",
                    help="train_vae.py's --output_dir (contains step_NNNNNN/ dirs and ph_ema/)."
                         " Defaults to the current directory.")
    ap.add_argument("--step", type=int, default=None,
                    help="Center step for a 5-checkpoint-wide sweep (that step plus"
                         " the 2 available steps before and after it, clamped to"
                         " what's actually on disk). Must match a step with a"
                         " ph_ema snapshot. Defaults to the last 5 available steps.")
    ap.add_argument("--test_img", default="/data/models/sampleimg-full-hf-512.png")
    ap.add_argument("--sigma_rels", type=float, nargs="+", default=None,
                    help="Post-hoc EMA lengths to reconstruct and evaluate at each step."
                         " Defaults to an evenly spaced sweep between the smallest"
                         " and largest sigma_rel anchor actually saved in ph_ema/.")
    ap.add_argument("--base_config", default=None,
                    help="Architecture source for EMA reconstructions."
                         " Defaults to each step's own raw checkpoint.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    ph_ema_dir = out_dir / "ph_ema"

    steps = step_window(available_steps(ph_ema_dir), args.step)
    sigma_rels = args.sigma_rels if args.sigma_rels is not None else auto_sigma_rels(ph_ema_dir)
    print(f"steps: {steps}")
    print(f"sigma_rels sweep: {sigma_rels}")

    x = load_image_tensor(Path(args.test_img), 512, 512)
    x = x.unsqueeze(0).to(device, dtype=torch.float32)

    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    rows = []
    for step in steps:
        ckpt_dir = out_dir / f"step_{step:06d}"
        if not ckpt_dir.is_dir():
            raise SystemExit(f"No checkpoint at {ckpt_dir}")
        base_config = Path(args.base_config) if args.base_config else ckpt_dir

        print(f"[step {step}] [raw] {ckpt_dir}")
        vae = load_autoencoder(ckpt_dir, device=device)
        rows.append((step, "raw", evaluate(vae, x, lpips_fn)))
        del vae

        for sr in sigma_rels:
            print(f"[step {step}] [ema sigma_rel={sr}] reconstructing from {base_config}")
            combined = reconstruct_state(ph_ema_dir, sr, max_t=step, verbose=False)
            vae = load_autoencoder(base_config, device=device)
            apply_state(vae, combined)
            rows.append((step, f"sr{sr}", evaluate(vae, x, lpips_fn)))
            del vae, combined

    print()
    header = f"{'step':>8} {'candidate':<12} {'l1':>8} {'lpips':>8} {'psnr':>8} {'ssim':>8}"
    print(header)
    print("-" * len(header))
    for step, name, m in rows:
        print(f"{step:8d} {name:<12} {m['l1']:8.4f} {m['lpips']:8.4f} {m['psnr']:8.2f} {m['ssim']:8.4f}")


if __name__ == "__main__":
    main()
