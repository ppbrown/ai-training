#!/usr/bin/env python3
"""
Synthesize an EMA model of any averaging length from the power-EMA
snapshots written during training (see --posthoc_ema in train_args.py).

Example:
    python reconstruct_ph_ema.py \
        --ph_ema_dir out/ph_ema \
        --base out/final \
        --sigma_rel 0.07 \
        --output_dir out/ema_sr0.07

Sweep several lengths and compare with your validation tooling:
    for sr in 0.02 0.05 0.08 0.12; do
        python reconstruct_ph_ema.py --ph_ema_dir out/ph_ema --base out/final \
            --sigma_rel $sr --output_dir out/ema_sr$sr
    done

The solved snapshot coefficients are printed; negative values and
values above 1 are normal and expected.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL

from train_posthoc_ema import sigma_rel_to_gamma, solve_weights

FNAME_RE = re.compile(r"ph_ema_t(\d+)_sr([0-9.]+)\.pt$")


def scan_snapshots(ph_ema_dir: Path):
    """Return [(t, sigma_rel, path)] for every snapshot in the dir."""
    snaps = []
    for f in sorted(ph_ema_dir.glob("ph_ema_t*_sr*.pt")):
        m = FNAME_RE.search(f.name)
        if m:
            snaps.append((int(m.group(1)), float(m.group(2)), f))
    if not snaps:
        raise SystemExit(f"No ph_ema_t*_sr*.pt snapshots found in {ph_ema_dir}")
    return snaps


def main():
    ap = argparse.ArgumentParser(
        description="Reconstruct a post-hoc EMA VAE from power-EMA snapshots.")
    ap.add_argument("--ph_ema_dir", required=True,
                    help="Directory of ph_ema_*.pt snapshots (<train output_dir>/ph_ema)")
    ap.add_argument("--base", required=True,
                    help="Saved checkpoint dir supplying config and any frozen"
                         " weights, e.g. <train output_dir>/final")
    ap.add_argument("--sigma_rel", type=float, required=True,
                    help="Target averaging length. Try sweeping 0.02 - 0.15.")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    snaps = scan_snapshots(Path(args.ph_ema_dir))
    t_i = np.array([s[0] for s in snaps], dtype=np.float64)
    gamma_i = np.array([sigma_rel_to_gamma(s[1]) for s in snaps], dtype=np.float64)
    t_r = t_i.max()
    gamma_r = sigma_rel_to_gamma(args.sigma_rel)

    x = solve_weights(t_i, gamma_i, t_r, gamma_r)
    print(f"Reconstructing sigma_rel={args.sigma_rel} at t={int(t_r)}"
          f" from {len(snaps)} snapshots:")
    for (t, sr, f), w in zip(snaps, x):
        print(f"  {f.name}: {w:+.5f}")

    # Stream snapshots one at a time; accumulate in fp32.
    combined = None
    for (t, sr, f), w in zip(snaps, x):
        data = torch.load(f, map_location="cpu")
        assert data["t"] == t and abs(data["sigma_rel"] - sr) < 1e-6, \
            f"metadata mismatch in {f.name}"
        state = data["state"]
        if combined is None:
            combined = {n: v.float() * w for n, v in state.items()}
        else:
            for n, v in state.items():
                combined[n] += v.float() * w
        del data, state

    try:
        vae = AutoencoderKL.from_pretrained(args.base, torch_dtype=torch.float32)
    except EnvironmentError:
        vae = AutoencoderKL.from_pretrained(
            args.base, subfolder="vae", torch_dtype=torch.float32)

    param_names = {n for n, _ in vae.named_parameters()}
    unmatched = [n for n in combined if n not in param_names]
    if unmatched:
        raise SystemExit(
            f"{len(unmatched)} snapshot params not found in base model,"
            f" e.g. {unmatched[0]} - wrong --base?")

    with torch.no_grad():
        for n, p in vae.named_parameters():
            if n in combined:
                p.copy_(combined[n])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(str(out))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
