#!/usr/bin/env python3
"""
Synthesize an EMA model of any averaging length from the power-EMA
snapshots written during training (see --posthoc_ema in train_args.py).

Example:
    python reconstruct_ph_ema.py \
        --ph_ema_dir out/ph_ema \
        --sigma_rel 0.07 \
        --output_dir out/ema_sr0.07

Sweep several lengths and compare with your validation tooling:
    for sr in 0.02 0.05 0.08 0.12; do
        python reconstruct_ph_ema.py --ph_ema_dir out/ph_ema \
            --sigma_rel $sr --output_dir out/ema_sr$sr
    done

The reconstruction target step is --step if given, else the newest
snapshot's step in --ph_ema_dir. --base_config is optional and defaults
to <ph_ema_dir>/../step_NNNNNN for that target step (the raw checkpoint
train_vae.py saves alongside each ph_ema snapshot batch). Pass
--base_config explicitly to borrow the architecture/frozen weights from
a different checkpoint instead.

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


def reconstruct_state(ph_ema_dir: Path, sigma_rel: float, step: int = None,
                       verbose: bool = True) -> dict:
    """
    Solve post-hoc EMA weights for sigma_rel from snapshots in ph_ema_dir.
    Reconstructs at step if given, else at the newest snapshot's step.
    Returns the combined state dict (param name -> fp32 tensor).
    """
    snaps = scan_snapshots(ph_ema_dir)
    if step is not None:
        snaps = [s for s in snaps if s[0] <= step]
        if not snaps:
            raise SystemExit(f"No snapshots with t <= {step} in {ph_ema_dir}")
    t_i = np.array([s[0] for s in snaps], dtype=np.float64)
    gamma_i = np.array([sigma_rel_to_gamma(s[1]) for s in snaps], dtype=np.float64)
    t_r = float(step) if step is not None else t_i.max()
    gamma_r = sigma_rel_to_gamma(sigma_rel)

    x = solve_weights(t_i, gamma_i, t_r, gamma_r)
    if verbose:
        print(f"Reconstructing sigma_rel={sigma_rel} at t={int(t_r)}"
              f" from {len(snaps)} snapshots:")
        for (t, sr, f), w in zip(snaps, x):
            print(f"  {f.name}: {w:+.5f}")

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
    return combined


def load_autoencoder(base_config, device="cpu", dtype=torch.float32) -> AutoencoderKL:
    """Load a VAE for its config/architecture, e.g. to receive a reconstructed state dict."""
    try:
        vae = AutoencoderKL.from_pretrained(str(base_config), torch_dtype=dtype)
    except EnvironmentError:
        vae = AutoencoderKL.from_pretrained(
            str(base_config), subfolder="vae", torch_dtype=dtype)
    return vae.to(device)


def apply_state(vae: AutoencoderKL, combined: dict) -> None:
    """Copy a reconstructed state dict into vae's matching named parameters in-place."""
    device = next(vae.parameters()).device
    param_names = {n for n, _ in vae.named_parameters()}
    unmatched = [n for n in combined if n not in param_names]
    if unmatched:
        raise SystemExit(
            f"{len(unmatched)} snapshot params not found in base model,"
            f" e.g. {unmatched[0]} - wrong --base_config?")
    with torch.no_grad():
        for n, p in vae.named_parameters():
            if n in combined:
                p.copy_(combined[n].to(device))


def main():
    ap = argparse.ArgumentParser(
        description="Reconstruct a post-hoc EMA VAE from power-EMA snapshots.")
    ap.add_argument("--ph_ema_dir", default="ph_ema",
                    help="Directory of ph_ema_*.pt snapshots from train_vae.py")
    ap.add_argument("-b", "--base_config", default=None,
                    help="Saved checkpoint dir this reconstruction borrows its"
                         " model config/architecture (and any non-EMA'd frozen"
                         " weights) from. Defaults to <ph_ema_dir>/../step_NNNNNN"
                         " for the target step being reconstructed (see --step)."
                         " Override to borrow architecture/frozen weights from a"
                         " different checkpoint instead.")
    ap.add_argument("-s","--sigma_rel", type=float, required=True,
                    help="Target averaging length. Try sweeping 0.02 - 0.15.")
    ap.add_argument("--step", type=int, default=None,
                    help="Only use snapshots with t <= this step, and"
                         " reconstruct at this step instead of the newest"
                         " snapshot's step. Use this to reconstruct as of an"
                         " earlier point in training.")
    ap.add_argument("-o","--output_dir", default=None,
                    help="Where to save the reconstructed VAE."
                         " Defaults to <base_config>_ema, e.g."
                         " step_NNNNNN_ema.")
    args = ap.parse_args()

    ph_ema_dir = Path(args.ph_ema_dir)
    if args.base_config:
        base_config = Path(args.base_config)
    else:
        t_r = args.step if args.step is not None else max(t for t, _, _ in scan_snapshots(ph_ema_dir))
        base_config = ph_ema_dir.parent / f"step_{t_r:06d}"

    combined = reconstruct_state(ph_ema_dir, args.sigma_rel, args.step)
    vae = load_autoencoder(base_config)
    apply_state(vae, combined)

    out = Path(args.output_dir) if args.output_dir else Path(f"{base_config}_ema")
    out.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(str(out))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
