#!/usr/bin/env python3
"""
reconstruct_ph_ema.py

Materialize a full, loadable EMA VAE from a direct-save EMA snapshot
(see train_posthoc_ema.py). Despite the historical name, there is no
post-hoc least-squares reconstruction anymore: snapshots are online EMAs
of the trainable params, saved fp32 as ema_t*_sr*.safetensors. This tool
merges one such shadow onto a raw checkpoint (for frozen params, buffers,
and config) and writes a standalone VAE dir.

Same CLI as before:

    reconstruct_ph_ema.py --ph_ema_dir <out>/ph_ema \
        --base <out>/step_084000 \
        --sigma_rel 0.05 \
        --output_dir step_084000_ema

--step is optional: pick the snapshot at that t (default: newest t present
for the requested sigma_rel).
"""

import re
import argparse
from pathlib import Path

import torch
import safetensors.torch as st
from diffusers import AutoencoderKL


SNAP_RE = re.compile(r"ema_t(\d+)_sr([0-9.]+)\.safetensors$")


def scan_snapshots(ph_ema_dir: Path):
    """Return [(t, sigma_rel, path)] for every EMA snapshot in the dir."""
    snaps = []
    for f in sorted(ph_ema_dir.glob("ema_t*_sr*.safetensors")):
        m = SNAP_RE.search(f.name)
        if m:
            snaps.append((int(m.group(1)), float(m.group(2)), f))
    if not snaps:
        raise SystemExit(f"No ema_t*_sr*.safetensors snapshots in {ph_ema_dir}")
    return snaps


def pick_snapshot(ph_ema_dir: Path, sigma_rel: float, step: int | None) -> Path:
    """Snapshot for sigma_rel at t==step, or newest t if step is None."""
    cands = [(t, f) for t, sr, f in scan_snapshots(ph_ema_dir)
             if abs(sr - sigma_rel) < 1e-6]
    if not cands:
        raise SystemExit(f"No snapshots with sigma_rel={sigma_rel:.4f} in {ph_ema_dir}")
    if step is None:
        return max(cands, key=lambda c: c[0])[1]
    for t, f in cands:
        if t == step:
            return f
    raise SystemExit(
        f"No sigma_rel={sigma_rel:.4f} snapshot at step {step}; "
        f"available: {sorted(t for t, _ in cands)}")


def load_ema_vae(base: Path, ema_file: Path, dtype=torch.float32) -> AutoencoderKL:
    """Raw checkpoint's frozen params + buffers + config, with trainable
    params overwritten by the EMA shadow."""
    try:
        vae = AutoencoderKL.from_pretrained(str(base), torch_dtype=dtype)
    except EnvironmentError:
        vae = AutoencoderKL.from_pretrained(str(base), subfolder="vae", torch_dtype=dtype)

    ema = st.load_file(str(ema_file))
    model_params = dict(vae.named_parameters())
    missing = [k for k in ema if k not in model_params]
    if missing:
        raise SystemExit(
            f"{len(missing)} EMA param(s) absent from --base model, e.g. "
            f"{missing[0]} -- wrong --base for this snapshot?")
    with torch.no_grad():
        for n, p in vae.named_parameters():
            if n in ema:
                p.copy_(ema[n].to(dtype=p.dtype))
    return vae


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ph_ema_dir", required=True,
                    help="Directory of ema_t*_sr*.safetensors snapshots"
                         " (<train output_dir>/ph_ema)")
    ap.add_argument("--base", required=True,
                    help="Raw step_NNNNNN/ checkpoint supplying config, buffers,"
                         " and any frozen (non-EMA'd) weights.")
    ap.add_argument("--sigma_rel", type=float, required=True,
                    help="Which saved EMA anchor to materialize.")
    ap.add_argument("--step", type=int, default=None,
                    help="Snapshot step t to use. Default: newest available.")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    ema_file = pick_snapshot(Path(args.ph_ema_dir), args.sigma_rel, args.step)
    print(f"Merging {ema_file.name} onto {args.base}")

    vae = load_ema_vae(Path(args.base), ema_file)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(str(out))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
