#!/usr/bin/env python3
"""
Compare raw checkpoints against online EMA snapshots across a 2D grid of
(step, sigma_rel), on a single test image.

This is the DIRECT-SAVE variant, paired with the rewritten
train_posthoc_ema.py. There is no post-hoc least-squares reconstruction:
each EMA snapshot is a full-precision online EMA of the trainable params
for one sigma_rel, saved as ema_t*_sr*.safetensors. Scoring merges that
shadow onto the step's raw checkpoint (for frozen params + buffers +
config) and evaluates the resulting VAE.

For each step, reports L1 / LPIPS(vgg) / PSNR / SSIM against the original
test image for:
  - the raw (non-EMA) checkpoint at that step
  - each requested sigma_rel's EMA snapshot at that step

Both axes self-adjust from what's actually in ph_ema/ unless overridden:
  - --step is optional. Without it, the sweep covers the last --step_sweep
    snapshot steps. With it, --step_sweep entries centered on that step's
    position in the available list, clamped to range.
  - --sigma_rels defaults to every sigma_rel anchor actually saved in
    ph_ema/. Only saved anchors are scorable (no interpolation).

Run from inside the training --output_dir (contains step_NNNNNN/ dirs and
ph_ema/):

    python eval_ema_checkpoints.py --test_img /data/test/sample.jpg

Center the --step_sweep window (default 5) on a specific step:

    python eval_ema_checkpoints.py --step 150000 --test_img /data/test/sample.jpg

Dump a scored EMA model to disk (self-verifying: re-scores the saved dir
and asserts it matches the row) for use as a training seed:

    python eval_ema_checkpoints.py --step 84000 --sigma_rels 0.05 \
        --dump step_084000_ema --test_img /data/test/sample.jpg

Test image is always evaluated at 512x512.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import safetensors.torch as st

from train_datatools import load_image_tensor
from train_posthoc_ema import load_ema_vae
from compare_loss import load_vgg


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


def available_steps(ph_ema_dir: Path) -> list:
    """Every distinct step with an EMA snapshot, sorted ascending."""
    return sorted({t for t, _, _ in scan_snapshots(ph_ema_dir)})


def available_sigma_rels(ph_ema_dir: Path) -> list:
    """Every distinct sigma_rel anchor saved, sorted ascending."""
    return sorted({sr for _, sr, _ in scan_snapshots(ph_ema_dir)})


def snapshot_for(ph_ema_dir: Path, step: int, sigma_rel: float) -> Path:
    """Exact snapshot file for (step, sigma_rel), or SystemExit."""
    for t, sr, f in scan_snapshots(ph_ema_dir):
        if t == step and abs(sr - sigma_rel) < 1e-6:
            return f
    raise SystemExit(
        f"No EMA snapshot for step={step} sigma_rel={sigma_rel:.4f} in {ph_ema_dir}")


def step_window(available: list, step: int = None, width: int = 5) -> list:
    """
    width-wide slice of `available` steps: last `width` if step is None,
    else `width` entries centered on `step`'s position, clamped to range.
    """
    n = len(available)
    if step is None:
        return available[-width:]
    if step not in available:
        raise SystemExit(f"No EMA snapshot at step {step}; available: {available}")
    idx = available.index(step)
    w = min(width, n)
    start = max(0, min(idx - width // 2, n - w))
    return available[start:start + w]


def load_raw_vae(ckpt_dir: Path, device, dtype=torch.float32):
    """Load a raw step checkpoint as a VAE (config + full weights)."""
    from diffusers import AutoencoderKL
    try:
        vae = AutoencoderKL.from_pretrained(str(ckpt_dir), torch_dtype=dtype)
    except EnvironmentError:
        vae = AutoencoderKL.from_pretrained(
            str(ckpt_dir), subfolder="vae", torch_dtype=dtype)
    return vae.to(device)


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


def quantize_delivered(x_01: torch.Tensor) -> torch.Tensor:
    """Round-trip through uint8 quantization, matching the actual delivered
    image (see tensor_to_pil_rgb in train_datatools.py) instead of scoring
    the raw float32 decoder output nobody will ever actually see."""
    return (x_01.clamp(0.0, 1.0) * 255.0).round() / 255.0


@torch.no_grad()
def evaluate(vae, x_pm1: torch.Tensor, lpips_fn) -> dict:
    """x_pm1: (1, 3, H, W) in [-1, 1]. Runs encode/decode, quantizes the
    decode to uint8 (the actual delivered image), scores against x_pm1."""
    was_training = vae.training
    vae.eval()

    # Match write_vae_sample_webp: full fp32 precision, no TF32 rounding,
    # since that's what actually produced the delivered image.
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dec = vae.decode(vae.encode(x_pm1).latent_dist.mean).sample

    torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
    torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
    if was_training:
        vae.train()

    x_01 = (x_pm1 / 2 + 0.5).clamp(0, 1)
    dec_01 = quantize_delivered(dec / 2 + 0.5)
    dec_pm1 = dec_01 * 2 - 1
    return {
        "l1": F.l1_loss(dec_01, x_01).item(),
        "lpips": lpips_fn(dec_pm1, x_pm1).mean().item(),
        "psnr": psnr(dec_01, x_01),
        "ssim": ssim(dec_01, x_01),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", default=".",
                    help="train_vae.py's --output_dir (contains step_NNNNNN/ and ph_ema/)."
                         " Defaults to the current directory.")
    ap.add_argument("--step", type=int, default=None,
                    help="Center step for a sweep (that step plus step_sweep//2"
                         " available steps before/after, clamped to disk). Must"
                         " match a step with an EMA snapshot. Defaults to the"
                         " last step_sweep available steps.")
    ap.add_argument("--step_sweep", type=int, default=5,
                    help="Width of the step window: this many steps total,"
                         " centered on --step (or most recent if omitted).")
    ap.add_argument("--test_img", default="/data/models/sampleimg-full-hf-512.png")
    ap.add_argument("--sigma_rels", type=float, nargs="+", default=None,
                    help="EMA sigma_rel anchors to evaluate at each step."
                         " Must match saved snapshot anchors (no interpolation)."
                         " Defaults to every anchor saved in ph_ema/.")
    ap.add_argument("--base_config", default=None,
                    help="Architecture / frozen-weight source for EMA merges."
                         " Defaults to each step's own raw checkpoint.")
    ap.add_argument("--dump", default=None,
                    help="If set with a single --step and single --sigma_rels,"
                         " save the scored EMA VAE to this dir, then re-score the"
                         " saved model and assert it reproduces the row.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    ph_ema_dir = out_dir / "ph_ema"

    steps = step_window(available_steps(ph_ema_dir), args.step, width=args.step_sweep)
    if args.sigma_rels is not None:
        sigma_rels = args.sigma_rels
    else:
        sigma_rels = available_sigma_rels(ph_ema_dir)
    print(f"steps: {steps}")
    print(f"sigma_rels: {sigma_rels}")

    if args.dump is not None and not (len(steps) == 1 and len(sigma_rels) == 1):
        raise SystemExit("--dump requires exactly one --step and one --sigma_rels")

    x = load_image_tensor(Path(args.test_img), 512, 512)
    x = x.unsqueeze(0).to(device, dtype=torch.float32)

    lpips_fn = load_vgg(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    rows = []
    for step in steps:
        ckpt_dir = out_dir / f"step_{step:06d}"
        if not ckpt_dir.is_dir():
            raise SystemExit(f"No checkpoint at {ckpt_dir}")
        base_dir = Path(args.base_config) if args.base_config else ckpt_dir

        print(f"[step {step}] [raw] {ckpt_dir}")
        vae = load_raw_vae(ckpt_dir, device)
        rows.append((step, "raw", evaluate(vae, x, lpips_fn)))
        del vae

        for sr in sigma_rels:
            snap = snapshot_for(ph_ema_dir, step, sr)
            print(f"[step {step}] [ema sigma_rel={sr}] {snap.name} <- {base_dir}")
            vae = load_ema_vae(base_dir, snap, device=device)
            metrics = evaluate(vae, x, lpips_fn)
            rows.append((step, f"sr{sr}", metrics))

            if args.dump is not None:
                dump_dir = Path(args.dump)
                dump_dir.mkdir(parents=True, exist_ok=True)
                vae.save_pretrained(str(dump_dir))
                print(f"[dump] saved {dump_dir}; verifying round-trip...")
                del vae
                check = load_raw_vae(dump_dir, device)
                cm = evaluate(check, x, lpips_fn)
                del check
                for k in ("l1", "lpips", "psnr", "ssim"):
                    if abs(cm[k] - metrics[k]) > 1e-3:
                        raise SystemExit(
                            f"[dump] round-trip mismatch on {k}: "
                            f"scored {metrics[k]:.4f}, reloaded {cm[k]:.4f}")
                print(f"[dump] verified: {dump_dir} reproduces the scored row")
            else:
                del vae

    print()
    print("Using test image of", args.test_img)
    header = f"{'step':>8} {'candidate':<12} {'l1':>8} {'lpips(vgg)':>10} {'psnr':>8} {'ssim':>8}"
    print(header)
    print("-" * len(header))
    for step, name, m in rows:
        print(f"{step:8d} {name:<12} {m['l1']:8.4f} {m['lpips']:10.4f} "
              f"{m['psnr']:8.2f} {m['ssim']:8.4f}")


if __name__ == "__main__":
    main()
