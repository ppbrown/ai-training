#!/usr/bin/env python3
"""
Compare raw checkpoints against online EMA snapshots across a 2D grid of
(step, sigma_rel), on a single test image.

Matches the DIRECT-SAVE EMA scheme in train_ema.py / train_vae.py: at each
checkpoint, train_vae.py swaps in each sigma_rel's EMA shadow and calls
vae.save_pretrained() directly, so every step_NNNNNN/ema/sr<SIGMA_REL>/ dir
is already a complete, standalone VAE checkpoint -- no post-hoc merging
onto a raw checkpoint is needed.

For each step, reports L1 / LPIPS(vgg) / PSNR / SSIM against the original
test image for:
  - the raw (non-EMA) checkpoint at that step
  - each requested sigma_rel's EMA snapshot at that step

Both axes self-adjust from what's actually on disk unless overridden:
  - --step is optional. Without it, the sweep covers the last --step_sweep
    step_NNNNNN/ dirs found under --output_dir. With it, --step_sweep
    entries centered on that step's position in the available list,
    clamped to range.
  - --sigma_rels defaults to every sigma_rel anchor actually saved (i.e.
    every step_NNNNNN/ema/sr*/ dir) among the swept steps. Only saved
    anchors are scorable (no interpolation).

Run from inside the training --output_dir (contains step_NNNNNN/ dirs,
each with its own ema/sr*/ subdirs):

    python eval_ema_checkpoints.py --test_img /data/test/sample.jpg

Center the --step_sweep window (default 5) on a specific step:

    python eval_ema_checkpoints.py --step 150000 --test_img /data/test/sample.jpg

Dump a scored EMA snapshot to disk (self-verifying: re-scores the copy
and asserts it matches the row) for use as a training seed:

    python eval_ema_checkpoints.py --step 84000 --sigma_rels 0.05 \
        --dump step_084000_ema --test_img /data/test/sample.jpg

Test image is always evaluated at 512x512.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

from train_datatools import load_image_tensor
from compare_loss import load_vgg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cache-utils"))
from show_vae_latent import decode_latent_to_pil


STEP_DIR_RE = re.compile(r"step_(\d+)$")
SR_DIR_RE = re.compile(r"sr([0-9.]+)$")


def available_steps(out_dir: Path) -> list:
    """Every step with a step_NNNNNN/ raw checkpoint dir, sorted ascending."""
    steps = []
    for d in out_dir.iterdir():
        if d.is_dir():
            m = STEP_DIR_RE.fullmatch(d.name)
            if m:
                steps.append(int(m.group(1)))
    if not steps:
        raise SystemExit(f"No step_NNNNNN/ checkpoints in {out_dir}")
    return sorted(steps)


def ema_root(out_dir: Path, step: int) -> Path:
    return out_dir / f"step_{step:06d}" / "ema"


def available_sigma_rels(out_dir: Path, steps: list) -> list:
    """Every distinct sigma_rel anchor saved under any of `steps`' ema/ dirs."""
    srs = set()
    for step in steps:
        root = ema_root(out_dir, step)
        if not root.is_dir():
            continue
        for d in root.iterdir():
            m = SR_DIR_RE.fullmatch(d.name)
            if m:
                srs.add(float(m.group(1)))
    if not srs:
        raise SystemExit(f"No sr*/ EMA dirs found under steps {steps}")
    return sorted(srs)


def ema_dir_for(out_dir: Path, step: int, sigma_rel: float) -> Path:
    """Exact EMA snapshot dir for (step, sigma_rel), or SystemExit."""
    d = ema_root(out_dir, step) / f"sr{sigma_rel:.4f}"
    if not d.is_dir():
        raise SystemExit(
            f"No EMA dir for step={step} sigma_rel={sigma_rel:.4f}: {d}")
    return d


def step_window(available: list, step: int = None, width: int = 5) -> list:
    """
    width-wide slice of `available` steps: last `width` if step is None,
    else `width` entries centered on `step`'s position, clamped to range.
    """
    n = len(available)
    if step is None:
        return available[-width:]
    if step not in available:
        raise SystemExit(f"No checkpoint at step {step}; available: {available}")
    idx = available.index(step)
    w = min(width, n)
    start = max(0, min(idx - width // 2, n - w))
    return available[start:start + w]


def load_vae_dir(ckpt_dir: Path, device, dtype=torch.float32):
    """Load a complete saved VAE dir -- either a step_NNNNNN/ raw checkpoint
    or a step_NNNNNN/ema/sr*/ standalone EMA snapshot -- config + full weights."""
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


@torch.no_grad()
def evaluate(vae, x_pm1: torch.Tensor, lpips_fn) -> dict:
    """x_pm1: (1, 3, H, W) in [-1, 1]. Runs encode/decode via
    show_vae_latent.decode_latent_to_pil (the same decode path the
    standalone cache-viewing tool uses), so scoring matches what that
    tool actually shows/writes, then scores the quantized result against
    x_pm1."""
    was_training = vae.training
    vae.eval()

    # Match write_vae_sample_webp: full fp32 precision, no TF32 rounding,
    # since that's what actually produced the delivered image.
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    latent = vae.encode(x_pm1).latent_dist.mean.squeeze(0)
    pil_img = decode_latent_to_pil(vae, latent)

    torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
    torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
    if was_training:
        vae.train()

    dec_01 = to_tensor(pil_img).unsqueeze(0).to(x_pm1.device, dtype=x_pm1.dtype)
    dec_pm1 = dec_01 * 2 - 1
    x_01 = (x_pm1 / 2 + 0.5).clamp(0, 1)
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
                    help="train_vae.py's --output_dir (contains step_NNNNNN/"
                         " dirs, each with its own ema/sr*/ subdirs)."
                         " Defaults to the current directory.")
    ap.add_argument("--step", type=int, default=None,
                    help="Center step for a sweep (that step plus step_sweep//2"
                         " available steps before/after, clamped to disk). Must"
                         " be a step_NNNNNN/ checkpoint on disk. Defaults to"
                         " the last step_sweep available steps.")
    ap.add_argument("--step_sweep", type=int, default=5,
                    help="Width of the step window: this many steps total,"
                         " centered on --step (or most recent if omitted).")
    ap.add_argument("--test_img", default="/data/models/sampleimg-full-hf-512.png")
    ap.add_argument("--sigma_rels", type=float, nargs="+", default=None,
                    help="EMA sigma_rel anchors to evaluate at each step."
                         " Must match saved step_NNNNNN/ema/sr*/ dirs (no"
                         " interpolation). Defaults to every anchor saved"
                         " among the swept steps.")
    ap.add_argument("--dump", default=None,
                    help="If set with a single --step and single --sigma_rels,"
                         " copy that step's EMA snapshot dir to this path, then"
                         " re-score the copy and assert it reproduces the row.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)

    steps = step_window(available_steps(out_dir), args.step, width=args.step_sweep)
    if args.sigma_rels is not None:
        sigma_rels = args.sigma_rels
    else:
        sigma_rels = available_sigma_rels(out_dir, steps)
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

        print(f"[step {step}] [raw] {ckpt_dir}")
        vae = load_vae_dir(ckpt_dir, device)
        rows.append((step, "raw", evaluate(vae, x, lpips_fn)))
        del vae

        for sr in sigma_rels:
            snap_dir = ema_dir_for(out_dir, step, sr)
            print(f"[step {step}] [ema sigma_rel={sr}] {snap_dir}")
            vae = load_vae_dir(snap_dir, device)
            metrics = evaluate(vae, x, lpips_fn)
            rows.append((step, f"sr{sr}", metrics))

            if args.dump is not None:
                dump_dir = Path(args.dump)
                if dump_dir.exists():
                    raise SystemExit(f"--dump target already exists: {dump_dir}")
                shutil.copytree(snap_dir, dump_dir)
                print(f"[dump] copied {snap_dir} -> {dump_dir}; verifying round-trip...")
                del vae
                check = load_vae_dir(dump_dir, device)
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
