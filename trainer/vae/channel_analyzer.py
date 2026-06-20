#!/usr/bin/env python3
"""
Analyze each latent channel of a KL VAE (AutoencoderKL) and classify it as:
  - zero       : all encoder/decoder weights for this channel are near-zero
  - weak       : non-zero but very small (likely sparse/early-stage initialization)
  - complex    : substantial weights on both encoder and decoder sides

Checks the four conv layers that directly gate latent channels:
  encoder.conv_out  : produces (mean, logvar) for each channel
  quant_conv        : mixes moments (mean/logvar)
  post_quant_conv   : maps latent -> decoder input
  decoder.conv_in   : first decoder conv, reads each latent channel
"""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL


ZERO_THRESH = 1e-6   # max_abs below this → zero
WEAK_THRESH = 0.10   # max_abs below this → weak (above → complex)


def weight_stats(t: torch.Tensor) -> dict:
    t = t.float()
    n = t.numel()
    return {
        "norm_per_param": t.norm().item() / max(n, 1),
        "max_abs": t.abs().max().item(),
        "std": t.std().item(),
        "n": n,
    }


def classify(max_abs: float, zero_thresh: float, weak_thresh: float) -> str:
    if max_abs < zero_thresh:
        return "zero"
    if max_abs < weak_thresh:
        return "weak"
    return "complex"


def analyze_vae(model_path: str, zero_thresh: float = ZERO_THRESH, weak_thresh: float = WEAK_THRESH) -> None:
    print(f"Loading VAE from: {model_path}")
    vae = AutoencoderKL.from_pretrained(model_path, torch_dtype=torch.float32)
    vae.eval()

    n = vae.config.latent_channels
    print(f"latent_channels = {n}\n")

    enc_conv_out    = vae.encoder.conv_out          # (2n, in_ch, k, k)
    quant_conv      = vae.quant_conv                # (2n, 2n, 1, 1)
    post_quant_conv = vae.post_quant_conv           # (n, n, 1, 1)
    dec_conv_in     = vae.decoder.conv_in           # (out_ch, n, k, k)

    rows = []
    for c in range(n):
        # --- encoder side ---
        # mean output for channel c
        enc_mean_w  = enc_conv_out.weight[c]
        enc_mean_b  = enc_conv_out.bias[c].item() if enc_conv_out.bias is not None else 0.0
        # logvar output for channel c
        enc_lv_w    = enc_conv_out.weight[n + c]
        enc_lv_b    = enc_conv_out.bias[n + c].item() if enc_conv_out.bias is not None else 0.0

        # quant_conv: output mean row c, output logvar row n+c
        qc_mean_w   = quant_conv.weight[c]
        qc_lv_w     = quant_conv.weight[n + c]
        qc_lv_b     = quant_conv.bias[n + c].item() if quant_conv.bias is not None else 0.0

        # --- decoder side ---
        # post_quant_conv: row c (what channel c produces)
        pqc_row_w   = post_quant_conv.weight[c]
        # post_quant_conv: col c (what reads from channel c)
        pqc_col_w   = post_quant_conv.weight[:, c]

        # decoder.conv_in: col c (how each decoder feature reads channel c)
        dec_col_w   = dec_conv_in.weight[:, c]

        enc_s  = weight_stats(enc_mean_w)
        dec_s  = weight_stats(dec_col_w)
        pqc_s  = weight_stats(pqc_col_w)

        # overall channel activity = worst of encoder and decoder norms
        enc_norm = enc_s["norm_per_param"]
        dec_norm = dec_s["norm_per_param"]
        combined_norm = min(enc_norm, dec_norm)
        combined_max  = min(enc_s["max_abs"], dec_s["max_abs"])

        label = classify(combined_max, zero_thresh, weak_thresh)

        # logvar bias: -30 means this channel was intentionally made near-deterministic
        lv_bias_note = ""
        if enc_lv_b < -20:
            lv_bias_note = f" [logvar_bias={enc_lv_b:.1f}→near-det]"
        elif qc_lv_b < -20:
            lv_bias_note = f" [qc_lv_bias={qc_lv_b:.1f}→near-det]"

        rows.append((c, label, enc_norm, dec_norm, pqc_s["norm_per_param"],
                     enc_s["max_abs"], dec_s["max_abs"], lv_bias_note))

    # --- print summary ---
    header = f"{'ch':>3}  {'label':>8}  {'enc_norm':>10}  {'dec_norm':>10}  {'pqc_norm':>10}  {'enc_max':>10}  {'dec_max':>10}  notes"
    print(header)
    print("-" * len(header))

    counts = {"zero": 0, "weak": 0, "complex": 0}
    for c, label, enc_norm, dec_norm, pqc_norm, enc_max, dec_max, note in rows:
        counts[label] += 1
        print(
            f"{c:>3}  {label:>8}  {enc_norm:>10.5f}  {dec_norm:>10.5f}  "
            f"{pqc_norm:>10.5f}  {enc_max:>10.5f}  {dec_max:>10.5f}{note}"
        )

    print()
    print(f"Summary: {counts['zero']} zero  |  {counts['weak']} weak  |  {counts['complex']} complex  (of {n} total)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze VAE latent channel utilization")
    ap.add_argument("model", nargs="?", default="kl-f8ch32-alpha01",
                    help="Path to AutoencoderKL model directory (default: kl-f8ch32-alpha01)")
    ap.add_argument("--zero-thresh", type=float, default=ZERO_THRESH,
                    help=f"max_abs below this → zero (default: {ZERO_THRESH})")
    ap.add_argument("--weak-thresh", type=float, default=WEAK_THRESH,
                    help=f"norm_per_param below this → weak (default: {WEAK_THRESH})")
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: model path not found: {args.model}", file=sys.stderr)
        return 1

    analyze_vae(args.model, zero_thresh=args.zero_thresh, weak_thresh=args.weak_thresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
