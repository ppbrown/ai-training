#!/bin/env python3

"""
merge_vaes_concat.py

Concatenate the latent channels of two or more independent sd1.5-style
AutoencoderKL VAEs (original config, 4 latent channels, double_z=True) into
a single larger VAE. N input VAEs -> N*4 latent channels; the i-th VAE's
4 channels (its own channels 0-3) become channels [4*i, 4*i+4) of the
merged model.

All subdirectories of SPLIT_DIR are scanned. Subdirectories that don't load
as an AutoencoderKL, or whose latent_channels / channel-aligned layer shapes
don't match the expected 4-channel sd1.5 layout, are skipped with a warning.
At least two valid candidates are required.

Backbone (non-channel-aligned) weights and config come entirely from the
first valid candidate (sorted by directory name).

CLI usage
---------
    python merge_vaes_concat.py SPLIT_DIR
    python merge_vaes_concat.py SPLIT_DIR --out OUTPUT_DIR

TIP: After you get the merged model, train it a bit with
the channels frozen, to align the backbone. eg:

    train_vae.py --freeze_all_channels ....

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL


LATENT_PER_MODEL = 4
CHANNEL_ALIGNED_PREFIXES = (
    "encoder.conv_out.",
    "quant_conv.",
    "post_quant_conv.",
    "decoder.conv_in.",
)


def _load_candidates(split_dir: Path) -> list[tuple[str, AutoencoderKL]]:
    """
    Load every subdirectory of split_dir as an AutoencoderKL, keeping only
    those with latent_channels == LATENT_PER_MODEL and channel-aligned layer
    shapes matching the standard sd1.5 (double_z) layout. Others are skipped
    with a warning on stderr.
    """
    candidates: list[tuple[str, AutoencoderKL]] = []
    for d in sorted(split_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            model = AutoencoderKL.from_pretrained(str(d))
            lc = model.config.latent_channels
            if lc != LATENT_PER_MODEL:
                raise ValueError(f"latent_channels={lc}, expected {LATENT_PER_MODEL}")
            if (model.encoder.conv_out.weight.shape[0] != 2 * LATENT_PER_MODEL
                    or model.quant_conv.weight.shape[:2] != (2 * LATENT_PER_MODEL,) * 2
                    or model.post_quant_conv.weight.shape[:2] != (LATENT_PER_MODEL,) * 2
                    or model.decoder.conv_in.weight.shape[1] != LATENT_PER_MODEL):
                raise ValueError("unexpected channel-aligned layer shapes")
        except Exception as e:
            print(f"  skipping {d.name}: {e}", file=sys.stderr)
            continue
        candidates.append((d.name, model))
    return candidates


def merge_concat(candidates: list[tuple[str, AutoencoderKL]]) -> AutoencoderKL:
    """
    Build a new AutoencoderKL with latent_channels = LATENT_PER_MODEL * len(candidates),
    using the first candidate's backbone and config, with each candidate's
    channel-aligned weights placed at channels [i*L, (i+1)*L).
    """
    L = LATENT_PER_MODEL
    n = len(candidates)
    latent_new = L * n

    base_name, base_model = candidates[0]
    config = {k: v for k, v in base_model.config.items() if not k.startswith("_")}
    config["latent_channels"] = latent_new
    merged = AutoencoderKL(**config)

    with torch.no_grad():
        base_sd = base_model.state_dict()
        for key, param in merged.state_dict().items():
            if not any(key.startswith(p) for p in CHANNEL_ALIGNED_PREFIXES):
                param.copy_(base_sd[key])
        # decoder.conv_in.bias is indexed by backbone features, not latent
        # channels, so its shape is unchanged -- keep the base model's value.
        merged.decoder.conv_in.bias.copy_(base_model.decoder.conv_in.bias)

        for layer in (merged.encoder.conv_out, merged.quant_conv, merged.post_quant_conv):
            layer.weight.zero_()
            layer.bias.zero_()
        merged.decoder.conv_in.weight.zero_()

        mean_old = slice(0, L)
        logvar_old = slice(L, 2 * L)
        for i, (_, model) in enumerate(candidates):
            ch = slice(i * L, (i + 1) * L)           # new latent channels for this candidate
            lv = slice(latent_new + i * L, latent_new + (i + 1) * L)  # corresponding logvar band

            eco_w, eco_b = model.encoder.conv_out.weight, model.encoder.conv_out.bias
            merged.encoder.conv_out.weight[ch] = eco_w[mean_old]
            merged.encoder.conv_out.weight[lv] = eco_w[logvar_old]
            merged.encoder.conv_out.bias[ch] = eco_b[mean_old]
            merged.encoder.conv_out.bias[lv] = eco_b[logvar_old]

            qc_w, qc_b = model.quant_conv.weight, model.quant_conv.bias
            for r_new, r_old in ((ch, mean_old), (lv, logvar_old)):
                for c_new, c_old in ((ch, mean_old), (lv, logvar_old)):
                    merged.quant_conv.weight[r_new, c_new] = qc_w[r_old, c_old]
            merged.quant_conv.bias[ch] = qc_b[mean_old]
            merged.quant_conv.bias[lv] = qc_b[logvar_old]

            merged.post_quant_conv.weight[ch, ch] = model.post_quant_conv.weight
            merged.post_quant_conv.bias[ch] = model.post_quant_conv.bias
            merged.decoder.conv_in.weight[:, ch] = model.decoder.conv_in.weight

    return merged


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Concatenate the latent channels of two or more independent "
                     "sd1.5-style (4-channel) AutoencoderKL VAEs into one larger VAE.",
    )
    ap.add_argument(
        "split_dir",
        type=str,
        help="Directory containing VAE subdirectories to merge.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for the merged VAE (default: split_dir itself).",
    )
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    print(f"scanning {split_dir} for sd1.5-style ({LATENT_PER_MODEL}-channel) VAEs...")
    candidates = _load_candidates(split_dir)

    if len(candidates) < 2:
        print(f"error: found {len(candidates)} valid candidate(s), need at least 2", file=sys.stderr)
        return 1

    n = len(candidates)
    print(f"merging {n} VAEs -> {LATENT_PER_MODEL * n} latent channels (backbone from {candidates[0][0]}):")
    for i, (name, _) in enumerate(candidates):
        lo, hi = i * LATENT_PER_MODEL, (i + 1) * LATENT_PER_MODEL
        print(f"  {name}: -> channels [{lo}, {hi})")

    merged = merge_concat(candidates)

    out_dir = Path(args.out) if args.out else split_dir
    print(f"saving merged model to {out_dir}...")
    merged.save_pretrained(out_dir)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
