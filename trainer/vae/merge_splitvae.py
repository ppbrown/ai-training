#!/bin/env python3

"""
merge_splitvae.py

Merge any number of channel-masked specialist VAEs (produced by split_vae.py
and then trained independently) into a single standard AutoencoderKL ready for
polish training.

All model_* subdirectories in SPLIT_DIR that contain a split_config.json are
discovered automatically.  The script reports how many specialist models were
found, which channel ranges each owns, and which channels (if any) have no
specialist coverage.

Conceptual model
----------------
An AutoencoderKL's weights split into two categories:

  - "Backbone" weights operate on inner feature maps that have no per-latent-
    channel identity. Most of the network (encoder downblocks, decoder
    upblocks, mid blocks, all norms) falls into this category.

  - "Channel-aligned" weights have at least one dimension whose index
    corresponds directly to a specific latent channel.

The merge rule is:

  - Backbone weights -> from --base (one source, chosen by the user).
  - Channel-aligned weights -> rebuilt by summing all specialists.
    split_vae.py physically zeroes inactive channels in each specialist
    before saving, so a simple element-wise sum places every specialist's
    trained weights in the correct positions with no per-channel indexing.

Where the channel-aligned weights live
--------------------------------------
In a diffusers AutoencoderKL there are exactly four layers with channel-
aligned weights:

  Encoder side - latent channel n corresponds to output channel n (mean)
  and output channel n + latent_channels (logvar):
      encoder.conv_out   produces 2*latent_channels of moments
      quant_conv         1x1 conv refining those moments

  Decoder side - latent channel n corresponds to input channel n:
      post_quant_conv    1x1 conv on the latent
      decoder.conv_in    first decoder conv consuming the latent

Every other layer is backbone.

The --base choice
-----------------
  NAME:  postfix of one of the discovered specialists (e.g. "lf", "hf").
         That specialist's full checkpoint is used for backbone weights.
  PATH:  external checkpoint (e.g. the original model before split).

CLI usage
---------
    python merge_splitvae.py SPLIT_DIR -o OUTPUT_DIR
    python merge_splitvae.py SPLIT_DIR -o OUTPUT_DIR --base lf
    python merge_splitvae.py SPLIT_DIR -o OUTPUT_DIR --base /path/to/phase4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _channel_ranges(channels: list[int]) -> list[str]:
    """Return compact range strings for a sorted list of channel indices."""
    if not channels:
        return []
    channels = sorted(channels)
    ranges: list[str] = []
    start = end = channels[0]
    for ch in channels[1:]:
        if ch == end + 1:
            end = ch
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = ch
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ranges


def _format_ranges(channels: list[int]) -> str:
    return ", ".join(_channel_ranges(channels))


def _format_zero_ranges(channels: list[int]) -> str:
    return ", ".join(f"{r}==0" for r in _channel_ranges(channels))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

Specialist = tuple[str, AutoencoderKL, list[int]]  # (dir_name, model, active_channels)


def load_specialists(split_dir: Path) -> tuple[int, list[Specialist]]:
    """
    Discover all model_* subdirs in split_dir that contain split_config.json,
    load each AutoencoderKL, and return (latent_channels, specialists) sorted
    by first active channel.

    Raises FileNotFoundError if none are found, ValueError if latent_channels
    disagrees across models.
    """
    dirs = sorted(
        p for p in split_dir.iterdir()
        if p.is_dir() and (p / "split_config.json").exists()
    )
    if not dirs:
        raise FileNotFoundError(f"No specialist models (model_*/split_config.json) found in {split_dir}")

    specialists: list[Specialist] = []
    latent: int | None = None
    for d in dirs:
        cfg = json.loads((d / "split_config.json").read_text())
        lc = cfg["latent_channels"]
        if latent is None:
            latent = lc
        elif lc != latent:
            raise ValueError(
                f"{d.name}: latent_channels={lc} disagrees with earlier value {latent}"
            )
        model = AutoencoderKL.from_pretrained(str(d))
        active = sorted(cfg["active_channels"])
        specialists.append((d.name, model, active))

    specialists.sort(key=lambda s: s[2][0] if s[2] else 0)
    return latent, specialists


# ---------------------------------------------------------------------------
# Splice
# ---------------------------------------------------------------------------

def _splice_channel_aligned_layers(
    merged: AutoencoderKL,
    specialists: list[Specialist],
    latent: int,
) -> None:
    """
    Rebuild the four channel-aligned layers in merged by zeroing them and
    then summing contributions from all specialists.

    Because split_vae.py physically zeroes inactive channels before saving,
    each specialist contributes non-zero values only for its own channels.
    The sum therefore assembles the correct weights for every channel with
    no per-channel index arithmetic.

    Decoder-side biases (post_quant_conv.bias, decoder.conv_in.bias) are
    indexed by decoder output features, not latent channels, so they are
    preserved unchanged from the base model.
    """
    # Save decoder biases before we zero the layers.
    pqc_bias = (merged.post_quant_conv.bias.clone()
                if merged.post_quant_conv.bias is not None else None)
    dec_bias = (merged.decoder.conv_in.bias.clone()
                if merged.decoder.conv_in.bias is not None else None)

    with torch.no_grad():
        for layer in (
            merged.encoder.conv_out,
            merged.quant_conv,
            merged.post_quant_conv,
            merged.decoder.conv_in,
        ):
            layer.weight.zero_()
            if layer.bias is not None:
                layer.bias.zero_()

        for _, model, _ in specialists:
            merged.encoder.conv_out.weight.add_(model.encoder.conv_out.weight)
            merged.quant_conv.weight.add_(model.quant_conv.weight)
            merged.post_quant_conv.weight.add_(model.post_quant_conv.weight)
            merged.decoder.conv_in.weight.add_(model.decoder.conv_in.weight)
            if merged.encoder.conv_out.bias is not None:
                merged.encoder.conv_out.bias.add_(model.encoder.conv_out.bias)
            if merged.quant_conv.bias is not None:
                merged.quant_conv.bias.add_(model.quant_conv.bias)

        # Restore decoder biases from base.
        if pqc_bias is not None:
            merged.post_quant_conv.bias.copy_(pqc_bias)
        if dec_bias is not None:
            merged.decoder.conv_in.bias.copy_(dec_bias)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_split(
    split_dir: str | Path,
    base: str | Path = "hf",
) -> AutoencoderKL:
    """
    Discover all specialist models in split_dir, report coverage, and merge
    their channel-aligned weights into a single AutoencoderKL.

    Args:
        split_dir: directory produced by split_vae.py containing model_*/ subdirs.
        base: source for backbone (non-channel-aligned) weights. Either the
              postfix of a discovered specialist (e.g. "lf", "hf") or the
              full name of a subdir (e.g. "model_lf"), or a path to another
              diffusers AutoencoderKL checkpoint.

    Returns:
        Merged AutoencoderKL with channel-aligned layers spliced from
        the appropriate specialist for each channel.
    """
    split_dir = Path(split_dir)
    latent, specialists = load_specialists(split_dir)

    n = len(specialists)
    print(f"Found {n} specialist model{'s' if n != 1 else ''}:")
    all_active: set[int] = set()
    for name, _, active in specialists:
        print(f"  {name}: channels {_format_ranges(active)}")
        all_active.update(active)

    zero = sorted(set(range(latent)) - all_active)
    if zero:
        print(f"  Uncovered channels: {_format_zero_ranges(zero)}")
    else:
        print("  All channels covered.")

    # Resolve base checkpoint
    candidate = split_dir / f"model_{base}"
    if not candidate.is_dir():
        candidate = split_dir / str(base)
    if candidate.is_dir():
        merged = AutoencoderKL.from_pretrained(str(candidate))
    else:
        merged = AutoencoderKL.from_pretrained(str(base))

    _splice_channel_aligned_layers(merged, specialists, latent)

    return merged


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_merge(
    merged: AutoencoderKL,
    split_dir: str | Path,
) -> dict:
    """
    Confirm that channel-aligned layers contain weights from the correct
    specialists, and that any uncovered channels are zero in the merged model.

    Returns a dict of check_name -> bool. Raises AssertionError on failure.
    """
    split_dir = Path(split_dir)
    latent, specialists = load_specialists(split_dir)

    checks: dict[str, bool] = {}
    all_active: set[int] = set()

    for name, model, active in specialists:
        idx = torch.tensor(active, dtype=torch.long)
        enc_idx = torch.cat([idx, idx + latent])
        all_active.update(active)

        checks[f"{name}.encoder.conv_out"] = torch.equal(
            merged.encoder.conv_out.weight[enc_idx],
            model.encoder.conv_out.weight[enc_idx],
        )
        checks[f"{name}.quant_conv"] = torch.equal(
            merged.quant_conv.weight[enc_idx],
            model.quant_conv.weight[enc_idx],
        )
        checks[f"{name}.post_quant_conv"] = torch.equal(
            merged.post_quant_conv.weight[:, idx],
            model.post_quant_conv.weight[:, idx],
        )
        checks[f"{name}.decoder.conv_in"] = torch.equal(
            merged.decoder.conv_in.weight[:, idx],
            model.decoder.conv_in.weight[:, idx],
        )

    # Verify uncovered channels are zero in the merged model.
    uncovered = sorted(set(range(latent)) - all_active)
    if uncovered:
        unc = torch.tensor(uncovered, dtype=torch.long)
        unc_enc = torch.cat([unc, unc + latent])
        checks["uncovered.encoder.conv_out"] = (merged.encoder.conv_out.weight[unc_enc].abs().max().item() == 0.0)
        checks["uncovered.quant_conv"]        = (merged.quant_conv.weight[unc_enc].abs().max().item() == 0.0)
        checks["uncovered.post_quant_conv"]   = (merged.post_quant_conv.weight[:, unc].abs().max().item() == 0.0)
        checks["uncovered.decoder.conv_in"]   = (merged.decoder.conv_in.weight[:, unc].abs().max().item() == 0.0)

    failures = [k for k, v in checks.items() if not v]
    assert not failures, f"splice verification failed for: {failures}"
    return checks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge channel-masked specialist VAEs into a single AutoencoderKL.",
    )
    ap.add_argument(
        "split_dir",
        type=str,
        help="Directory containing model_*/ subdirs produced by split_vae.py.",
    )
    ap.add_argument(
        "-o", "--out-dir",
        type=str,
        required=True,
        help="Output directory for the merged AutoencoderKL.",
    )
    ap.add_argument(
        "--base",
        type=str,
        default="hf",
        help="Source for backbone (non-channel-aligned) weights. Postfix of a "
             "discovered specialist (e.g. 'lf', 'hf'), full subdir name, or a "
             "path to another AutoencoderKL checkpoint. Default: hf.",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Check that spliced weights match the source specialists.",
    )
    args = ap.parse_args()

    print(f"merging {args.split_dir} (base={args.base})...")
    merged = merge_split(args.split_dir, base=args.base)

    if args.verify:
        print("verifying splice...")
        verify_merge(merged, args.split_dir)
        print("  all spliced layers match their source specialists.")

    out_dir = Path(args.out_dir)
    print(f"saving merged model to {out_dir}...")
    merged.save_pretrained(out_dir)
    print("done. ready for polish training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
