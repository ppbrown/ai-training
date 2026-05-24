#!/bin/env python3

"""
merge_vae.py

Merge two channel-masked specialist VAEs (produced by split_vae.py and then
trained independently) into a single standard AutoencoderKL ready for polish
training.

Conceptual model
----------------
An AutoencoderKL's weights split into two categories:

  - "Backbone" weights operate on inner feature maps that have no per-latent-
    channel identity. Most of the network (encoder downblocks, decoder
    upblocks, mid blocks, all norms) falls into this category.

  - "Channel-aligned" weights have at least one dimension whose index
    corresponds directly to a specific latent channel. For latent channel n,
    these are the weights that produce or consume channel n of the latent.

The merge rule is:

  - Backbone weights -> from --base (one source, chosen by the user).
  - Channel-aligned weights for channel n -> from whichever specialist
    owns channel n (vae_lf owns [0, split_at), vae_hf owns [split_at, latent)).

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

Note: quant_conv and post_quant_conv are 1x1 convs whose BOTH dims are
latent-aligned. For a weight element W[i, j] where i and j have different
owners (one from lf, one from hf), there's no clean per-channel assignment.
The implementation uses output-dim ownership for encoder-side layers and
input-dim ownership for decoder-side layers, so cross-owned elements come
from the row or column source respectively. These layers are typically
near-identity in trained VAEs, so off-diagonal mass is small and the choice
matters little in practice.

The --base choice
-----------------
  "hf":  use vae_hf as the base. Encoder backbone is HF-trained, so the
         HF-owned channels stay internally consistent end-to-end; only the
         LF-owned channels are operating on a backbone they weren't quite
         trained against. Recommended when HF fidelity is the priority.
  "lf":  mirror.
  PATH:  external checkpoint (e.g. the original model before split). 
         Both halves get equal representational drift from their training
         and polish has to reconcile both.

CLI usage
---------
    python merge_vae.py SPLIT_DIR -o OUTPUT_DIR
    python merge_vae.py SPLIT_DIR -o OUTPUT_DIR --base lf
    python merge_vae.py SPLIT_DIR -o OUTPUT_DIR --base /path/to/phase4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


def _splice_encoder_layer(
    target: nn.Conv2d,
    lf: nn.Conv2d,
    hf: nn.Conv2d,
    split_at: int,
    latent: int,
) -> None:
    """
    Splice by OUTPUT channel for an encoder-side conv whose output is
    interpreted as (mean[0:latent], logvar[0:latent]).
    """
    with torch.no_grad():
        # Means: output channels [0, latent)
        target.weight[:split_at].copy_(lf.weight[:split_at])
        target.weight[split_at:latent].copy_(hf.weight[split_at:latent])
        # Logvars: output channels [latent, 2*latent)
        target.weight[latent:latent + split_at].copy_(lf.weight[latent:latent + split_at])
        target.weight[latent + split_at:2 * latent].copy_(hf.weight[latent + split_at:2 * latent])
        if target.bias is not None:
            target.bias[:split_at].copy_(lf.bias[:split_at])
            target.bias[split_at:latent].copy_(hf.bias[split_at:latent])
            target.bias[latent:latent + split_at].copy_(lf.bias[latent:latent + split_at])
            target.bias[latent + split_at:2 * latent].copy_(hf.bias[latent + split_at:2 * latent])


def _splice_decoder_layer(
    target: nn.Conv2d,
    lf: nn.Conv2d,
    hf: nn.Conv2d,
    split_at: int,
    latent: int,
) -> None:
    """
    Splice by INPUT channel for a decoder-side conv whose input channels
    are the latent channels.
    """
    with torch.no_grad():
        target.weight[:, :split_at].copy_(lf.weight[:, :split_at])
        target.weight[:, split_at:latent].copy_(hf.weight[:, split_at:latent])
        # Bias is per output channel, unaffected by input-channel splicing.


def merge_split(
    split_dir: str | Path,
    base: str | Path = "hf",
) -> AutoencoderKL:
    """
    Merge a split_dir produced by split_vae.py into a single AutoencoderKL.

    Args:
        split_dir: directory containing model_lf/ and model_hf/ subdirs.
        base: source for non-spliced weights. One of:
            - "lf": use vae_lf's full checkpoint as the base.
            - "hf": use vae_hf's full checkpoint as the base.
            - PATH: path to another diffusers AutoencoderKL checkpoint.

    Returns:
        A single AutoencoderKL with:
            - latent channels [0, split_at) routed via vae_lf's projection
            - latent channels [split_at, latent) routed via vae_hf's projection
            - everything else inherited from `base`
    """
    split_dir = Path(split_dir)
    lf_dir = split_dir / "model_lf"
    hf_dir = split_dir / "model_hf"

    lf = AutoencoderKL.from_pretrained(str(lf_dir))
    hf = AutoencoderKL.from_pretrained(str(hf_dir))

    lf_cfg = json.loads((lf_dir / "split_config.json").read_text())
    hf_cfg = json.loads((hf_dir / "split_config.json").read_text())
    latent = lf_cfg["latent_channels"]
    assert hf_cfg["latent_channels"] == latent, "lf/hf disagree on latent_channels"

    lf_active = sorted(lf_cfg["active_channels"])
    hf_active = sorted(hf_cfg["active_channels"])
    # Require the standard contiguous-low / contiguous-high split. If you want
    # a fancier split pattern, rewrite the splice helpers.
    assert lf_active == list(range(0, len(lf_active))), (
        f"lf active channels must be contiguous [0, k), got {lf_active}"
    )
    assert hf_active == list(range(len(lf_active), latent)), (
        f"hf active channels must be contiguous [k, latent), got {hf_active}"
    )
    split_at = len(lf_active)

    if base == "lf":
        merged = AutoencoderKL.from_pretrained(str(lf_dir))
    elif base == "hf":
        merged = AutoencoderKL.from_pretrained(str(hf_dir))
    else:
        merged = AutoencoderKL.from_pretrained(str(base))

    _splice_encoder_layer(merged.encoder.conv_out, lf.encoder.conv_out, hf.encoder.conv_out, split_at, latent)
    _splice_encoder_layer(merged.quant_conv, lf.quant_conv, hf.quant_conv, split_at, latent)
    _splice_decoder_layer(merged.post_quant_conv, lf.post_quant_conv, hf.post_quant_conv, split_at, latent)
    _splice_decoder_layer(merged.decoder.conv_in, lf.decoder.conv_in, hf.decoder.conv_in, split_at, latent)

    return merged


def verify_merge(
    merged: AutoencoderKL,
    split_dir: str | Path,
) -> dict:
    """
    Confirm that the spliced layers actually contain the expected weights.
    Loads the two specialists again and checks per-channel equality on the
    four spliced layers.
    """
    split_dir = Path(split_dir)
    lf = AutoencoderKL.from_pretrained(str(split_dir / "model_lf"))
    hf = AutoencoderKL.from_pretrained(str(split_dir / "model_hf"))
    lf_cfg = json.loads((split_dir / "model_lf" / "split_config.json").read_text())
    latent = lf_cfg["latent_channels"]
    split_at = len(lf_cfg["active_channels"])

    checks = {}
    # Encoder side: check mean band and logvar band, both halves.
    for layer_name, m, l, h in [
        ("encoder.conv_out", merged.encoder.conv_out, lf.encoder.conv_out, hf.encoder.conv_out),
        ("quant_conv",       merged.quant_conv,       lf.quant_conv,       hf.quant_conv),
    ]:
        checks[f"{layer_name}.mean_lf"]   = torch.equal(m.weight[:split_at],                           l.weight[:split_at])
        checks[f"{layer_name}.mean_hf"]   = torch.equal(m.weight[split_at:latent],                     h.weight[split_at:latent])
        checks[f"{layer_name}.logvar_lf"] = torch.equal(m.weight[latent:latent + split_at],            l.weight[latent:latent + split_at])
        checks[f"{layer_name}.logvar_hf"] = torch.equal(m.weight[latent + split_at:2 * latent],        h.weight[latent + split_at:2 * latent])

    # Decoder side: check input channel halves.
    for layer_name, m, l, h in [
        ("post_quant_conv", merged.post_quant_conv, lf.post_quant_conv, hf.post_quant_conv),
        ("decoder.conv_in", merged.decoder.conv_in, lf.decoder.conv_in, hf.decoder.conv_in),
    ]:
        checks[f"{layer_name}.in_lf"] = torch.equal(m.weight[:, :split_at],          l.weight[:, :split_at])
        checks[f"{layer_name}.in_hf"] = torch.equal(m.weight[:, split_at:latent],    h.weight[:, split_at:latent])

    failures = [k for k, v in checks.items() if not v]
    assert not failures, f"splice verification failed for: {failures}"
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge channel-masked specialist VAEs into a single AutoencoderKL.",
    )
    ap.add_argument(
        "split_dir",
        type=str,
        help="Directory containing model_lf/ and model_hf/ produced by split_vae.py.",
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
        help="Source for non-spliced layers. 'lf', 'hf', or a path to another "
             "AutoencoderKL checkpoint. Default: hf.",
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
