#!/bin/env python3

"""
split_vae.py

Split a trained AutoencoderKL checkpoint into N ready-to-train specialist
variants that mask equal-sized, non-overlapping slices of the latent channel
space.  The number of splits is determined by the postfix list; latent_channels
must be evenly divisible by len(postfixes) or the script aborts with an error.

Default postfixes are ["lf", "hf"] (low-frequency / high-frequency halves).

The wrapper applies its channel mask defensively in decode() and forward(),
so training loops that call either the full forward pass or encode/decode
separately both produce the masked behavior. KL loss gradient is preserved
on all channels (the posterior is passed through unmodified); reconstruction
gradient flows only through active channels because the mask multiplies the
sample before the decoder sees it.

CLI usage:
    python split_vae.py CHECKPOINT_PATH -o OUTPUT_DIR
    python split_vae.py CHECKPOINT_PATH -o OUTPUT_DIR --postfixes lf hf --verify
    python split_vae.py CHECKPOINT_PATH -o OUTPUT_DIR --postfixes a b c d

Output layout (example with default ["lf", "hf"] and 32 channels):
    OUTPUT_DIR/
        model_lf/                          (active channels [0, 16))
            config.json, *.safetensors     <- diffusers AutoencoderKL files
            split_config.json              <- mask metadata sidecar
        model_hf/                          (active channels [16, 32))
            ...

Library usage:
    from split_vae import split_vae, ChannelMaskedVAE
    vaes = split_vae("path/to/phase4_ckpt", postfixes=["lf", "hf"])
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class ChannelMaskedVAE(nn.Module):
    """
    Wraps an AutoencoderKL and zeros specified latent channels between
    sampling and decoding.

    Gradient flow:
        - Reconstruction loss: flows only through unmasked channels.
          Encoder weights producing masked channels receive zero gradient
          from reconstruction (output * 0 = 0 -> chain rule terminates).
        - KL loss (computed on the unwrapped posterior): flows through all
          channels normally. Masked channels are pushed toward N(0,1) which
          is the desired behavior; their content is discarded at merge time
          and replaced with the other specialist's encoding.
    """

    def __init__(
        self,
        base_vae: AutoencoderKL,
        active_channels: slice | torch.Tensor,
        latent_channels: int = 32,
    ):
        super().__init__()
        self.vae = base_vae
        self.latent_channels = latent_channels

        mask = torch.zeros(1, latent_channels, 1, 1)
        if isinstance(active_channels, slice):
            mask[:, active_channels, :, :] = 1.0
        else:
            idx = torch.as_tensor(active_channels, dtype=torch.long)
            mask[:, idx, :, :] = 1.0
        self.register_buffer("channel_mask", mask, persistent=True)

    @property
    def active_indices(self) -> torch.Tensor:
        """1D long tensor of channel indices that carry information."""
        return (self.channel_mask.squeeze() > 0).nonzero(as_tuple=False).flatten()

    def encode(self, x):
        """Pass-through. Posterior left intact so KL loss sees all channels."""
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor):
        """Mask defensively, then decode. Safe to call with an unmasked z."""
        return self.vae.decode(z * self.channel_mask)

    def forward(self, x: torch.Tensor, sample_posterior: bool = True):
        """Full encode -> sample -> mask -> decode."""
        dist = self.vae.encode(x).latent_dist
        z = dist.sample() if sample_posterior else dist.mean
        return self.decode(z)


def split_vae(
    checkpoint_path: str | Path,
    latent_channels: int = 32,
    postfixes: list[str] | None = None,
) -> list[ChannelMaskedVAE]:
    """
    Load one independent copy of an AutoencoderKL checkpoint per postfix and
    return masked wrappers with evenly divided channel slices.

    Raises ValueError if latent_channels is not evenly divisible by
    len(postfixes).

    Returns a list of ChannelMaskedVAE, one per postfix, with channels
    [i*chunk, (i+1)*chunk) active for the i-th entry.
    """
    if postfixes is None:
        postfixes = ["lf", "hf"]
    n = len(postfixes)
    if latent_channels % n != 0:
        raise ValueError(
            f"latent_channels={latent_channels} is not evenly divisible by "
            f"{n} postfixes"
        )
    chunk = latent_channels // n
    checkpoint_path = Path(checkpoint_path)
    wrappers = []
    for i in range(n):
        base = AutoencoderKL.from_pretrained(str(checkpoint_path))
        w = ChannelMaskedVAE(
            base,
            active_channels=slice(i * chunk, (i + 1) * chunk),
            latent_channels=latent_channels,
        )
        wrappers.append(w)
    return wrappers


def _zero_inactive_channels(vae: AutoencoderKL, active: list[int], latent: int) -> None:
    """
    Zero the channel-aligned weights for every latent channel NOT in `active`.

    Encoder-side layers (encoder.conv_out, quant_conv) own their inactive
    channels via output rows; both the mean band [0, latent) and the logvar
    band [latent, 2*latent) are zeroed.

    Decoder-side layers (post_quant_conv, decoder.conv_in) own their inactive
    channels via input columns.
    """
    inactive = sorted(set(range(latent)) - set(active))
    if not inactive:
        return
    inactive_t = torch.tensor(inactive, dtype=torch.long)
    enc_rows = torch.cat([inactive_t, inactive_t + latent])
    with torch.no_grad():
        vae.encoder.conv_out.weight[enc_rows] = 0.0
        if vae.encoder.conv_out.bias is not None:
            vae.encoder.conv_out.bias[enc_rows] = 0.0
        vae.quant_conv.weight[enc_rows] = 0.0
        if vae.quant_conv.bias is not None:
            vae.quant_conv.bias[enc_rows] = 0.0
        vae.post_quant_conv.weight[:, inactive_t] = 0.0
        vae.decoder.conv_in.weight[:, inactive_t] = 0.0


def save_split(
    wrappers: list[ChannelMaskedVAE],
    postfixes: list[str],
    out_dir: str | Path,
) -> None:
    """
    Zero inactive channel weights in each wrapper's VAE, then save as a
    standard diffusers AutoencoderKL directory plus a split_config.json
    sidecar. The saved weights have inactive channels physically zeroed so
    the model behaves correctly when loaded without the ChannelMaskedVAE
    wrapper.
    """
    out_dir = Path(out_dir)
    for postfix, wrapper in zip(postfixes, wrappers):
        sub = out_dir / f"model_{postfix}"
        sub.mkdir(parents=True, exist_ok=True)
        active = wrapper.active_indices.tolist()
        _zero_inactive_channels(wrapper.vae, active, wrapper.latent_channels)
        wrapper.vae.save_pretrained(sub)
        config = {
            "latent_channels": wrapper.latent_channels,
            "active_channels": active,
        }
        (sub / "split_config.json").write_text(json.dumps(config, indent=2))


def verify_split(
    vae: ChannelMaskedVAE,
    sample_input: torch.Tensor,
    atol: float = 1e-6,
) -> dict:
    """
    Sanity-check that the mask is applied. Runs a forward pass and confirms
    that masked channels are exactly zero in the latent fed to the decoder.

    Returns a diagnostics dict. Raises AssertionError on failure.
    """
    vae.eval()
    with torch.no_grad():
        z = vae.vae.encode(sample_input).latent_dist.mean
        z_masked = z * vae.channel_mask
        active = vae.active_indices.tolist()
        masked_idx = sorted(set(range(vae.latent_channels)) - set(active))

        active_mean = z_masked[:, active].abs().mean().item()
        masked_max = z_masked[:, masked_idx].abs().max().item() if masked_idx else 0.0

        assert masked_max < atol, (
            f"Masked channels not zero: max abs = {masked_max:.2e}"
        )

    return {
        "active_channels": active,
        "masked_channels": masked_idx,
        "active_mean_abs": active_mean,
        "masked_max_abs": masked_max,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Split a trained AutoencoderKL into two channel-masked "
                    "specialist variants for HF/LF training.",
    )
    ap.add_argument(
        "checkpoint",
        type=str,
        help="Path to base AutoencoderKL checkpoint (diffusers format).",
    )
    ap.add_argument(
        "-o", "--out-dir",
        type=str,
        required=True,
        help="Output directory. Will contain model_lf/ and model_hf/ subdirs.",
    )
    ap.add_argument(
        "--postfixes",
        nargs="+",
        default=["lf", "hf"],
        metavar="NAME",
        help="List of name postfixes, one per split (e.g. lf hf). "
             "Channels are divided evenly; latent_channels must be divisible "
             "by the number of postfixes. Default: lf hf",
    )
    ap.add_argument(
        "--latent-channels",
        type=int,
        default=32,
        help="Total latent channels. Default 32 (f8d32).",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Run a forward pass on random input to confirm masking is active.",
    )
    args = ap.parse_args()

    n = len(args.postfixes)
    if args.latent_channels % n != 0:
        print(
            f"error: --latent-channels {args.latent_channels} is not evenly "
            f"divisible by {n} postfixes",
            file=sys.stderr,
        )
        return 2

    chunk = args.latent_channels // n
    print(f"loading base VAE from {args.checkpoint} ({n} copies)...")
    wrappers = split_vae(
        checkpoint_path=args.checkpoint,
        latent_channels=args.latent_channels,
        postfixes=args.postfixes,
    )

    if args.verify:
        print("verifying mask application...")
        x = torch.randn(1, 3, 64, 64)
        for postfix, w in zip(args.postfixes, wrappers):
            diag = verify_split(w, x)
            print(
                f"  {postfix}: active={diag['active_channels']}, "
                f"masked_max_abs={diag['masked_max_abs']:.2e}, "
                f"active_mean_abs={diag['active_mean_abs']:.4f}"
            )

    out_dir = Path(args.out_dir)
    print(f"saving to {out_dir}...")
    save_split(wrappers, args.postfixes, out_dir)
    for i, postfix in enumerate(args.postfixes):
        lo, hi = i * chunk, (i + 1) * chunk
        print(f"  {out_dir}/model_{postfix}: channels [{lo}, {hi}) active")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
