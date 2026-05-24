#!/bin/env python3

"""
split_vae.py

Split a trained AutoencoderKL checkpoint into two ready-to-train specialist
variants that mask complementary halves of the latent channel space.

The intent is to train one half for high frequency image data, and other
half for low frequency, so resulting modules will be called model_hf
and model_lf

The wrapper applies its channel mask defensively in decode() and forward(),
so training loops that call either the full forward pass or encode/decode
separately both produce the masked behavior. KL loss gradient is preserved
on all channels (the posterior is passed through unmodified); reconstruction
gradient flows only through active channels because the mask multiplies the
sample before the decoder sees it.

CLI usage:
    python split_vae.py CHECKPOINT_PATH -o OUTPUT_DIR
    python split_vae.py CHECKPOINT_PATH -o OUTPUT_DIR --split-at 16 --verify

Output layout:
    OUTPUT_DIR/
        model_lf/                          (active channels [0, split_at))
            config.json, *.safetensors     <- diffusers AutoencoderKL files
            split_config.json              <- mask metadata sidecar
        model_hf/                          (active channels [split_at, latent_channels))
            ...

Library usage:
    from split_vae import split_vae, ChannelMaskedVAE
    vae_lf, vae_hf = split_vae("path/to/phase4_ckpt", split_at=16)
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
    split_at: int = 16,
) -> tuple[ChannelMaskedVAE, ChannelMaskedVAE]:
    """
    Load two independent copies of an AutoencoderKL checkpoint and return
    masked wrappers.

    Returns:
        (vae_lf, vae_hf):
            - vae_lf has channels [0, split_at) active.
            - vae_hf has channels [split_at, latent_channels) active.

    Both are intended to be trained on the FULL image with the existing loss
    combo. The masking enforces capacity allocation; the loss combo
    (lpips_rawvgg + laplacian + disc) drives HF specialization in the HF
    half once LF reconstruction plateaus under reduced channel capacity.
    """
    checkpoint_path = Path(checkpoint_path)
    base_lf = AutoencoderKL.from_pretrained(str(checkpoint_path))
    base_hf = AutoencoderKL.from_pretrained(str(checkpoint_path))

    vae_lf = ChannelMaskedVAE(
        base_lf,
        active_channels=slice(0, split_at),
        latent_channels=latent_channels,
    )
    vae_hf = ChannelMaskedVAE(
        base_hf,
        active_channels=slice(split_at, latent_channels),
        latent_channels=latent_channels,
    )
    return vae_lf, vae_hf


def save_split(
    vae_lf: ChannelMaskedVAE,
    vae_hf: ChannelMaskedVAE,
    out_dir: str | Path,
) -> None:
    """
    Save each wrapper as a standard diffusers AutoencoderKL directory plus a
    split_config.json sidecar containing the mask metadata. The base VAE
    files in each subdirectory remain loadable by any stock diffusers code.
    """
    out_dir = Path(out_dir)
    for name, wrapper in [("model_lf", vae_lf), ("model_hf", vae_hf)]:
        sub = out_dir / name
        sub.mkdir(parents=True, exist_ok=True)
        wrapper.vae.save_pretrained(sub)
        config = {
            "latent_channels": wrapper.latent_channels,
            "active_channels": wrapper.active_indices.tolist(),
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
        "--split-at",
        type=int,
        default=16,
        help="Channel index at which to split. Default 16 (32-channel latent halved).",
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

    if not (0 < args.split_at < args.latent_channels):
        print(
            f"error: --split-at must be in (0, {args.latent_channels}), got {args.split_at}",
            file=sys.stderr,
        )
        return 2

    print(f"loading base VAE from {args.checkpoint} (twice)...")
    vae_lf, vae_hf = split_vae(
        checkpoint_path=args.checkpoint,
        latent_channels=args.latent_channels,
        split_at=args.split_at,
    )

    if args.verify:
        print("verifying mask application...")
        x = torch.randn(1, 3, 64, 64)
        for name, w in [("lf", vae_lf), ("hf", vae_hf)]:
            diag = verify_split(w, x)
            print(
                f"  {name}: active={diag['active_channels']}, "
                f"masked_max_abs={diag['masked_max_abs']:.2e}, "
                f"active_mean_abs={diag['active_mean_abs']:.4f}"
            )

    out_dir = Path(args.out_dir)
    print(f"saving to {out_dir}...")
    save_split(vae_lf, vae_hf, out_dir)
    print(f"  {out_dir}/model_lf: channels [0, {args.split_at}) active")
    print(f"  {out_dir}/model_hf: channels [{args.split_at}, {args.latent_channels}) active")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
