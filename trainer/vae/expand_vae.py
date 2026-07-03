#!/usr/bin/env python3

"""
Expand a VAE's latent channel count by initializing new channel weights to
the mean of all existing channels, then adding Gaussian noise.

Example: expand a 4ch VAE to 8ch by adding 4 new channels
    python expand_vae.py --model my-4ch-vae --out my-8ch-vae --add-ch 4
"""

import jsonargparse as argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL


def _load_vae(model_path: Path, dtype: torch.dtype) -> AutoencoderKL:
    if (model_path / "vae" / "config.json").is_file():
        return AutoencoderKL.from_pretrained(str(model_path), subfolder="vae", torch_dtype=dtype)
    if (model_path / "config.json").is_file():
        return AutoencoderKL.from_pretrained(str(model_path), torch_dtype=dtype)
    raise SystemExit(f"Could not find a VAE at: {model_path}")


def _mean_and_noise(
    src: torch.Tensor,
    dim: int,
    add_c: int,
    std_scale: float,
) -> torch.Tensor:
    """
    Expand tensor along `dim` by appending add_c channels, each initialized to
    the mean of all existing channels along that dim, plus Gaussian noise
    scaled to src's overall std.
    """
    base = src.mean(dim=dim, keepdim=True)
    repeats = [1] * src.dim()
    repeats[dim] = add_c
    extra = base.repeat(repeats)

    weight_std = float(src.detach().std().item())
    noise_std = weight_std * std_scale if weight_std > 0 else std_scale
    with torch.no_grad():
        extra = extra + torch.randn_like(extra) * noise_std

    return torch.cat([src, extra], dim=dim)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Expand a VAE's latent channel count by mean-init + noise."
    )
    ap.add_argument("--model", required=True, help="Input VAE (pipeline dir with vae/ subdir, or VAE-only dir)")
    ap.add_argument("--out", required=True, help="Output VAE-only directory")
    ap.add_argument("--add-ch", type=int, required=True, help="Number of latent channels to add")
    ap.add_argument(
        "--noise-std-scale",
        type=float,
        default=0.1,
        help="Noise std as a fraction of each existing weight tensor's std (1.0 = 100%%)",
    )
    args = ap.parse_args()

    in_path = Path(args.model)
    out_path = Path(args.out)

    if out_path.exists() and any(out_path.iterdir()):
        raise SystemExit(f"Refusing to write into non-empty directory: {out_path}")

    vae = _load_vae(in_path, dtype=torch.float32)

    old_c = int(vae.config.latent_channels)
    add_c = int(args.add_ch)
    new_c = old_c + add_c
    s = args.noise_std_scale

    print(f"Expanding VAE: {old_c} -> {new_c} latent channels (adding {add_c})")

    sd = vae.state_dict()
    new_sd = {}

    for key, val in sd.items():
        t = val.clone()

        # ------------------------------------------------------------------ #
        # encoder.conv_out  shape: [2*old_c, enc_in_ch, kH, kW]
        #   first old_c rows = means, next old_c rows = logvars
        # ------------------------------------------------------------------ #
        if key == "encoder.conv_out.weight":
            means   = t[:old_c]
            logvars = t[old_c:]
            new_means   = _mean_and_noise(means,   0, add_c, s)
            new_logvars = _mean_and_noise(logvars, 0, add_c, s)
            t = torch.cat([new_means, new_logvars], dim=0)

        elif key == "encoder.conv_out.bias":
            means   = t[:old_c]
            logvars = t[old_c:]
            new_means   = _mean_and_noise(means,   0, add_c, s)
            new_logvars = _mean_and_noise(logvars, 0, add_c, s)
            t = torch.cat([new_means, new_logvars], dim=0)

        # ------------------------------------------------------------------ #
        # quant_conv  shape: [2*old_c, 2*old_c, 1, 1]
        #   maps (mean,logvar) -> (mean,logvar), interleaved same as conv_out
        # ------------------------------------------------------------------ #
        elif key == "quant_conv.weight":
            # expand rows first, then cols
            m_out  = t[:old_c];   lv_out = t[old_c:]
            new_m_out  = _mean_and_noise(m_out,  0, add_c, s)
            new_lv_out = _mean_and_noise(lv_out, 0, add_c, s)
            t = torch.cat([new_m_out, new_lv_out], dim=0)   # rows done

            m_in  = t[:, :old_c]; lv_in = t[:, old_c:]
            new_m_in  = _mean_and_noise(m_in,  1, add_c, s)
            new_lv_in = _mean_and_noise(lv_in, 1, add_c, s)
            t = torch.cat([new_m_in, new_lv_in], dim=1)     # cols done

        elif key == "quant_conv.bias":
            means   = t[:old_c]
            logvars = t[old_c:]
            new_means   = _mean_and_noise(means,   0, add_c, s)
            new_logvars = _mean_and_noise(logvars, 0, add_c, s)
            t = torch.cat([new_means, new_logvars], dim=0)

        # ------------------------------------------------------------------ #
        # post_quant_conv  shape: [old_c, old_c, 1, 1]
        #   latent -> latent, plain channel dims
        # ------------------------------------------------------------------ #
        elif key == "post_quant_conv.weight":
            t = _mean_and_noise(t, 0, add_c, s)  # out channels
            t = _mean_and_noise(t, 1, add_c, s)  # in  channels

        elif key == "post_quant_conv.bias":
            t = _mean_and_noise(t, 0, add_c, s)

        # ------------------------------------------------------------------ #
        # decoder.conv_in  shape: [out_ch, old_c, kH, kW]
        #   in-channel dim is the latent dim
        # ------------------------------------------------------------------ #
        elif key == "decoder.conv_in.weight":
            t = _mean_and_noise(t, 1, add_c, s)  # in channels

        # everything else is unchanged
        new_sd[key] = t

    # ------------------------------------------------------------------ #
    # Patch config: latent_channels and (if present) in_channels
    # ------------------------------------------------------------------ #
    new_config = dict(vae.config)
    new_config["latent_channels"] = new_c
    if "in_channels" in new_config and new_config["in_channels"] == old_c:
        new_config["in_channels"] = new_c

    out_path.mkdir(parents=True, exist_ok=True)

    vae_out = AutoencoderKL(**{k: v for k, v in new_config.items() if not k.startswith("_")})
    missing, unexpected = vae_out.load_state_dict(new_sd, strict=False)
    if unexpected:
        print(f"WARNING: unexpected keys in state dict: {unexpected}")
    if missing:
        print(f"WARNING: missing keys in state dict: {missing}")

    vae_out.save_pretrained(str(out_path), safe_serialization=True)

    print(f"Loaded from:       {in_path}")
    print(f"latent_channels:   {old_c} -> {new_c}")
    print(f"Noise std scale:   {s} (fraction of each tensor's own std)")
    print(f"Saved to:          {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
