#!/usr/bin/env python3
"""
Fine-tune a VAE on real-world images using reconstruction loss.
"""

from train_args import parseargs
args = parseargs()

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lpips
import time

from train_datatools import (
    ImageReconDataset,
    parse_dataset_spec,
    load_tile_batch,
    collate_path_tensor,
    write_vae_sample_webp,
)
from train_hard_region import HardRegionMiner
from train_discriminator import (
    NLayerDiscriminator,
    hinge_d_loss,
    generator_hinge_loss,
    adopt_weight,
    weights_init,
)
from diffusers import AutoencoderKL


# -----------------------------
# Loss helpers
# -----------------------------

def highpass_box(img: torch.Tensor, k: int) -> torch.Tensor:
    """
    High-pass filter: img - blur(img) using avg_pool2d box blur.
    img: (B, C, H, W) in [-1, 1]
    """
    if k <= 1:
        return img
    if k % 2 == 0:
        raise ValueError(f"highpass kernel size must be odd; got k={k}")
    blur = F.avg_pool2d(img, kernel_size=k, stride=1, padding=k // 2)
    return img - blur


def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) in [-1,1] -> luma (B, 1, H, W) using BT.601."""
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def auto_scale_k(x: torch.Tensor) -> int:
    """Pick box blur kernel size based on image resolution."""
    m = min(x.shape[-2], x.shape[-1])
    if m <= 640:
        return 5
    elif m <= 1024:
        return 7
    return 9


def fft_mag_log1p(x: torch.Tensor) -> torch.Tensor:
    f = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    return torch.log1p(torch.abs(f))


def fft_mag_loss(dec_hf: torch.Tensor, x_hf: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(fft_mag_log1p(dec_hf.float()), fft_mag_log1p(x_hf.float()))


def fft_phase_loss(dec_hf: torch.Tensor, x_hf: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    fd = torch.fft.rfft2(dec_hf.float(), dim=(-2, -1), norm="ortho")
    fx = torch.fft.rfft2(x_hf.float(), dim=(-2, -1), norm="ortho")
    fdn = fd / (torch.abs(fd) + eps)
    fxn = fx / (torch.abs(fx) + eps)
    return F.l1_loss(torch.view_as_real(fdn), torch.view_as_real(fxn))


# -----------------------------
# Loader bookkeeping
# -----------------------------

@dataclass
class LoaderPack:
    name: str
    loader: DataLoader
    it: object
    tw: int
    th: int
    epoch: int = 0

    def next_batch(self):
        """Get next (paths, tensor_batch) from the DataLoader, restarting if exhausted."""
        try:
            return next(self.it)
        except StopIteration:
            self.epoch += 1
            self.it = iter(self.loader)
            return next(self.it)


# -----------------------------
# Core loss computation
# Extracted so it can be called for both whole-image and tile passes.
# -----------------------------

def compute_loss(
    vae,
    x: torch.Tensor,
    args,
    device: torch.device,
    lpips_fn,
    lap_kernel: Optional[torch.Tensor],
    miner,
    disc,
    step: int,
) -> torch.Tensor:
    """
    Full forward pass + loss computation for a batch x.
    Returns (loss, dec, l1, lp, edge_l1, lap_loss) for logging.
    """
    enc = vae.encode(x)
    posterior = enc.latent_dist
    latents = posterior.mean
    dec = vae.decode(latents).sample

    l1 = F.l1_loss(dec, x)

    kl = None
    if args.kl_weight != 0:
        kl = posterior.kl().mean()

    # High-frequency domain selection
    if args.hf_luma_only:
        dec_hf = rgb_to_luma(dec)
        x_hf = rgb_to_luma(x)
        dec_lp = dec_hf.repeat(1, 3, 1, 1)
        x_lp = x_hf.repeat(1, 3, 1, 1)
    else:
        dec_hf = dec
        x_hf = x
        dec_lp = dec
        x_lp = x

    lp = None
    if args.lpips_weight > 0:
        if args.lpips_shapeonly:
            k = auto_scale_k(x_lp)
            lp = lpips_fn(highpass_box(dec_lp, k), highpass_box(x_lp, k)).mean()
        else:
            lp = lpips_fn(dec_lp, x_lp).mean()

    # Edge L1
    dx_dec = dec_hf[:, :, :, 1:] - dec_hf[:, :, :, :-1]
    dy_dec = dec_hf[:, :, 1:, :] - dec_hf[:, :, :-1, :]
    dx_x = x_hf[:, :, :, 1:] - x_hf[:, :, :, :-1]
    dy_x = x_hf[:, :, 1:, :] - x_hf[:, :, :-1, :]
    edge_l1 = (F.l1_loss(dx_dec, dx_x) + F.l1_loss(dy_dec, dy_x)) * 0.5 if args.edge_l1_weight else 0.0

    # FFT losses
    hf_fft_dec = hf_fft_x = None
    if args.fft_weight > 0 or args.fft_phase_weight > 0 or args.hf_energy_weight > 0:
        k = auto_scale_k(x_hf)
        hf_fft_dec = highpass_box(dec_hf, k)
        hf_fft_x = highpass_box(x_hf, k)

    fft_loss = fft_mag_loss(hf_fft_dec, hf_fft_x) if args.fft_weight > 0 else None
    fft_phase = fft_phase_loss(hf_fft_dec, hf_fft_x) if args.fft_phase_weight > 0 else None

    grad_energy_loss = None
    if args.grad_energy_weight > 0:
        grad_energy_loss = (
            F.l1_loss(dx_dec.square(), dx_x.square())
            + F.l1_loss(dy_dec.square(), dy_x.square())
        ) * 0.5

    # Laplacian
    lap_loss = None
    if lap_kernel is not None:
        if args.hf_luma_only:
            lap_k = lap_kernel[:1, :1].contiguous()
            lap_dec = F.conv2d(dec_hf, lap_k, padding=1, groups=1)
            lap_x = F.conv2d(x_hf, lap_k, padding=1, groups=1)
        else:
            lap_dec = F.conv2d(dec, lap_kernel, padding=1, groups=3)
            lap_x = F.conv2d(x, lap_kernel, padding=1, groups=3)
        lap_loss = F.l1_loss(lap_dec, lap_x)

    hf_energy_loss = None
    if args.hf_energy_weight > 0:
        hf_energy_loss = F.l1_loss(hf_fft_dec.square(), hf_fft_x.square())

    # Assemble loss
    loss = args.l1_weight * l1 + args.edge_l1_weight * edge_l1

    if lp is not None:
        loss = loss + args.lpips_weight * lp
    if fft_loss is not None:
        loss = loss + args.fft_weight * fft_loss
    if fft_phase is not None:
        loss = loss + args.fft_phase_weight * fft_phase
    if kl is not None:
        loss = loss + args.kl_weight * kl.float()
    if lap_loss is not None:
        loss = loss + args.laplacian_weight * lap_loss
    if hf_energy_loss is not None:
        loss = loss + args.hf_energy_weight * hf_energy_loss
    if grad_energy_loss is not None:
        loss = loss + args.grad_energy_weight * grad_energy_loss

    # Hard region mining
    if miner is not None:
        if args.lpips_weight > 0:
            if args.crop_shapeonly:
                k = auto_scale_k(x)
                crop_lpips = lambda a, b: lpips_fn(highpass_box(a, k), highpass_box(b, k))
            else:
                crop_lpips = lpips_fn
        else:
            crop_lpips = None

        crop_l = miner.crop_loss(
            vae=vae,
            x=x,
            dec=dec,
            l1_weight=args.l1_weight,
            lpips_fn=crop_lpips,
            lpips_weight=args.lpips_weight,
        )
        loss = loss + args.crop_mining_weight * crop_l

    # Discriminator generator loss
    if disc is not None:
        d_weight = adopt_weight(args.disc_weight, step, threshold=args.disc_start)
        if d_weight > 0:
            g_loss = generator_hinge_loss(disc(dec))
            loss = loss + d_weight * g_loss

    return loss, dec, l1, lp, edge_l1, lap_loss


def disc_update(disc, opt_d, x, dec, args, step):
    """Discriminator update step. Always called after VAE optimizer step."""
    d_weight = adopt_weight(args.disc_weight, step, threshold=args.disc_start)
    if d_weight > 0:
        opt_d.zero_grad(set_to_none=True)
        d_loss = hinge_d_loss(disc(x.detach()), disc(dec.detach()))
        d_loss.backward()
        opt_d.step()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    print("Using VAE:", args.model)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Enabled TF32")
    else:
        print("Disabled TF32 for maximum precision")

    if args.lpips_shapeonly and args.lpips_weight <= 0:
        raise SystemExit("--lpips_shapeonly requires --lpips_weight > 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        vae = AutoencoderKL.from_pretrained(
            args.model,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(device)
    except EnvironmentError:
        raise SystemExit("ERROR: Cannot load model from", args.model)

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing.")
        vae.enable_gradient_checkpointing()

    vae.train()

    lpips_fn = None
    if args.lpips_weight > 0:
        print(f"Using LPIPS at {args.lpips_weight}" + (" (shape only)" if args.lpips_shapeonly else ""))
        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)

    lap_kernel = None
    if args.laplacian_weight != 0:
        print(f"Using Laplacian loss weight={args.laplacian_weight}")
        k = torch.tensor(
            [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
            device=device, dtype=torch.float32,
        ).view(1, 1, 3, 3)
        lap_kernel = k.repeat(3, 1, 1, 1)

    miner = None
    if args.crop_mining_weight > 0:
        miner = HardRegionMiner(
            crop_size=args.crop_size,
            num_crops=args.num_crops,
            patch_size=args.crop_patch_size,
            temperature=args.crop_temperature,
        )

    disc = None
    opt_d = None
    if args.disc_weight > 0:
        disc = NLayerDiscriminator(
            input_nc=3, ndf=64, n_layers=args.disc_layers,
        ).apply(weights_init).to(device)
        disc.train()
        opt_d = torch.optim.AdamW(
            disc.parameters(),
            lr=args.disc_lr,
            betas=(0.5, 0.9),
            weight_decay=0.0,
        )
        print(f"Discriminator enabled, starts at step {args.disc_start}")

    if args.hires_tiling:
        print("Tiling enabled: each image produces 1 whole-image + 4 tile optimizer steps.")

    # Build data loaders
    import os
    os_cpu_count = Path("/proc/cpuinfo").read_text().count("processor") if Path("/proc/cpuinfo").exists() else 4
    num_workers = min(8, os_cpu_count)

    packs: List[LoaderPack] = []
    for spec in args.dataset:
        root, tw, th = parse_dataset_spec(spec)
        print(f"Loading image paths from {root}")
        ds = ImageReconDataset(root, tw, th)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(torch.cuda.is_available()),
            drop_last=True,
            collate_fn=collate_path_tensor,
        )
        packs.append(LoaderPack(
            name=f"{root.name}:{tw}x{th}",
            loader=loader,
            it=iter(loader),
            tw=tw,
            th=th,
        ))

    opt = torch.optim.AdamW(
        vae.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    _, sample_tw, sample_th = parse_dataset_spec(args.dataset[0])
    sample_img_path = Path(args.sample_img) if args.sample_img else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    last_log_t = time.perf_counter()
    last_log_step = 0
    step = 0

    # Keep last logged loss values for display
    log_l1 = log_lp = log_edge = log_lap = 0.0

    while step < args.train_steps:
        pack = random.choice(packs)
        path_strs, x = pack.next_batch()
        paths = [Path(p) for p in path_strs]

        if args.skip_steps > 0:
            step += 1
            args.skip_steps -= 1
            if args.skip_steps == 0:
                print("Skipped to step", step)
            continue

        # ------------------------------------------------------------------
        # Step A: whole-image pass at target resolution (e.g. 512x512)
        # ------------------------------------------------------------------
        x = x.to(device, non_blocking=True)

        loss, dec, l1, lp, edge_l1, lap_loss = compute_loss(
            vae, x, args, device, lpips_fn, lap_kernel, miner, disc, step,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if disc is not None:
            disc_update(disc, opt_d, x, dec, args, step)

        step += 1

        # ------------------------------------------------------------------
        # Step B: tile passes (if tiling enabled)
        # Each tile is a 512x512 native-resolution crop from the 1024x1024 version.
        # NW=0, NE=1, SW=2, SE=3
        # ------------------------------------------------------------------
        if args.hires_tiling:
            for tile_idx in range(4):
                x_tile = load_tile_batch(paths, tile_idx, device)
                if x_tile is None:
                    continue  # all images in batch were too small, skip this tile

                tile_loss, tile_dec, _, _, _, _ = compute_loss(
                    vae, x_tile, args, device, lpips_fn, lap_kernel, miner, disc, step,
                )

                opt.zero_grad(set_to_none=True)
                tile_loss.backward()
                opt.step()

                if disc is not None:
                    disc_update(disc, opt_d, x_tile, tile_dec, args, step)

                step += 1

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if (step % 50 == 0 or step == 1):
            now = time.perf_counter()
            denom = max(1, step - last_log_step)
            sps = (now - last_log_t) / denom
            last_log_t = now
            last_log_step = step

            lp_val = lp.item() if lp is not None else 0.0
            lap_val = lap_loss.item() if lap_loss is not None else 0.0
            edge_val = edge_l1.item() if isinstance(edge_l1, torch.Tensor) else edge_l1

            print(
                f"step {step}/{args.train_steps} "
                f"l1={l1.item():.4f} lpips={lp_val:.4f} "
                f"edge={edge_val:.4f} lap={lap_val:.4f} "
                f"s/step={sps:.4f}",
                flush=True,
            )

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        if args.save_every > 0 and (step % args.save_every == 0 or step == args.train_steps):
            ckpt_dir = out_dir / f"step_{step:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            vae.save_pretrained(str(ckpt_dir))
            print(f"Saved: {ckpt_dir}", flush=True)
            print(" Datasets in use:")
            for pack in packs:
                print(f"   {pack.name}: {len(pack.loader.dataset)} images")

            if sample_img_path is not None:
                write_vae_sample_webp(
                    vae_model=vae,
                    sample_img=sample_img_path,
                    target_w=sample_tw,
                    target_h=sample_th,
                    out_path=ckpt_dir / "vae_sample.webp",
                )

    # Final save
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(str(final_dir))
    print(f"saved: {final_dir}", flush=True)

    if sample_img_path is not None:
        write_vae_sample_webp(
            vae_model=vae,
            sample_img=sample_img_path,
            target_w=sample_tw,
            target_h=sample_th,
            out_path=final_dir / "vae_sample.webp",
        )


if __name__ == "__main__":
    main()
