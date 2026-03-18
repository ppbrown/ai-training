#!/usr/bin/env python3
"""
Fine-tune SDXL VAE on real-world images using reconstruction loss only.
"""


from train_args import parseargs
"""Call this early for faster usage response """
args = parseargs()

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVF
from torchvision.transforms import InterpolationMode as IM

from diffusers import AutoencoderKL
import lpips
import time

from train_hard_region import HardRegionMiner

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# -----------------------------
# Loss helpers
# -----------------------------

def highpass_box(img: torch.Tensor, k: int) -> torch.Tensor:
    """
    High-pass filter: img - blur(img), where blur is a cheap box blur (avg_pool2d).
    img: (B,C,H,W) in [-1,1]
    Used to help LPIPS focus on shapes instead of color
    """
    if k <= 1:
        return img
    if k % 2 == 0:
        raise ValueError(f"highpass kernel size must be odd; got k={k}")
    blur = F.avg_pool2d(img, kernel_size=k, stride=1, padding=k // 2)
    return img - blur

def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    """
    Convert (B,3,H,W) in [-1,1] to luma (B,1,H,W).
    Uses BT.601 coefficients; good enough for our purpose.
    """
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b

# Auto blur size based on image resolution
def auto_scale_k(x):
    h = int(x.shape[-2])
    w = int(x.shape[-1])
    m = min(h, w)
    if m <= 640:
        k = 5
    elif m <= 1024:
        k = 7
    else:
        k = 9
    return k

def fft_mag_log1p(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,C,H,W) float32
    Returns: log1p(|rfft2(x)|) to compress dynamic range.
    """
    # rfft2 over H,W
    f = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    mag = torch.abs(f)  # float32
    return torch.log1p(mag)


def fft_mag_loss(dec_hf: torch.Tensor, x_hf: torch.Tensor) -> torch.Tensor:
    """
    L1 distance between log-magnitude spectra.
    Use float32 for FFT stability (bf16 FFT support can be spotty).
    """
    a = fft_mag_log1p(dec_hf.float())
    b = fft_mag_log1p(x_hf.float())
    return F.l1_loss(a, b)

def fft_phase_loss(dec_hf: torch.Tensor, x_hf: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Phase-aware FFT loss via normalized complex spectra.

    - Uses float32 FFT for stability.
    - Normalizes each FFT bin by its magnitude, so the loss emphasizes alignment/phase
      more than raw energy.
    - Returns L1 over (real, imag) components.
    """
    fd = torch.fft.rfft2(dec_hf.float(), dim=(-2, -1), norm="ortho")
    fx = torch.fft.rfft2(x_hf.float(), dim=(-2, -1), norm="ortho")

    fdn = fd / (torch.abs(fd) + eps)
    fxn = fx / (torch.abs(fx) + eps)

    # view_as_real -> (..., 2) where last dim is (real, imag)
    return F.l1_loss(torch.view_as_real(fdn), torch.view_as_real(fxn))
# -----------------------------
# Data
# -----------------------------

def parse_dataset_spec(s: str) -> Tuple[Path, int, int]:
    if ":" not in s:
        raise ValueError(f"Dataset spec missing ':': {s}")
    root_s, size_s = s.rsplit(":", 1)
    if "x" not in size_s:
        raise ValueError(f"Dataset size must be WIDTHxHEIGHT: {s}")
    w_s, h_s = size_s.split("x", 1)
    root = Path(root_s)
    w = int(w_s)
    h = int(h_s)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid target size in spec: {s}")
    return root, w, h


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    if not paths:
        raise RuntimeError(f"No images found under {root}")
    return paths


def resize_min_and_center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = img.size
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    if new_w != w or new_h != h:
        img = TVF.resize(img, [new_h, new_w], interpolation=IM.BICUBIC, antialias=True)

    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    img = TVF.crop(img, top=top, left=left, height=target_h, width=target_w)
    return img


class ImageReconDataset(Dataset):
    def __init__(self, root: Path, target_w: int, target_h: int):
        self.root = root
        self.target_w = target_w
        self.target_h = target_h
        self.paths = list_images(root)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = resize_min_and_center_crop(img, self.target_w, self.target_h)

        x = TVF.to_tensor(img)      # [0,1]
        x = x.mul(2.0).sub(1.0)     # [-1,1]
        return x


# -----------------------------
# Sample image helpers
# -----------------------------

# Resize to height/width cover, preserving aspect ratio, then center crop.
def make_cover_resize_center_crop(target_w: int, target_h: int):
    def _f(img: Image.Image) -> Image.Image:
        src_w, src_h = img.width, img.height
        scale = max(target_w / src_w, target_h / src_h)
        resized_w = round(src_w * scale)
        resized_h = round(src_h * scale)
        img2 = TVF.resize(img, [resized_h, resized_w], interpolation=IM.BICUBIC, antialias=True)
        return TVF.center_crop(img2, [target_h, target_w])
    return _f


def get_transform(width: int, height: int):
    return TVT.Compose(
        [
            lambda im: im.convert("RGB"),
            make_cover_resize_center_crop(width, height),
            TVT.ToTensor(),
            # Must be in [-1, 1] before VAE encode
            TVT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def tensor_to_pil_rgb(img_chw: torch.Tensor) -> Image.Image:
    """
    img_chw: float tensor in [0, 1], shape (3, H, W)
    """
    img_chw = img_chw.clamp(0.0, 1.0)
    img_hwc_u8 = (img_chw.permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(img_hwc_u8)


@torch.no_grad()
def write_vae_sample_webp(
    *,
    vae_model,
    sample_img: Path,
    target_w: int,
    target_h: int,
    out_path: Path,
) -> None:
    tfm = get_transform(target_w, target_h)

    img = Image.open(sample_img)
    x = tfm(img)  # (3,H,W) in [-1,1]

    device = next(vae_model.parameters()).device
    x = x.unsqueeze(0).to(device)  # (1,3,H,W)

    enc = vae_model.encode(x)
    latents = enc.latent_dist.mean  # raw latent, no scaling
    dec = vae_model.decode(latents).sample  # reconstructed in [-1,1] (typical)

    dec_01 = (dec[0] / 2.0 + 0.5).clamp(0.0, 1.0).detach().cpu()  # (3,H,W) in [0,1]
    pil = tensor_to_pil_rgb(dec_01)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(str(out_path), format="WEBP", lossless=True, quality=100, method=6)


# -----------------------------
# Train
# -----------------------------

@dataclass
class LoaderPack:
    name: str
    loader: DataLoader
    it: object
    sampler: DistributedSampler | None
    epoch: int


def ddp_init_if_needed() -> Tuple[bool, int, int]:
    """
    Returns: (use_ddp, rank, local_rank)
    DDP is enabled when launched via torchrun (RANK/LOCAL_RANK/WORLD_SIZE env vars).
    """
    if not torch.cuda.is_available():
        return False, 0, 0
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        return False, 0, 0

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank


def is_rank0(use_ddp: bool, rank: int) -> bool:
    return (not use_ddp) or (rank == 0)


def main() -> None:
    print("Using VAE:", args.model)

    if args.allow_tf32 is True:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Enabled TF32 for speed over precision")
    else:
        print("Disabled TF32 for maximum precision")
    if args.lpips_weight > 0:
        print("Using LPIPS at", args.lpips_weight)
    if args.lpips_shapeonly:
        if args.lpips_weight <= 0:
            raise SystemExit("--lpips_shapeonly requires --lpips_weight > 0")
        print("Using LPIPS Shape-Only")
    if args.laplacian_weight != 0:
        print(f"Using Laplacian loss weight={args.laplacian_weight}")
    elif args.lpips_weight == 0:
        print("WARNING: suggest using either lpips or laplacian values")

    use_ddp, rank, local_rank = ddp_init_if_needed()

    # Make each rank's RNG different but reproducible-ish
    seed = args.seed + (rank if use_ddp else 0)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    try:
        vae = AutoencoderKL.from_pretrained(
            args.model,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(device)
    except EnvironmentError:
        raise SystemExit("ERROR: Cannot load model from", args.model)

    if args.gradient_checkpointing:
        if is_rank0(use_ddp, rank):
            print("Enabling gradient checkpointing (less VRAM, slower).", flush=True)
        vae.enable_gradient_checkpointing()

    vae.train()

    # LPIPS (frozen perceptual loss). Expects inputs in [-1, 1], which matches this script.
    if args.lpips_weight > 0:
        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)

    if use_ddp:
        vae = DDP(vae, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    packs: List[LoaderPack] = []
    # your existing worker heuristic
    os_cpu_count = (Path("/proc/cpuinfo").read_text().count("processor")
                    if Path("/proc/cpuinfo").exists() else 4)
    num_workers = min(8, os_cpu_count)

    # Use the first dataset's size for the sample image transform (if enabled)
    _, sample_tw, sample_th = parse_dataset_spec(args.dataset[0])
    sample_img_path = Path(args.sample_img) if args.sample_img else None

    for spec in args.dataset:
        root, tw, th = parse_dataset_spec(spec)
        ds = ImageReconDataset(root, tw, th)

        sampler = DistributedSampler(ds, shuffle=True) if use_ddp else None

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        packs.append(LoaderPack(
            name=f"{root.name}:{tw}x{th}",
            loader=loader,
            it=iter(loader),
            sampler=sampler,
            epoch=0,
        ))

    opt = torch.optim.AdamW(
        vae.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # scaling factor
    vae_for_config = vae.module if use_ddp else vae
    sf = float(getattr(vae_for_config.config, "scaling_factor", 1.0))

    out_dir = Path(args.output_dir)
    if is_rank0(use_ddp, rank):
        out_dir.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()

    # Laplacian kernel (4-neighbor). Use groups=3 for per-channel conv.
    lap_kernel = None
    if args.laplacian_weight != 0:
        k = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        lap_kernel = k.repeat(3, 1, 1, 1)  # (3,1,3,3)

    miner = None
    if args.crop_mining_weight > 0:
        miner = HardRegionMiner(
            crop_size=args.crop_size,
            num_crops=args.num_crops,
            patch_size=args.crop_patch_size,
            temperature=args.crop_temperature,
        )

    # timing (rank0 logs only)
    last_log_t = time.perf_counter()
    last_log_step = 0
    step = 0

    while step < args.train_steps:
        pack = random.choice(packs)

        try:
            x = next(pack.it)
        except StopIteration:
            pack.epoch += 1
            if pack.sampler is not None:
                pack.sampler.set_epoch(pack.epoch)
            pack.it = iter(pack.loader)
            x = next(pack.it)

        # Do this here so we skip over input images in the order desired
        if args.skip_steps >0:
            step += 1
            args.skip_steps -= 1
            if args.skip_steps == 0:
                print("Skipped to step", step)
            continue

        x = x.to(device, non_blocking=True)

        enc = vae.module.encode(x) if use_ddp else vae.encode(x)
        posterior = enc.latent_dist
        latents = posterior.mean

        dec = (vae.module.decode(latents).sample if use_ddp else vae.decode(latents).sample)

        l1 = F.l1_loss(dec, x)

        kl = None
        if args.kl_weight != 0:
            kl = posterior.kl().mean()


        # -----------------------------
        # High-frequency loss domain selection
        # -----------------------------
        if args.hf_luma_only:
            dec_hf = rgb_to_luma(dec)  # (B,1,H,W)
            x_hf = rgb_to_luma(x)      # (B,1,H,W)
            # (B,1,H,W) -> (B,3,H,W) for LPIPS
            dec_lp = dec_hf.repeat(1, 3, 1, 1)
            x_lp   = x_hf.repeat(1, 3, 1, 1)
        else:
            dec_hf = dec              # (B,3,H,W)
            x_hf = x                  # (B,3,H,W)
            dec_lp = dec
            x_lp   = x

        if args.lpips_weight > 0:
            if args.lpips_shapeonly:
                k = auto_scale_k(x_lp)
                hp_dec = highpass_box(dec_lp, k)
                hp_x = highpass_box(x_lp, k)
                lp = lpips_fn(hp_dec, hp_x).mean()
            else:
                lp = lpips_fn(dec_lp, x_lp).mean()

        # -----------------------------
        #  ...
        # -----------------------------
        dx_dec = dec_hf[:, :, :, 1:] - dec_hf[:, :, :, :-1]
        dy_dec = dec_hf[:, :, 1:, :] - dec_hf[:, :, :-1, :]
        dx_x = x_hf[:, :, :, 1:] - x_hf[:, :, :, :-1]
        dy_x = x_hf[:, :, 1:, :] - x_hf[:, :, :-1, :]

        if args.edge_l1_weight == 0:
            edge_l1 = 0.0
        else:
            edge_l1 = (F.l1_loss(dx_dec, dx_x) + F.l1_loss(dy_dec, dy_x)) * 0.5

        hf_fft_dec = None
        hf_fft_x = None
        if args.fft_weight > 0 or args.fft_phase_weight > 0 or args.hf_energy_weight > 0:
            k = auto_scale_k(x_hf)
            hf_fft_dec = highpass_box(dec_hf, k)
            hf_fft_x = highpass_box(x_hf, k)

        fft_loss = None
        if args.fft_weight > 0:
            fft_loss = fft_mag_loss(hf_fft_dec, hf_fft_x)

        fft_phase = None
        if args.fft_phase_weight > 0:
            fft_phase = fft_phase_loss(hf_fft_dec, hf_fft_x)

        grad_energy_loss = None
        if args.grad_energy_weight > 0:
            grad_energy_loss = (
                F.l1_loss(dx_dec.square(), dx_x.square())
                + F.l1_loss(dy_dec.square(), dy_x.square())
            ) * 0.5

        # loss = l1 # loss = 0.5 * l1 + 0.5 * mse
        # loss = l1 + (1e-6 * kl)
        loss = (
            (args.l1_weight * l1)
            + (args.edge_l1_weight * edge_l1)
        )

        if args.lpips_weight > 0:
            loss = loss + (args.lpips_weight * lp)

        if fft_loss is not None:
            loss = loss + (args.fft_weight * fft_loss)

        if fft_phase is not None:
            loss = loss + (args.fft_phase_weight * fft_phase)


        if kl is not None:
            loss = loss + (args.kl_weight * kl.float())

        # -----------------------------
        # Laplacian (2nd-derivative) loss
        # -----------------------------
        lap_loss = None
        if lap_kernel is not None:
            if args.hf_luma_only:
                # dec_hf/x_hf are (B,1,H,W); lap_kernel is assumed (3,1,3,3) for RGB.
                lap_k = lap_kernel[:1, :1].contiguous()  # (1,1,3,3)
                lap_dec = F.conv2d(dec_hf, lap_k, padding=1, groups=1)
                lap_x = F.conv2d(x_hf, lap_k, padding=1, groups=1)
            else:
                lap_dec = F.conv2d(dec, lap_kernel, padding=1, groups=3)
                lap_x = F.conv2d(x, lap_kernel, padding=1, groups=3)

            lap_loss = F.l1_loss(lap_dec, lap_x)

        # High-frequency "energy" match: compare magnitudes so blur is penalized
        hf_energy_loss = None
        if args.hf_energy_weight > 0:
            # Match high-frequency energy (microcontrast) rather than raw highpass
            hf_energy_loss = F.l1_loss(hf_fft_dec.square(), hf_fft_x.square())


        ## mse = F.mse_loss(dec, x)

        """
        mse style loss averages, and targets large scale things,
        but may cause blur.
        L1 is nitpicky absolute calculations better for small details
        """

        if lap_loss is not None:
            loss = loss + (args.laplacian_weight * lap_loss)

        if hf_energy_loss is not None:
            loss = loss + (args.hf_energy_weight * hf_energy_loss)

        if grad_energy_loss is not None:
            loss = loss + (args.grad_energy_weight * grad_energy_loss)


        if miner is not None:
            # For crops, we may want shapeonly off even if global is on
            # Pass a wrapped lpips that does/doesn't highpass based on crop_shapeonly
            if args.lpips_weight > 0:
                if args.crop_shapeonly:
                    k = auto_scale_k(x)
                    def crop_lpips(a, b):
                        return lpips_fn(highpass_box(a, k), highpass_box(b, k))
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
                use_ddp=use_ddp,
            )
            loss = loss + args.crop_mining_weight * crop_l

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        step += 1

        if (step % 50 == 0 or step == 1) and is_rank0(use_ddp, rank):
            now = time.perf_counter()
            denom = max(1, step - last_log_step)
            sps = (now - last_log_t) / denom
            last_log_t = now
            last_log_step = step

            lp_val = lp.item() if args.lpips_weight else 0.0
            lap_val = lap_loss.item() if lap_loss is not None else 0.0
            hf_val = hf_energy_loss.item() if hf_energy_loss is not None else 0.0
            ge_val = grad_energy_loss.item() if grad_energy_loss is not None else 0.0
            fft_val = fft_loss.item() if fft_loss is not None else 0.0
            kl_val = kl.item() if kl is not None else 0.0


            print(f"step {step}/{args.train_steps} "
                  f"l1/kl/lpips/edge/lap="
                  f"{l1.item():.4f}/{kl_val:.4f}/{lp_val:.4f}/"
                  f"{edge_l1.item():.4f}/{lap_val:.4f} ",
                  flush=True)
            print(
                    f"s/step={sps:.4f} hf={hf_val:.4f}"
                    f" fft={fft_val:.4f}",
                    f" ge={ge_val:.4f} dataset={pack.name}",
                  flush=True)

        if args.save_every > 0 and (step % args.save_every == 0 or step == args.train_steps):
            if is_rank0(use_ddp, rank):
                ckpt_dir = out_dir / f"step_{step:06d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                vae_to_save = vae.module if use_ddp else vae
                vae_to_save.save_pretrained(str(ckpt_dir))
                print(f"saved: {ckpt_dir}", flush=True)

                if sample_img_path is not None:
                    write_vae_sample_webp(
                        vae_model=vae_to_save,
                        sample_img=sample_img_path,
                        target_w=sample_tw,
                        target_h=sample_th,
                        out_path=ckpt_dir / "vae_sample.webp",
                    )

            if use_ddp:
                dist.barrier()

    if is_rank0(use_ddp, rank):
        final_dir = out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        vae_to_save = vae.module if use_ddp else vae
        vae_to_save.save_pretrained(str(final_dir))
        print(f"saved: {final_dir}", flush=True)

        if sample_img_path is not None:
            write_vae_sample_webp(
                vae_model=vae_to_save,
                sample_img=sample_img_path,
                target_w=sample_tw,
                target_h=sample_th,
                out_path=final_dir / "vae_sample.webp",
            )

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
