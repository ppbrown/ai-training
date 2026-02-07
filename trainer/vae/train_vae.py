#!/usr/bin/env python3
"""
Fine-tune SDXL VAE on real-world images using reconstruction loss only.
"""

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVF
from torchvision.transforms import InterpolationMode as IM

from diffusers import AutoencoderKL
import lpips
import time

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


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
    # Pillow 13 deprecates the explicit mode= parameter here; omit it.
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
    ap = argparse.ArgumentParser(description="Fine-tune SDXL VAE with image reconstruction loss only.")
    ap.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Repeatable. Format: /path/to/dataset:WIDTHxHEIGHT (e.g., /data/laion:1024x1024)",
    )
    ap.add_argument("--output_dir", required=True, help="Where to save the fine-tuned VAE.")
    ap.add_argument("--train_steps", type=int, required=True, help="Number of optimizer steps.")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--allow_tf32", action="store_true",
                   help="Speed optimization. (Possibly bad at extremely low LR?)")
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size per step (per dataset, per GPU).")
    ap.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    ap.add_argument("--save_every", type=int, default=2000, help="Save checkpoint every N steps.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--lpips_weight", type=float,
                    help="high quality but high cost. Range 0.2 - 0.05")
    ap.add_argument(
        "--laplacian_weight",
        type=float,
        default=0.05,
        help="Weight for Laplacian (2nd-derivative) loss term (default: 0.05). Set 0 to disable.",
    )
    ap.add_argument(
        "--sample_img",
        type=str,
        default=None,
        help="Optional. If set, encode/decode this image through the VAE at each save and write vae_sample.webp in the save dir.",
    )
    args = ap.parse_args()

    print("Using VAE", args.model)
    if args.allow_tf32 is True:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Enabled TF32 for speed over precision")
    else:
        print("Disabled TF32 for maximum precision")
    if args.lpips_weight:
        print("Using LPIPS at", args.lpips_weight)
    if args.laplacian_weight != 0:
        print(f"Using Laplacian loss weight={args.laplacian_weight}")

    use_ddp, rank, local_rank = ddp_init_if_needed()

    # Make each rank's RNG different but reproducible-ish
    seed = args.seed + (rank if use_ddp else 0)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    vae = AutoencoderKL.from_pretrained(
        args.model,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)

    if args.gradient_checkpointing:
        if is_rank0(use_ddp, rank):
            print("Enabling gradient checkpointing (less VRAM, slower).", flush=True)
        vae.enable_gradient_checkpointing()

    vae.train()

    # LPIPS (frozen perceptual loss). Expects inputs in [-1, 1], which matches this script.
    if args.lpips_weight:
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

    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    # scaling factor
    vae_for_config = vae.module if use_ddp else vae
    sf = float(getattr(vae_for_config.config, "scaling_factor", 1.0))

    out_dir = Path(args.output_dir)
    if is_rank0(use_ddp, rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure out_dir exists before non-rank0 tries to save (it won't, but be safe)
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

        x = x.to(device, non_blocking=True)

        enc = vae.module.encode(x) if use_ddp else vae.encode(x)
        posterior = enc.latent_dist
        latents = posterior.mean * sf

        dec = (vae.module.decode(latents / sf).sample if use_ddp else vae.decode(latents / sf).sample)

        l1 = F.l1_loss(dec, x)
        kl = posterior.kl().mean()
        if args.lpips_weight:
            lp = lpips_fn(dec, x).mean()
        mse = F.mse_loss(dec, x)

        # edge / gradient loss (finite differences)
        dx_dec = dec[:, :, :, 1:] - dec[:, :, :, :-1]
        dy_dec = dec[:, :, 1:, :] - dec[:, :, :-1, :]
        dx_x   = x[:,   :, :, 1:] - x[:,   :, :, :-1]
        dy_x   = x[:,   :, 1:, :] - x[:,   :, :-1, :]
        edge_l1 = (F.l1_loss(dx_dec, dx_x) + F.l1_loss(dy_dec, dy_x)) * 0.5

        # In theory could/should do LPIPS loss calc. 
        # But that would be large performance hit

        # loss = l1 # loss = 0.5 * l1 + 0.5 * mse
        # loss = l1 + (1e-6 * kl)
        if args.lpips_weight:
            loss = l1 + (0.1 * edge_l1) + (args.lpips_weight * lp) + (1e-6 * kl)
        else:
            loss = l1 + (0.1 * edge_l1) + (1e-6 * kl)

        # Laplacian (2nd-derivative) loss
        lap_loss = None
        if lap_kernel is not None:
            lap_dec = F.conv2d(dec, lap_kernel, padding=1, groups=3)
            lap_x   = F.conv2d(x,   lap_kernel, padding=1, groups=3)
            lap_loss = F.l1_loss(lap_dec, lap_x)

        """ mse style loss averges, and targets large scale things,
        but may cause blur.
        L1 is nitpicky absolute calculations better for small details
        """

        if lap_loss is not None:
            loss = loss + (args.laplacian_weight * lap_loss)

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

            print(f"step {step}/{args.train_steps} "
                  f"l1/kl/lpips={l1.item():.6f}/{kl.item():.6f}/{lp_val:.6f} "
                  f"s/step={sps:.4f} dataset={pack.name}", 
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
