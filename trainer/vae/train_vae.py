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
import torchvision.transforms.functional as TVF
from torchvision.transforms import InterpolationMode as IM

from diffusers import AutoencoderKL

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
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size per step (per dataset, per GPU).")
    ap.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    ap.add_argument("--save_every", type=int, default=2000, help="Save checkpoint every N steps.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = ap.parse_args()

    use_ddp, rank, local_rank = ddp_init_if_needed()

    # Make each rank's RNG different but reproducible-ish
    seed = args.seed + (rank if use_ddp else 0)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)

    if args.gradient_checkpointing:
        if is_rank0(use_ddp, rank):
            print("Enabling gradient checkpointing (less VRAM, slower).", flush=True)
        vae.enable_gradient_checkpointing()

    vae.train()

    if use_ddp:
        vae = DDP(vae, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    packs: List[LoaderPack] = []
    # your existing worker heuristic
    os_cpu_count = (Path("/proc/cpuinfo").read_text().count("processor")
                    if Path("/proc/cpuinfo").exists() else 4)
    num_workers = min(8, os_cpu_count)

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
        latents = enc.latent_dist.mean * sf

        dec = (vae.module.decode(latents / sf).sample if use_ddp else vae.decode(latents / sf).sample)

        loss = F.mse_loss(dec, x)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        step += 1

        if (step % 50 == 0 or step == 1) and is_rank0(use_ddp, rank):
            print(f"step {step}/{args.train_steps}  loss={loss.item():.6f}  dataset={pack.name}", flush=True)

        if args.save_every > 0 and (step % args.save_every == 0 or step == args.train_steps):
            if is_rank0(use_ddp, rank):
                ckpt_dir = out_dir / f"step_{step:08d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                vae_to_save = vae.module if use_ddp else vae
                vae_to_save.save_pretrained(str(ckpt_dir))
                print(f"saved: {ckpt_dir}", flush=True)
            if use_ddp:
                dist.barrier()

    if is_rank0(use_ddp, rank):
        final_dir = out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        vae_to_save = vae.module if use_ddp else vae
        vae_to_save.save_pretrained(str(final_dir))
        print(f"saved: {final_dir}", flush=True)

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
