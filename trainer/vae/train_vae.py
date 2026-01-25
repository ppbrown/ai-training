#!/usr/bin/env python3
"""
Fine-tune SDXL VAE on real-world images using reconstruction loss only.

Example:
  python3 train_vae_only.py \
    --dataset /data/laion:1024x1024 \
    --dataset /data/cc12m:1024x1024 \
    --dataset /data/pexels:1024x1024 \
    --output_dir ./vae_finetuned \
    --train_steps 20000 \
    --batch_size 2 \
    --lr 1e-5
"""

import argparse
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


# -----------------------------
# Data
# -----------------------------

def parse_dataset_spec(s: str) -> Tuple[Path, int, int]:
    # Format: /path/to/dataset:WIDTHxHEIGHT
    # Example: /data/laion:1024x1024  or  /data/laion:448x576
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
    # Your structure is top/00/, top/01/, etc; rglob is simplest and robust.
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    if not paths:
        raise RuntimeError(f"No images found under {root}")
    return paths


def resize_min_and_center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    # Minimal resize preserving aspect so that both dims >= target, then center crop.
    w, h = img.size
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    if new_w != w or new_h != h:
        img = TVF.resize(img, [new_h, new_w], interpolation=IM.BICUBIC, antialias=True)

    # Center crop
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

        # ToTensor gives [0,1]. SDXL VAE expects [-1,1].
        x = TVF.to_tensor(img)                 # (3,H,W), float32, [0,1]
        x = x.mul(2.0).sub(1.0)                # [-1,1]
        return x


# -----------------------------
# Train
# -----------------------------

@dataclass
class LoaderPack:
    name: str
    loader: DataLoader
    it: object


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
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size per step (per dataset).")
    ap.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    ap.add_argument("--save_every", type=int, default=2000, help="Save checkpoint every N steps.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)

    vae.train()

    # Build one loader per dataset (targets differ => shapes differ => separate loaders).
    packs: List[LoaderPack] = []
    for spec in args.dataset:
        root, tw, th = parse_dataset_spec(spec)
        ds = ImageReconDataset(root, tw, th)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(8, (os_cpu_count := (Path("/proc/cpuinfo").read_text().count("processor") if Path("/proc/cpuinfo").exists() else 4))),
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        packs.append(LoaderPack(name=f"{root.name}:{tw}x{th}", loader=loader, it=iter(loader)))

    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    # SDXL VAE scaling factor (diffusers stores this on config).
    # We use the standard encode->(latent*sf)->decode(latent/sf) convention.
    sf = float(getattr(vae.config, "scaling_factor", 1.0))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    while step < args.train_steps:
        pack = random.choice(packs)
        try:
            x = next(pack.it)
        except StopIteration:
            pack.it = iter(pack.loader)
            x = next(pack.it)

        x = x.to(device, non_blocking=True)

        # Encode -> latent (use mean for stable reconstruction training)
        enc = vae.encode(x)
        latents = enc.latent_dist.mean * sf

        # Decode
        dec = vae.decode(latents / sf).sample

        # Reconstruction loss in pixel space (both in [-1,1])
        loss = F.mse_loss(dec, x)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        step += 1

        if step % 50 == 0 or step == 1:
            print(f"step {step}/{args.train_steps}  loss={loss.item():.6f}  dataset={pack.name}")

        if args.save_every > 0 and (step % args.save_every == 0 or step == args.train_steps):
            ckpt_dir = out_dir / f"step_{step:08d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            vae.save_pretrained(str(ckpt_dir))
            print(f"saved: {ckpt_dir}")

    # Final save (also convenient if save_every is large)
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(str(final_dir))
    print(f"saved: {final_dir}")


if __name__ == "__main__":
    import os
    main()
