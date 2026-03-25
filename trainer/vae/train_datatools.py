"""
vae_dataset.py

Dataset and image loading helpers for VAE reconstruction training.

Key design:
  - ImageReconDataset returns file paths only, not tensors.
  - load_batch() does the actual image loading and tensor conversion.
  - load_tile_batch() loads the same images as 512x512 native-resolution tiles.
  - Tiling pipeline: resize+crop to 1024x1024, split into 4x 512x512 quadrants.
    Tile index: 0=NW, 1=NE, 2=SW, 3=SE
"""

from pathlib import Path
from typing import List, Tuple
from PIL import Image

import torch
import torchvision.transforms.functional as TVF
from torchvision.transforms import InterpolationMode as IM
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Image helpers
# -----------------------------

def resize_min_and_center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Scale image so shorter side covers target, then center crop.
    Works for any input aspect ratio and any target size.
    """
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


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB image -> float32 tensor (3, H, W) in [-1, 1]."""
    x = TVF.to_tensor(img)
    x = x.mul(2.0).sub(1.0)
    return x


def load_image_tensor(path: Path, target_w: int, target_h: int) -> torch.Tensor:
    """Load one image from disk, resize+crop, return (3, H, W) in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    img = resize_min_and_center_crop(img, target_w, target_h)
    return pil_to_tensor(img)


def make_tiles(path: Path) -> List[torch.Tensor]:
    """
    Load image and split into four 512x512 quadrants.

    - If shortest edge >= 2048: resize+crop to 1024x1024, then tile.
    - If shortest edge >= 1024: center crop to 1024x1024 directly, no upscale.
    - If shortest edge < 1024: logs error and returns empty list (caller must handle).

    Returns list of 4 tensors (3, 512, 512) in [-1, 1], order: NW, NE, SW, SE.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    short_edge = min(w, h)

    if short_edge < 1024:
        print(f"[WARN] Skipping tile for image too small ({w}x{h}): {path}", flush=True)
        return []

    if short_edge >= 2048:
        # Original behavior: scale down so shorter edge = 1024, then center crop
        img = resize_min_and_center_crop(img, 1024, 1024)
    else:
        # Between 1024 and 2048: just center crop at 1024x1024, no scaling
        left = (w - 1024) // 2
        top = (h - 1024) // 2
        img = TVF.crop(img, top=top, left=left, height=1024, width=1024)

    t = pil_to_tensor(img)  # (3, 1024, 1024)
    nw = t[:, :512, :512]
    ne = t[:, :512, 512:]
    sw = t[:, 512:, :512]
    se = t[:, 512:, 512:]
    return [nw, ne, sw, se]


# -----------------------------
# Batch loading
# -----------------------------

def load_batch(paths: List[Path], target_w: int, target_h: int, device: torch.device) -> torch.Tensor:
    """
    Load a list of image paths into a stacked batch tensor.
    Returns (N, 3, H, W) in [-1, 1] on device.
    """
    tensors = [load_image_tensor(p, target_w, target_h) for p in paths]
    return torch.stack(tensors).to(device, non_blocking=True)


def load_tile_batch(paths: List[Path], tile_idx: int, device: torch.device) -> torch.Tensor | None:
    """
    Load a list of image paths as a specific 512x512 tile from their 1024x1024 crop.
    tile_idx: 0=NW, 1=NE, 2=SW, 3=SE
    Returns (N, 3, 512, 512) in [-1, 1] on device, or None if all images were too small.
    """
    assert 0 <= tile_idx <= 3, f"tile_idx must be 0-3, got {tile_idx}"
    tensors = []
    for p in paths:
        tiles = make_tiles(p)
        if tiles:
            tensors.append(tiles[tile_idx])
    if not tensors:
        return None
    return torch.stack(tensors).to(device, non_blocking=True)


# -----------------------------
# Dataset
# -----------------------------

def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    if not paths:
        raise RuntimeError(f"No images found under {root}")
    return paths


def parse_dataset_spec(s: str) -> Tuple[Path, int, int]:
    """
    Parse a dataset spec string like '/path/to/images:512x512'.
    Returns (root_path, width, height).
    """
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


class ImageReconDataset(Dataset):
    """
    Returns (path_str, tensor) tuples.
    Workers load the whole-image tensor in the background.
    The path is kept so the training loop can load tiles on demand.
    Use collate_path_tensor as the DataLoader collate_fn.
    """
    def __init__(self, root: Path, target_w: int, target_h: int):
        self.root = root
        self.target_w = target_w
        self.target_h = target_h
        self.paths = list_images(root)
        print(f"  {root}: {len(self.paths)} images")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        tensor = load_image_tensor(p, self.target_w, self.target_h)
        return str(p), tensor


def collate_path_tensor(batch):
    """
    Custom collate for ImageReconDataset.
    Returns (list_of_path_strings, stacked_tensor_batch).
    """
    paths = [item[0] for item in batch]
    tensors = torch.stack([item[1] for item in batch])
    return paths, tensors


# -----------------------------
# Sample image helpers
# -----------------------------

def tensor_to_pil_rgb(img_chw: torch.Tensor) -> Image.Image:
    """float tensor (3, H, W) in [0, 1] -> PIL RGB image."""
    img_chw = img_chw.clamp(0.0, 1.0)
    arr = (img_chw.permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(arr)


@torch.no_grad()
def write_vae_sample_webp(
    *,
    vae_model,
    sample_img: Path,
    target_w: int,
    target_h: int,
    out_path: Path,
) -> None:
    device = next(vae_model.parameters()).device
    x = load_image_tensor(sample_img, target_w, target_h)
    x = x.unsqueeze(0).to(device)

    enc = vae_model.encode(x)
    latents = enc.latent_dist.mean
    dec = vae_model.decode(latents).sample

    dec_01 = (dec[0] / 2.0 + 0.5).clamp(0.0, 1.0).detach().cpu()
    pil = tensor_to_pil_rgb(dec_01)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(str(out_path), format="WEBP", lossless=True, quality=100, method=6)
