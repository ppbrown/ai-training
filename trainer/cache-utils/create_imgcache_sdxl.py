#!/usr/bin/env python3

"""
    Create latent cache files from img files.
    This is half of what train_lion_cached.py needs

    For large runs, maximum throughput is obtained by running at least 2
    seperate processes of this, after splitting up dataset accordingly
"""

import os
import sys
import subprocess

# This stuff required for the "deterministic" settings
ENV_VAR = "CUBLAS_WORKSPACE_CONFIG"
DESIRED = ":4096:8"  # Or ":16:8" if preferred

if os.environ.get(ENV_VAR) != DESIRED:
    os.environ[ENV_VAR] = DESIRED
    print(f"[INFO] Setting {ENV_VAR}={DESIRED} and re-executing.")
    args = [sys.executable] + sys.argv
    # Use os.execvpe to replace the current process (no zombie parent)
    os.execvpe(args[0], args, os.environ)
    sys.exit(1) # If exec fails, exit explicitly

import argparse
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torchvision.transforms as TVT
from torchvision.transforms import InterpolationMode as IM
import torchvision.transforms.functional as F

import safetensors.torch as st
from diffusers import DiffusionPipeline
from PIL import Image



device = "cuda" if torch.cuda.is_available() else "cpu"

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0",
                   help="HF repo or local dir (default=stabilityai/stable-diffusion-xl-base-1.0)")
    p.add_argument("--data_root", required=True,
                   help="Directory containing images (recursively searched)")
    p.add_argument("--out_suffix", default=".img_sdxl", 
                   help="File suffix for saved latents(default: .img_sdxl)")
    p.add_argument("--target_width", type=int, default=512, help="Width Resolution for images (default: 512)")
    p.add_argument("--target_height", type=int, default=512, help="Height Resolution for images (default: 512)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--extensions", nargs="+", default=["jpg", "jpeg", "png", "webp"])
    p.add_argument("--custom", action="store_true",help="Treat model as custom pipeline")
    return p.parse_args()

def find_images(input_dir, exts):
    images = []
    for ext in exts:
        images += list(Path(input_dir).rglob(f"*.{ext}"))
    return sorted(images)


# Resize to height, while preserving aspect ratio.
# Then crop to width
def make_cover_resize_center_crop(target_width: int, target_height: int):
    def _f(img):
        src_height, src_width = img.height, img.width
        scale = max(target_width / src_width, target_height / src_height)
        resized_width, resized_height = round(src_width * scale), round(src_height * scale)
        img = F.resize(img, (resized_height, resized_width), interpolation=IM.BICUBIC, antialias=True)
        return F.center_crop(img, (target_height, target_width))
    return _f

def get_transform(width, height):
    return TVT.Compose([
        lambda im: im.convert("RGB"),
        make_cover_resize_center_crop(width, height),
        TVT.ToTensor(),
        # Have to do this before appplying VAE!!
        TVT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


def _load_vae_fp32(model_id: str):
    global device

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    vae.to(device)
    vae.eval()
    return vae

@torch.no_grad()
def main():
    args = parse_args()


    # Load pipeline (for VAE)
    vae = _load_vae_fp32(args.model)

    # Collect images
    all_image_paths = find_images(args.data_root, args.extensions)
    image_paths = []
    skipped = 0
    for path in all_image_paths:
        out_path = path.with_name(path.stem + args.out_suffix)
        if out_path.exists():
            skipped += 1
            continue
        image_paths.append(path)
    if not image_paths:
        print("No new images to process (all cache files exist).")
        return
    if skipped:
        print(f"Skipped {skipped} files with existing cache.")


    tfm = get_transform(args.target_width, args.target_height, )


    print(f"Processing {len(image_paths)} images from {args.data_root}")
    print("Batch size is",args.batch_size)
    print(f"Using {args.model} to {args.out_suffix}...")
    print("")

    for i in tqdm(range(0, len(image_paths), args.batch_size)):
        batch_paths = image_paths[i:i+args.batch_size]
        batch_imgs = []
        valid_paths = []
        for path in batch_paths:
            try:
                img = Image.open(path)
                img = tfm(img)
                batch_imgs.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Could not load {path}: {e}")

        if not batch_imgs:
            continue

        batch_tensor = torch.stack(batch_imgs).to(device)
        latents = vae.encode(batch_tensor).latent_dist.mean.cpu()  # raw latent, no scaling

        # Save each latent as its own safetensors file
        for j, path in enumerate(valid_paths):
            out_path = path.with_name(path.stem + args.out_suffix)
            st.save_file({"latent": latents[j]}, str(out_path))

if __name__ == "__main__":
    main()
