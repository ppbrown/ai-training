#!/bin/env python
"""Utility to decode a VAE cache file

Default is to show the given filenames on display. However,
with --writepreview, it will write a companion file,
   "{file}.webp"
The cache generator does not write these files by default, because
out of 1000s of files, you probably only need to check certain
small categories (eg: humans)

Usage:

 Pass in one or more latent image cachefile names
    (safetensors with key "latent")
  --model MODEL   Diffusers model directory or repo (must have VAE)
  --custom        Look for custom pipeline in the model
  --writepreview  Write out a .webp file instead of displaying

"""


import sys
device = "cpu"  # or: torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, file=sys.stderr)
# I want this to print FAST, so before other imports

import argparse
from pathlib import Path

import torch
import safetensors.torch as st
from diffusers import DiffusionPipeline, AutoencoderKL
from torchvision.transforms.functional import to_pil_image

import tkinter as tk
from PIL import ImageTk


# ---- config ----
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        help="Diffusers model directory or repo (must have VAE)",
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    p.add_argument("--vae", action="store_true", help="Treat model as direct vae, not full pipeline")
    p.add_argument(
        "--custom",
        action="store_true",
        help="Look for custom pipeline in the model",
    )
    p.add_argument(
        "--writepreview",
        action="store_true",
        help="Write out a .webp file instead of display",
    )
    p.add_argument(
        "files",
        nargs="+",
        help="Path(s) to VAE cache file(s) (.safetensors) containing key 'latent'",
    )
    return p

def load_latent(file_path: str):
    try:
        latent = st.load_file(file_path)["latent"].to(device)
    except Exception:
        print("ERROR: could not load file", file_path)
        exit(1)
    return latent

def decode_latent_to_pil(vae_model, latent):
    p = next(vae_model.parameters())
    latent = latent.to(device=p.device, dtype=p.dtype)

    # Expect latent shape like (C, H, W); add batch dim for decode()
    decoded = vae_model.decode(latent.unsqueeze(0)).sample

    decoded = (decoded / 2 + 0.5).clamp(0, 1)  # Undo normalization
    decoded = decoded.squeeze(0).cpu()         # Remove batch dimension

    pil_img = to_pil_image(decoded)
    return pil_img


class Viewer:
    def __init__(self, root: tk.Tk, vae_model, files: list[str]):
        self.root = root
        self.vae_model = vae_model
        self.files = files
        self.idx = 0

        self.label = tk.Label(root)
        self.label.pack()

        self.status = tk.Label(root, anchor="w", justify="left")
        self.status.pack(fill="x")

        # Key bindings
        root.bind("<Right>", self.next_image)
        root.bind("<space>", self.next_image)
        root.bind("n", self.next_image)

        root.bind("<Left>", self.prev_image)
        root.bind("p", self.prev_image)

        root.bind("q", self.quit)
        root.bind("<Escape>", self.quit)

        self._photo = None  # keep reference
        self._normal_cursor = root.cget("cursor")
        self.show_current()

    def _set_busy_cursor(self, busy: bool) -> None:
        self.root.configure(cursor="watch" if busy else self._normal_cursor)
        # Force the cursor change to appear immediately
        self.root.update_idletasks()

    def show_current(self):
        self._set_busy_cursor(True)
        try:
            file_path = self.files[self.idx]
            latent = load_latent(file_path)
            latent_shape = latent.shape
            pil_img = decode_latent_to_pil(self.vae_model, latent)

            # Convert to Tk image
            self._photo = ImageTk.PhotoImage(pil_img)
            self.label.configure(image=self._photo)

            # Title bar shows filename
            title = f"{Path(file_path).name}  ({self.idx + 1}/{len(self.files)})"
            self.root.title(title)

            self.status.configure(
                text=f"File: {file_path}\nLatent shape: {latent_shape}\nKeys: Next=Right/Space/n  Prev=Left/p  Quit=q/Esc"
            )
        finally:
            self._set_busy_cursor(False)

    def next_image(self, _event=None):
        self.idx = (self.idx + 1) % len(self.files)
        self.show_current()

    def prev_image(self, _event=None):
        self.idx = (self.idx - 1) % len(self.files)
        self.show_current()

    def quit(self, _event=None):
        self.root.destroy()

def _load_vae_fp32(model_id: str, vae: bool):

    if vae:
        vae = AutoencoderKL.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

    vae.to("cpu")
    vae.eval()
    return vae


def WritePreviews(vae_model, files: list[str]) -> list[Path]:
    """
    Write lossless preview images next to each input cache file.

    Output filename: <input_file>.webp
    Encoding: lossless WebP (space-efficient, lossless).
    """
    written: list[Path] = []

    for file_path in files:
        in_path = Path(file_path)
        out_path = Path(file_path + ".webp")
        latent = load_latent(file_path)
        pil_img = decode_latent_to_pil(vae_model, latent)

        pil_img.save(out_path, format="WEBP", lossless=True, method=6)
        written.append(out_path)
        print("Saved preview to", out_path)

    return written


def main():
    args = build_argparser().parse_args()

    print(f"Using model {args.model} on {len(args.files)} file(s)")

    vae_model = _load_vae_fp32(args.model, args.vae)
    if args.writepreview:
        WritePreviews(vae_model, args.files)
        exit(0)

    rootwin = tk.Tk()
    Viewer(rootwin, vae_model, args.files)
    rootwin.mainloop()


if __name__ == "__main__":
    main()
