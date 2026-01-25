#!/bin/env python

# Pass in one or more latent image cachefiles (.safetensors with key "latent")
# Decode via the model VAE and view interactively:
#   - Next: Right / Space / n
#   - Prev: Left / p
#   - Quit: q / Esc

import sys
device = "cpu"  # or: torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, file=sys.stderr)
# I want this to print FAST, so before other imports

import argparse
from pathlib import Path

import torch
import safetensors.torch as st
from diffusers import DiffusionPipeline
from torchvision.transforms.functional import to_pil_image

import tkinter as tk
from PIL import ImageTk


# ---- config ----
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        help="Diffusers model directory or repo (must have VAE)",
        default="/BLUE/t5-train/models/sdxl-orig",
    )
    p.add_argument(
        "--custom",
        action="store_true",
        help="Look for custom pipeline in the model",
    )
    p.add_argument(
        "files",
        nargs="+",
        help="Path(s) to VAE cache file(s) (.safetensors) containing key 'latent'",
    )
    return p


def decode_latent_to_pil(vae_model, file_path: str):
    cached = st.load_file(file_path)["latent"].to(device)

    # Expect latent shape like (C, H, W); add batch dim for decode()
    decoded = vae_model.decode(cached.unsqueeze(0)).sample

    decoded = (decoded / 2 + 0.5).clamp(0, 1)  # Undo normalization
    decoded = decoded.squeeze(0).cpu()         # Remove batch dimension

    pil_img = to_pil_image(decoded)
    return pil_img, tuple(cached.shape)


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
            pil_img, latent_shape = decode_latent_to_pil(self.vae_model, file_path)

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

def _load_vae_fp32(model_id: str):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    vae.to("cpu")
    vae.eval()
    return vae


def main():
    args = build_argparser().parse_args()

    print(f"Using model {args.model} on {len(args.files)} file(s)")

    vae_model = _load_vae_fp32(args.model)

    root = tk.Tk()
    Viewer(root, vae_model, args.files)
    root.mainloop()


if __name__ == "__main__":
    main()
