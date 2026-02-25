#!/bin/env python

# Encode, and then DECODE an image,
# to make sure that the vae is not buggy

# fyi: uses "tempfile"

# ---------- arg section -------------------------------------------------#
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", 
                      help="Diffusers model directory or repo (must have VAE). Default is sd-base",
                      default="/BLUE/t5-train/models/sd-base")
parser.add_argument("--vae", help="VAE model directory or repo (VAE only)")

parser.add_argument("--imgfile", required=True, help="Path to an image file")
parser.add_argument("--custom", action="store_true",help="Treat model as custom pipeline")
parser.add_argument("--res", type=int, default=512,help="Length of res square. Default=512")

args = parser.parse_args()

# -----------------------------------------------------------#

from pathlib import Path
import torch
import safetensors.torch as st
import torchvision.transforms as transforms
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL
from PIL import Image

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

if args.vae:
    print("Using VAE", args.vae)
    vae_model = AutoencoderKL.from_pretrained(args.vae)
else:
    print("Using model", args.model)
    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        custom_pipeline=args.model if args.custom else None,
    )
    vae_model = pipe.vae

vae_model = vae_model.to(device).eval()

input_image = Image.open(args.imgfile).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((args.res, args.res)),  # Resize to 512x512 for consistency
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Mandatory Coordinate transformation for VAE input
])

input_tensor = transform(input_image).unsqueeze(0).to(device) 

# Encode the image
with torch.no_grad():
    encoded = vae_model.encode(input_tensor).latent_dist.sample()

    print("Latent shape",encoded.shape)
    st.save_file({"latent": encoded}, "tempfile")

    cached = st.load_file("tempfile")["latent"].to(device)
    decoded_image = vae_model.decode(cached).sample

decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)  # Undo transformation
decoded_image = decoded_image.squeeze(0).cpu()

from torchvision.transforms.functional import to_pil_image

pil_image = to_pil_image(decoded_image)
pil_image.show()
