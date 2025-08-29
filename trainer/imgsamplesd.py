#!/bin/env python

# This is to specifically use "normal", non-T5 SD

def argproc():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--seed",  type=int, default=90)
    p.add_argument("--steps",  type=int, default=30)
    p.add_argument("--vae_scaling",  type=float)
    p.add_argument("--prompt", nargs="+", type=str,
                   default="woman", help="one or more prompt strings")
    p.add_argument("--output_directory", type=str)
    return p.parse_args()

args=argproc()

from diffusers import DiffusionPipeline
import torch.nn as nn, torch, types
import os,sys

from PIL import Image, PngImagePlugin


MODEL = args.model

print("HAND HACKING FLOWMATCH MODULE")
from diffusers import FlowMatchEulerDiscreteScheduler
def scale_model_input(self, sample, timestep):
    return sample
FlowMatchEulerDiscreteScheduler.scale_model_input = scale_model_input

print(f"Loading from {MODEL}")
if MODEL.endswith(".safetensors") or MODEL.endswith(".st"):
    raise ValueError("Cannot acccept single-file models. "
    "Need diffusers directory tree or hf reference")
else:
    pipe = DiffusionPipeline.from_pretrained(
        MODEL, use_safetensors=True,
        safety_checker=None, requires_safety_checker=False,
        torch_dtype=torch.bfloat16,
    )

if args.vae_scaling:
    pipe.vae.config.scaling_factor = args.vae_scaling

#pipe.safety_checker = dummy_safety_checker

pipe.safety_checker=None

print("model initialized. ")
pipe.enable_sequential_cpu_offload()
# The above obviates the need for to("cuda") I guess..
#pipe.to("cuda")

prompt=args.prompt
seed=args.seed

generator = torch.Generator(device="cuda").manual_seed(seed)

print(f"Trying render of '{prompt}' using seed {seed}...")
images = pipe(prompt, num_inference_steps=args.steps, generator=generator).images

if args.output_directory:
    OUTDIR=args.output_directory
else:
    OUTDIR=MODEL if os.path.isdir(MODEL) else "./"

for i,image in enumerate(images):
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Comment", f"prompt={prompt}")
    fname=f"{OUTDIR}/sample{i}_s{seed}.png"
    print(f"saving to {fname}")
    image.save(fname, pnginfo=meta)

