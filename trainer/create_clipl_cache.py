#!/bin/env python

""" util to work with old classic SD1.5 CLIP_L embeddings """

import argparse, gc
from pathlib import Path


CACHE_POSTFIX = "_clipl"  # distinguish from your T5 cache

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="directory tree that contains *.txt caption files")
    p.add_argument("--model",
                   default="runwayml/stable-diffusion-v1-5",
                   help="HF repo / local dir of a SD1.5/2.1 pipeline (CLIP)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--overwrite", action="store_true",
                   help=f"re-encode even if *{CACHE_POSTFIX} already exists")
    return p.parse_args()

#faster usage return
args = cli()

import torch, safetensors.torch as st
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


@torch.inference_mode()
def encode_gpu(captions, pipe, precision):
    cast = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.autocast("cuda", cast):
        emb = pipe.encode_prompt(
            captions,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    return emb[0].to(torch.bfloat16, copy=False).cpu()

def main():
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Loading", args.model)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.dtype == "fp16" else torch.bfloat16,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Free modules unused during caption encoding to save VRAM
    for attr in ("vae", "unet", "image_encoder", "scheduler"):
        if hasattr(pipe, attr):
            setattr(pipe, attr, None)

    pipe.to("cuda")
    gc.collect(); torch.cuda.empty_cache()

    root = Path(args.data_root).expanduser().resolve()
    txt_files = sorted(root.rglob("*.txt"))

    print("Parsing", root)
    def needs_cache(p: Path) -> bool:
        return args.overwrite or not (p.with_suffix(p.suffix + CACHE_POSTFIX)).exists()

    txt_files = [p for p in txt_files if needs_cache(p)]
    print(f"🟢 Encoding {len(txt_files):,} captions on GPU")

    bs = args.batch_size
    for start in tqdm(range(0, len(txt_files), bs), unit="batch"):
        batch_files = txt_files[start : start + bs]
        captions = [p.read_text(encoding="utf-8").strip() for p in batch_files]

        emb = encode_gpu(captions, pipe, args.dtype)

        for path, vec in zip(batch_files, emb):
            st.save_file({"emb": vec}, path.with_suffix(path.suffix + CACHE_POSTFIX))

    print("✅ All caches written.")

if __name__ == "__main__":
    main()
