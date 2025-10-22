#!/usr/bin/env python3
"""
build_t5cache_gpu.py
~~~~~~~~~~~~~~~~~~~~
Generate T5-caption caches using your custom Diffusers pipeline.

For every  caption.txt  under  --data_root
    caption.txt_t5cache   (bf16 safetensors) is written.

Key points:
• Tight-fit per file: one caption per encode (no padding saved).
• Multi-threaded I/O/saving; GPU work is serialized by default (safe on 24GB 4090).
• Uses pipeline.encode_prompt() (no manual tokenizer/encoder calls).
• Default model: /BLUE/t5-train/models/t5-sd  (change via --model)
"""

import argparse
import concurrent.futures
import gc
import os
import threading
from pathlib import Path

import safetensors.torch as st
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm

# This is a hidden dependancy. Put this to force cleaner errormsg
import sentencepiece  

CACHE_POSTFIX = "_t5cache"  # keep for training backend compatibility


# --------------------------------------------------------------------------- #
def cli():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        required=True,
        help="Directory tree that contains *.txt caption files",
    )
    p.add_argument(
        "--model",
        default="/BLUE/t5-train/models/t5-sd",
        help="HF repo / local dir of your pipeline(or take default)",
    )
    p.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="GPU compute precision",
    )
    # Kept for backward compatibility only; unused in single-item encodes
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="(Deprecated/unused) Previously batched encodes. Ignored.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=f"Re-encode even if *{CACHE_POSTFIX} already exists",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=min(8, (os.cpu_count() or 4)),
        help="CPU threads for file I/O and saving. Default=min(8, CPU cores)",
    )
    p.add_argument(
        "--gpu_concurrency",
        type=int,
        default=1,
        help="Max concurrent GPU encodes. 1 is safest on a single 24GB GPU.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
@torch.inference_mode()
def encode_gpu_single(caption: str, pipe, precision: str) -> torch.Tensor:
    """
    Encode ONE caption on GPU and return bf16 CPU tensor [T, D] (tight; no pads saved).
    """
    cast = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.autocast("cuda", cast):
        emb = pipe.encode_prompt(
            [caption],  # single-item batch
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            truncation=True,
            padding="do_not_pad",  # tight; no extra tokens produced
        )  # (1, T, D) on GPU
    return emb.squeeze(0).to(torch.bfloat16, copy=False).cpu()  # (T, D) on CPU


# --------------------------------------------------------------------------- #
def main():
    args = cli()
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Loading", args.model)
    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        custom_pipeline=args.model,
        torch_dtype=torch.float16 if args.dtype == "fp16" else torch.bfloat16,
    )

    # Optional info; skip if projection absent in your custom pipeline
    if hasattr(pipe, "t5_projection") and hasattr(pipe.t5_projection, "config"):
        try:
            sf = pipe.t5_projection.config.scaling_factor
            print("T5 (projection layer) scaling factor is", sf)
        except Exception:
            pass

    # Free modules unused during caption encoding to save VRAM
    for attr in ("vae", "unet", "image_encoder", "scheduler"):
        if hasattr(pipe, attr):
            setattr(pipe, attr, None)

    pipe.to("cuda")
    if hasattr(pipe, "tokenizer"):
        pipe.tokenizer.padding_side = "right"  # standard; no effect with do_not_pad
    gc.collect()
    torch.cuda.empty_cache()

    # ---------- gather caption files -------------------------------------- #
    root = Path(args.data_root).expanduser().resolve()
    txt_files = sorted(root.rglob("*.txt"))

    print(f"Parsing {root} while skipping existing cache files...")

    def needs_cache(p: Path) -> bool:
        return args.overwrite or not (p.with_suffix(p.suffix + CACHE_POSTFIX)).exists()

    txt_files = [p for p in txt_files if needs_cache(p)]
    total = len(txt_files)
    print(
        f"🟢 Encoding {total:,} captions with {args.workers} CPU worker(s); "
        f"GPU concurrency={args.gpu_concurrency}"
    )

    # Semaphore gates how many threads may run GPU encode at once
    gpu_sem = threading.Semaphore(args.gpu_concurrency)

    def process_one(path: Path):
        # Returns True on success; returns Exception instance on failure
        try:
            caption = path.read_text(encoding="utf-8").strip()
            if not caption:
                # Save empty tensor to mark processed, or skip; here we skip
                return True
            with gpu_sem:  # ensure limited concurrent GPU work
                vec = encode_gpu_single(caption, pipe, args.dtype)  # [T, D] bf16 on CPU
            out = path.with_suffix(path.suffix + CACHE_POSTFIX)
            st.save_file({"emb": vec}, out)
            return True
        except Exception as e:
            return e

    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in tqdm(ex.map(process_one, txt_files), total=total, unit="file"):
            if isinstance(res, Exception):
                errors += 1

    if errors:
        print(f"⚠️ Completed with {errors} error(s). See logs above if printed.")
    else:
        print("✅ All caches written.")

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
