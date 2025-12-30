#!/usr/bin/env python3
"""
create_clipl_cache.py

Create per-caption SD1.5 CLIP-L prompt-embedding caches.

Behavior (no launcher required):
- Always COMPUTE in full fp32 (no autocast, no bf16/fp16, TF32 disabled).
- Always STORE fp32 (safetensors).
- Automatically uses all visible GPUs on this single machine (one worker process per GPU).
- If only 1 GPU is visible, runs single-GPU.
"""

import argparse
import gc
import os
from pathlib import Path

CACHE_POSTFIX = "_clipl"


def cli():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        required=True,
        help="directory tree that contains *.txt caption files",
    )
    p.add_argument(
        "--model",
        default="runwayml/stable-diffusion-v1-5",
        help="HF repo / local dir of a SD1.5/2.1 pipeline (CLIP)",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=f"re-encode even if *{CACHE_POSTFIX} already exists",
    )
    return p.parse_args()


def build_file_list(root: Path, overwrite: bool) -> list[Path]:
    txt_files = sorted(root.rglob("*.txt"))

    def needs_cache(p: Path) -> bool:
        return overwrite or not (p.with_suffix(p.suffix + CACHE_POSTFIX)).exists()

    return [p for p in txt_files if needs_cache(p)]


def _load_pipe_fp32(model_id: str, device: str):
    import torch
    from diffusers import StableDiffusionPipeline

    # Enforce fp32 math (no TF32).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Free modules unused during caption encoding to save VRAM
    for attr in ("vae", "unet", "image_encoder", "scheduler"):
        if hasattr(pipe, attr):
            setattr(pipe, attr, None)

    pipe.to(device)
    pipe.text_encoder.eval()

    gc.collect()
    torch.cuda.empty_cache()
    return pipe


def _encode_prompt_embeds_fp32(captions, pipe, device):
    import torch

    with torch.inference_mode():
        out = pipe.encode_prompt(
            captions,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = out[0] if isinstance(out, (tuple, list)) else out
        return prompt_embeds.to(dtype=torch.float32).cpu()


def worker(local_rank: int, world_size: int, files: list[str], model_id: str, batch_size: int):
    import torch
    import safetensors.torch as st

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    pipe = _load_pipe_fp32(model_id, device)

    # Shard work deterministically across GPUs
    my_files = files[local_rank::world_size]

    for start in range(0, len(my_files), batch_size):
        batch_files = [Path(p) for p in my_files[start : start + batch_size]]
        captions = [p.read_text(encoding="utf-8").strip() for p in batch_files]

        emb = _encode_prompt_embeds_fp32(captions, pipe, device)

        # emb is [B, T, C]; save per-sample
        for path, vec in zip(batch_files, emb):
            st.save_file({"emb": vec}, path.with_suffix(path.suffix + CACHE_POSTFIX))


def main():
    args = cli()

    root = Path(args.data_root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"data_root does not exist: {root}")

    files = build_file_list(root, args.overwrite)
    if not files:
        print("No captions to encode (nothing missing; use --overwrite to force).")
        return

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")

    world_size = torch.cuda.device_count()
    print(f"Found {world_size} visible GPU(s). Encoding {len(files):,} caption(s).")

    # Convert Paths to plain strings for safer multiprocessing pickling.
    file_strs = [str(p) for p in files]

    if world_size <= 1:
        worker(0, 1, file_strs, args.model, args.batch_size)
        print("All caches written.")
        return

    # No torchrun. Pure python entrypoint.
    torch.multiprocessing.spawn(
        fn=worker,
        args=(world_size, file_strs, args.model, args.batch_size),
        nprocs=world_size,
        join=True,
    )
    print("All caches written.")


if __name__ == "__main__":
    # Ensure spawn start method behaves consistently.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
