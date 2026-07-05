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
from pathlib import Path

CACHE_POSTFIX_DEFAULT = "txt_clipl"


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        required=True,
        help="Directory tree that contains caption files",
    )
    p.add_argument(
        "--model",
        default="runwayml/stable-diffusion-v1-5",
        help="HF repo / local dir of a SD1.5/2.1 pipeline (CLIP text encoder)",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--ext", default="txt", help="Extension of caption file (no dot). Default: txt")
    p.add_argument(
        "--postfix",
        default=CACHE_POSTFIX_DEFAULT,
        help=f"Cache extension to write (no dot). Default: {CACHE_POSTFIX_DEFAULT}",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode even if cache already exists",
    )
    return p.parse_args()


def _cache_path(caption_path: Path, postfix: str) -> Path:
    postfix = postfix.lstrip(".")
    # foo.txtqss -> foo.txt_clipl   (replace suffix, do NOT append)
    return caption_path.with_suffix(f".{postfix}")


def build_file_list(root: Path, ext: str, postfix: str, overwrite: bool) -> list[Path]:
    ext = ext.lstrip(".")
    txt_files = sorted(root.rglob(f"*.{ext}"))

    def needs_cache(p: Path) -> bool:
        return overwrite or not _cache_path(p, postfix).exists()

    return [p for p in txt_files if needs_cache(p)]


def _load_pipe_fp32(model_id: str, device: str):
    import torch
    from diffusers import StableDiffusionPipeline

    # Enforce fp32 math (no TF32)
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


def _encode_prompt_embeds_fp32(captions: list[str], pipe, device: str):
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


def worker(
    local_rank: int,
    world_size: int,
    files: list[str],
    model_id: str,
    batch_size: int,
    postfix: str,
):
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
        for path, vec in zip(batch_files, emb, strict=True):
            out_path = _cache_path(path, postfix)
            st.save_file({"emb": vec.contiguous()}, out_path)


def main() -> None:
    args = cli()

    root = Path(args.data_root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"data_root does not exist: {root}")

    files = build_file_list(root, args.ext, args.postfix, args.overwrite)
    if not files:
        print("No captions to encode (nothing missing; use --overwrite to force).")
        return

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")

    world_size = torch.cuda.device_count()
    print(f"Found {world_size} visible GPU(s). Encoding {len(files):,} caption(s).")

    file_strs = [str(p) for p in files]

    if world_size <= 1:
        worker(0, 1, file_strs, args.model, args.batch_size, args.postfix)
        print("All caches written.")
        return

    torch.multiprocessing.spawn(
        fn=worker,
        args=(world_size, file_strs, args.model, args.batch_size, args.postfix),
        nprocs=world_size,
        join=True,
    )
    print("All caches written.")


if __name__ == "__main__":
    main()
