#!/usr/bin/env python3
"""
freq_split_hf.py

Single-band high-frequency split for VAE training dataset generation.
Variant of freq_split.py that only ever computes and writes a single
high-frequency split-off. Calculated by:

  HF = original - blur(hf_sigma)

Output is highly regularized specifically for my VAE training.

Every image is first center-square-cropped 
(the very first operation after load, before any other transform),
then bucketed to a fixed size based on the square's side length:

  side < 1536            -> dropped (warning printed, nothing written)
  side in [1536, 2048]   -> center-crop to 1536
  side in (2048, 3071]   -> center-crop to 2048
  side > 3071             -> resize to 1536 (LANCZOS)

--out_resize BASE_DIR writes the center-cropped/bucketed image itself
(lossless webp, mirroring the input tree) and skips frequency-band
splitting entirely.

Optional arg --512 forces final output to be again scaled down,
specifically to 512x512

Optional arg --reg_2048 raises the minimum accepted source size (post
center-square-crop) to 2048 and replaces the bucketing above with:

  side < 2048           -> dropped (warning printed, nothing written)
  side in [2048, 4095]  -> center-crop to 2048
  side > 4095             -> resize to 2048 (LANCZOS)

See freq_split.py for shared defaults, --preserve_hf semantics, and
--analyze mode (already HF-based, reused here unmodified).
"""

import jsonargparse as argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from freq_split import (
    DEFAULT_LF_SIGMA,
    HF_TOP_BANDS,
    _preserve_hf_band,
    add_analyze_args,
    add_preserve_hf_args,
    collect_images,
    run_analyze,
    run_split,
    save_band,
    split_image,
)

# Expected target size is 512x512. Splitting at higher resolution and
# downstream-downscaling gives better, anti-alias style results than
# splitting at 512 directly - but our dataset's minimum image size is
# 1536, so the bucketing below caters to that floor.
MIN_SIDE = 1536
REG_2048_MIN_SIDE = 2048


# ---------------------------------------------------------------------------
# Size normalization
# ---------------------------------------------------------------------------

def _center_square_crop(img: Image.Image) -> Image.Image:
    """Crop the largest centered square out of img."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _center_crop_to(img: Image.Image, size: int) -> Image.Image:
    """Center-crop a square image down to size x size (no scaling)."""
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _bucket_size(img: Image.Image, side: int, reg_2048: bool = False) -> Image.Image:
    """Bucket an already-square image (side >= min_side) to a fixed size."""
    if reg_2048:
        if side <= 4095:
            return img
        # Only resize if can get superclean resize, which
        # requires size=(target x2)
        return img.resize((2048, 2048), Image.LANCZOS)

    # Otherwise, use buckets intended for final-size=512
    # Except we are planning to also apply sigma, so 
    # try to make it close to some uniform size. 
    # In this case, ideally between 1536 <-> 2048. 
    # But we make allowances for 2560
    if side < 2048:
        return _center_crop_to(img, 1536)
    if side == 2048:
        return img
    if side < 2560:
        return _center_crop_to(img, 2048)
    if side < 3072:
        return _center_crop_to(img, 2560)
    if side < 4096:
        return img.resize((1536, 1536), Image.LANCZOS)
    # side >= 4096
    return img.resize((2048, 2048), Image.LANCZOS)


def _load_and_normalize(img_path: Path, reg_2048: bool = False) -> tuple[Image.Image | None, int]:
    """Load, center-square-crop (first operation), then size-bucket.

    Returns (None, side) if the cropped side is below the active minimum
    (MIN_SIDE, or 2048 when reg_2048 is set).
    """
    img = Image.open(img_path).convert("RGB")
    img = _center_square_crop(img)
    side = img.size[0]
    min_side = REG_2048_MIN_SIDE if reg_2048 else MIN_SIDE
    if side < min_side:
        return None, side
    return _bucket_size(img, side, reg_2048), side


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _worker_hf(args: tuple) -> str | None:
    """Process a single image into its HF band. Returns error/skip string or None."""
    img_path, out_dirs, hf_sigma, input_root, preserve_hf, preserve_hf_threshold, use_hf_abs, downsize_512, reg_2048 = args
    try:
        rel = img_path.relative_to(input_root)
        out_paths = {band: out_dirs[band] / rel.with_suffix(".png") for band in out_dirs}
        if all(p.exists() for p in out_paths.values()):
            return None
        img, side = _load_and_normalize(img_path, reg_2048)
        if img is None:
            min_side = REG_2048_MIN_SIDE if reg_2048 else MIN_SIDE
            return f"SKIP:{img_path}: side {side}px < {min_side}, dropped"
        arr = np.array(img, dtype=np.float32) / 255.0
        bands_data = split_image(arr, hf_sigma)
        orig = bands_data.pop("_orig")
        if preserve_hf:
            for top in HF_TOP_BANDS:
                if top in bands_data:
                    bands_data[top] = _preserve_hf_band(
                        bands_data[top], orig, preserve_hf_threshold, use_hf_abs
                    )
        resize_to = 512 if downsize_512 else None
        for band, band_arr in bands_data.items():
            if band not in out_paths:
                continue
            out_path = out_paths[band]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_band(band_arr, out_path, resize_to=resize_to)
        return None
    except Exception as e:
        return f"{img_path}: {e}"


def _worker_resize(args: tuple) -> str | None:
    """Write the center-cropped/bucketed image itself, no frequency split."""
    img_path, out_root, input_root, downsize_512, reg_2048 = args
    try:
        rel = img_path.relative_to(input_root)
        out_path = (out_root / rel).with_suffix(".webp")
        if out_path.exists():
            return None
        img, side = _load_and_normalize(img_path, reg_2048)
        if img is None:
            min_side = REG_2048_MIN_SIDE if reg_2048 else MIN_SIDE
            return f"SKIP:{img_path}: side {side}px < {min_side}, dropped"
        if downsize_512:
            img = img.resize((512, 512), Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, lossless=True)
        return None
    except Exception as e:
        return f"{img_path}: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split images into a single high-frequency band for VAE training."
    )
    parser.add_argument("--input", "-i", required=True, type=Path,
                        help="Directory of source images")

    parser.add_argument("-o", "--out_hf", type=Path,
                        help="Output root for the HF band")
    parser.add_argument("--out_resize", type=Path,
                        help="Write center-cropped/resized images (lossless webp, same tree "
                             "layout) instead of doing any frequency split")

    parser.add_argument("-s", "--hf_sigma", type=float, default=DEFAULT_LF_SIGMA,
                        help=f"Gaussian sigma for the HF split in pixels (default: {DEFAULT_LF_SIGMA})")

    parser.add_argument("--512", dest="downsize_512", action="store_true",
                        help="Downsize the final output to 512x512 (LANCZOS, highest quality) "
                             "after crop/bucket. Applies to both --out_hf and --out_resize.")

    parser.add_argument("--reg_2048", action="store_true",
                        help="Raise the minimum accepted source size to 2048 (instead of "
                             f"{MIN_SIDE}) and bucket accordingly: crop to 2048 up to side "
                             "4095, resize to 2048 above that."
                            " This idea needs more work")

    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                        help="Parallel worker count (default: cpu count - 1)")
    add_preserve_hf_args(parser)
    parser.add_argument("--preview", action="store_true",
                        help="Process first 5 images only, for visual inspection")
    add_analyze_args(parser)
    args = parser.parse_args()

    # Validate input
    if not args.input.is_dir():
        print(f"ERROR: input directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.reg_2048:
        print("--reg_2048 used. Will discard anything smaller than 2048x2048")

    if args.analyze:
        mode = "analyze"
    elif args.out_resize:
        mode = "out_resize"
    elif args.out_hf:
        mode = "split"
    else:
        print("ERROR: specify --out_hf, --out_resize, or --analyze", file=sys.stderr)
        sys.exit(1)

    # Collect images
    images = collect_images(args.input, preview=args.preview)
    if not images:
        print(f"ERROR: no images found in {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.preview:
        print(f"Preview mode: processing {len(images)} images")

    # Print config — all to stderr in analyze mode so stdout stays pipeline-clean
    info = sys.stderr if args.analyze else sys.stdout
    print(f"Input:     {args.input}  ({len(images)} images)", file=info)
    if mode == "analyze":
        print(f"Mode:      analyze  (no files written)", file=info)
        print(f"sigma:     {args.analyze_sigma} px", file=info)
        print(f"threshold: {args.analyze_threshold}", file=info)
    elif mode == "out_resize":
        print(f"Mode:      out_resize  (no frequency split)")
        print(f"Out:       {args.out_resize}")
    else:
        print(f"Out HF:    {args.out_hf}")
        print(f"hf_sigma:  {args.hf_sigma} px")
    if mode != "analyze" and args.downsize_512:
        print(f"Downsize:  512x512 (LANCZOS)")
    min_side = REG_2048_MIN_SIDE if args.reg_2048 else MIN_SIDE
    if mode != "analyze" and args.reg_2048:
        print(f"reg_2048:  on (min side {min_side}px)")
    print(f"Workers:   {args.workers}", file=info)
    print(file=info)

    if mode == "analyze":
        run_analyze(images, args.analyze_sigma, args.analyze_threshold, args.workers)
        return

    if mode == "out_resize":
        work = [(p, args.out_resize, args.input, args.downsize_512, args.reg_2048) for p in images]
        worker_fn = _worker_resize
    else:
        out_dirs = {"hf": args.out_hf}
        work = [
            (p, out_dirs, args.hf_sigma, args.input, args.preserve_hf, args.preserve_hf_threshold, args.use_hf_abs, args.downsize_512, args.reg_2048)
            for p in images
        ]
        worker_fn = _worker_hf

    results = run_split(work, args.workers, worker_fn=worker_fn)
    skips = [r for r in results if r.startswith("SKIP:")]
    errors = [r for r in results if not r.startswith("SKIP:")]

    if skips:
        print(f"\n{len(skips)} image(s) dropped (side < {min_side}px):", file=sys.stderr)
        for s in skips:
            print(f"  {s[len('SKIP:'):]}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    else:
        written = len(images) - len(skips)
        if mode == "out_resize":
            print(f"\nDone. {written} images resized -> {args.out_resize}")
        else:
            print(f"\nDone. {written} images -> band: hf")


if __name__ == "__main__":
    main()
