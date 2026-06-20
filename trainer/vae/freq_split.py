#!/usr/bin/env python3
"""
freq_split.py

Frequency-band split for VAE training dataset generation.

2-band mode (--out_lf / --out_hf):
  Splits images into LF and HF using a single Gaussian blur at lf_sigma.
    LF  = blur(lf_sigma)
    HF  = original - blur(lf_sigma)

4-band mode (--out_4split BASE_DIR):
  Splits images into 4 bands using 3 Gaussian blurs.
    LF  = blur(lf_sigma)          - same as 2-band LF
    HF1 = blur(hf1_sigma) - LF   - medium-high frequencies
    HF2 = blur(hf2_sigma) - blur(hf1_sigma)  - high frequencies
    HF3 = original - blur(hf2_sigma)          - finest detail
  Output directories: BASE_DIR/lf/, BASE_DIR/hf1/, BASE_DIR/hf2/, BASE_DIR/hf3/
  Sigma order must satisfy: hf2_sigma < hf1_sigma < lf_sigma

Sigma values are FIXED in pixel space for consistent band boundaries
across the entire dataset.

Usage:
  # 2-band split
  python freq_split.py --input /data/originals --out_lf /data/lf --out_hf /data/hf

  # 4-band split
  python freq_split.py --input /data/originals --out_4split /data/split4

  # 4-band with custom sigmas
  python freq_split.py --input /data/originals --out_4split /data/split4 \\
      --lf_sigma 10.0 --hf1_sigma 4.0 --hf2_sigma 1.5

All bands are saved as PNG, clipped to [0,1].

--preserve_hf:
  Instead of clipping the top HF band (hf in 2-band, hf3 in 4-band) to [0,1],
  active pixels (where any channel of the signed residual exceeds the threshold)
  are replaced with the original source pixel; inactive pixels are written as black.
  This preserves true color and brightness at HF-active locations without
  grey-shifting or losing the sign information via clip.

Default sigma values (for 512x512 images):
  lf_sigma  = 10.0 px  (broad color/luminance blobs)
  hf1_sigma =  5.0 px  (hf1/hf2 boundary, one octave below lf)
  hf2_sigma =  2.5 px  (hf2/hf3 boundary, one octave below hf1)
"""

import jsonargparse as argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_LF_SIGMA  = 10.0
DEFAULT_HF1_SIGMA =  5.0   # octave below lf_sigma
DEFAULT_HF2_SIGMA =  2   # octave below hf1_sigma
DEFAULT_ANALYZE_SIGMA     =  1.0
DEFAULT_ANALYZE_THRESHOLD =  0.02
DEFAULT_PRESERVE_HF_THRESHOLD = 0.03  # ~8/255; signed HF per-pixel max above this = active
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# top HF band name per mode
HF_TOP_BANDS = {"hf", "hf3"}


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------

def _gblur(arr: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([
        gaussian_filter(arr[:, :, c], sigma=sigma)
        for c in range(3)
    ], axis=2)


def _resize_shortest_side(img: Image.Image, target: int = 1024) -> Image.Image:
    """Downscale (never upscale) so the shortest side equals `target`, preserving aspect ratio."""
    w, h = img.size
    short_side = min(w, h)
    if short_side <= target:
        return img
    scale = target / short_side
    new_size = (round(w * scale), round(h * scale))
    return img.resize(new_size, Image.BICUBIC)


def _resize_arr(arr: np.ndarray, target: int) -> np.ndarray:
    """Downscale float32 HxWx3 array so shortest side equals target. Preserves negative values."""
    h, w = arr.shape[:2]
    short = min(h, w)
    if short <= target:
        return arr
    scale = target / short
    new_w, new_h = round(w * scale), round(h * scale)
    return np.stack([
        np.array(Image.fromarray(arr[:, :, c]).resize((new_w, new_h), Image.BICUBIC))
        for c in range(3)
    ], axis=2)


def split_image(img_path: Path, lf_sigma: float) -> dict[str, np.ndarray]:
    """2-band split: returns {'lf', 'hf', '_orig'}."""
    arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    lf = _gblur(arr, lf_sigma)
    return {"lf": lf, "hf": arr - lf, "_orig": arr}


def split_image_4(
    img_path: Path,
    lf_sigma: float,
    hf1_sigma: float,
    hf2_sigma: float,
) -> dict[str, np.ndarray]:
    """4-band split: returns {'lf', 'hf1', 'hf2', 'hf3', '_orig'}."""
    arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    lf   = _gblur(arr, lf_sigma)
    mid1 = _gblur(arr, hf1_sigma)
    mid2 = _gblur(arr, hf2_sigma)
    return {
        "lf":  lf,
        "hf1": mid1 - lf,
        "hf2": mid2 - mid1,
        "hf3": arr  - mid2,
        "_orig": arr,
    }


def _preserve_hf_band(hf_signed: np.ndarray, orig: np.ndarray, threshold: float) -> np.ndarray:
    """Return orig pixels where signed HF exceeds threshold, black elsewhere."""
    #mask = hf_signed.max(axis=2) > threshold
    mask = np.abs(hf_signed).max(axis=2) > threshold
    out = np.zeros_like(orig)
    out[mask] = orig[mask]
    return out


def save_band(arr: np.ndarray, out_path: Path) -> None:
    clipped = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((clipped * 255).astype(np.uint8))
    #img.save(out_path.with_suffix(".png"), compress_level=9)
    #img.save(out_path.with_suffix(args.suffix))
    #img.save(out_path.with_suffix(".jpg", quality=95, subsampling=0))
    img.save(out_path.with_suffix(".webp"), lossless=True)


def measure_sharpness(img_path: Path, sigma: float, resize_target: int | None = None) -> float:
    """RMS of the HF component at `sigma`. Higher = sharper."""
    arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    hf = arr - _gblur(arr, sigma)
    return float(np.sqrt(np.mean(hf ** 2)))


def _analyze_worker(args: tuple) -> tuple[str, float]:
    img_path, sigma = args
    try:
        return (str(img_path), measure_sharpness(img_path, sigma))
    except Exception:
        return (str(img_path), -1.0)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> str | None:
    """Process a single image. Returns error string or None on success."""
    img_path, out_dirs, lf_sigma, input_root, hf1_sigma, hf2_sigma, resize_target, preserve_hf, preserve_hf_threshold = args
    try:
        rel = img_path.relative_to(input_root)
        out_paths = {band: out_dirs[band] / rel.with_suffix(".png") for band in out_dirs}
        if all(p.exists() for p in out_paths.values()):
            return None
        if hf1_sigma is not None:
            bands_data = split_image_4(img_path, lf_sigma, hf1_sigma, hf2_sigma)
        else:
            bands_data = split_image(img_path, lf_sigma)
        orig = bands_data.pop("_orig")
        if preserve_hf:
            for top in HF_TOP_BANDS:
                if top in bands_data:
                    bands_data[top] = _preserve_hf_band(
                        bands_data[top], orig, preserve_hf_threshold
                    )
        for band, arr in bands_data.items():
            if band not in out_paths:
                continue
            out_path = out_paths[band]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if resize_target:
                clipped_img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))
                w, h = clipped_img.size
                short = min(w, h)
                if short > resize_target:
                    scale = resize_target / short
                    new_size = (round(w * scale), round(h * scale))
                    clipped_img = clipped_img.resize(new_size, Image.BICUBIC)
                clipped_img.save(out_path.with_suffix(".webp"), lossless=True)
            else:
                save_band(arr, out_path)
        return None
    except Exception as e:
        return f"{img_path}: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split images into absolute frequency bands for VAE training."
    )
    parser.add_argument("--input", "-i", required=True, type=Path,
                        help="Directory of source images")
    parser.add_argument("--suffix", default=".png")
    resize_group = parser.add_mutually_exclusive_group()
    resize_group.add_argument("--1024", dest="resize_target", action="store_const", const=1024,
                              help="Downscale so shortest side is 1024px before splitting (never upscales)")
    resize_group.add_argument("--512", dest="resize_target", action="store_const", const=512,
                              help="Downscale so shortest side is 512px before splitting (never upscales)")

    # 2-band mode
    parser.add_argument("--out_lf", type=Path,
                        help="(2-band) Output root for LF images")
    parser.add_argument("--out_hf", type=Path,
                        help="(2-band) Output root for HF images")

    # 4-band mode
    parser.add_argument("--out_4split", type=Path,
                        help="(4-band) Base output directory; creates lf/, hf1/, hf2/, hf3/ subdirs")

    # Sigma controls
    parser.add_argument("--lf_sigma", type=float, default=DEFAULT_LF_SIGMA,
                        help=f"Gaussian sigma for LF band in pixels (default: {DEFAULT_LF_SIGMA})")
    parser.add_argument("--hf1_sigma", type=float, default=DEFAULT_HF1_SIGMA,
                        help=f"(4-band) Sigma for HF1/HF2 boundary (default: {DEFAULT_HF1_SIGMA})")
    parser.add_argument("--hf2_sigma", type=float, default=DEFAULT_HF2_SIGMA,
                        help=f"(4-band) Sigma for HF2/HF3 boundary (default: {DEFAULT_HF2_SIGMA}. best is probably 1.5)")

    parser.add_argument("--workers", type=int, default=mp.cpu_count(),
                        help="Parallel worker count (default: cpu count)")
    parser.add_argument("--preserve_hf", action="store_true",
                        help="Write original pixel values at active HF locations instead of "
                             "clipping the signed residual; inactive (near-zero) pixels become black. "
                             "Applies only to the top HF band (hf in 2-band, hf3 in 4-band).")
    parser.add_argument("--preserve_hf_threshold", type=float, default=DEFAULT_PRESERVE_HF_THRESHOLD,
                        help=f"Signed HF per-pixel max above which a pixel is considered active "
                             f"for --preserve_hf (default: {DEFAULT_PRESERVE_HF_THRESHOLD})")
    parser.add_argument("--preview", action="store_true",
                        help="Process first 5 images only, for visual inspection")
    parser.add_argument("--analyze", action="store_true",
                        help="Report soft/blurry images without writing any files")
    parser.add_argument("--analyze_sigma", type=float, default=DEFAULT_ANALYZE_SIGMA,
                        help=f"HF sigma for sharpness measurement in pixels (default: {DEFAULT_ANALYZE_SIGMA})")
    parser.add_argument("--analyze_threshold", type=float, default=DEFAULT_ANALYZE_THRESHOLD,
                        help=f"Minimum RMS sharpness score to pass (default: {DEFAULT_ANALYZE_THRESHOLD})")
    args = parser.parse_args()

    # Validate input
    if not args.input.is_dir():
        print(f"ERROR: input directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Determine mode and build out_dirs
    if args.analyze:
        hf1_sigma = None
        hf2_sigma = None
        out_dirs = {}
    elif args.out_4split:
        if args.hf2_sigma >= args.hf1_sigma or args.hf1_sigma >= args.lf_sigma:
            print(
                f"ERROR: sigmas must satisfy hf2_sigma < hf1_sigma < lf_sigma "
                f"(got {args.hf2_sigma} < {args.hf1_sigma} < {args.lf_sigma}?)",
                file=sys.stderr,
            )
            sys.exit(1)
        hf1_sigma = args.hf1_sigma
        hf2_sigma = args.hf2_sigma
        out_dirs = {
            "lf":  args.out_4split / "lf",
            "hf1": args.out_4split / "hf1",
            "hf2": args.out_4split / "hf2",
            "hf3": args.out_4split / "hf3",
        }
    elif args.out_lf or args.out_hf:
        hf1_sigma = None
        hf2_sigma = None
        out_dirs = {}
        if args.out_lf:
            out_dirs["lf"] = args.out_lf
        if args.out_hf:
            out_dirs["hf"] = args.out_hf
    else:
        print(
            "ERROR: specify either --out_4split BASE_DIR  or at least one of --out_lf / --out_hf",
            file=sys.stderr,
        )
        sys.exit(1)

    # Collect images
    images = sorted([
        p for p in args.input.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ])
    if not images:
        print(f"ERROR: no images found in {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.preview:
        images = images[:5]
        print(f"Preview mode: processing {len(images)} images")

    # Print config — all to stderr in analyze mode so stdout stays pipeline-clean
    info = sys.stderr if args.analyze else sys.stdout
    print(f"Input:     {args.input}  ({len(images)} images)", file=info)
    if args.analyze:
        print(f"Mode:      analyze  (no files written)", file=info)
        print(f"sigma:     {args.analyze_sigma} px", file=info)
        print(f"threshold: {args.analyze_threshold}", file=info)
    elif args.out_4split:
        print(f"Out base:  {args.out_4split}  (lf/, hf1/, hf2/, hf3/)")
        print(f"lf_sigma:  {args.lf_sigma} px")
        print(f"hf1_sigma: {args.hf1_sigma} px")
        print(f"hf2_sigma: {args.hf2_sigma} px")
    else:
        if args.out_lf:
            print(f"Out LF:    {args.out_lf}")
        if args.out_hf:
            print(f"Out HF:    {args.out_hf}")
        print(f"lf_sigma:  {args.lf_sigma} px")
    print(f"Workers:   {args.workers}", file=info)
    print(file=info)

    if args.analyze:
        awork = [(p, args.analyze_sigma) for p in images]
        soft: list[tuple[str, float]] = []
        read_errors: list[str] = []
        with mp.Pool(args.workers) as pool:
            for path, score in tqdm(
                pool.imap_unordered(_analyze_worker, awork), total=len(awork)
            ):
                if score < 0:
                    read_errors.append(path)
                elif score < args.analyze_threshold:
                    soft.append((path, score))
        if read_errors:
            print(f"{len(read_errors)} file(s) could not be read:", file=sys.stderr)
            for p in sorted(read_errors):
                print(f"  {p}", file=sys.stderr)
        for path, score in sorted(soft):
            print(path)
        print(
            f"{len(soft)}/{len(images)} failed "
            f"(sigma={args.analyze_sigma}, threshold={args.analyze_threshold})",
            file=sys.stderr,
        )
        return

    work = [
        (p, out_dirs, args.lf_sigma, args.input, hf1_sigma, hf2_sigma, args.resize_target, args.preserve_hf, args.preserve_hf_threshold)
        for p in images
    ]

    errors = []
    with mp.Pool(args.workers) as pool:
        for err in tqdm(pool.imap_unordered(_worker, work), total=len(work)):
            if err:
                errors.append(err)

    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    else:
        bands = list(out_dirs.keys())
        print(f"\nDone. {len(images)} images -> bands: {', '.join(bands)}")


if __name__ == "__main__":
    main()
