#!/usr/bin/env python3
"""
find_blurry_grainy.py

You must pass either -b or -g (or both).

Behavior:
  - If -b: print JPEGs that FAIL the blur/sharpness checks.
  - If -g: print JPEGs that are grainy (noise_score >= threshold).
Each filename is printed at most once even if it fails both checks.

Default grainy values are somewhat permissive. I may use
  --noise-thr 0.6 --flat-percentile 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# -------------------------- blur logic (unchanged) -------------------------- #

def laplacian_variance(gray_u8: np.ndarray) -> float:
    """Simple 3x3 Laplacian variance (higher => sharper)."""
    a = gray_u8.astype(np.float32)
    p = np.pad(a, 1, mode="edge")
    lap = (
        p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1]
        - 4.0 * p[1:-1, 1:-1]
    )
    return float(lap.var())


def count_sharp_patches(
    gray_u8: np.ndarray,
    patch_size_px: int,
    sharp_patch_thr: float,
) -> int:
    """Count how many non-overlapping patches are "sharp enough"."""
    h, w = gray_u8.shape
    h2 = (h // patch_size_px) * patch_size_px
    w2 = (w // patch_size_px) * patch_size_px
    if h2 < patch_size_px or w2 < patch_size_px:
        return 0

    g = gray_u8[:h2, :w2]
    sharp_count = 0

    for y in range(0, h2, patch_size_px):
        for x in range(0, w2, patch_size_px):
            patch = g[y : y + patch_size_px, x : x + patch_size_px]
            if laplacian_variance(patch) >= sharp_patch_thr:
                sharp_count += 1

    return sharp_count


def is_blurry_fail(
    gray_u8: np.ndarray,
    width_px: int,
    height_px: int,
    min_short_side: int,
    patch_size_px: int,
    sharp_patch_thr: float,
    min_sharp_patches: int,
) -> bool:
    """Return True if the image FAILS the blur/sharpness checks."""
    if min(width_px, height_px) < min_short_side:
        return True
    sharp_patches = count_sharp_patches(
        gray_u8=gray_u8,
        patch_size_px=patch_size_px,
        sharp_patch_thr=sharp_patch_thr,
    )
    return sharp_patches < min_sharp_patches


# ------------------------- grain logic (unchanged) -------------------------- #

def _blur5(gray_u8: np.ndarray) -> np.ndarray:
    """Separable 5-tap blur: [1,4,6,4,1]/16 applied horizontally then vertically."""
    k = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    a = gray_u8.astype(np.float32)

    p = np.pad(a, ((0, 0), (2, 2)), mode="edge")
    h = (
        k[0] * p[:, 0:-4] +
        k[1] * p[:, 1:-3] +
        k[2] * p[:, 2:-2] +
        k[3] * p[:, 3:-1] +
        k[4] * p[:, 4:  ]
    )

    p = np.pad(h, ((2, 2), (0, 0)), mode="edge")
    v = (
        k[0] * p[0:-4, :] +
        k[1] * p[1:-3, :] +
        k[2] * p[2:-2, :] +
        k[3] * p[3:-1, :] +
        k[4] * p[4:  , :]
    )
    return v


def _gradient_mag(gray_u8: np.ndarray) -> np.ndarray:
    """Fast gradient magnitude proxy using abs diffs; returns float32 (H,W)."""
    a = gray_u8.astype(np.int16)
    dx = np.abs(a[:, 1:] - a[:, :-1]).astype(np.float32)
    dy = np.abs(a[1:, :] - a[:-1, :]).astype(np.float32)
    dx = np.pad(dx, ((0, 0), (0, 1)), mode="edge")
    dy = np.pad(dy, ((0, 1), (0, 0)), mode="edge")
    return dx + dy


def noise_score(gray_u8: np.ndarray, flat_percentile: float = 30.0) -> float:
    """
    Robust noise estimate (sigma-like, in 8-bit units).
    Higher => grainier.

    flat_percentile selects the flattest pixels by gradient magnitude.
    """
    gmag = _gradient_mag(gray_u8)
    thr = np.percentile(gmag, flat_percentile)
    flat = gmag <= thr

    resid = gray_u8.astype(np.float32) - _blur5(gray_u8)
    r = resid[flat]
    if r.size < 1024:
        r = resid.ravel()

    med = np.median(r)
    mad = np.median(np.abs(r - med))
    return float(1.4826 * mad)  # MAD -> sigma for normal noise


# ------------------------------ shared utils ------------------------------- #

def iter_jpegs(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"):
            yield p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Scan JPEGs and print filenames that fail blur checks (-b) and/or look grainy (-g)."
    )
    ap.add_argument("roots", nargs="+", help="Root directories to scan recursively.")

    ap.add_argument("-b", "--blurry", action="store_true", help="Run blur/sharpness test (prints failing files).")
    ap.add_argument("-g", "--grainy", action="store_true", help="Run grain/high-ISO noise test (prints grainy files).")

    ap.add_argument("--min-short-side", type=int, default=512, help="default=512")
    ap.add_argument("--patch-size", type=int, default=64, help="default=64")
    ap.add_argument("--sharp-patch-thr", type=float, default=120.0, help="default=120.0")
    ap.add_argument("--min-sharp-patches", type=int, default=6, help="default=6")

    ap.add_argument("--noise-thr", type=float, default=2.0, 
                    help="Safe ranges: (Strict 0.4) < (Permissive 2.0) < (Special Cases 6.0). default=2.0")
    ap.add_argument("--flat-percentile", type=float, default=15.0, 
                    help="For noise detection, related to edge detection. Typical ranges: (Permissive 10) <> (Strict 60). default=15.0")

    args = ap.parse_args()
    if not args.blurry and not args.grainy:
        ap.error("must pass at least one of: -b (blurry), -g (grainy)")
    return args


def main() -> None:
    args = parse_args()

    for root in args.roots:
        for p in iter_jpegs(Path(root)):
            print_it = False

            try:
                with Image.open(p) as im:
                    w, h = im.size
                    gray = np.array(im.convert("L"), dtype=np.uint8)

                if args.blurry and is_blurry_fail(
                    gray_u8=gray,
                    width_px=w,
                    height_px=h,
                    min_short_side=args.min_short_side,
                    patch_size_px=args.patch_size,
                    sharp_patch_thr=args.sharp_patch_thr,
                    min_sharp_patches=args.min_sharp_patches,
                ):
                    print_it = True

                if args.grainy and noise_score(gray, flat_percentile=args.flat_percentile) >= args.noise_thr:
                    print_it = True

            except Exception:
                if args.blurry:
                    print_it = True

            if print_it:
                print(str(p))


if __name__ == "__main__":
    main()
