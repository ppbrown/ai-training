#!/usr/bin/env python3
"""
detect_poor_capture_dof_ok.py

Detect likely poor-capture images while allowing shallow depth-of-field.
** DO NOT TRUST THIS ON ITS OWN! 
   Use as a filter to hand-check a smaller subset of your images

We do NOT score blur over the entire frame. Instead we:
  - divide the image into square patches (e.g. 64x64 pixels)
  - compute a sharpness score per patch (Laplacian variance)
  - keep the image if there are "enough" sharp patches

This way, an image with a sharp subject and blurred background is kept.

Default behavior: prints filenames that FAIL the checks.

Example:
  python3 detect_poor_capture_dof_ok.py /data/images --min-short-side 512 \
      --patch-size 64 --sharp-patch-thr 120 --min-sharp-patches 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


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


def iter_jpegs(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"):
            yield p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Print JPEGs likely too low-quality for HQ training (DOF-friendly sharpness test)."
    )
    ap.add_argument("roots", nargs="+", help="Root directories to scan recursively.")
    ap.add_argument(
        "--min-short-side",
        type=int,
        default=512,
        help="Reject if min(width,height) is below this (pixels). Default: 512",
    )
    ap.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Square patch size in pixels (patch_size x patch_size). Default: 64",
    )
    ap.add_argument(
        "--sharp-patch-thr",
        type=float,
        default=120.0,
        help="Patch is 'sharp' if Laplacian variance >= this. Default: 120.0",
    )
    ap.add_argument(
        "--min-sharp-patches",
        type=int,
        default=6,
        help="Image passes if it has at least this many sharp patches. Default: 6",
    )
    ap.add_argument(
        "--print-passing",
        action="store_true",
        help="Print files that PASS instead of files that FAIL.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    for root in args.roots:

        for p in iter_jpegs(Path(root)):
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    gray = np.array(im.convert("L"), dtype=np.uint8)

                too_small = min(w, h) < args.min_short_side
                sharp_patches = 0 if too_small else count_sharp_patches(
                    gray_u8=gray,
                    patch_size_px=args.patch_size,
                    sharp_patch_thr=args.sharp_patch_thr,
                )
                passes = (not too_small) and (sharp_patches >= args.min_sharp_patches)

                if args.print_passing:
                    if passes:
                        print(str(p))
                else:
                    if not passes:
                        print(str(p))

            except Exception:
                # Unreadable/corrupt -> treat as failing unless user asked for passing-only.
                if not args.print_passing:
                    print(str(p))


if __name__ == "__main__":
    main()

