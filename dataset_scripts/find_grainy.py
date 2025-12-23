#!/usr/bin/env python3
"""
detect_grainy_jpegs.py

Recursively scan a directory tree and print JPEG filenames that appear
grainy / high-ISO noisy.

Method (no ML, no OpenCV):
  - Convert to grayscale.
  - Estimate "noise" as robust MAD of a high-frequency residual
    (image - small blur) measured only on the flattest pixels
    (low gradient), so real texture/edges don't dominate.
  - Flag if noise_score >= threshold.

Usage:
  python3 detect_grainy_jpegs.py /path/to/images
  python3 detect_grainy_jpegs.py /path/to/images --noise-thr 6.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


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


def iter_jpegs(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"):
            yield p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Print JPEG filenames that appear grainy / high-ISO noisy.")
    ap.add_argument("roots", nargs="+", help="Root directory to scan recursively.")
    ap.add_argument(
        "--noise-thr",
        type=float,
        default=6.0,
        help="Flag image if noise_score >= this. Default: 6.0",
    )
    ap.add_argument(
        "--flat-percentile",
        type=float,
        default=30.0,
        help="Use flattest X%% pixels (by gradient) for noise estimation. Default: 30.0",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    for root in args.roots:
        for p in iter_jpegs(Path(root)):
            try:
                with Image.open(p) as im:
                    gray = np.array(im.convert("L"), dtype=np.uint8)
                if noise_score(gray, flat_percentile=args.flat_percentile) >= args.noise_thr:
                    print(str(p))
            except Exception:
                continue


if __name__ == "__main__":
    main()

