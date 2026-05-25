#!/usr/bin/env python3
"""
Frequency separation of images in a source tree.

For each image found under SRC, writes two PNGs to DEST:
  <stem>-lf.png  — low-frequency (blurred) component, normal image
  <stem>-hf.png  — high-frequency residual, centered at mid-gray (128)
                   so edges appear lighter/darker than the neutral gray background

Directory structure under SRC is preserved under DEST.

Example:
  freq_split.py /data/images /data/split --ratio 0.1
  produces /data/split/00/foo-lf.png and /data/split/00/foo-hf.png
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def load_float(path):
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def process_image(src_path, dest_dir, ratio):
    arr = load_float(src_path)
    h, w = arr.shape[:2]
    sigma = ratio * min(h, w)

    low = gaussian_filter(arr, sigma=[sigma, sigma, 0])
    high = arr - low

    lf_uint8 = (np.clip(low, 0, 1) * 255).astype(np.uint8)
    # Center HF at 0.5 (mid-gray=128): negative diffs go dark, positive go bright
    hf_uint8 = (np.clip(high + 0.5, 0, 1) * 255).astype(np.uint8)

    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = src_path.stem
    Image.fromarray(lf_uint8).save(dest_dir / f'{stem}-lf.png')
    Image.fromarray(hf_uint8).save(dest_dir / f'{stem}-hf.png')


def iter_images(root):
    for p in Path(root).rglob('*'):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('src', help='Source directory tree')
    parser.add_argument('dest', help='Destination directory for output pairs')
    parser.add_argument(
        '--ratio', type=float, default=0.001,
        help='Gaussian sigma as fraction of shorter image dimension (default: 0.1)',
    )
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dest = Path(args.dest).resolve()

    if not src.is_dir():
        print(f'Error: {args.src!r} is not a directory', file=sys.stderr)
        sys.exit(1)

    jobs = []
    for img_path in iter_images(src):
        rel = img_path.parent.relative_to(src)
        jobs.append((img_path, dest / rel))

    if not jobs:
        print('No images found.')
        sys.exit(1)

    total = len(jobs)
    done = 0
    errors = 0

    workers = os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_image, p, d, args.ratio): p for p, d in jobs}
        for fut in as_completed(futures):
            img_path = futures[fut]
            try:
                fut.result()
                done += 1
                print(f'\r{done}/{total}', end='', flush=True)
            except Exception as exc:
                print(f'\nWARN: {img_path}: {exc}', file=sys.stderr)
                errors += 1

    print(f'\nDone. {done} written, {errors} errors.')


if __name__ == '__main__':
    main()
