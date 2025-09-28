#!/usr/bin/env python3

# util that identifies which images have a high amount of text content in them,
# suggesting that you probably want to cull them.
# you need to "pip install pytesseract", but also install
# on ubuntu, for example, "apt install tesseract-ocr"
#
# Yes there are VLMs that do this sort of thing. This is NOT a VLM, yet its somewhat fast.
# This means you can run a GPU job at the same time you are running this over some large
# other dataset,and neither job will slow down the other.
# (Unless you tweak this to run on gpu. Theoretically possible, I have no idea how though.)
#
# Note that by default this uses 16 cpu threads. I think it has about 1 image/thread/second
# throughput, but I havent accurately measured it.


import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal

# deps: pip install opencv-python pytesseract
import cv2
import pytesseract
from datetime import datetime

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def text_coverage(img_path, min_conf: float):
    im = cv2.imread(str(img_path))
    if im is None:
        return 0.0, 0, 0.0
    h, w = im.shape[:2]
    data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
    boxes, chars = [], 0
    for i, conf in enumerate(data["conf"]):
        try:
            conf = float(conf)
        except ValueError:
            continue
        if conf >= min_conf and data["text"][i].strip():
            x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            boxes.append((x, y, bw, bh))
            chars += len(data["text"][i])
    area = sum(bw * bh for (_, _, bw, bh) in boxes)
    bandH = int(0.2 * h)
    band_area = sum(bw * bh for (x, y, bw, bh) in boxes if y < bandH or y + bh > h - bandH)
    denom = max(w * h, 1)
    return area/denom, chars, band_area/denom

def looks_text_heavy(path, min_conf=60, cov_thr=0.015, nchar_thr=30, band_thr=0.008):
    cov, nchar, band_cov = text_coverage(path, min_conf)
    return (cov > cov_thr) or (nchar > nchar_thr) or (band_cov > band_thr)

def _worker(args):
    path, min_conf, cov_thr, nchar_thr, band_thr = args
    try:
        if looks_text_heavy(path, min_conf, cov_thr, nchar_thr, band_thr):
            return str(path)
    except Exception:
        # Treat failures as non-text to avoid crashing the whole run
        pass
    return None

def main():
    ap = argparse.ArgumentParser(
        description="Flag text-heavy images (posters/memes/logos) and write to list.textheavy"
    )
    ap.add_argument("topdir", type=Path, help="Top-level directory to scan")
    ap.add_argument("--workers", type=int, default=16, help="CPU workers")
    ap.add_argument("--min-conf", type=float, default=60, help="OCR min confidence (0-100): default=60")
    ap.add_argument("--cov-thr", type=float, default=0.015, help="Min text area ratio")
    ap.add_argument("--nchar-thr", type=int, default=30, help="Min OCR character count: default=30")
    ap.add_argument("--band-thr", type=float, default=0.008, help="Min top/bottom band text area ratio")
    ap.add_argument("--report-every", type=int, default=100, help="Print progress every N images")
    ap.add_argument("--out", default="list.textheavy", help="Output file path. Default=list.textheavy")
    ap.add_argument("--append", action="store_true", help="Append to output file instead of overwrite")
    args = ap.parse_args()

    if not args.topdir.is_dir():
        print("Error: topdir is not a directory", file=sys.stderr)
        sys.exit(2)

    # Check tesseract is available
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        print("Error: 'tesseract' binary not found. Install with: sudo apt install tesseract-ocr", file=sys.stderr)
        sys.exit(3)

    # Collect images
    paths = [p for p in args.topdir.rglob("*") if p.suffix.lower() in EXTS]
    total = len(paths)
    if total == 0:
        print("No images found.")
        return

    # Graceful Ctrl-C
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    mode = "a" if args.append else "w"
    processed = 0
    flagged = 0
    start = datetime.now()

    print(f"[{start.strftime('%H:%M:%S')}] Scanning {total} images with {args.workers} workers...")
    with open(args.out, mode, encoding="utf-8") as fout:
        jobs = [(p, args.min_conf, args.cov_thr, args.nchar_thr, args.band_thr) for p in paths]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_worker, j) for j in jobs]
            for fut in as_completed(futures):
                res = fut.result()
                processed += 1
                if res:
                    fout.write(res + "\n")
                    flagged += 1
                if processed % args.report_every == 0:
                    # Print compact progress line
                    print(f"Processed {processed}/{total} | flagged {flagged}", flush=True)

    end = datetime.now()
    dur = (end - start).total_seconds()
    print(f"[{end.strftime('%H:%M:%S')}] Done. Processed {processed}, flagged {flagged}. "
          f"Results -> {args.out}. Elapsed {dur:.1f}s.")

if __name__ == "__main__":
    main()
