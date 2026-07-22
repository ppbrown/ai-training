#!/usr/bin/env python3
"""
gutenberg_glyph_dataset.py

Build a VAE glyph-training dataset from a Project Gutenberg book:

    1. Download plaintext from Gutenberg
    2. Strip PG header/footer boilerplate
    3. Normalize to ASCII-only (drops/transliterates unicode -- no CJK,
       accented letters, curly quotes, em-dashes, etc.)
    4. Chunk into paragraph-sized blocks
    5. Render each chunk to a PNG via Typst
       (based on value of --font-size)

Usage:
    pip install typst --break-system-packages
    python3 this.py --book pride_and_prejudice --out ./glyph_ds
    python3 this.py --gutenberg-id 84 --out ./glyph_ds --augment
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

try:
    import typst
except ImportError:
    typst = None

GUTENBERG_BOOKS = {
    "pride_and_prejudice": 1342,
    "frankenstein": 84,
    "moby_dick": 2701,
    "sherlock_holmes": 1661,
    "dracula": 345,
    "alice_in_wonderland": 11,
}

GUTENBERG_URL_TEMPLATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]

START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)

# Curly punctuation etc -> ASCII equivalents, applied before the hard
# unicode strip so meaning survives instead of just vanishing.
ASCII_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": ",",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "--", "\u2015": "--",
    "\u2026": "...", "\u00a0": " ", "\u2032": "'", "\u2033": '"',
}

TYPST_ESCAPE_RE = re.compile(r'([\\#*_$`@<>\[\]])')

AUGMENT_FONT_CHOICES = [
    "Libertinus Serif",
    "New Computer Modern",
    "DejaVu Sans",
    "DejaVu Serif",
    "Linux Libertine",
]

# Color bands for randomized fg/bg (--bw disables this). Each channel is
# drawn independently from whichever band, so hue varies while staying
# clearly on the dark or light side -- the gap between the bands (86-169)
# is the excluded midtone range.
DARK_COLOR_RANGE = (0, 85)
LIGHT_COLOR_RANGE = (170, 255)

BASE_PPI = 144  # arbitrary internal constant for px->pt conversion; cancels
                # out mathematically and has no effect on final pixel output
                # (see render_with_typst). Not exposed as a CLI knob.

SUPERSAMPLE = 2.0  # fixed internal anti-aliasing factor: render at 2x the
                   # target resolution, then downsample. Not user-facing --
                   # --font-size is the knob that actually matters for
                   # controlling glyph size/density on the page.


def fetch_gutenberg_text(gutenberg_id: int) -> str:
    last_err = None
    for template in GUTENBERG_URL_TEMPLATES:
        url = template.format(id=gutenberg_id)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not fetch Gutenberg id={gutenberg_id}: {last_err}")


def strip_boilerplate(text: str) -> str:
    start_match = START_RE.search(text)
    if start_match:
        text = text[start_match.end():]
    end_match = END_RE.search(text)
    if end_match:
        text = text[:end_match.start()]
    return text.strip()


def to_ascii_english(text: str) -> str:
    for uni, ascii_eq in ASCII_MAP.items():
        text = text.replace(uni, ascii_eq)
    # Transliterate accented latin letters to their base form (e.g. e -> e)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    # Keep printable ASCII + newlines only.
    text = "".join(c for c in text if c == "\n" or 32 <= ord(c) <= 126)
    return text


def split_paragraphs(text: str) -> list[str]:
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = []
    for p in raw_paragraphs:
        p = " ".join(p.split())  # collapse internal whitespace/newlines
        if p:
            paragraphs.append(p)
    return paragraphs


def chunk_paragraphs(paragraphs: list[str], min_chars: int, max_chars: int) -> list[str]:
    """Merge short paragraphs up to min_chars; hard-split long ones at
    sentence boundaries (falling back to word boundaries) if they exceed
    max_chars."""
    chunks: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf:
            chunks.append(buf.strip())
            buf = ""

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            chunks.extend(_split_long(para, max_chars))
            continue
        if buf and len(buf) + 1 + len(para) > max_chars:
            flush()
        buf = f"{buf} {para}".strip()
        if len(buf) >= min_chars:
            flush()
    flush()
    return [c for c in chunks if c]


def _split_long(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?]) ", text)
    out: list[str] = []
    buf = ""
    for s in sentences:
        if len(s) > max_chars:
            # single sentence too long -- hard-wrap on words
            words = s.split(" ")
            w_buf = ""
            for w in words:
                if len(w_buf) + 1 + len(w) > max_chars:
                    out.append(w_buf.strip())
                    w_buf = w
                else:
                    w_buf = f"{w_buf} {w}".strip()
            if w_buf:
                out.append(w_buf.strip())
            continue
        if buf and len(buf) + 1 + len(s) > max_chars:
            out.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        out.append(buf.strip())
    return out


def typst_escape(text: str) -> str:
    return TYPST_ESCAPE_RE.sub(r"\\\1", text)


def _random_side_color(side: str) -> str:
    lo, hi = DARK_COLOR_RANGE if side == "dark" else LIGHT_COLOR_RANGE
    r, g, b = (random.randint(lo, hi) for _ in range(3))
    return f"#{r:02x}{g:02x}{b:02x}"


def random_fg_bg_colors() -> tuple[str, str]:
    """Pick a fg color from the dark or light band, then a bg color from
    the opposite band, so contrast is always high but hue still varies."""
    fg_side = random.choice(("dark", "light"))
    bg_side = "light" if fg_side == "dark" else "dark"
    return _random_side_color(fg_side), _random_side_color(bg_side)


def px_to_pt(px: float, ppi: int) -> float:
    """Typst page/margin sizes are in points (1pt = 1/72in); PNG export
    rasterizes at --ppi. To land on an exact pixel size regardless of ppi,
    the pt size must be derived from it: pt = px * 72 / ppi."""
    return px * 72.0 / ppi


# ppi for the height-measurement pass below. Only page geometry (points),
# not pixel detail, is being read back, so a small raster is enough and
# keeps the extra compile cheap regardless of the real target resolution.
MEASURE_PPI = 36


def measure_content_height_px(text: str, font: str, font_size_px: float,
                               width_px: float, margin_lr_px: float) -> float:
    """Ask Typst itself how tall this text renders for this font/size/width,
    via page height: auto, instead of guessing from a fixed average-char-width
    ratio (which undershoots wider-set fonts like DejaVu Serif and let text
    overflow onto a second page). Same font/size/justify/width as the real
    render, so line breaks match and the reported height is exact rather than
    estimated."""
    width_pt = px_to_pt(width_px, BASE_PPI)
    margin_lr_pt = px_to_pt(margin_lr_px, BASE_PPI)
    font_size_pt = px_to_pt(font_size_px, BASE_PPI)
    escaped = typst_escape(text)
    margin = f"(top: 0pt, bottom: 0pt, left: {margin_lr_pt}pt, right: {margin_lr_pt}pt)"
    source = (
        f'#set page(width: {width_pt}pt, height: auto, margin: {margin})\n'
        f'#set text(font: "{font}", size: {font_size_pt}pt)\n'
        f'#set par(justify: true)\n'
        f"{escaped}\n"
    )
    raw_bytes = typst.compile(source.encode("utf-8"), format="png", ppi=float(MEASURE_PPI))
    if isinstance(raw_bytes, list):
        raw_bytes = raw_bytes[0]
    from PIL import Image
    import io
    height_px_at_measure_ppi = Image.open(io.BytesIO(raw_bytes)).height
    height_pt = height_px_at_measure_ppi * 72.0 / MEASURE_PPI
    return height_pt * BASE_PPI / 72.0


def build_typst_source(text: str, font: str, font_size_px: float,
                        width_px: float, height_px: float,
                        margin_top_px: float, margin_bottom_px: float,
                        margin_left_px: float, margin_right_px: float,
                        fg_color: str = "#000000", bg_color: str = "#ffffff") -> str:
    width_pt = px_to_pt(width_px, BASE_PPI)
    height_pt = px_to_pt(height_px, BASE_PPI)
    mt_pt = px_to_pt(margin_top_px, BASE_PPI)
    mb_pt = px_to_pt(margin_bottom_px, BASE_PPI)
    ml_pt = px_to_pt(margin_left_px, BASE_PPI)
    mr_pt = px_to_pt(margin_right_px, BASE_PPI)
    font_size_pt = px_to_pt(font_size_px, BASE_PPI)
    escaped = typst_escape(text)
    margin = f"(top: {mt_pt}pt, bottom: {mb_pt}pt, left: {ml_pt}pt, right: {mr_pt}pt)"
    return (
        f'#set page(width: {width_pt}pt, height: {height_pt}pt, margin: {margin}, fill: rgb("{bg_color}"))\n'
        f'#set text(font: "{font}", size: {font_size_pt}pt, fill: rgb("{fg_color}"))\n'
        f'#set par(justify: true)\n'
        f"{escaped}\n"
    )


def render_with_typst(source: str, png_path: Path, target_width_px: int, target_height_px: int) -> None:
    """Render at SUPERSAMPLE x the target resolution, then downsample with a
    quality filter for anti-aliased edges. (ppi has no effect on final pixel
    size in this script -- page geometry is derived from px, so it cancels
    out -- this fixed internal supersample is what actually smooths edges.)"""
    render_ppi = BASE_PPI * SUPERSAMPLE
    try:
        raw_bytes = typst.compile(source.encode("utf-8"), format="png", ppi=float(render_ppi))
        if isinstance(raw_bytes, list):
            # typst.compile() returns one bytes object per page when the
            # content overflows onto multiple pages -- treat as the same
            # overflow case the caller already retries on.
            raise RuntimeError(f"content overflowed onto multiple pages ({len(raw_bytes)})")
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        im = im.resize((target_width_px, target_height_px), Image.LANCZOS)
        im.save(png_path)
    except Exception as e:  # noqa: BLE001 -- typst.TypstError etc.
        raise RuntimeError(f"typst failed on {png_path.stem}: {e}") from e


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    book_group = ap.add_mutually_exclusive_group(required=True)
    book_group.add_argument("--book", choices=sorted(GUTENBERG_BOOKS), help="named book from built-in list")
    book_group.add_argument("--gutenberg-id", type=int, help="arbitrary Gutenberg ebook id")
    book_group.add_argument("--input-file", type=Path, help="use a local plaintext file instead of downloading")

    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--min-chars", type=int, default=280, help="min chars per rendered chunk")
    ap.add_argument("--max-chars", type=int, default=600, help="max chars per rendered chunk")
    ap.add_argument("--max-images", type=int, default=0, help="cap number of images (0 = no cap)")

    ap.add_argument("--page-width", type=float, default=512.0, help="output image width in PIXELS")
    ap.add_argument("--page-height", type=float, default=512.0, help="output image height in PIXELS")
    ap.add_argument("--margin", type=float, default=24.0, help="base page margin in PIXELS (all sides)")
    ap.add_argument("--margin-stretch", type=float, default=80.0,
                     help="extra left/right margin (px) applied to the shortest chunks, tapering to 0 "
                          "at --max-chars, so short paragraphs sit in a narrower text column instead of "
                          "always using the same footprint as long ones")
    ap.add_argument("--font", type=str, default="Libertinus Serif")
    ap.add_argument("--font-size", type=float, default=18.0,
                     help="font size in the same pixel-scale convention as --page-width (default 18)")
    ap.add_argument("--no-vary-position", action="store_true",
                     help="disable randomized vertical placement (default: text lands at a random "
                          "vertical offset on the page instead of always starting at the top)")

    ap.add_argument("--bw", action="store_true",
                     help="disable color randomization: always render black text on a white "
                          "background (the previous default). Without this flag, each chunk gets "
                          "a random fg/bg pair -- fg is drawn from the dark or light side, bg from "
                          "the opposite side, so contrast stays high but hue varies")

    ap.add_argument("--augment", action="store_true",
                     help="randomize font/font-size/margin per-chunk from a fixed pool, for training diversity")
    ap.add_argument("--seed", type=int, default=None,
                     help="seed the RNG for reproducible colors/placement/augmentation across runs. "
                          "Default: unseeded, so each run is genuinely random.")
    ap.add_argument("--keep-typ", action="store_true", help="keep intermediate .typ source files")

    args = ap.parse_args()

    if typst is None:
        print("ERROR: the `typst` python package is not installed.\n"
              "Install with: pip install typst --break-system-packages",
              file=sys.stderr)
        return 1

    if args.seed is not None:
        random.seed(args.seed)

    if args.input_file:
        raw_text = args.input_file.read_text(encoding="utf-8", errors="replace")
        book_label = args.input_file.stem
    else:
        gid = args.gutenberg_id if args.gutenberg_id else GUTENBERG_BOOKS[args.book]
        cache_path = Path(f"gutenberg.{gid}")
        if cache_path.exists():
            print(f"Using cached file {cache_path} ...")
            raw_text = cache_path.read_text(encoding="utf-8", errors="replace")
        else:
            print(f"Fetching Gutenberg id={gid} ...")
            raw_text = fetch_gutenberg_text(gid)
            cache_path.write_text(raw_text, encoding="utf-8")
            print("Wrote", cache_path)
        book_label = args.book or f"gutenberg_{gid}"

    text = strip_boilerplate(raw_text)
    text = to_ascii_english(text)
    paragraphs = split_paragraphs(text)
    chunks = chunk_paragraphs(paragraphs, args.min_chars, args.max_chars)

    if args.max_images:
        chunks = chunks[:args.max_images]

    print(f"{len(paragraphs)} paragraphs -> {len(chunks)} chunks")

    out_dir = args.out
    img_dir = out_dir / "images"
    typ_dir = out_dir / "typ"
    img_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_typ:
        typ_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    n_ok, n_fail = 0, 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for i, chunk in enumerate(chunks):
            stem = f"{book_label}_{i:05d}"
            png_path = img_dir / f"{stem}.png"

            font = args.font
            font_size = args.font_size
            if args.augment:
                font = random.choice(AUGMENT_FONT_CHOICES)
                font_size = round(random.uniform(11.0, 18.0), 1)

            # Shorter chunks get wider left/right margins (narrower column)
            # so they don't just sit at the same footprint as long chunks.
            length_frac = min(1.0, len(chunk) / args.max_chars)
            stretch = args.margin_stretch * (1.0 - length_frac)
            margin_lr = args.margin + stretch
            if args.augment:
                margin_lr += random.uniform(-8.0, 8.0)
            margin_lr = max(4.0, margin_lr)

            # Randomize vertical placement: measure how tall the rendered
            # text actually is (via Typst's own layout, see
            # measure_content_height_px), then pick a random top margin
            # within whatever slack remains so text doesn't always start at
            # the top of the page. Bottom margin stays at the base value;
            # Typst just leaves blank space below short text, it doesn't
            # need to match.
            try:
                est_height = measure_content_height_px(chunk, font, font_size, args.page_width, margin_lr)
            except Exception as e:  # noqa: BLE001 -- typst.TypstError etc.
                print(f"[warn] height measurement failed for chunk {i}, using top-of-page layout: {e}",
                      file=sys.stderr)
                est_height = args.page_height  # forces the top-of-page fallback below
            max_top_margin = args.page_height - args.margin - est_height
            if args.no_vary_position or max_top_margin <= args.margin:
                margin_top = args.margin
            else:
                margin_top = random.uniform(args.margin, max_top_margin)
            margin_bottom = args.margin

            if args.bw:
                fg_color, bg_color = "#000000", "#ffffff"
            else:
                fg_color, bg_color = random_fg_bg_colors()

            src = build_typst_source(chunk, font, font_size, args.page_width, args.page_height,
                                      margin_top, margin_bottom, margin_lr, margin_lr,
                                      fg_color, bg_color)

            if args.keep_typ:
                (typ_dir / f"{stem}.typ").write_text(src, encoding="utf-8")

            try:
                render_with_typst(src, png_path, int(args.page_width), int(args.page_height))
            except RuntimeError as e:
                if "multiple pages" in str(e):
                    # height estimate was wrong for this chunk -- retry once
                    # with a safe top-of-page layout instead of dropping it.
                    src = build_typst_source(chunk, font, font_size, args.page_width, args.page_height,
                                              args.margin, args.margin, args.margin, args.margin,
                                              fg_color, bg_color)
                    margin_top, margin_lr = args.margin, args.margin
                    try:
                        render_with_typst(src, png_path, int(args.page_width), int(args.page_height))
                    except RuntimeError as e2:
                        print(f"[skip] {e2}", file=sys.stderr)
                        n_fail += 1
                        continue
                else:
                    print(f"[skip] {e}", file=sys.stderr)
                    n_fail += 1
                    continue

            manifest.write(json.dumps({
                "file": str(png_path.relative_to(out_dir)),
                "text": chunk,
                "book": book_label,
                "font": font,
                "font_size": font_size,
                "fg_color": fg_color,
                "bg_color": bg_color,
                "margin_top_px": round(margin_top, 1),
                "margin_lr_px": round(margin_lr, 1),
                "chars": len(chunk),
            }) + "\n")
            n_ok += 1

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(chunks)} rendered")

    print(f"Done: {n_ok} images written to {img_dir}, {n_fail} failures. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
