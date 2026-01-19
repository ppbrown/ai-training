#!/usr/bin/env python3
""" Find all .txt files. Presume they have tags in them.
Go through and remove all duplicate tags
"""

from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

for p in root.rglob("*.txt"):
    tags = [t.strip() for t in p.read_text(encoding="utf-8").split(",") if t.strip()]
    seen = set()
    out = []
    changed = False
    for t in tags:
        if t in seen:
            changed = True
            continue
        seen.add(t)
        out.append(t)

    if changed:
        p.write_text(", ".join(out), encoding="utf-8")
        print(f"updated: {p}")
