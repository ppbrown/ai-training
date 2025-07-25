#!/bin/env python


"""
Script to convert a disorganized set of images/etc, such as those
downloaded by dataset2img, into a more deterministic hex based naming format.
This will have the side effect of deduplication. Very useful for a large
dataset2img set.

Give this script the path to the top of a new tree.
Pass in a list of img files on stdin, and make sure to have a json file matching each img file,
that contains an md5 checksum of the image.
(You can use the "jsonupdatemd5.py" util if needed)

It will then create a symlink from an appropriate cache dir under the new tree
to the original source image.
It will also link any matching .txt file

Example:
A jpg file with chksum ac53454543534  will be linked from
 DESTDIR/ac/ac5345454543534.jpg

IMPORTANT:
The easy way to call this is:

    find /path/to/topsrc -name '*jpg' | prog /new/dir/top

However, if you want to use relative symlinks, you have to get
a bit fancy and do something like

    mkdir /new/dir/top/tmpdir && cd /new/dir/top/tmpdir
    find ../../topsrc -name '*jpg' | prog /new/dir/top
"""

import os
import sys
import hashlib
import glob
import shutil
import subprocess
import json

# Allow setting of destdir by arg, or env var
DESTDIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DESTDIR", "")

def ensure_hash_dirs(destdir):
    for a in "0123456789abcdef":
        for b in "0123456789abcdef":
            path = os.path.join(destdir, f"{a}{b}")
            os.makedirs(path, exist_ok=True)

def get_md5_from_json(jsonfile):
    try:
        with open(jsonfile, 'r') as f:
            data = json.load(f)
            return data.get("md5")
    except (json.JSONDecodeError, FileNotFoundError):
        return None

def get_md5_from_file(filename):
    h = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def symlink_force(src, dst):
    try:
        os.symlink(os.path.abspath(src), dst)
    except FileExistsError:
        os.remove(dst)
        os.symlink(os.path.abspath(src), dst)

def main():
    if not DESTDIR or not os.path.isdir(DESTDIR):
        print(f"ERROR: DESTDIR {DESTDIR} missing")
        sys.exit(1)
    ensure_hash_dirs(DESTDIR)
    # Accept files from stdin, or use glob
    files = [line.strip() for line in sys.stdin if line.strip()]
    for imgfile in files:
        basename = os.path.basename(imgfile)
        ext = os.path.splitext(basename)[1]
        longbase = os.path.splitext(imgfile)[0]
        md5 = None
        jsonfile = longbase + ".json"
        if os.path.isfile(jsonfile):
            md5 = get_md5_from_json(jsonfile)
        if not md5 or md5 == "None":
            md5 = get_md5_from_file(imgfile)
        if not md5:
            print(f"ERROR: cant find md5 for {imgfile}")
            sys.exit(1)
        hashcode = md5[:2]
        target_img = os.path.join(DESTDIR, hashcode, f"{md5}{ext}")
        target_json = os.path.join(DESTDIR, hashcode, f"{md5}.json")
        symlink_force(imgfile, target_img)
        if os.path.isfile(jsonfile):
            symlink_force(jsonfile, target_json)
        txtfile = longbase + ".txt"
        if os.path.isfile(txtfile):
            target_txt = os.path.join(DESTDIR, hashcode, f"{md5}.txt")
            symlink_force(txtfile, target_txt)

if __name__ == "__main__":
    main()
