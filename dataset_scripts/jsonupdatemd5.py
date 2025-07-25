#!/bin/env python

"""
Take jpg filenames from stdin.
read in image. find the md5 hash.
open matching .json file
update data.
write out json
"""


import sys
import json
from PIL import Image

import hashlib

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def update_json_with_md5(jpg_file):
    try:
        # Open the image and get its hash
        sum = md5(jpg_file)

        # Determine the corresponding JSON filename
        json_file = jpg_file.rsplit('.', 1)[0] + '.json'

        # Open the JSON file and update/add md5 checksum
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}  # If the file doesn't exist, start with an empty dictionary

        data['md5'] = sum

        # Write back the updated JSON
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Updated {json_file} with md5={sum}")

    except Exception as e:
        print(f"Error processing {jpg_file}: {e}")

if __name__ == "__main__":
    for line in sys.stdin:
        jpg_file = line.strip()
        if jpg_file:
            update_json_with_md5(jpg_file)

