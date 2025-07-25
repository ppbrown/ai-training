#!/bin/env python

import torch
import glob
import os,sys
from tqdm import tqdm
from safetensors.torch import load_file

# Set your path and target std
cache_root = sys.argv[1]
target_std = 0.2   # Typical for SD1.5/CLIP

# Gather all cache files (recursively)
ext = ".txt_t5cache"

print(f"Scanning {cache_root} for cache files {ext}...")
cache_files = glob.glob(os.path.join(cache_root, f"**/*{ext}"), recursive=True)
if cache_files == []:
    print("No cache files found")
    exit(0)

all_embs = []
perfile_stds = []

for path in tqdm(cache_files):
    cache = load_file(path)
    emb = cache["emb"].flatten()
    perfile_stds.append(emb.std().item())
    all_embs.append(emb)

# Stack to [N, 768]
all_embs = torch.stack(all_embs)
global_std = all_embs.std().item()
mean_std = sum(perfile_stds) / len(perfile_stds)

print(f"Number of files: {len(cache_files)}")
print("(Expected std range for SD1.5 is 0.18-0.22)")
print(f"Global std across all embeddings: {global_std:.6f}")
print(f"Mean per-file std: {mean_std:.6f}")
print(f"Typical SD/CLIP/SDXL embedding std: {target_std}")

scaling_factor = target_std / (global_std + 1e-8)
print(f"\nOptimal scaling factor to apply: {scaling_factor:.4f}")

# Optionally, print a histogram of stds
try:
    import matplotlib.pyplot as plt
    plt.hist(perfile_stds, bins=40)
    plt.title("Per-file std distribution")
    plt.xlabel("std")
    plt.ylabel("Count")
    plt.show()
except ImportError:
    pass  # matplotlib optional
