#!/usr/bin/env python3
"""
 Meant to be used as an alternative to tensorboard tracking of loss
 during VAE training.
 When you have enabled saving of a test image with each checkpoint, 
 use this tool to compare original image to each checkpoint's sample.

Usage: python compare_losss.py <original> <sample1> [sample2 ...]

Note: lazy hardcode to use CPU not cuda so safe to use while training
is running. You really dont need cuda anyway, its reasonably fast
"""

import sys
import torch
import torch.nn.functional as F
import lpips
from PIL import Image
from torchvision import transforms

#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
print("Using",DEVICE)
PADDING = 50

to_tensor = transforms.ToTensor()  # [0,1]
to_model  = transforms.Normalize([0.5]*3, [0.5]*3)  # [-1,1] for lpips


def load(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    img = Image.open(path).convert("RGB")
    t01 = to_tensor(img).unsqueeze(0).to(DEVICE)
    t11 = to_model(t01)
    return t01, t11


def laplacian_loss(a: torch.Tensor, b: torch.Tensor) -> float:
    kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                          dtype=torch.float32, device=DEVICE)
    kernel = kernel.view(1,1,3,3).expand(3,-1,-1,-1)
    la = F.conv2d(a, kernel, padding=1, groups=3)
    lb = F.conv2d(b, kernel, padding=1, groups=3)
    return F.l1_loss(la, lb).item()


def edge_loss(a: torch.Tensor, b: torch.Tensor) -> float:
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                      dtype=torch.float32, device=DEVICE).view(1,1,3,3).expand(3,-1,-1,-1)
    ky = kx.transpose(2,3)
    def edges(t):
        ex = F.conv2d(t, kx, padding=1, groups=3)
        ey = F.conv2d(t, ky, padding=1, groups=3)
        return (ex**2 + ey**2).sqrt()
    return F.l1_loss(edges(a), edges(b)).item()


def main():
    if len(sys.argv) < 3:
        print("Usage: rate_samples.py <original> <sample1> [sample2 ...]")
        sys.exit(1)

    original_path = sys.argv[1]
    sample_paths  = sys.argv[2:]

    print(f"Loading VGG (rawvgg mode)...", flush=True)
    vgg_loss = lpips.LPIPS(net="vgg", lpips=False).to(DEVICE)

    orig_01, orig_11 = load(original_path)

    # Resize samples to original size if needed
    h, w = orig_01.shape[2], orig_01.shape[3]

    header = f"{'image':<{PADDING}} {'l1':>8} {'rawvgg':>10} {'edge':>8} {'lap':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for path in sample_paths:
        s01, s11 = load(path)

        if s01.shape[2:] != orig_01.shape[2:]:
            raise ValueError(f"Size mismatch: original {(h,w)}, {path} {tuple(s01.shape[2:])}")

        l1     = F.l1_loss(s01, orig_01).item()
        vgg    = vgg_loss(s11, orig_11).item()
        edge   = edge_loss(s01, orig_01)
        lap    = laplacian_loss(s01, orig_01)

#        name = path.split("/")[-1][:PADDING]
        name = path
        print(f"{name:<{PADDING}} {l1:>8.4f} {vgg:>10.4f} {edge:>8.4f} {lap:>8.4f}")


if __name__ == "__main__":
    main()
