#!/usr/bin/env python3
"""
 Meant to be used as an alternative to tensorboard tracking of loss
 during VAE training.
 When you have enabled saving of a test image with each checkpoint,
 use this tool to compare original image to each checkpoint's sample.

Usage: python compare_loss.py <original> <sample1> [sample2 ...]

Can also be imported and used as a subroutine from another program:

    from compare_loss import compute_losses, load_vgg

    vgg_loss = load_vgg()  # load once, reuse across many calls
    results = compute_losses(original_path, sample_paths, vgg_loss)
    for r in results:
        print(r["path"], r["l1"], r["rawvgg"], r["edge"], r["lap"])

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
PADDING = 40

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


def load_vgg(device: str = DEVICE) -> lpips.LPIPS:
    """Load the VGG-based LPIPS model. Call once and pass to compute_losses
    when making repeated calls, to avoid reloading the network each time."""
    print("Loading VGG (rawvgg mode)...", flush=True)
    return lpips.LPIPS(net="vgg", lpips=False).to(device)


def compute_losses(original_path: str, sample_paths: list[str],
                    vgg_loss: lpips.LPIPS | None = None) -> list[dict]:
    """Compare an original image to one or more sample images.

    Returns a list of dicts (one per sample path) with keys:
    path, l1, rawvgg, edge, lap.
    """
    if vgg_loss is None:
        vgg_loss = load_vgg()

    orig_01, orig_11 = load(original_path)
    h, w = orig_01.shape[2], orig_01.shape[3]

    results = []
    for path in sample_paths:
        s01, s11 = load(path)

        if s01.shape[2:] != orig_01.shape[2:]:
            raise ValueError(f"Size mismatch: original {(h,w)}, {path} {tuple(s01.shape[2:])}")

        results.append({
            "path":   path,
            "l1":     F.l1_loss(s01, orig_01).item(),
            "rawvgg": vgg_loss(s11, orig_11).item(),
            "edge":   edge_loss(s01, orig_01),
            "lap":    laplacian_loss(s01, orig_01),
        })

    return results


def print_results(results: list[dict]) -> None:
    header = f"{'image':<{PADDING}} {'l1':>8} {'rawvgg':>10} {'edge':>8} {'lap':>8}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        print(f"{r['path']:<{PADDING}} {r['l1']:>8.4f} {r['rawvgg']:>10.4f} "
              f"{r['edge']:>8.4f} {r['lap']:>8.4f}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_loss.py <original> <sample1> [sample2 ...]")
        sys.exit(1)

    print("Using", DEVICE)
    original_path = sys.argv[1]
    sample_paths  = sys.argv[2:]

    results = compute_losses(original_path, sample_paths)
    print_results(results)


if __name__ == "__main__":
    main()
