#!/usr/bin/env python3

import jsonargparse, json
import os

def parseargs():
    """
    Define this early for faster usage response 
    Note that unlike original argparse, jsonargparse automatically 
    shows default values
    """
    ap = jsonargparse.ArgumentParser(description="Fine-tune SDXL type VAE.")
    ap.add_argument("--config", action="config")
    ap.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Repeatable. Format: /path/to/dataset:WIDTHxHEIGHT (e.g., /data/laion:1024x1024)",
    )
    ap.add_argument("--output_dir", required=True, help="Where to save the fine-tuned VAE.")
    ap.add_argument("--train_steps", type=int, required=True, help="Number of optimizer steps.")
    ap.add_argument("--skip_steps", type=int, default=0,
                    help="For scheduler purposes, skip this many steps."
                    " Tiled steps are not counted so factor x5 if using --hires_tiling.")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    ap.add_argument("--hires_tiling", action="store_true",
                    help="Presuming high res dataset, add additional 4x highres tile processing."
                    " Note1: This then counts 5 steps per image instead of 1."
                    " Note2: Hardcoded to rescale to 1024x1024 fullsize, then make 512x512 tiles.")
    ap.add_argument("--pixel_shift", type=int, default=0,
               help="Expand each image to a batch of pixel shifted crops. N yields (N+1)^2 imgs. Try 1 or 2.")

    ap.add_argument("--bf16", action="store_true", help="Allow mixed precision training")
    ap.add_argument("--allow_tf32", action="store_true",
                   help="Speed optimization. (Possibly bad at extremely low LR?)")

    ap.add_argument("--batch_size", type=int, default=1, help="Batch size per step (per dataset, per GPU).")
    ap.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    ap.add_argument("--save_every", type=int, default=2000, help="Save checkpoint every N steps.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")

    ap.add_argument(
        "--optimizer",
        choices=["adamw", "sgd"],
        default="adamw",
        help="Optimizer for VAE training.",
    )
    ap.add_argument(
        "--sgd_momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer.",
    )
    ap.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay. For late-stage VAE polish you may want 0.",
    )
    ap.add_argument("--kl_weight", type=float, default=1e-6)
    ap.add_argument("--edge_l1_weight", type=float, default=0.1)
    ap.add_argument("--l1_weight", type=float, default=0.8)
    ap.add_argument( "--fft_weight", type=float, default=0.0,
        help="Match FFT magnitude (texture). Try 0.01-0.10 (0.02 is typical).",
    )
    ap.add_argument( "--fft_phase_weight", type=float, default=0.0,
        help="FFT alignment loss: fixes tiny detail placement;"
        " combine with --fft_weight to also match texture strength."
        " Try 0.03-0.08.",
    )
    ap.add_argument("--lpips_weight", type=float, default=0.0,
                    help="high quality but high cost. Range 0.2 - 0.05")
    ap.add_argument("--lpips_shapeonly", action="store_true",
                    help="Compute LPIPS on shape only, ignoring color matching")
    ap.add_argument(
        "--hf_luma_only", action="store_true",
        help="Compute edge/laplacian losses on luma (Y) only"
            " instead of RGB to avoid chroma-driven 'sharpness' cheats.",
    )
    ap.add_argument(
        "--hf_energy_weight", type=float, default=0.0,
        help="Weight for matching high-frequency energy on luma (helps prevent blurred recon).",
    )
    ap.add_argument(
        "--laplacian_weight", type=float, default=0.0,
        help="Weight for Laplacian loss term."
            " Range 0.005-0.03 (0.01 is common)",
    )
    ap.add_argument(
        "--grad_energy_weight", type=float, default=0.0,
        help="Match gradient energy (dx^2, dy^2). Helps reduce over-smoothing.",
    )

    ap.add_argument("--crop_mining_weight", type=float, default=0.0)
    ap.add_argument("--crop_size",          type=int,   default=128)
    ap.add_argument("--crop_patch_size",    type=int,   default=32)
    ap.add_argument("--num_crops",          type=int,   default=3)
    ap.add_argument("--crop_temperature",   type=float, default=2.0)
    ap.add_argument("--crop_shapeonly",     action="store_true")

    ap.add_argument("--disc_weight",  type=float, default=0.0,
                    help="Enable 'Discriminator' (aka GAN based) loss calc")
    ap.add_argument("--disc_start",   type=int,   default=50000,
                    help="What step it should kick in at")
    ap.add_argument("--disc_lr",      type=float, default=2e-4,
                    help="Default lr for GAN is 2e-4")
    ap.add_argument("--disc_layers",  type=int,   default=3)


    ap.add_argument(
        "--sample_img",
        type=str,
        default=None,
        help="Optional. If set, encode/decode this image through the VAE"
            " at each save and write vae_sample.webp in the save dir.",
    )

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Config saved to: {config_path}")
    return args

if __name__ == "__main__":
    # This is to test usage message
    parseargs()
