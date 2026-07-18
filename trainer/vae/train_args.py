#!/usr/bin/env python3

import jsonargparse
import os
import sys


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
    ap.add_argument("--freeze_all_channels", action="store_true",
                    help="Freeze channel-aligned weights (encoder.conv_out, quant_conv,"
                         " post_quant_conv, decoder.conv_in) and train only the backbone."
                         " Use after merging specialist channel VAEs back together."
                         " Mutually exclusive with --freeze_backbone and --freeze_channels.")
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze all backbone weights and train only the channel-aligned layers"
                         " (encoder.conv_out, quant_conv, post_quant_conv, decoder.conv_in).")
    ap.add_argument("--freeze_channels", type=str, default=None,
                    help="Freeze channel-aligned weights for a specific range of latent channels."
                         " Format: START-END (inclusive), e.g. '0-7'."
                         " Zeros gradients for those channel rows/cols in encoder.conv_out,"
                         " quant_conv, post_quant_conv, decoder.conv_in."
                         " Cannot be combined with --freeze_all_channels.")

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
    ap.add_argument("--warmup_steps", type=int, default=0,
                    help="Linearly warm up LR from 0 to --lr over this many optimizer steps. 0 disables warmup.")
    ap.add_argument("--max_grad_norm", type=float, default=1.0,
                    help="Clip VAE gradient norm to this before each optimizer step."
                         " 0 disables clipping.")
    ap.add_argument("--use_ema", action="store_true",
                    help="Track power-function EMA accumulators (EDM2-style) of the"
                         " trainable weights. At each checkpoint, a full EMA model is"
                         " saved under <ckpt>/ema/sr<SIGMA_REL> for every value in"
                         " --ema_sigma_rels, each with its own vae_sample.webp.")
    ap.add_argument("--ema_sigma_rels", type=float, nargs="+", default=[0.05, 0.10],
                    help="sigma_rel of the tracked power-EMA accumulators."
                         " The defaults (0.05, 0.10) span most useful lengths."
                         " Only used with --use_ema.")
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
    ap.add_argument("--lpips_rawvgg", action="store_true",
                    help="Use LPIPS module but set lpips=False. Which oddly, gives you 'raw VGG'")

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

    ap.add_argument(
        "--freq_lowbound", type=float, default=0.0,
        help="Fixed Gaussian sigma for the coarse (low-freq) edge of the training band. "
             "Content coarser than this sigma is excluded. 0 = no lower bound (include all coarse content). "
             "Typical range: 2.0-20.0 px. Must be < --freq_lowbound if both are set.",
    )
    ap.add_argument(
        "--freq_highbound", type=float, default=0.0,
        help="Fixed Gaussian sigma for higher edge of the training band. "
             "Content higher than this sigma is excluded. 0 = no high bound (include all high detail). "
             "Typical range: 0.5-5.0 px. Use with --freq_lowbound to isolate a mid band.",
    )

    ap.add_argument("--crop_mining_weight", type=float, default=0.0)
    ap.add_argument("--crop_size",          type=int,   default=128)
    ap.add_argument("--crop_patch_size",    type=int,   default=32)
    ap.add_argument("--num_crops",          type=int,   default=3)
    ap.add_argument("--crop_temperature",   type=float, default=2.0)
    ap.add_argument("--crop_shapeonly",     action="store_true")

    ap.add_argument("--disc_weight",  type=float, default=0.0,
                    help="Enable 'Discriminator' (aka GAN based) loss calc."
                         " By default this multiplies an adaptive scale that matches"
                         " the GAN gradient to the recon gradient at decoder.conv_out"
                         " (taming-transformers style); 0.5 is the usual LDM value.")
    ap.add_argument("--disc_no_adaptive", action="store_true",
                    help="Use --disc_weight as a fixed scale on the generator loss"
                         " instead of multiplying the adaptive gradient-ratio scale.")
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

    ap.add_argument(
        "--continue_steps", type=int, default=0,
        help="Resume training for this many additional steps. Exclusive:"
             " must be the only argument, e.g. 'train_vae.py --continue_steps"
             " 5000'. Must be run from the prior run's top-level directory"
             " (the one containing ./config.json and ./final/, i.e. its"
             " --output_dir) -- --model, --output_dir, and every other"
             " setting are read unchanged from ./config.json. --train_steps"
             " from the saved config is ignored -- the new total is (step"
             " already reached) + --continue_steps.",
    )

    argv = sys.argv[1:]
    if argv and argv[0] == "--continue_steps":
        if len(argv) != 2:
            raise SystemExit("--continue_steps must be the only argument: --continue_steps N")
        if not os.path.exists("config.json"):
            raise SystemExit(
                "--continue_steps must be run from the prior run's top-level"
                " directory (no ./config.json found in the current directory)"
            )
        argv = ["--config", "config.json", "--output_dir", ".", "--model", "final",
                "--continue_steps", argv[1]]

    args = ap.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "config.json")
    ap.save(args, config_path, format="json_indented", overwrite=True)
    print(f"Config saved to: {config_path}")
    return args

if __name__ == "__main__":
    # This is to test usage message
    parseargs()
