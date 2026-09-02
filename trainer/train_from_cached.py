#!/usr/bin/env python

# train_from_cached.py


# --------------------------------------------------------------------------- #
# 1. CLI                                                                      #
# --------------------------------------------------------------------------- #

from train_args import parse_args
from train_multiloader import InfiniteLoader

# Put this super-early, so that usage message procs fast
args = parse_args()

from train_state import TrainState
from train_core import train_micro_batch
from train_checkpointandsave import checkpointandsave

# --------------------------------------------------------------------------- #

import os, math, signal
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedDataParallelKwargs
from diffusers import DiffusionPipeline

from torch.optim.lr_scheduler import LinearLR, SequentialLR
# from pytorch_optimizer.lr_scheduler.rex import REXScheduler - this is not compatible
from axolotl.utils.schedulers import RexLR

# diffusers optimizers dont have a min_lr arg,
# so dont use that scheduler
# from diffusers.optimization import get_scheduler
from transformers import get_scheduler

from torch.utils.tensorboard import SummaryWriter

import lion_pytorch
from optimi import Lion  # torch-optimi pip module

# Speed boost for fp32 training.
# We give up "strict fp32 math", for an alleged negligable
# accuracy difference, and 30% speed boost.
if args.allow_tf32 and args.bf16:
    print("--allow_tf32 has no effect under --bf16 (all UNet compute is already"
          " native bf16, not fp32); ignoring it")
elif args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("Enabled TF32 for speed over precision")
else:
    print("Disabled TF32 for maximum precision")

# This enables profiling on first batch,
# which then SPEEDS UP subsequent runs automaticaly
torch.backends.cudnn.benchmark = True

# --------------------------------------------------------------------------- #
# 2. Utils                                                                    #
# --------------------------------------------------------------------------- #

from train_captiondata import CaptionImgDataset

from train_utils import collate_fn, sample_img, sample_without_checkpoint


#####################################################
# Main                                              #
#####################################################

_interrupt_requested = False


def _request_interrupt(signum, frame):
    """First Ctrl-C: wait for a clean step, then save. Second Ctrl-C: abort now."""
    global _interrupt_requested
    if _interrupt_requested:
        raise KeyboardInterrupt
    _interrupt_requested = True
    print("\nCtrl-C caught; will save full training state at the next clean step."
          " Press Ctrl-C again to abort immediately without saving.")


def main():
    torch.manual_seed(args.seed)
    peak_lr = args.learning_rate

    if args.fp32:
        print("Training type: fp32")
    elif args.bf16:
        print("Training type: pure bf16 (whole pipeline, no fp32 master weights)")
    else:
        print("Training type: mixed precision (fp32 master weights, bf16 autocast)")

    resume_state = None
    resume_model_dir = None
    if args.continue_steps > 0:
        resume_model_dir = os.path.join(args.output_dir, "final")
        if not os.path.exists(os.path.join(resume_model_dir, "training_state.pt")):
            fallback_dir = os.path.join(args.output_dir, "interrupted")
            if os.path.exists(os.path.join(fallback_dir, "training_state.pt")):
                resume_model_dir = fallback_dir
                print(f"--continue_steps: no final/training_state.pt found;"
                      f" falling back to {resume_model_dir}/")
            else:
                print("ERROR: --continue_steps: no training_state.pt found under",
                      os.path.join(args.output_dir, "final"), "or", fallback_dir)
                exit(1)
        resume_state = torch.load(os.path.join(resume_model_dir, "training_state.pt"),
                                  map_location="cpu", weights_only=False)
        print(f"--continue_steps: resuming from batch {resume_state['batch_count']},"
              f" loading model from {resume_model_dir}")

    model_dtype = torch.bfloat16 if args.bf16 else torch.float32
    compute_dtype = torch.float32 if args.fp32 else torch.bfloat16  # runtime math dtype

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accum,
        # Accelerate's mixed_precision autocast wraps the model's forward
        # with ConvertOutputsToFp32, forcibly upcasting its return value to
        # fp32 regardless of the model's own parameter dtype. That's the
        # point in the default mode (fp32 master weights), but under --bf16
        # the model is already natively bf16 everywhere, and that forced
        # upcast just reintroduces an fp32 tensor into an otherwise uniform
        # bf16 pipeline (e.g. it broke sample_without_checkpoint()'s call
        # into a stock diffusers pipeline, whose denoising loop then handed
        # fp32 latents to a bf16 VAE and crashed on decode).
        mixed_precision="no" if (args.fp32 or args.bf16) else "bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )
    device = accelerator.device

    # ----- load pipeline --------------------------------------------------- #

    if args.is_custom:
        custom_pipeline = args.pretrained_model
    else:
        custom_pipeline = None

    model_path = resume_model_dir if resume_state is not None else args.pretrained_model
    print(f"Loading '{model_path}' Custom pipeline? {custom_pipeline}")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            custom_pipeline=custom_pipeline,
            torch_dtype=model_dtype
        )
    except Exception as e:
        print("Error loading model", model_path)
        print(e)
        exit(0)

    # -- unet trainable selection -- #
    if args.targetted_training:
        print("Limiting Unet training to targetted area(s)")
        pipe.unet.requires_grad_(False)
    else:
        print("Training full prior Unet")
        pipe.unet.requires_grad_(True)

    if args.force_txtcache:
        print("Forcing use of single txtcache file", args.force_txtcache)

    if args.reinit_unet:
        print("Training Unet from scratch")
        """ This does not work!!
        BASEUNET="models/sd-base/unet"
        # Note: the config from pipe.unet seems to get corrupted.
        # SO, Load a fresh one instead
        conf=UNet2DConditionModel.load_config(BASEUNET)
        new_unet=UNet2DConditionModel.from_config(conf)
        print("UNet cross_attention_dim:", new_unet.config.cross_attention_dim)
        new_unet.to(torch_dtype)
        pipe.unet=new_unet
        """
        print("Attempting to reset ALL layers of Unet")
        from train_reinit import reinit_all_unet
        reinit_all_unet(pipe.unet)
    elif args.reinit_qk:
        print("Attempting to reset Q/K layers of Unet")
        from train_reinit import reinit_qk
        reinit_qk(pipe.unet)
    elif args.reinit_crossattn:
        print("Attempting to reset Cross Attn layers of Unet")
        from train_reinit import reinit_cross_attention
        reinit_cross_attention(pipe.unet)
    elif args.reinit_crossattnout:
        print("Attempting to reset Cross Attn OUT layers of Unet")
        from train_reinit import reinit_cross_attention_outproj
        reinit_cross_attention_outproj(pipe.unet)
    elif args.reinit_attention:
        print("Attempting to reset attention layers of Unet")
        from train_reinit import reinit_all_attention
        reinit_all_attention(pipe.unet)

    if args.reinit_out:
        print("Attempting to reset Out layers of Unet")
        from train_reinit import retrain_out
        retrain_out(pipe.unet, reset=True)
    elif args.unfreeze_out:
        print("Attempting to unfreeze Out layers of Unet")
        from train_reinit import retrain_out
        retrain_out(pipe.unet, reset=False)
    if args.reinit_in:
        print("Attempting to reset In layers of Unet")
        from train_reinit import retrain_in
        retrain_in(pipe.unet, reset=True)
    elif args.unfreeze_in:
        print("Attempting to unfreeze In layers of Unet")
        from train_reinit import retrain_in
        retrain_in(pipe.unet, reset=False)
    if args.reinit_time:
        print("Attempting to reset time layers of Unet")
        from train_reinit import retrain_time
        retrain_time(pipe.unet, reset=True)
    elif args.unfreeze_time:
        print("Attempting to unfreeze time layers of Unet")
        from train_reinit import retrain_time
        retrain_time(pipe.unet, reset=False)

    if args.unfreeze_attn2:
        from train_reinit import unfreeze_attn2
        unfreeze_attn2(pipe.unet)

    if args.unfreeze_up_blocks:
        print(f"Attempting to unfreeze (({args.unfreeze_up_blocks}))"
              " upblocks of Unet")
        from train_reinit import unfreeze_up_blocks
        unfreeze_up_blocks(pipe.unet, args.unfreeze_up_blocks, reset=False)
    if args.unfreeze_down_blocks:
        print(f"Attempting to unfreeze (({args.unfreeze_down_blocks}))"
              " downblocks of Unet")
        from train_reinit import unfreeze_down_blocks
        unfreeze_down_blocks(pipe.unet, args.unfreeze_down_blocks, reset=False)
    if args.unfreeze_mid_block:
        print(f"Attempting to unfreeze mid block of Unet")
        from train_reinit import unfreeze_mid_block
        unfreeze_mid_block(pipe.unet)
    if args.unfreeze_norms:
        print(f"Attempting to unfreeze normals components of Unet")
        from train_reinit import unfreeze_norms
        unfreeze_norms(pipe.unet)

    if args.unfreeze_attention:
        print("Attempting to unfreeze attention layers of Unet")
        from train_reinit import unfreeze_all_attention
        unfreeze_all_attention(pipe.unet)
    # ------------------------------------------ #

    if args.save_start > 0:
        print("save_start limit set to", args.save_start)
    if args.cpu_offload:
        print("Enabling cpu offload")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if args.gradient_topk:
        print("Gradient sparsification(gradient_topk) set to", args.gradient_topk)

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing in UNet")
        pipe.unet.enable_gradient_checkpointing()
        if args.disc_weight > 0:
            # Only worth doing when the discriminator is on. That is the
            # one path that backprops through the decoder; everywhere else
            # the VAE runs under no_grad, where checkpointing saves nothing
            # and just costs a recompute.
            print("Enabling gradient checkpointing in VAE decoder")
            pipe.vae.enable_gradient_checkpointing()

    if args.vae_scaling_factor:
        pipe.vae.config.scaling_factor = args.vae_scaling_factor

    vae, unet = pipe.vae.eval(), pipe.unet

    noise_sched = pipe.scheduler
    print("Pipe is using noise scheduler", type(noise_sched).__name__)
    """
    It was once suggested to swap out  PNDMScheduler for DDPMScheduler,
    JUST for training.
    DO NOT DO THIS. It screwed everything up.
    """

    if hasattr(noise_sched, "add_noise"):
        print("DEBUG: add_noise present. Normal noise sched.")
    else:
        print("DEBUG: add_noise not present: presuming FlowMatch desired")

    latent_scaling = vae.config.scaling_factor
    print("VAE scaling factor is", latent_scaling)

    # Freeze VAE (and T5) so only UNet is optimised; comment-out to train all.
    for p in vae.parameters():                p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():  p.requires_grad_(False)
    if hasattr(pipe, "t5_projection"):
        print("T5 (projection layer) scaling factor is", pipe.t5_projection.config.scaling_factor)
        for p in pipe.t5_projection.parameters(): p.requires_grad_(False)

    # Gather just-trainable parameters
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if not trainable_params:
        print("ERROR: no layers selected for training")
        exit(0)
    print(
        f"Align-phase: {sum(p.numel() for p in trainable_params) / 1e6:.2f} M "
        "parameters will be updated"
    )

    if args.bf16:
        # bf16 rounds an in-place add back to the original value whenever
        # it's smaller than ~half a ULP, i.e. ~0.4% of the weight's own
        # magnitude. Measured over trainable_params specifically (matching
        # log_unet_l2_norm()'s own requires_grad filter in train_utils.py),
        # since frozen weights never get an optimizer step and don't belong
        # in this floor at all.
        n = sum(p.numel() for p in trainable_params)
        sumsq = sum(p.detach().float().pow(2).sum().item() for p in trainable_params)
        rms = (sumsq / n) ** 0.5
        lr_floor = rms * 0.004

        print(f"--bf16: trainable UNet weights have no fp32 master copy. Their "
              f"per-weight RMS is ~{rms:.3g}, so any optimizer step below "
              f"~{lr_floor:.1e} rounds away to nothing (frozen, not slow) "
              f"instead of accumulating.")
        margin = args.learning_rate / lr_floor if lr_floor > 0 else float("inf")
        if args.optimizer == "opt_lion":
            # torch-optimi's Lion auto-enables Kahan (compensated) summation
            # whenever a param's dtype is fp16/bf16 (see optimi/lion.py):
            # each step's rounding error is banked in a compensation buffer
            # and fed back in on the next step, instead of being discarded.
            # That is the actual fix for the floor above, not a workaround --
            # sub-floor updates accumulate over steps rather than vanishing.
            print(f"--optimizer opt_lion: torch-optimi auto-enables Kahan "
                  f"summation for bf16 params, which banks each step's "
                  f"rounding error and re-applies it next step instead of "
                  f"discarding it. The {lr_floor:.1e} floor above mostly "
                  f"does not apply here -- this is the safer Lion choice "
                  f"for --bf16, no extra flag needed.")
        elif args.optimizer in ("py_lion", "d_lion"):
            # Neither has any Kahan-style compensation: lion_pytorch's update
            # is a single `p.data.add_(update, alpha=-dlr)` on whatever dtype
            # p already is, and dadaptation's DAdaptLion (dadapt_lion.py) is
            # the same shape (`torch.zeros_like(p)` for its exp_avg/s state,
            # no dtype override, same bare `p.data.add_`). Confirmed against
            # the installed packages -- both run fine on bf16 params (no
            # crash), they just get the full floor risk below with nothing
            # to counteract it. Either way, Lion's step is exactly
            # sign(momentum) * lr for every weight, every step -- unlike
            # AdamW's variance-normalized step, there's no adaptive scaling
            # that might occasionally cross the floor anyway. So this is a
            # hard pass/fail against the actual --learning_rate, not a
            # "likely" statement.
            if args.learning_rate >= lr_floor:
                print(f"--optimizer {args.optimizer}: step is exactly +/-lr per "
                      f"weight, so this is a hard threshold. Your "
                      f"--learning_rate={args.learning_rate:.1e} clears the "
                      f"{lr_floor:.1e} floor ({margin:.1f}x), so updates should "
                      f"register -- but that lr is well above Lion's usual tuned "
                      f"range (typically 3-10x below an AdamW lr), so watch for "
                      f"instability instead of the freeze.")
            else:
                print(f"--optimizer {args.optimizer}: step is exactly +/-lr per "
                      f"weight, so this is a hard threshold, not a maybe. Your "
                      f"--learning_rate={args.learning_rate:.1e} is only "
                      f"{margin:.2f}x the {lr_floor:.1e} floor -- most of the "
                      f"UNet WILL freeze outright under bf16. You would need "
                      f"lr >= {lr_floor:.1e} to clear it, which is likely too "
                      f"high for Lion to stay stable at this scale.")
        elif args.optimizer in ("adamw", "adamw8"):
            # Adam's step is lr * m_hat / (sqrt(v_hat) + eps), not an exact
            # +/-lr like Lion's -- the ratio varies per weight with gradient
            # noise, but empirically sits within about a factor of 1 of lr.
            # So this is "expect trouble", not the hard guarantee Lion gets.
            if args.learning_rate >= lr_floor:
                print(f"--optimizer {args.optimizer}: step size (lr * m/sqrt(v)) "
                      f"usually runs close to lr itself. Your "
                      f"--learning_rate={args.learning_rate:.1e} clears the "
                      f"{lr_floor:.1e} floor ({margin:.1f}x), so most weights "
                      f"should get real updates.")
            else:
                print(f"--optimizer {args.optimizer}: step size (lr * m/sqrt(v)) "
                      f"usually runs close to lr itself, not exactly it, so this "
                      f"is a likelihood rather than the hard guarantee Lion gets. "
                      f"Your --learning_rate={args.learning_rate:.1e} is only "
                      f"{margin:.2f}x the {lr_floor:.1e} floor -- expect larger-"
                      f"magnitude weights (norm/gain params near 1.0 especially, "
                      f"since their floor scales with their own size) to freeze "
                      f"first. Raise --learning_rate toward {lr_floor:.1e} or "
                      f"above to avoid it.")
        print(f"--batch_size {args.batch_size}: fewer steps/epoch means fewer "
              f"chances to clear that floor per weight.")

    # ----- discriminator (optional) ---------------------------------------- #
    # Built here, deliberately BEFORE the RNG restore below: weight init
    # consumes RNG, and doing it afterwards would knock a resumed run off
    # the random walk it was on at checkpoint time.
    disc = None
    opt_d = None
    if args.disc_weight > 0:
        from train_discriminator import NLayerDiscriminator, weights_init
        # 3, not latent_channels: this discriminator judges decoded RGB.
        disc = NLayerDiscriminator(
            input_nc=3, ndf=64, n_layers=args.disc_layers,
        ).apply(weights_init).to(device)
        disc.train()
        opt_d = torch.optim.AdamW(
            disc.parameters(),
            lr=args.disc_lr,
            betas=(0.5, 0.9),
            weight_decay=0.0,
        )
        if resume_state is not None and "disc" in resume_state:
            disc.load_state_dict(resume_state["disc"])
            opt_d.load_state_dict(resume_state["opt_d"])
            print("Restored discriminator + opt_d state")
        else:
            # An ordinary run started from a checkpoint dir has no
            # training_state.pt, but every checkpoint now carries a
            # disc_state.pt. Without this the UNet spends the first stretch
            # of every such run being judged by a random discriminator.
            disc_path = os.path.join(model_path, "disc_state.pt")
            if os.path.exists(disc_path):
                dstate = torch.load(disc_path, map_location="cpu",
                                    weights_only=False)
                disc.load_state_dict(dstate["disc"])
                opt_d.load_state_dict(dstate["opt_d"])
                print("Restored discriminator + opt_d state from", disc_path)
        mode = "fixed" if args.disc_no_adaptive else "adaptive"
        print(f"Pixel discriminator enabled ({mode} weight {args.disc_weight}):"
              f" 3ch RGB, {args.disc_layers} layer(s), lr={args.disc_lr}")
        print(f"  Trains from step {args.disc_start};"
              f" steers the UNet from step {args.disc_start + args.disc_warmup}"
              f" (warmup {args.disc_warmup})")
        print(f"  Applied to noise levels <= {args.disc_max_noise},"
              f" {args.disc_decode_batch} samples per microbatch")

    # ----- load data, set training params ------------------------------------------------ #

    if resume_state is not None:
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state" in resume_state:
            cuda_rng_state = [t.cpu() for t in resume_state["cuda_rng_state"]]
            torch.cuda.set_rng_state_all(cuda_rng_state)
        print("Restored RNG state for dataset shuffling continuity")

    bs = args.batch_size
    accum = args.gradient_accum
    effective_batch_size = bs * accum
    dataloaders = []
    micro_steps_per_epoch = 0  # "micro" means "size directly run on gpu"
    ebs_steps_per_epoch = 0  # Effective-batch_size
    unsupervised = True if args.force_txtcache else False
    for dirs in args.train_data_dir:
        # remember, dirs can contain more than one dirname
        ds = CaptionImgDataset(dirs,
                               batch_size=bs,
                               txtcache_suffix=args.txtcache_suffix,
                               imgcache_suffix=args.imgcache_suffix,
                               gradient_accum=accum,
                               unsupervised=unsupervised,
                               )

        # Yes keep this using microbatch not effective batch size
        # If you want to be fancy, maybe aim for accum = number of dataloaders
        dl = DataLoader(ds, batch_size=bs,
                        shuffle=True,
                        drop_last=True,
                        num_workers=8, persistent_workers=True,
                        pin_memory=True, collate_fn=collate_fn,
                        prefetch_factor=4)
        if len(dl) < 1:
            raise ValueError("Error: dataset invalid")

        dataloaders.append(dl)
    mix_loader = InfiniteLoader(*dataloaders)

    shortest_dl_len = mix_loader.get_shortest_len()
    micro_steps_per_epoch = shortest_dl_len * len(dataloaders)
    # dl count already divided by micro batch size.
    # So now calculate EBS steps
    ebs_steps_per_epoch = micro_steps_per_epoch // accum
    print(f"Shortest dataset = {shortest_dl_len} microbatches")
    print(f"   {len(dataloaders)} datasets x ({shortest_dl_len} x {bs}) ... ")
    print("    => Using",
          micro_steps_per_epoch * bs,
          "as image count per epoch:",
          ebs_steps_per_epoch, "steps per epoch")

    if resume_state is not None:
        max_steps = resume_state["batch_count"] + args.continue_steps
        print(f"--continue_steps: training to step {max_steps}")
    elif args.max_steps and args.max_steps.endswith("e"):
        max_steps = float(args.max_steps.removesuffix("e"))
        max_steps = max_steps * ebs_steps_per_epoch
    else:
        max_steps = int(args.max_steps)
    if args.warmup_steps.endswith("e"):
        warmup_steps = float(args.warmup_steps.removesuffix("e"))
        warmup_steps = warmup_steps * ebs_steps_per_epoch
    else:
        warmup_steps = int(args.warmup_steps)


    # Common args that may or may not be defined
    # Allow fall-back to optimizer-specific defaults
    opt_args = {
        **({'weight_decay': args.weight_decay} if args.weight_decay is not None else {}),
        **({'betas': tuple(args.betas)} if args.betas else {}),
        **({'d0': args.initial_d} if args.initial_d else {}),
    }
    if args.optimizer == "py_lion":
        optim = lion_pytorch.Lion(trainable_params,
                                  lr=peak_lr,
                                  **opt_args
                                  )
    elif args.optimizer == "opt_lion":
        optim = Lion(trainable_params,
                     lr=peak_lr,
                     **opt_args
                     )
    elif args.optimizer == "d_lion":
        from dadaptation import DAdaptLion
        # D-Adapt controls the step size; a large/base LR is expected.
        # 1.0 is the common choice; fall back to peak_lr if you've set one.
        base_lr = peak_lr
        if base_lr < 0.1:
            print("WARNING: Typically, DAdaptLion expects LR of 1.0")
        if args.initial_d:
            print("Note; initial_d set to", args.initial_d)

        optim = DAdaptLion(
            trainable_params,
            lr=base_lr,
            **opt_args
        )
    elif args.optimizer == "adamw8":
        import bitsandbytes as bnb
        optim = bnb.optim.AdamW8bit(trainable_params,
                                    lr=peak_lr,
                                    **opt_args
                                    )
    elif args.optimizer == "adamw":
        from torch.optim import AdamW
        optim = AdamW(
            trainable_params,
            lr=peak_lr,
            **opt_args
        )
    else:
        print("ERROR: unrecognized optimizer setting")
        exit(1)

    if resume_state is not None:
        optim.load_state_dict(resume_state["optimizer"])
        print("Restored optimizer state")

    # -- optimizer settings...
    print("Using optimizer", args.optimizer)
    if args.use_snr:
        if hasattr(noise_sched, "alphas_cumprod"):
            print(f"  Using MinSNR with gamma of {args.noise_gamma}")
        else:
            print("  Skipping --use_snr: invalid with scheduler", type(noise_sched))
            args.use_snr = False

    print(
        f"  NOTE: peak_lr = {peak_lr}, lr_scheduler={args.scheduler}, total steps={max_steps}(steps/Epoch={ebs_steps_per_epoch})")
    print(f"        batch={bs}, accum={accum}, effective batchsize={effective_batch_size}")
    print(f"        warmup={warmup_steps}, betas=",
          args.betas if args.betas else "(default)",
          " weight_decay=",
          args.weight_decay if args.weight_decay else "(default)",
          )

    unet, dl, optim = accelerator.prepare(
        pipe.unet,
        mix_loader,
        optim)
    unet.train()

    scheduler_args = {
        "optimizer": optim,
        "num_warmup_steps": warmup_steps,
        "num_training_steps": max_steps,
        "scheduler_specific_kwargs": {},
    }

    if args.scheduler == "cosine_with_min_lr":
        scheduler_args["scheduler_specific_kwargs"]["min_lr_rate"] = args.min_lr_ratio
        print(f"  Setting min_lr_ratio to {args.min_lr_ratio}")
    if args.num_cycles:
        # technically this should only be used for cosine types?
        scheduler_args["scheduler_specific_kwargs"]["num_cycles"] = args.num_cycles
        print(f"  Setting num_cycles to {args.num_cycles}")

    if args.scheduler.lower() == "rex":
        rex = RexLR(
            optim,
            total_steps=max_steps - warmup_steps,
            max_lr=peak_lr,
            min_lr=peak_lr * args.min_lr_ratio,
        )
        if warmup_steps > 0:
            warmup = LinearLR(
                optim,
                start_factor=args.rex_start_factor,
                end_factor=args.rex_end_factor,
                total_iters=warmup_steps,
            )
            lr_sched = SequentialLR(optim, [warmup, rex], milestones=[warmup_steps])
        else:
            lr_sched = rex

    elif args.scheduler.lower() == "linear_with_min_lr":
        from transformers import get_polynomial_decay_schedule_with_warmup
        base_lr = args.learning_rate
        floor_lr = base_lr * args.min_lr_ratio

        lr_sched = get_polynomial_decay_schedule_with_warmup(
            optimizer=optim,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=floor_lr
        )

    else:
        lr_sched = get_scheduler(args.scheduler, **scheduler_args)

    if resume_state is not None and "scheduler" in resume_state:
        lr_sched.load_state_dict(resume_state["scheduler"])
        print("Restored LR scheduler state")

    lr_sched = accelerator.prepare(lr_sched)

    tstate = TrainState(args=args,
                        device=device,
                        compute_dtype=compute_dtype,
                        latent_scaling=latent_scaling,
                        noise_sched=noise_sched,
                        )
    tstate.disc = disc
    tstate.opt_d = opt_d
    tstate.vae = vae
    if resume_state is not None:
        tstate.global_step = resume_state["global_step"]
        tstate.batch_count = resume_state["batch_count"]

    run_name = os.path.basename(args.output_dir)
    tstate.tb_writer = SummaryWriter(log_dir=os.path.join("tensorboard/", run_name))

    #
    # ----- training loop --------------------------------------------------- #

    tstate.total_epochs = math.ceil(max_steps / ebs_steps_per_epoch)
    mix_iter = iter(mix_loader)

    # Ctrl-C waits for a clean step (gradient_accum boundary) before saving,
    # so we never save mid-accumulation. A second Ctrl-C aborts immediately.
    global _interrupt_requested
    _interrupt_requested = False
    old_sigint_handler = signal.signal(signal.SIGINT, _request_interrupt)

    try:
        for epoch_count in range(tstate.total_epochs):
            tstate.epoch_count = epoch_count

            if args.save_on_epoch:
                checkpointandsave(pipe, unet, accelerator, tstate)

            if args.scheduler_at_epoch:
                # Implement a stair-stepped decay, updating on epoch to what the smooth would be at this point
                lr_sched.step(tstate.batch_count)

            tstate.pbar = tqdm(range(ebs_steps_per_epoch),
                               desc=f"E{epoch_count}/{tstate.total_epochs}",
                               bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} {rate_fmt}{postfix}",
                               dynamic_ncols=True,
                               leave=True)

            # "batch" is actually micro-batch
            # yes this will stop at end of shortest dataset.
            # Every dataset will get equal value. I'm not messing around with
            #  custom "balancing"
            for _ in range(micro_steps_per_epoch):
                if tstate.batch_count >= max_steps:
                    break
                step, batch_paths = next(mix_iter)
                try:
                    # this bumps tstate.batch_count only for EBS size
                    train_micro_batch(unet, accelerator, batch_paths, tstate,
                                      optim, lr_sched, ebs_steps_per_epoch)
                except torch.OutOfMemoryError:
                    print("OUT OF VRAM Problem in Batch:", batch_paths)
                    exit(0)

                # Now save if trigger present, OR if right stepcount
                if tstate.global_step % args.gradient_accum == 0:
                    if _interrupt_requested:
                        # Ignore further SIGINT for the rest of shutdown: an
                        # impatient second Ctrl-C landing mid-write would
                        # corrupt the very checkpoint we're trying to
                        # protect. Normal handling is restored in the
                        # `finally` below, only once the save AND the
                        # dataloader worker teardown have both finished -
                        # so by the time the process actually exits, there's
                        # nothing left running in the background.
                        signal.signal(signal.SIGINT, signal.SIG_IGN)
                        print("Clean step reached; saving full training state to 'interrupted'...")
                        if accelerator.is_main_process:
                            checkpointandsave(pipe, unet, accelerator, tstate,
                                              optim=optim, lr_sched=lr_sched,
                                              save_training_state=True, tag="interrupted",
                                              skip_sample=True)
                        raise KeyboardInterrupt

                    trigger_path = os.path.join(args.output_dir, "trigger.checkpoint")
                    numbered_trigger = os.path.join(
                        args.output_dir,
                        f"trigger.checkpoint.{tstate.batch_count:05}")

                    if os.path.exists(trigger_path):
                        print("trigger.checkpoint detected. ...")
                        checkpointandsave(pipe, unet, accelerator, tstate)
                        try:
                            os.remove(trigger_path)
                        except Exception as e:
                            print("warning: got exception", e)

                    elif os.path.exists(numbered_trigger):
                        print(f"{os.path.basename(numbered_trigger)} "
                              f"matched current step.")
                        checkpointandsave(pipe, unet, accelerator, tstate)
                        try:
                            os.remove(numbered_trigger)
                        except Exception as e:
                            print("warning: got exception", e)

                    elif args.save_steps and (tstate.batch_count % args.save_steps == 0):
                        if tstate.batch_count > 0 and tstate.batch_count >= int(args.save_start):
                            print(f"Saving @{tstate.batch_count:05} (save every {args.save_steps} steps)")
                            checkpointandsave(pipe, unet, accelerator, tstate)

                    elif (args.sample_prompt and args.sample_steps
                          and (tstate.batch_count % args.sample_steps == 0)):
                        if tstate.batch_count > 0 and tstate.batch_count >= int(args.save_start):
                            print(f"Sampling @{tstate.batch_count:05} (sample every {args.sample_steps} steps,"
                                  " no checkpoint)")
                            sample_dir = os.path.join(args.output_dir, "samples",
                                                      f"step-{tstate.batch_count:05}")
                            sample_without_checkpoint(pipe, unet, accelerator, tstate,
                                                      args.sample_prompt, args.seed,
                                                      args.sampler_steps, sample_dir,
                                                      tstate.device)

            tstate.pbar.close()
            if tstate.batch_count >= max_steps:
                break
    finally:
        # Tear down the dataloader worker processes deterministically,
        # before we tell the user (and the shell) that we're done - rather
        # than leaving that to GC/atexit timing after the prompt is back.
        print("Shutting down dataloader workers...")
        mix_loader.shutdown()
        signal.signal(signal.SIGINT, old_sigint_handler)

    if accelerator.is_main_process:
        if False:
            pipe.save_pretrained(args.output_dir, safe_serialization=True)
            sample_img(args, args.seed, args.output_dir,
                       custom_pipeline)
            print(f"finished:model saved to {args.output_dir}")
        else:
            checkpointandsave(pipe, unet, accelerator, tstate,
                              optim=optim, lr_sched=lr_sched,
                              save_training_state=True, tag="final")
        if tstate.tb_writer is not None:
            tstate.tb_writer.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting.")
        # just fall off end?
