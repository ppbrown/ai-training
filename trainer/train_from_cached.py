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


# --------------------------------------------------------------------------- #

import os, math, shutil
from pathlib import Path
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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# This enables profiling on first batch,
# which then SPEEDS UP subsequent runs automaticaly
torch.backends.cudnn.benchmark = True

# --------------------------------------------------------------------------- #
# 2. Utils                                                                    #
# --------------------------------------------------------------------------- #

from train_captiondata import CaptionImgDataset

from train_utils import collate_fn, sample_img, log_unet_l2_norm

#####################################################
# Main                                              #
#####################################################

def main():
    torch.manual_seed(args.seed)
    peak_lr = args.learning_rate

    print("Training type:", "fp32" if args.fp32 else "mixed precision")

    model_dtype = torch.float32  # Always load master in full fp32
    compute_dtype = torch.float32 if args.fp32 else torch.bfloat16  # runtime math dtype

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accum,
        mixed_precision="no" if args.fp32 else "bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )
    device = accelerator.device

    # ----- read arguments and load pipeline -------------- #

    if args.is_custom:
        custom_pipeline = args.pretrained_model
    else:
        custom_pipeline = None

    print(f"Loading '{args.pretrained_model}' Custom pipeline? {custom_pipeline}")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            args.pretrained_model,
            custom_pipeline=custom_pipeline,
            torch_dtype=model_dtype
        )
    except Exception as e:
        print("Error loading model", args.pretrained_model)
        print(e)
        exit(0)

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

    # ----- load data, set training params ------------------------------------------------ #


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

    if args.max_steps and args.max_steps.endswith("e"):
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

    lr_sched = accelerator.prepare(lr_sched)



    tstate = TrainState()

    ## These are now all bundled into tstate, and
    ##   initialized there
    ## global_step = 0  # micro-batch count
    ## batch_count = 0  # effective-batch-size count, aka num of fullsize batches
    ## accum_loss = 0.0
    ## accum_mse = 0.0
    ## accum_qk = 0.0
    ## accum_norm = 0.0
    ## latent_paths = []

    run_name = os.path.basename(args.output_dir)
    tstate.tb_writer = SummaryWriter(log_dir=os.path.join("tensorboard/", run_name))

    def checkpointandsave():
        if tstate.global_step % args.gradient_accum != 0:
            print("INTERNAL ERROR: checkpointandsave() not called on clean step")
            return

        ckpt_dir = os.path.join(args.output_dir,
                                f"checkpoint-{tstate.batch_count:05}")
        if os.path.exists(ckpt_dir):
            print(f"Checkpoint {ckpt_dir} already exists. Skipping redundant save")
            return
        pinned_te, pinned_unet = pipe.text_encoder, pipe.unet
        pipe.unet = accelerator.unwrap_model(unet)
        log_unet_l2_norm(pipe.unet, tstate.tb_writer, tstate.batch_count)

        print(f"Saving checkpoint to {ckpt_dir}")
        pipe.save_pretrained(ckpt_dir, safe_serialization=True)
        pipe.text_encoder, pipe.unet = pinned_te, pinned_unet
        if args.sample_prompt is not None:
            sample_img(args, args.seed, ckpt_dir,
                       custom_pipeline)
        if args.copy_config:
            savefile = os.path.join(args.output_dir, args.copy_config)
            if not os.path.exists(savefile):
                tqdm.write(f"Copying {args.copy_config} to {args.output_dir}")
                shutil.copy(args.copy_config, args.output_dir)

                import yaml
                savefile = os.path.join(args.output_dir, "args.yaml")
                tqdm.write(f"Saving commandline to  {savefile}")
                Path(savefile).write_text(yaml.safe_dump(vars(args), sort_keys=True))

        savefile = os.path.join(ckpt_dir, "tstate.latent_paths")
        with open(savefile, "w") as f:
            f.write('\n'.join(tstate.latent_paths) + '\n')
            f.close()
        print("Wrote", len(tstate.latent_paths), "loglines to", savefile)
        tstate.latent_paths = []

    #
    # ----- training loop --------------------------------------------------- #
    """ Old way
    ebar = tqdm(range(math.ceil(max_steps / len(dl))),
                desc="Epoch", unit="", dynamic_ncols=True,
                position=0,
                leave=True)
    """

    total_epochs = math.ceil(max_steps / ebs_steps_per_epoch)
    mix_iter = iter(mix_loader)

    for epoch_count in range(total_epochs):
        if args.save_on_epoch:
            checkpointandsave()

        if args.scheduler_at_epoch:
            # Implement a stair-stepped decay, updating on epoch to what the smooth would be at this point
            lr_sched.step(tstate.batch_count)

        tstate.pbar = tqdm(range(ebs_steps_per_epoch),
                    desc=f"E{epoch_count}/{total_epochs}",
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
                train_micro_batch(unet, accelerator, batch_paths, tstate)
            except torch.OutOfMemoryError:
                print("OUT OF VRAM Problem in Batch:", batch_paths)
                exit(0)

        tstate.pbar.close()
        if tstate.batch_count >= max_steps:
            break

    if accelerator.is_main_process:
        if tstate.tb_writer is not None:
            tstate.tb_writer.close()
        if False:
            pipe.save_pretrained(args.output_dir, safe_serialization=True)
            sample_img(args, args.seed, args.output_dir,
                       custom_pipeline)
            print(f"finished:model saved to {args.output_dir}")
        else:
            checkpointandsave()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting.")
        # just fall off end?
