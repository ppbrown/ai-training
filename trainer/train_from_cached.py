#!/usr/bin/env python

# train_with_caching.py


# --------------------------------------------------------------------------- #
# 1. CLI                                                                      #
# --------------------------------------------------------------------------- #
import argparse
def parse_args():
    p = argparse.ArgumentParser(epilog="Touch 'trigger.checkpoint' in the output_dir to dynamically trigger checkpoint save")
    p.add_argument("--fp32", action="store_true",
                   help="Override default mixed precision fp32/bf16, to force everything full fp32")
    p.add_argument("--cpu_offload", action="store_true",
                   help="Enable cpu offload at pipe level")
    p.add_argument("--pretrained_model", required=True,  help="HF repo or local dir")
    p.add_argument("--is_custom", action="store_true",
                   help="Model provides a 'custom pipeline'")
    p.add_argument("--train_data_dir",  nargs="+", required=True, action="append",
                   help="Directory tree(s) containing *.jpg + *.txt.\nCan use more than once, but make sure same resolution for each")
    p.add_argument("--scheduler",      type=str, default="constant", help="Default=constant")
    p.add_argument("--scale_loss_with_accum", action="store_true", 
                   help="When accum >1, scale each microbatch loss /accum")
    p.add_argument("--scheduler_at_epoch", action="store_true", 
                   help="Only consult scheduler at epoch boundaries. Useful if less than 15 epochs")
    p.add_argument("--optimizer",      type=str, choices=["adamw","adamw8","opt_lion","py_lion", "d_lion"], default="adamw8", 
                   help="opt_lion is recommended over py_lion")
    p.add_argument("--num_cycles",     type=float, help="Typically only used with cosine decay")
    p.add_argument("--min_sigma",      type=float, default=1e-5, 
                   help="For FlowMatch. Default=1e-5. If you are using effective batchsize <256, consider a higher value like 2e-4")
    p.add_argument("--copy_config",    type=str, help="Config file to archive with training, if model load succeeds")
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--force_toklen",   type=int, 
                   help="Force token length to a single value, like 256. Use for T5 cache")
    p.add_argument("--gradient_accum", type=int, default=1, help="Default=1")
    p.add_argument('--gradient_checkpointing', action='store_true',
                   help="Enable grad checkpointing in unet")
    p.add_argument("--learning_rate",  type=float, default=1e-5, help="Default=1e-5")
    p.add_argument("--min_lr_ratio",   type=float, default=0.1, 
                   help="Actually a ratio, not hard number. Only used if 'min_lr' type schedulers are used")
    p.add_argument("--rex_start_factor", type=float, default=1.0, help="Only used with REX Scheduler during warmup steps. Must be greater than 0. Default=1")
    p.add_argument("--rex_end_factor", action='store_const', const=1.0, default=1.0,
                   help='[read-only] fixed at 1.0; providing a value is an error')
                   #end factor is fixed at 1.0 to avoid odd LR jumps messing things up


    p.add_argument("--learning_rate_decay", type=float,
                   help="Subtract this every epoch, if schedler==constant")
    p.add_argument("--weight_decay",   type=float)
    p.add_argument("--vae_scaling_factor", type=float, help="Override vae scaling factor")
    p.add_argument("--text_scaling_factor", type=float, help="Override embedding scaling factor")
    p.add_argument("--warmup_steps",   default=0, 
                   help="Measured in effective batchsize steps (b * a) default=0")
    p.add_argument("--max_steps",      default=10_000, 
                   help="Maximum EFFECTIVE BATCHSIZE steps(b * accum) default=10_000. May use '2e' for whole epochs")
    p.add_argument("--save_steps",     type=int, help="Measured in effective batchsize(b * a)")
    p.add_argument("--save_start",     type=int, help="Dont start saving until past this step")
    p.add_argument("--save_on_epoch",  action="store_true")
    p.add_argument("--noise_gamma",    type=float, default=5.0)
    p.add_argument("--betas",  type=float, nargs=2, metavar=("BETA1","BETA2"),
                   help="Typical LION default is '0.9, 0.99'." \
                   "For instability issues,  use 0.95 0.98")
    p.add_argument("--use_snr", action="store_true",
                   help="Use Min SNR noise adjustments")
    p.add_argument( "--gradient_clip", type=float, default=1.0,
                        help="Max global grad norm. Set <=0 to disable gradient clipping.")

    p.add_argument("--targetted_training", action="store_true",
                   help="Only train reset layers")
    p.add_argument("--reinit_crossattn", action="store_true",
                   help="Attempt to reset cross attention weights for text realign")
    p.add_argument("--reinit_crossattnout", action="store_true",
                   help="Attempt to reset just the 'out' cross attention weights")
    p.add_argument("--reinit_attention", action="store_true",
                   help="Attempt to reset ALL attention weights for text realign")
    p.add_argument("--reinit_qk", action="store_true",
                   help="Attempt to reset just qk weights for text realign")
    p.add_argument("--reinit_out", action="store_true",
                   help="Attempt to reset just out blocks")
    p.add_argument("--unfreeze_out", action="store_true",
                   help="Just make the out blocks trainable")
    p.add_argument("--reinit_in", action="store_true",
                   help="Attempt to reset just in blocks")
    p.add_argument("--unfreeze_in", action="store_true",
                   help="Just make the in blocks trainable")
    p.add_argument("--reinit_time", action="store_true",
                   help="Attempt to reset just noise schedule layer")
    p.add_argument("--unfreeze_time", action="store_true",
                   help="Attempt to unfreeze just noise schedule layer")
    p.add_argument("--unfreeze_up_blocks", type=int, nargs="+",
                   help="Just unfreeze, dont reinit. Give 1 or more space-seperated numbers ranged [0-3]")
    p.add_argument("--unfreeze_down_blocks", type=int, nargs="+",
                   help="Just unfreeze, dont reinit. Give 1 or more space-seperated numbers ranged [0-3]")
    p.add_argument("--unfreeze_mid_block", action="store_true",
                   help="Just unfreeze, dont reinit.")
    p.add_argument("--unfreeze_norms", action="store_true",
                   help="Just unfreeze, dont reinit.")
    p.add_argument("--reinit_unet", action="store_true",
                   help="Train from scratch unet (Do not use, this is broken)")
    p.add_argument("--unfreeze_attention", action="store_true",
                   help="Just unfreeze, dont reinit.")

    p.add_argument("--sample_prompt", nargs="+", type=str, help="Prompt to use for a checkpoint sample image")
    p.add_argument("--seed",        type=int, default=90)
    p.add_argument("--txtcache_suffix", type=str, default=".txt_t5cache", help="Default=.txt_t5cache")
    p.add_argument("--imgcache_suffix", type=str, default=".img_sdvae", help="Default=.img_sdvae")

    return p.parse_args()


# Put this super-early, so that usage message procs fast
args = parse_args()

# --------------------------------------------------------------------------- #

import os, math, shutil
from itertools import chain
from pathlib import Path
from tqdm.auto import tqdm

import torch
import safetensors.torch as st
from torch.utils.data import Dataset, DataLoader


from accelerate import Accelerator, DistributedDataParallelKwargs
from diffusers import DiffusionPipeline, UNet2DConditionModel
from diffusers import DDPMScheduler, PNDMScheduler
from diffusers.models.attention import Attention as CrossAttention
from diffusers.training_utils import compute_snr


# diffusers optimizers dont have a min_lr arg,
# so dont use that scheduler
#from diffusers.optimization import get_scheduler
from transformers import get_scheduler

from torch.utils.tensorboard import SummaryWriter

import lion_pytorch

# Speed boost, as long as we dont need "strict fp32 math"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------- #
# 2. Utils                                                                    #
# --------------------------------------------------------------------------- #

from caption_dataset import CaptionImgDataset

def collate_fn(examples):
    return {
        "img_cache": [e["img_cache"] for e in examples],
        "txt_cache": [e["txt_cache"] for e in examples],
    }


# PIPELINE_CODE_DIR is typicaly the dir of original model
def sample_img(prompt, seed, CHECKPOINT_DIR, PIPELINE_CODE_DIR):
    tqdm.write(f"Trying render of '{prompt}' using seed {seed} ..")
    pipe = DiffusionPipeline.from_pretrained(
        CHECKPOINT_DIR, 
        custom_pipeline=PIPELINE_CODE_DIR, 
        use_safetensors=True,
        safety_checker=None, requires_safety_checker=False,
        #torch_dtype=torch.bfloat16,
    )
    pipe.safety_checker=None
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_sequential_cpu_offload()

    # Make sure that prompt order doesnt change effective seed
    generator = [torch.Generator(device="cuda").manual_seed(seed) for _ in range(len(prompt))]

    images = pipe(prompt, num_inference_steps=30, generator=generator).images
    for ndx, image in enumerate(images):
        fname=f"sample-{seed}-{ndx}.png"
        outname=f"{CHECKPOINT_DIR}/{fname}"
        image.save(outname)
        print(f"Saved {outname}")



def log_unet_l2_norm(unet, tb_writer, step):
    """
    Util to log the overall average parameter size.
    Purpose is to determine if we maybe need weight decay or not.
    (If it grows significantly over time, we prob need it)
    """
    # Gather all parameters as a single vector
    params = [p.data.flatten() for p in unet.parameters() if p.requires_grad]
    all_params = torch.cat(params)
    l2_norm = torch.norm(all_params, p=2).item()
    tb_writer.add_scalar('unet/L2_norm', l2_norm, step)


#####################################################
# Main                                              #
#####################################################

def main():
    torch.manual_seed(args.seed)
    peak_lr       = args.learning_rate

    print("Training type:", "fp32" if args.fp32 else "mixed precision")

    model_dtype   = torch.float32 # Always load master in full fp32
    compute_dtype = torch.float32 if args.fp32 else torch.bfloat16 # runtime math dtype

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accum,
        mixed_precision="no" if args.fp32 else "bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )
    device = accelerator.device
    


    # ----- load pipeline --------------------------------------------------- #

    if args.is_custom:
        custom_pipeline=args.pretrained_model
    else:
        custom_pipeline=None

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


    # -- unet trainable selection -- #
    if args.targetted_training:
        print("Limiting Unet training to targetted area(s)")
        pipe.unet.requires_grad_(False)
    else:
        print("Training full prior Unet")
        pipe.unet.requires_grad_(True)

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
        from reinit import reinit_all_unet
        reinit_all_unet(pipe.unet)
    elif args.reinit_qk:
        print("Attempting to reset Q/K layers of Unet")
        from reinit import reinit_qk
        reinit_qk(pipe.unet)
    elif args.reinit_crossattn:
        print("Attempting to reset Cross Attn layers of Unet")
        from reinit import reinit_cross_attention
        reinit_cross_attention(pipe.unet)
    elif args.reinit_crossattnout:
        print("Attempting to reset Cross Attn OUT layers of Unet")
        from reinit import reinit_cross_attention_outproj
        reinit_cross_attention_outproj(pipe.unet)
    elif args.reinit_attention:
        print("Attempting to reset attention layers of Unet")
        from reinit import reinit_all_attention
        reinit_all_attention(pipe.unet)

    if args.reinit_out:
        print("Attempting to reset Out layers of Unet")
        from reinit import retrain_out
        retrain_out(pipe.unet, reset=True)
    elif args.unfreeze_out:
        print("Attempting to unfreeze Out layers of Unet")
        from reinit import retrain_out
        retrain_out(pipe.unet, reset=False)
    if args.reinit_in:
        print("Attempting to reset in layers of Unet")
        from reinit import retrain_in
        retrain_in(pipe.unet, reset=True)
    elif args.unfreeze_in:
        print("Attempting to unfreeze in layers of Unet")
        from reinit import retrain_in
        retrain_in(pipe.unet, reset=False)
    if args.reinit_time:
        print("Attempting to reset time layers of Unet")
        from reinit import retrain_time
        retrain_time(pipe.unet, reset=True)
    elif args.unfreeze_time:
        print("Attempting to unfreeze time layers of Unet")
        from reinit import retrain_time
        retrain_time(pipe.unet, reset=False)

    if args.unfreeze_up_blocks:
        print(f"Attempting to unfreeze (({args.unfreeze_up_blocks})) upblocks of Unet")
        from reinit import unfreeze_up_blocks
        unfreeze_up_blocks(pipe.unet, args.unfreeze_up_blocks, reset=False)
    if args.unfreeze_down_blocks:
        print(f"Attempting to unfreeze (({args.unfreeze_down_blocks})) downblocks of Unet")
        from reinit import unfreeze_down_blocks
        unfreeze_down_blocks(pipe.unet, args.unfreeze_down_blocks, reset=False)
    if args.unfreeze_mid_block:
        print(f"Attempting to unfreeze mid block of Unet")
        from reinit import unfreeze_mid_block
        unfreeze_mid_block(pipe.unet)
    if args.unfreeze_norms:
        print(f"Attempting to unfreeze normals components of Unet")
        from reinit import unfreeze_norms
        unfreeze_norms(pipe.unet)

    if args.unfreeze_attention:
        print("Attempting to unfreeze attention layers of Unet")
        from reinit import unfreeze_all_attention
        unfreeze_all_attention(pipe.unet)
    # ------------------------------------------ #

    if args.save_start:
        print("save_start limit set to",args.save_start)
    if args.cpu_offload:
        print("Enabling cpu offload")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing in UNet")
        pipe.unet.enable_gradient_checkpointing()

    if args.vae_scaling_factor:
        pipe.vae.config.scaling_factor = args.vae_scaling_factor

    vae, unet = pipe.vae.eval(), pipe.unet

    noise_sched = pipe.scheduler
    print("Pipe is using noise scheduler", type(noise_sched))
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
    print("VAE scaling factor is",latent_scaling)


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
        f"Align-phase: {sum(p.numel() for p in trainable_params)/1e6:.2f} M "
        "parameters will be updated"
    )

    # ----- load data ------------------------------------------------------------ #
    bs = args.batch_size ; accum = args.gradient_accum
    effective_batch_size = bs * accum
    dataloaders = []; steps_per_epoch = 0

    for dirs in args.train_data_dir:
        ds = CaptionImgDataset(dirs, 
                               batch_size=bs,
                               txtcache_suffix=args.txtcache_suffix,
                               imgcache_suffix=args.imgcache_suffix,
                               gradient_accum=accum
                               )

        # Yes keep this using microbatch not effective batch
        # drop_last=True not needed: ds should have self-truncated
        dl = DataLoader(ds, batch_size=bs, 
                        shuffle=True,
                        num_workers=8, persistent_workers=True,
                        pin_memory=True, collate_fn=collate_fn,
                        prefetch_factor=4)
        if len(dl)<1:
            raise ValueError("Error: dataset invalid")

        dataloaders.append(dl)

    shortest_dl_len = min(len(dl) for dl in dataloaders)
    if len(dataloaders) > 1:
        print("Will truncate all to shortest length:", shortest_dl_len * bs)
        # Truncation is effectively done by use of zip, lower down.
        # We dont actually change the objs here. But we DO use this to calculate
        # steps_per_epoch, which is important

    steps_per_epoch = shortest_dl_len * len(dataloaders)
    # dl count already divided by mini batch
    steps_per_epoch = steps_per_epoch // accum

    if args.max_steps and args.max_steps.endswith("e"):
        max_steps = float(args.max_steps.removesuffix("e"))
        max_steps = max_steps * steps_per_epoch
    else:
        max_steps = int(args.max_steps)
    if args.warmup_steps.endswith("e"):
        warmup_steps = float(args.warmup_steps.removesuffix("e"))
        warmup_steps = warmup_steps * steps_per_epoch
    else:
        warmup_steps  = int(args.warmup_steps)

    # Common args that may or may not be defined
    # Allow fall-back to optimizer-specific defaults
    opt_args = {
        **({'weight_decay': args.weight_decay} if args.weight_decay is not None else {}),
        **({'betas': tuple(args.betas)} if args.betas else {}),
    }
    if args.optimizer == "py_lion":
        import lion_pytorch
        optim = lion_pytorch.Lion(trainable_params, 
                                  lr=peak_lr, 
                                  **opt_args
                                  )
    elif args.optimizer == "opt_lion":
        from optimi import Lion # torch-optimi pip module
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
    print("Using optimizer",args.optimizer)
    if args.use_snr:
        if hasattr(noise_sched, "alphas_cumprod"):
            print(f"  Using MinSNR with gamma of {args.noise_gamma}")
        else:
            print("  Skipping --use_snr: invalid with scheduler", type(noise_sched))
            args.use_snr = False

    print(f"  NOTE: peak_lr = {peak_lr}, lr_scheduler={args.scheduler}, total steps={max_steps}(steps/Epoch={steps_per_epoch})")
    print(f"        batch={bs}, accum={accum}, effective batchsize={effective_batch_size}")
    print(f"        warmup={warmup_steps}, betas=",
          args.betas if args.betas else "(default)",
          " weight_decay=",
          args.weight_decay if args.weight_decay else "(default)",
    )

    unet, dl, optim = accelerator.prepare(pipe.unet, dl, optim)
    unet.train()

    scheduler_args = {
        "optimizer": optim,
        "num_warmup_steps": warmup_steps,
        "num_training_steps": max_steps,
    }

    scheduler_args["scheduler_specific_kwargs"] = {}
    if args.scheduler == "cosine_with_min_lr":
        scheduler_args["scheduler_specific_kwargs"]["min_lr_rate"] = args.min_lr_ratio 
        print(f"  Setting min_lr_ratio to {args.min_lr_ratio}")
    if args.num_cycles:
        #technically this should only be used for cosine types?
        scheduler_args["scheduler_specific_kwargs"]["num_cycles"] = args.num_cycles 
        print(f"  Setting num_cycles to {args.num_cycles}")


    if args.scheduler.lower() == "rex":
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        # from pytorch_optimizer.lr_scheduler.rex import REXScheduler - this is not compatible
        from axolotl.utils.schedulers import RexLR 

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
        base_lr  = args.learning_rate
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

    global_step = 0 
    batch_count = 0
    accum_loss = 0.0; accum_mse = 0.0; accum_qk = 0.0; accum_norm = 0.0
    latent_paths = []

    run_name = os.path.basename(args.output_dir)
    tb_writer = SummaryWriter(log_dir=os.path.join("tensorboard/",run_name))

    def checkpointandsave():
        nonlocal latent_paths
        if global_step % args.gradient_accum != 0:
            print("INTERNAL ERROR: checkpointandsave() not called on clean step")
            return

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{batch_count:05}")
        if os.path.exists(ckpt_dir):
            print(f"Checkpoint {ckpt_dir} already exists. Skipping redundant save")
            return
        pinned_te, pinned_unet = pipe.text_encoder, pipe.unet
        pipe.unet = accelerator.unwrap_model(unet)
        log_unet_l2_norm(pipe.unet, tb_writer, batch_count)


        print(f"Saving checkpoint to {ckpt_dir}")
        pipe.save_pretrained(ckpt_dir, safe_serialization=True)
        pipe.text_encoder, pipe.unet = pinned_te, pinned_unet
        if args.sample_prompt is not None:
            sample_img(args.sample_prompt, args.seed, ckpt_dir, 
                       custom_pipeline)
        if args.copy_config:
            savefile = os.path.join(args.output_dir, args.copy_config)
            if not os.path.exists(savefile):
                tqdm.write(f"Copying {args.copy_config} to {args.output_dir}")
                shutil.copy(args.copy_config, args.output_dir)

        savefile = os.path.join(ckpt_dir, "latent_paths")
        with open(savefile, "w") as f:
            f.write('\n'.join(latent_paths) + '\n')
            f.close()
        print("Wrote",len(latent_paths),"loglines to",savefile)
        latent_paths = []

    def train_micro_batch(unet):
        nonlocal batch_count, global_step, accum_loss, accum_mse, accum_qk, accum_norm, epoch_count
        nonlocal latent_paths

        with accelerator.accumulate(unet):
            # --- Load latents & prompt embeddings from cache ---
            latents = []
            for cache_file in batch["img_cache"]:
                latent = st.load_file(cache_file)["latent"]
                latent_paths.append(cache_file)
                latents.append(latent)
            latents = torch.stack(latents).to(device, dtype=compute_dtype) * latent_scaling

            embeds = []
            for cache_file in batch["txt_cache"]:
                if Path(cache_file).suffix == ".h5":
                    arr = h5f["emb"][:]
                    emb = torch.from_numpy(arr)
                else:
                    try:
                        emb = st.load_file(cache_file)["emb"]
                    except Exception as e:
                        print("Error loading these files...")
                        print(cache_file)
                        exit(0)
                emb = emb.to(device, dtype=compute_dtype)
                embeds.append(emb)

            if args.force_toklen:
                # Text embeddings have to all be same length otherwise we cant batch train.
                # zero-pad where needed. Truncate where needed.
                MAX_TOK = args.force_toklen
                D = embeds[0].size(1)
                fixed = []
                for e in embeds:
                    T = e.size(0)
                    if T >= MAX_TOK:
                        fixed.append(e[:MAX_TOK])
                    else:
                        pad = torch.zeros((MAX_TOK - T, D), dtype=e.dtype, device=e.device)
                        fixed.append(torch.cat([e, pad], dim=0))
                prompt_emb = torch.stack(fixed, dim=0).to(device, dtype=compute_dtype) # [B, MAX_TOK, D]
            else:
                # Take easy path, if using CLIP cache or something where length is already forced
                prompt_emb = torch.stack(embeds).to(device, dtype=compute_dtype)

            # --- Add noise ---

            if hasattr(noise_sched, "add_noise"):
                # Standard DDPM/PNDM-style
                noise = torch.randn_like(latents)
                bsz = latents.size(0)
                timesteps = torch.randint(
                    0, noise_sched.config.num_train_timesteps,
                    (bsz,), device=device, dtype=torch.long
                )
                noisy_latents = noise_sched.add_noise(latents, noise, timesteps)
                noise_target = noise
            else:
                # Flow Matching: continuous s in [epsilon, 1 - epsilon]
                bsz = latents.size(0)
                eps = args.min_sigma  # avoid divide-by-zero magic
                noise = torch.randn_like(latents)
                
                s = torch.rand(bsz, device=device).mul_(1 - 2*eps).add_(eps)
                timesteps = s.to(torch.float32).mul(999.0)

                s = s.view(-1, *([1] * (latents.dim() - 1)))  # broadcasting [B,1,1,1]

                noisy_latents = s * noise + (1 - s) * latents
                noise_target = noise - latents

            # --- UNet forward & loss ---
            model_pred = unet(noisy_latents, timesteps,
                              encoder_hidden_states=prompt_emb).sample

            mse = torch.nn.functional.mse_loss(
                    model_pred.float(), noise_target.float(), reduction="none")
            mse = mse.view(mse.size(0), -1).mean(dim=1)
            raw_mse_loss = mse.mean()

            if args.use_snr:
                snr = compute_snr(noise_sched, timesteps)
                gamma = args.noise_gamma
                gamma_tensor = torch.full_like(snr, gamma)
                weights = torch.minimum(snr, gamma_tensor) / (snr + 1e-8)
                loss = (weights * mse).mean()
            else:
                loss = raw_mse_loss

            if args.scale_loss_with_accum:
                loss = loss / args.gradient_accum

            accelerator.wait_for_everyone()
            accelerator.backward(loss)

        # -----logging & ckp save  ----------------------------------------- #
        if accelerator.is_main_process:

            for n, p in unet.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    print(f"NaN grad: {n}")

            current_lr = float(optim.param_groups[0]["lr"])
            if args.optimizer.startswith("d_"):
                current_lr *= float(optim.param_groups[0]["d"])

            if tb_writer is not None:
                # overly complicated if gr accum==1, but nice to skip an "if"
                accum_loss += loss.item()
                accum_mse += raw_mse_loss.item()


            pbar.set_description_str((
                    f"E{epoch}/{total_epochs}"
                    f"({batch_count:05})"  #PROBLEM HERE
                    ))
            pbar.set_postfix_str((f" l: {loss.item():.3f}"
                                  f" raw: {raw_mse_loss.item():.3f}"
                                  f" lr: {current_lr:.1e}"
                                  #f" qk: {qk_grad_sum:.1e}"
                                  #f" gr: {total_norm:.1e}"
                                  ))


        # Accelerate will make sure this only gets called on full-batch boundaries
        if accelerator.sync_gradients:
            accum_qk = sum(
                    p.grad.abs().mean().item()
                    for n,p in unet.named_parameters()
                    if p.grad is not None and (".to_q" in n or ".to_k" in n))
            total_norm = 0.0
            for p in unet.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            accum_norm = total_norm ** 0.5

            if tb_writer is not None:
                try:
                    tb_writer.add_scalar("train/learning_rate", current_lr, batch_count)
                    if args.use_snr:
                        tb_writer.add_scalar("train/loss_snr", accum_loss / args.gradient_accum, batch_count)
                    tb_writer.add_scalar("train/loss_raw", accum_mse / args.gradient_accum, batch_count)
                    tb_writer.add_scalar("train/qk_grads_av", accum_qk, batch_count)
                    tb_writer.add_scalar("train/grad_norm", accum_norm, batch_count)
                    accum_loss = 0.0; accum_mse = 0.0; accum_qk = 0.0; accum_norm = 0.0
                    tb_writer.add_scalar("train/epoch_progress", epoch_count / steps_per_epoch,  batch_count)
                except Exception as e:
                    print("Error logging to tensorboard")



            if args.gradient_clip is not None and args.gradient_clip > 0:
                accelerator.clip_grad_norm_(unet.parameters(), args.gradient_clip)


        global_step += 1
        # We have to take into account gradient accumilation!!
        # This is one reason it has to default to "1", not "0"
        if global_step % args.gradient_accum == 0:
            optim.step(); 
            optim.zero_grad()
            if not args.scheduler_at_epoch:
                lr_sched.step(); 
            pbar.update(1)
            batch_count += 1
            epoch_count += 1

            trigger_path = os.path.join(args.output_dir, "trigger.checkpoint")
            if os.path.exists(trigger_path):
                print("trigger.checkpoint detected. ...")
                # It is tempting to put this in the same place as the other save.
                # But, we want to include this one in the 
                #   "did we complete a full batch?"
                # logic
                checkpointandsave()
                try:
                    os.remove(trigger_path)
                except Exception as e:
                    print("warning: got exception", e)

            elif args.save_steps and (batch_count % args.save_steps == 0):
                if batch_count > int(args.save_start or 0):
                    print(f"Saving @{batch_count:05} (save every {args.save_steps} steps)")
                    checkpointandsave()


    ################## end of def train_micro_batch():



    # ----- training loop --------------------------------------------------- #
    """ Old way
    ebar = tqdm(range(math.ceil(max_steps / len(dl))), 
                desc="Epoch", unit="", dynamic_ncols=True,
                position=0,
                leave=True)
    """

    total_epochs = math.ceil(max_steps / steps_per_epoch)

    for epoch in range(total_epochs):
        if args.save_on_epoch:
            checkpointandsave()

        if args.scheduler_at_epoch:
            # Implement a stair-stepped decay, updating on epoch to what the smooth would be at this point
            lr_sched.step(batch_count)

        pbar = tqdm(range(steps_per_epoch),
                    desc=f"E{epoch}/{total_epochs}", 
                    bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} {rate_fmt}{postfix}", 
                    dynamic_ncols=True,
                    leave=True)

        epoch_count =  0

        # "batch" is actually micro-batch

        # yes this will stop at end of shortest dataset.
        # Every dataset will get equal value. I'm not messing around with
        #  custom "balancing"
        for batch in chain.from_iterable(zip(*dataloaders)):
            if batch_count >= max_steps:
                break
            train_micro_batch(unet)

        pbar.close()
        if batch_count >= max_steps:
            break

    if accelerator.is_main_process:
        if tb_writer is not None:
            tb_writer.close()
        if False:
            pipe.save_pretrained(args.output_dir, safe_serialization=True)
            sample_img(args.sample_prompt, args.seed, args.output_dir, 
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
