#!/usr/bin/env python

# train_with_caching.py



# --------------------------------------------------------------------------- #
# 1. CLI                                                                      #
# --------------------------------------------------------------------------- #
import argparse
def parse_args():
    p = argparse.ArgumentParser(epilog="Touch 'trigger.checkpoint' in the output_dir to dynamically trigger checkpoint save")
    p.add_argument("--pretrained_model", required=True,  help="HF repo or local dir")
    p.add_argument("--train_data_dir",  nargs="+", required=True,  help="Directory tree(s) containing *.jpg + *.txt")
    p.add_argument("--optimizer",      type=str, choices=["adamw8","lion"], default="adamw8")
    p.add_argument("--copy_config",    type=str, help="Config file to archive with training, if model load succeeds")
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--gradient_accum", type=int, default=1, help="Default=1")
    p.add_argument('--gradient_checkpointing', action='store_true',
                   help="Enable grad checkpointing in unet")
    p.add_argument("--learning_rate",   type=float, default=1e-5, help="Default=1e-5")
    p.add_argument("--epsilon",   type=float, default=1e-7, help="Default=1e-7")
    p.add_argument("--min_learning_rate",   type=float, default=0.1, help="Only used if 'min_lr' type schedulers are used")
    p.add_argument("--fp32", action="store_true",
                   help="Override default mixed precision bf16")
    p.add_argument("--is_custom", action="store_true",
                   help="Model provides a 'custom pipeline'")
    p.add_argument("--weight_decay",   type=float)
    p.add_argument("--vae_scaling_factor", type=float, help="Override vae scaling factor")
    p.add_argument("--text_scaling_factor", type=float, help="Override embedding scaling factor")
    p.add_argument("--learning_rate_decay", type=float,
                   help="Subtract this every epoch, if schedler==constant")
    p.add_argument("--max_steps",       default=10_000, 
                   help="Maximum EFFECTIVE BATCHSIZE steps(b * accum) default=10_000. May use '2e' for whole epochs")
    p.add_argument("--save_steps",    type=int, help="Measured in effective batchsize(b * a)")
    p.add_argument("--save_on_epoch", action="store_true")
    p.add_argument("--warmup_steps",    type=int, default=0, help="Measured in effective batchsize steps (b * a) default=0")
    p.add_argument("--noise_gamma",     type=float, default=5.0)
    p.add_argument("--cpu_offload", action="store_true",
                   help="Enable cpu offload at pipe level")
    p.add_argument("--use_snr", action="store_true",
                   help="Use Min SNR noise adjustments")

    p.add_argument("--reinit_crossattn", action="store_true",
                   help="Attempt to reset cross attention weights for text realign")
    p.add_argument("--reinit_attn", action="store_true",
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
    p.add_argument("--reinit_unet", action="store_true",
                   help="Train from scratch unet (Do not use, this is broken)")

    parser.add_argument( "--gradient_clip", type=float, default=1.0,
                        help="Max global grad norm. Set <=0 to disable gradient clipping.")

    p.add_argument("--targetted_training", action="store_true",
                   help="Only train reset layers")
    p.add_argument("--sample_prompt", nargs="+", type=str, help="Prompt to use for a checkpoint sample image")
    p.add_argument("--scheduler", type=str, default="constant", help="Default=constant")
    p.add_argument("--seed",        type=int, default=90)
    p.add_argument("--txtcache_suffix", type=str, default=".txt_t5cache", help="Default=.txt_t5cache")
    p.add_argument("--imgcache_suffix", type=str, default=".img_sdvae", help="Default=.img_sdvae")

    return p.parse_args()


# Put this super-early, so that usage message procs fast
args = parse_args()

# --------------------------------------------------------------------------- #

import os, math, shutil
from pathlib import Path
from tqdm.auto import tqdm

import torch
import safetensors.torch as st
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------- #
print("HAND HACKING FLOWMATCH MODULE")
from diffusers import FlowMatchEulerDiscreteScheduler
def scale_model_input(self, sample, timestep):
    return sample

FlowMatchEulerDiscreteScheduler.scale_model_input = scale_model_input
# --------------------------------------------------------------------------- #
    

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

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


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
        torch_dtype=torch.bfloat16,
    )
    pipe.safety_checker=None
    pipe.set_progress_bar_config(disable=True)

    pipe.enable_sequential_cpu_offload()
    generator = torch.Generator(device="cuda").manual_seed(seed)

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
    warmup_steps  = args.warmup_steps

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accum,
        mixed_precision="bf16" if not args.fp32 else "no",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )
    device = accelerator.device
    torch_dtype=torch.bfloat16 if accelerator.mixed_precision=="bf16" else torch.float32

    print("Training type:",torch_dtype)

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
            torch_dtype=torch_dtype
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
    elif args.reinit_attn:
        print("Attempting to reset Attn layers of Unet")
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
        print(f"Attempting to unfreeze (({args.unfreeze_down_blocks})) upblocks of Unet")
        from reinit import unfreeze_down_blocks
        unfreeze_down_blocks(pipe.unet, args.unfreeze_down_blocks, reset=False)
    if args.unfreeze_mid_block:
        print(f"Attempting to unfreeze mid block of Unet")
        from reinit import unfreeze_mid_block
        unfreeze_mid_block(pipe.unet)

    # ------------------------------------------ #

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
    print("Pipe wants to use noise scheduler", type(noise_sched))
    if isinstance(noise_sched, PNDMScheduler):
        print("Overriding noise scheduler from PNDMScheduler to DDPMScheduler")
        noise_sched = DDPMScheduler(
                num_train_timesteps=1000, # DIFFERENT from lr_sched num_training_steps
                # beta_schedule="cosine", # "cosine not implemented for DDPMScheduler"
                clip_sample=False
                )

    if hasattr(noise_sched, "add_noise"):
        print("DEBUG: add_noise present")
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



    # ----- load data ------------------------------------------------------------ #
    ds = CaptionImgDataset(args.train_data_dir, 
                           txtcache_suffix=args.txtcache_suffix,
                           imgcache_suffix=args.imgcache_suffix,
                           batch_size=args.batch_size,
                           gradient_accum=args.gradient_accum
                           )
    dl = DataLoader(ds, batch_size=args.batch_size, 
                    shuffle=True, drop_last=True,
                    num_workers=8, persistent_workers=True,
                    pin_memory=True, collate_fn=collate_fn,
                    prefetch_factor=4)

    bs = args.batch_size ; accum = args.gradient_accum
    effective_batch_size = bs * accum
    steps_per_epoch = len(dl) // accum # dl count already divided by mini batch
    if args.max_steps and args.max_steps.endswith("e"):
        max_steps = int(args.max_steps.removesuffix("e"))
        max_steps = max_steps * steps_per_epoch
    else:
        max_steps = int(args.max_steps)

    # Gather just-trainable parameters
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if args.optimizer == "lion":
        import lion_pytorch
        if args.weight_decay:
            weight_decay=args.weight_decay
        else:
        # lion doesnt use decay?
            weight_decay=0.00
        optim = lion_pytorch.Lion(trainable_params, lr=peak_lr, weight_decay=weight_decay, betas=(0.95,0.98))
        #optim = lion_pytorch.Lion(trainable_params, lr=peak_lr, weight_decay=0.00, betas=(0.93,0.95))
    elif args.optimizer == "adamw8":
        import bitsandbytes as bnb
        if args.weight_decay:
            weight_decay=args.weight_decay
        else:
            weight_decay=0.01
        optim = bnb.optim.AdamW8bit(trainable_params, weight_decay=weight_decay, lr=peak_lr, betas=(0.95,0.98))
    else:
        print("ERROR: unrecognized optimizer setting")
        exit(1)

    # -- optimizer settings...
    print("Using optimizer",args.optimizer,"weight decay:",weight_decay)
    if args.use_snr:
        if hasattr(noise_sched, "alphas_cumprod"):
            print(f"  Using MinSNR with gamma of {args.noise_gamma}")
        else:
            print("  Skipping --use_snr: invalid with scheduler", type(noise_sched))
            args.use_snr = False

    print(f"  NOTE: peak_lr = {peak_lr}, lr_scheduler={args.scheduler}, total steps={max_steps}(steps/Epoch={steps_per_epoch})")
    print(f"        batch={bs}, accum={accum}, effective batchsize={effective_batch_size}")

    unet, dl, optim = accelerator.prepare(pipe.unet, dl, optim)
    unet.train()

    scheduler_args = {
        "optimizer": optim,
        "num_warmup_steps": warmup_steps,
        "num_training_steps": max_steps,
    }

    if args.scheduler == "cosine_with_min_lr":
        scheduler_args["scheduler_specific_kwargs"] = {"min_lr_rate": args.min_learning_rate }
        print(f"  Setting default min_lr to {args.min_learning_rate}")

    lr_sched = get_scheduler(args.scheduler, **scheduler_args)
    lr_sched = accelerator.prepare(lr_sched)

    print(
        f"Align-phase: {sum(p.numel() for p in trainable_params)/1e6:.2f} M "
        "parameters will be updated"
    )

    global_step = 0 
    batch_count = 0
    accum_loss = 0.0; accum_mse = 0.0; accum_qk = 0.0; accum_norm = 0.0

    run_name = os.path.basename(args.output_dir)
    tb_writer = SummaryWriter(log_dir=os.path.join("tensorboard/",run_name))

    def checkpointandsave():
        if global_step % args.gradient_accum != 0:
            print("INTERNAL ERROR: checkpointandsave() not called on clean step")
            return

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{batch_count:05}")
        pinned_te, pinned_unet = pipe.text_encoder, pipe.unet
        pipe.unet = accelerator.unwrap_model(unet)
        log_unet_l2_norm(pipe.unet, tb_writer, batch_count)

        print(f"Saving checkpoint to {ckpt_dir}")
        pipe.save_pretrained(ckpt_dir, safe_serialization=True)
        pipe.text_encoder, pipe.unet = pinned_te, pinned_unet
        if args.sample_prompt is not None:
            sample_img(args.sample_prompt, args.seed, ckpt_dir, 
                       custom_pipeline)
            if global_step == 0:
                if args.copy_config:
                    tqdm.write(f"Archiving {args.copy_config}")
                    shutil.copy(args.copy_config, args.output_dir)

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

        pbar = tqdm(range(steps_per_epoch),
                    desc=f"E{epoch}/{total_epochs}", 
                    bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} {rate_fmt}{postfix}", 
                    dynamic_ncols=True,
                    leave=True)

        epoch_step =  0
        # "batch" is actually micro-batch
        for batch in dl:
            with accelerator.accumulate(unet):
                # --- Load latents & prompt embeddings from cache ---
                latents = []
                for cache_file in batch["img_cache"]:
                    latent = st.load_file(cache_file)["latent"]
                    latents.append(latent)
                latents = torch.stack(latents).to(device, dtype=torch_dtype) * latent_scaling

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
                    emb = emb.to(device, dtype=torch_dtype)
                    embeds.append(emb)
                prompt_emb = torch.stack(embeds).to(device, dtype=torch_dtype)

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
                    eps = args.epsilon  # avoid divide-by-zero magic
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

                accelerator.wait_for_everyone()
                accelerator.backward(loss)

            # -----logging & ckp save  ----------------------------------------- #
            if accelerator.is_main_process:
                qk_grad_sum = sum(
                        p.grad.abs().mean().item()
                        for n,p in unet.named_parameters()
                        if p.grad is not None and (".to_q" in n or ".to_k" in n))
                total_norm = 0.0
                for p in unet.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                for n, p in unet.named_parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        print(f"NaN grad: {n}")

                current_lr = lr_sched.get_last_lr()[0]
                pbar.set_postfix({"l": f"{loss.item():.3f}",
                                  "raw": f"{raw_mse_loss.item():.3f}",
                                  "qk": f"{qk_grad_sum:.1e}",
                                  "gr": f"{total_norm:.1e}",
                                  #"lr": f"{current_lr:.1e}",
                                  })

                if tb_writer is not None:
                    # overly complicated if gr accum==1, but nice to skip an "if"
                    accum_loss += loss.item()
                    accum_mse += raw_mse_loss.item()
                    accum_qk += qk_grad_sum
                    accum_norm += total_norm
                    if global_step % args.gradient_accum == 0:
                        tb_writer.add_scalar("train/learning_rate", current_lr, batch_count)
                        if args.use_snr:
                            tb_writer.add_scalar("train/loss_snr", accum_loss / args.gradient_accum, batch_count)
                        tb_writer.add_scalar("train/loss_raw", accum_mse / args.gradient_accum, batch_count)
                        tb_writer.add_scalar("train/qk_grads_av", accum_qk / args.gradient_accum, batch_count)
                        tb_writer.add_scalar("train/grad_norm", accum_norm / args.gradient_accum, batch_count)
                        accum_loss = 0.0; accum_mse = 0.0; accum_qk = 0.0; accum_norm = 0.0
                        tb_writer.add_scalar("train/epoch_progress", epoch_step / steps_per_epoch,  batch_count)

            # Accelerate will make sure this only gets called on full-batch boundaries
            if accelerator.sync_gradients and (args.gradient_clip is not None and args.gradient_clip > 0):
                    accelerator.clip_grad_norm_(unet.parameters(), args.gradient_clip)


            global_step += 1
            # We have to take into account gradient accumilation!!
            # This is one reason it has to default to "1", not "0"
            if global_step % args.gradient_accum == 0:
                optim.step(); lr_sched.step(); optim.zero_grad()
                pbar.update(1)
                batch_count += 1
                epoch_step += 1

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

                elif batch_count % args.save_steps == 0:
                    print(f"Saving @{batch_count:05} (save every {args.save_steps} steps)")
                    checkpointandsave()



            if batch_count >= max_steps:
                break

        # ----- end of "for batch in pbar" loop ------
        pbar.close()
        if batch_count >= max_steps:
            break

    if accelerator.is_main_process:
        if tb_writer is not None:
            tb_writer.close()
        pipe.save_pretrained(args.output_dir, safe_serialization=True)
        sample_img(args.sample_prompt, args.seed, args.output_dir, 
                   custom_pipeline)
        print(f"finished:model saved to {args.output_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting.")
        # just fall off end?
