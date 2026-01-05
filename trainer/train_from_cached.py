#!/usr/bin/env python

# train_from_cached.py


# --------------------------------------------------------------------------- #
# 1. CLI                                                                      #
# --------------------------------------------------------------------------- #

from train_args import parse_args
from train_multiloader import InfiniteLoader

# Put this super-early, so that usage message procs fast
args = parse_args()

# --------------------------------------------------------------------------- #

import os, math, shutil
from pathlib import Path
from tqdm.auto import tqdm

import torch
import safetensors.torch as st
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedDataParallelKwargs
from diffusers import DiffusionPipeline

from diffusers.training_utils import compute_snr

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

    # ----- load pipeline --------------------------------------------------- #

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

    # ----- load data ------------------------------------------------ #
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

    unet, dl, optim = accelerator.prepare(pipe.unet, mix_loader, optim)
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

    if args.gradient_topk:
        from train_grad_topk import sparsify_sd15_gradients

    global_step = 0  # micro-batch count
    batch_count = 0  # effective-batch-size count, aka num of fullsize batches
    accum_loss = 0.0
    accum_mse = 0.0
    accum_qk = 0.0
    accum_norm = 0.0
    latent_paths = []

    run_name = os.path.basename(args.output_dir)
    tb_writer = SummaryWriter(log_dir=os.path.join("tensorboard/", run_name))

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

        savefile = os.path.join(ckpt_dir, "latent_paths")
        with open(savefile, "w") as f:
            f.write('\n'.join(latent_paths) + '\n')
            f.close()
        print("Wrote", len(latent_paths), "loglines to", savefile)
        latent_paths = []

    #######################################################
    #    Core training code. Very Long!!                  #
    #######################################################
    def train_micro_batch(unet, batch):
        nonlocal batch_count, global_step, accum_loss, accum_mse, accum_qk, accum_norm, epoch_count
        nonlocal latent_paths

        with accelerator.accumulate(unet):
            # --- Load latents & prompt embeddings from cache ---
            latents = []
            for cache_file in batch["img_cache"]:
                latent = st.load_file(cache_file)["latent"]
                latent_paths.append(cache_file)
                latents.append(latent)
            try:
                latents = torch.stack(latents).to(device, dtype=compute_dtype) * latent_scaling
            except RuntimeError as e:
                print("Problem loading this latest batch")
                print(e)
                print(batch)
                exit(1)

            embeds = []
            for cache_file in batch["txt_cache"]:
                if args.force_txtcache:
                    cache_file = args.force_txtcache
                if Path(cache_file).suffix == ".h5":
                    raise NotImplementedError(cache_file, "not supported yet")
                    # arr = h5f["emb"][:]
                    # emb = torch.from_numpy(arr)
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
                prompt_emb = torch.stack(fixed, dim=0).to(device, dtype=compute_dtype)  # [B, MAX_TOK, D]
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

                s = torch.rand(bsz, device=device).mul_(1 - 2 * eps).add_(eps)
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
            if args.gradient_topk:
                sparsify_sd15_gradients(unet, keep_frac=args.gradient_topk)

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
                f"E{epoch_count}/{total_epochs}"
                f"({batch_count:05})"  # PROBLEM HERE
            ))
            pbar.set_postfix_str((f" l: {loss.item():.3f}"
                                  f" raw: {raw_mse_loss.item():.3f}"
                                  f" lr: {current_lr:.1e}"
                                  # f" qk: {qk_grad_sum:.1e}"
                                  # f" gr: {total_norm:.1e}"
                                  ))

        # Accelerate will make sure this only gets called on full-batch boundaries
        if accelerator.sync_gradients:
            accum_qk = sum(
                p.grad.abs().mean().item()
                for n, p in unet.named_parameters()
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
                    accum_loss = 0.0
                    accum_mse = 0.0
                    accum_qk = 0.0
                    accum_norm = 0.0
                    tb_writer.add_scalar("train/epoch_progress", epoch_count / ebs_steps_per_epoch, batch_count)
                except Exception as e:
                    print("Error logging to tensorboard")

            if args.gradient_clip is not None and args.gradient_clip > 0:
                accelerator.clip_grad_norm_(unet.parameters(), args.gradient_clip)

        global_step += 1
        # We have to take into account gradient accumilation!!
        # This is one reason it has to default to "1", not "0"
        if global_step % args.gradient_accum == 0:
            optim.step()
            optim.zero_grad()
            if not args.scheduler_at_epoch:
                lr_sched.step()
            pbar.update(1)
            batch_count += 1

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
                if batch_count >= int(args.save_start):
                    print(f"Saving @{batch_count:05} (save every {args.save_steps} steps)")
                    checkpointandsave()

    ###################################################
    #     end of def train_micro_batch()
    ###################################################

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
            lr_sched.step(batch_count)

        pbar = tqdm(range(ebs_steps_per_epoch),
                    desc=f"E{epoch_count}/{total_epochs}",
                    bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} {rate_fmt}{postfix}",
                    dynamic_ncols=True,
                    leave=True)

        # "batch" is actually micro-batch
        # yes this will stop at end of shortest dataset.
        # Every dataset will get equal value. I'm not messing around with
        #  custom "balancing"
        for _ in range(micro_steps_per_epoch):
            if batch_count >= max_steps:
                break
            step, batch = next(mix_iter)
            try:
                # this bumps batch_count only for EBS size
                train_micro_batch(unet, batch)
            except torch.OutOfMemoryError as e:
                print("OUT OF VRAM Problem in Batch", batch)
                exit(0)

        pbar.close()
        if batch_count >= max_steps:
            break

    if accelerator.is_main_process:
        if tb_writer is not None:
            tb_writer.close()
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
