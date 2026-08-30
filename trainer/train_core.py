# train_core.py

"""
Contains the "main loop" for training
"""

from train_state import TrainState

from pathlib import Path
import torch
from accelerate import Accelerator
import safetensors.torch as st

from diffusers.training_utils import compute_snr

from train_discriminator import (
    hinge_d_loss,
    generator_hinge_loss,
    calculate_adaptive_weight,
    adopt_weight,
    noise_level,
    predict_x0,
)


#######################################################
#    Core training code. Very Long!!                  #
#######################################################
def train_micro_batch(unet, accelerator: Accelerator, batch_paths, tstate: TrainState,
                      optim, lr_sched, ebs_steps_per_epoch):
    args = tstate.args
    noise_sched = tstate.noise_sched

    with (accelerator.accumulate(unet)):
        # --- Load latents & prompt embeddings from cache ---
        latents = []
        for cache_file in batch_paths["img_cache"]:
            latent = st.load_file(cache_file)["latent"]
            tstate.latent_paths.append(cache_file)
            latents.append(latent)
        try:
            latents = torch.stack(latents).to(
                tstate.device,
                dtype=tstate.compute_dtype) * tstate.latent_scaling
        except RuntimeError as e:
            print("Problem loading this latest batch")
            print(e)
            print(batch_paths)
            exit(1)

        embeds = []
        for cache_file in batch_paths["txt_cache"]:
            if args.force_txtcache:
                cache_file = args.force_txtcache
            if Path(cache_file).suffix == ".h5":
                raise NotImplementedError(cache_file, "not supported yet")
                # arr = h5f["emb"][:]
                # emb = torch.from_numpy(arr)
            else:
                try:
                    emb = st.load_file(cache_file)["emb"]
                except Exception:
                    print("Error loading these files...")
                    print(cache_file)
                    exit(0)
            emb = emb.to(tstate.device, dtype=tstate.compute_dtype)
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
            prompt_emb = torch.stack(fixed, dim=0).to(tstate.device, dtype=tstate.compute_dtype)  # [B, MAX_TOK, D]
        else:
            # Take easy path, if using CLIP cache or something where length is already forced
            prompt_emb = torch.stack(embeds).to(tstate.device, dtype=tstate.compute_dtype)

        # --- Add noise ---

        if hasattr(noise_sched, "add_noise"):
            # Standard DDPM/PNDM-style
            noise = torch.randn_like(latents)
            bsz = latents.size(0)
            timesteps = torch.randint(
                0, noise_sched.config.num_train_timesteps,
                (bsz,), device=tstate.device, dtype=torch.long
            )
            noisy_latents = noise_sched.add_noise(latents, noise, timesteps)
            noise_target = noise
            sigmas = None  # only the FlowMatch branch has one
        else:
            # Flow Matching: continuous s in [epsilon, 1 - epsilon]
            bsz = latents.size(0)
            eps = args.min_sigma  # avoid divide-by-zero magic
            noise = torch.randn_like(latents)

            s = torch.rand(bsz, device=tstate.device).mul_(1 - 2 * eps).add_(eps)
            timesteps = s.to(torch.float32).mul(999.0)

            s = s.view(-1, *([1] * (latents.dim() - 1)))  # broadcasting [B,1,1,1]

            noisy_latents = s * noise + (1 - s) * latents
            noise_target = noise - latents
            sigmas = s

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

        # --- Adversarial (GAN) term ------------------------------------- #
        # MSE alone converges on the conditional MEAN latent, and the mean
        # of every plausible fine detail is mush - which the VAE then
        # decodes faithfully as mush. This term pushes the prediction back
        # onto the real manifold, where the texture lives.
        # The judging happens in PIXEL space: the implied clean latent is
        # decoded to RGB and the discriminator looks at that, same as the
        # VAE trainer does.
        # See train_discriminator.py for the reasoning and tuning notes.
        disc_pair = None
        g_loss = None
        if tstate.disc is not None:
            d_weight = adopt_weight(args.disc_weight, tstate.batch_count,
                                    threshold=args.disc_start)
            # Warmup: the discriminator trains, but the UNet does not listen
            # yet. A freshly-initialized critic's opinion is close to noise,
            # and the adaptive weight would hand that noise the same gradient
            # norm as the diffusion loss. See train_discriminator.py.
            d_warmup = tstate.batch_count < args.disc_start + args.disc_warmup
            if d_weight > 0:
                # Only judge the low-noise tail. Higher up the schedule the
                # implied clean latent is a rough guess at global layout,
                # and the epsilon inversion amplifies its error badly.
                level = noise_level(noise_sched, timesteps, sigmas)
                keep = (level <= args.disc_max_noise).nonzero(as_tuple=True)[0]
                if keep.numel() > 0:
                    # The cached latents were multiplied by latent_scaling
                    # on load; the decoder wants them unscaled.
                    inv_scale = 1.0 / tstate.latent_scaling

                    # Decoded WITH grad once warmup is over: that is the path
                    # the adversarial gradient takes back into the UNet. The
                    # VAE's own parameters have requires_grad False, so
                    # nothing lands on the decoder itself.
                    # During warmup nothing reaches the UNet, so we skip
                    # building the graph at all. That decoder pass, plus the
                    # two adaptive-weight traversals of it, is very nearly
                    # the whole cost of this block - which is what makes a
                    # long warmup affordable.
                    with torch.set_grad_enabled(not d_warmup):
                        x0_pred = predict_x0(noise_sched, noisy_latents,
                                             model_pred, timesteps, sigmas)
                        fake = tstate.vae.decode(
                            x0_pred[keep] * inv_scale).sample

                    with torch.no_grad():
                        # Decode the REAL LATENT, not the source jpg. Both
                        # sides then carry an identical VAE round trip, so
                        # the discriminator cannot win by learning to spot
                        # decoder artifacts instead of judging texture.
                        real = tstate.vae.decode(
                            latents[keep].float() * inv_scale).sample

                    if not d_warmup:
                        # Freeze the discriminator's own weights for this
                        # pass. Gradient still flows THROUGH it into fake and
                        # back into the UNet; it just never lands on the
                        # discriminator itself.
                        tstate.disc.requires_grad_(False)
                        g_loss = generator_hinge_loss(tstate.disc(fake))

                        if args.disc_no_adaptive:
                            scale = d_weight
                        else:
                            scale = d_weight * calculate_adaptive_weight(
                                loss, g_loss, model_pred, index=keep)
                        loss = loss + scale * g_loss
                        tstate.accum_gloss += g_loss.item()
                        tstate.accum_dweight += float(scale)

                    # Held back for the discriminator's own update below.
                    # Both detached: the UNet must get nothing from d_loss.
                    # This happens during warmup too - training the critic is
                    # the entire point of that phase.
                    disc_pair = (real.detach(), fake.detach())
                    tstate.accum_dcount += 1

        if args.scale_loss_with_accum:
            loss = loss / args.gradient_accum

        accelerator.wait_for_everyone()
        accelerator.backward(loss)
        if args.gradient_topk:
            from train_grad_topk import sparsify_sd15_gradients
            sparsify_sd15_gradients(unet, keep_frac=args.gradient_topk)

        # Discriminator update. Both inputs are detached, so this backward
        # reaches only the discriminator's own weights.
        # Divided by gradient_accum by hand rather than going through
        # accelerator.backward(): the discriminator is deliberately not a
        # prepared model, and opt_d steps once per effective batch
        # alongside the UNet optimizer.
        if disc_pair is not None:
            real, fake = disc_pair
            tstate.disc.requires_grad_(True)
            d_loss = hinge_d_loss(tstate.disc(real), tstate.disc(fake))
            (d_loss / args.gradient_accum).backward()
            tstate.accum_dloss += d_loss.item()

    # -----logging & ckp save  ----------------------------------------- #
    if accelerator.is_main_process:

        for n, p in unet.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f"NaN grad: {n}")

        current_lr = float(optim.param_groups[0]["lr"])
        if args.optimizer.startswith("d_"):
            current_lr *= float(optim.param_groups[0]["d"])

        if tstate.tb_writer is not None:
            # overly complicated if gr accum==1, but nice to skip an "if"
            tstate.accum_loss += loss.item()
            tstate.accum_mse += raw_mse_loss.item()

        tstate.pbar.set_description_str((
            f"E{tstate.epoch_count}/{tstate.total_epochs}"
            f"({tstate.batch_count:05})"
        ))
        postfix = (f" l: {loss.item():.3f}"
                   f" raw: {raw_mse_loss.item():.3f}"
                   f" lr: {current_lr:.1e}")
        if disc_pair is not None:
            # g_loss is None while the discriminator is still warming up.
            if g_loss is not None:
                postfix += f" g: {g_loss.item():.2f}"
            postfix += f" d: {d_loss.item():.2f}"
        tstate.pbar.set_postfix_str(postfix)

    # Accelerate will make sure this only gets called on full-batch boundaries
    if accelerator.sync_gradients:
        tstate.accum_qk = sum(
            p.grad.abs().mean().item()
            for n, p in unet.named_parameters()
            if p.grad is not None and (".to_q" in n or ".to_k" in n))
        total_norm = 0.0
        for p in unet.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        tstate.accum_norm = total_norm ** 0.5

        try:
            if tstate.tb_writer is not None:
                tstate.tb_writer.add_scalar("train/learning_rate", current_lr, tstate.batch_count)
                if args.use_snr:
                    tstate.tb_writer.add_scalar("train/loss_snr", tstate.accum_loss / args.gradient_accum, tstate.batch_count)
                tstate.tb_writer.add_scalar("train/loss_raw", tstate.accum_mse / args.gradient_accum, tstate.batch_count)
                tstate.tb_writer.add_scalar("train/qk_grads_av", tstate.accum_qk, tstate.batch_count)
                tstate.tb_writer.add_scalar("train/grad_norm", tstate.accum_norm, tstate.batch_count)
                if tstate.accum_dcount > 0:
                    # Averaged over contributing microbatches only, not over
                    # gradient_accum: the noise-level filter means some
                    # microbatches have no qualifying samples at all.
                    ndisc = tstate.accum_dcount
                    tstate.tb_writer.add_scalar("disc/g_loss", tstate.accum_gloss / ndisc, tstate.batch_count)
                    tstate.tb_writer.add_scalar("disc/d_loss", tstate.accum_dloss / ndisc, tstate.batch_count)
                    tstate.tb_writer.add_scalar("disc/weight", tstate.accum_dweight / ndisc, tstate.batch_count)

            tstate.reset_accums()

            if tstate.tb_writer is not None:
                tstate.tb_writer.add_scalar("train/epoch_progress", tstate.epoch_count / ebs_steps_per_epoch, tstate.batch_count)
        except Exception:
            print("Error logging to tensorboard")

        if args.gradient_clip is not None and args.gradient_clip > 0:
            accelerator.clip_grad_norm_(unet.parameters(), args.gradient_clip)

    tstate.global_step += 1
    # We have to take into account gradient accumilation!!
    # This is one reason it has to default to "1", not "0"
    if tstate.global_step % args.gradient_accum == 0:
        optim.step()
        optim.zero_grad()
        if tstate.opt_d is not None:
            # No-op while batch_count < disc_start: nothing has run a
            # backward through the discriminator yet, so its grads are None
            # and AdamW skips those params.
            tstate.opt_d.step()
            tstate.opt_d.zero_grad(set_to_none=True)
        if not args.scheduler_at_epoch:
            lr_sched.step()
        tstate.pbar.update(1)
        tstate.batch_count += 1


###################################################
#     end of def train_micro_batch()
###################################################
