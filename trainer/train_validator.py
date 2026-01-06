# train_validation.py

"""
Specialized 'validation image dataset' handling code.
The Validator class encapsulates the dataset.
  (with the help of build_val_dataloader() )
It also keeps track of training statistics.

Main loop should call Validator.maybe_run_validation() at
some desired frequency.
That will print out a user notice if results have failed to improve for some number
of samples.
"""

import torch
import safetensors.torch as st
from torch.utils.data import DataLoader

from diffusers.training_utils import compute_snr
from train_captiondata import CaptionImgDataset


def build_val_dataloader(args, batch_size, unsupervised, collate_fn):
    """
    Optional validation dataloader builder.
    Returns None if val_steps or val_dataset are not set.
    """
    if not getattr(args, "val_steps", None) or not getattr(args, "val_dataset", None):
        return None

    print("Validation enabled: using", args.val_dataset)
    val_ds = CaptionImgDataset(
        args.val_dataset,
        batch_size=batch_size,
        txtcache_suffix=args.txtcache_suffix,
        imgcache_suffix=args.imgcache_suffix,
        gradient_accum=1,  # no grad accum for val
        unsupervised=unsupervised,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2,
    )
    if len(val_dl) < 1:
        raise ValueError("Error: validation dataset invalid")

    return val_dl


class Validator:
    """
    Lightweight validation helper.

    - Holds a val dataloader and unet reference.
    - On maybe_run_validation(), runs forward-only val, logs metrics,
      and prints a suggestion to lower LR when a simple plateau heuristic triggers.
    """

    def __init__(
            self,
            args,
            device,
            compute_dtype,
            latent_scaling,
            noise_sched,
            val_dl,
            accelerator,
            tb_writer,
            val_min_improve=0.01,  # 1% relative improvement
            val_patience=2,  # plateau if no improvement for 2 validations
    ):
        self.args = args
        self.device = device
        self.compute_dtype = compute_dtype
        self.latent_scaling = latent_scaling
        self.noise_sched = noise_sched
        self.val_dl = val_dl
        self.accelerator = accelerator
        self.tb_writer = tb_writer

        self.val_best_loss = None
        self.val_plateau_count = 0
        self.val_min_improve = val_min_improve
        self.val_patience = val_patience

    def maybe_run_validation(self, ebs_steps: int, unet):
        """
        Call this from the training loop, only at the completion of
        a full EBS step.
        "ebs_steps" is "effective batch size" steps, not microbatch steps
        This will:
        - return immediately, if no val_dl loaded
        - return immediately, if not at configured multiple of steps for validation
        - Run validation only on main process.
        - Print a "Propose that you lower LR now" message on plateau.
        """
        if self.val_dl is None:
            return
        if not getattr(self.args, "val_steps", None):
            return
        if ebs_steps == 0:
            return
        if ebs_steps % self.args.val_steps != 0:
            return

        self._run_validation(ebs_steps, unet)

    def _run_validation(self, ebs_steps: int, unet):
        """
        Note: "ebs_steps" is "effective batch size" steps, not microbatch steps
        """
        if not self.accelerator.is_main_process:
            return

        unet_was_training = unet.training
        unet.eval()

        total_loss = 0.0
        total_raw_mse = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in self.val_dl:
                # --- Load latents ---
                latents = []
                for cache_file in batch["img_cache"]:
                    latent = st.load_file(cache_file)["latent"]
                    latents.append(latent)
                latents = torch.stack(latents).to(
                    self.device, dtype=self.compute_dtype
                ) * self.latent_scaling

                # --- Load text embeddings (mirror train_micro_batch) ---
                embeds = []
                for cache_file in batch["txt_cache"]:
                    cf = cache_file
                    if self.args.force_txtcache:
                        cf = self.args.force_txtcache
                    emb = st.load_file(cf)["emb"]
                    emb = emb.to(self.device, dtype=self.compute_dtype)
                    embeds.append(emb)

                if self.args.force_toklen:
                    MAX_TOK = self.args.force_toklen
                    D = embeds[0].size(1)
                    fixed = []
                    for e in embeds:
                        T = e.size(0)
                        if T >= MAX_TOK:
                            fixed.append(e[:MAX_TOK])
                        else:
                            pad = torch.zeros(
                                (MAX_TOK - T, D),
                                dtype=e.dtype,
                                device=e.device,
                            )
                            fixed.append(torch.cat([e, pad], dim=0))
                    prompt_emb = torch.stack(fixed, dim=0).to(
                        self.device, dtype=self.compute_dtype
                    )
                else:
                    prompt_emb = torch.stack(embeds).to(
                        self.device, dtype=self.compute_dtype
                    )

                # --- Noise + forward (same structure as train_micro_batch, no grad) ---
                if hasattr(self.noise_sched, "add_noise"):
                    noise = torch.randn_like(latents)
                    bsz = latents.size(0)
                    timesteps = torch.randint(
                        0,
                        self.noise_sched.config.num_train_timesteps,
                        (bsz,),
                        device=self.device,
                        dtype=torch.long,
                    )
                    noisy_latents = self.noise_sched.add_noise(latents, noise, timesteps)
                    noise_target = noise
                else:
                    bsz = latents.size(0)
                    eps = self.args.min_sigma
                    noise = torch.randn_like(latents)

                    s = torch.rand(bsz, device=self.device).mul_(1 - 2 * eps).add_(eps)
                    timesteps = s.to(torch.float32).mul(999.0)
                    s = s.view(-1, *([1] * (latents.dim() - 1)))
                    noisy_latents = s * noise + (1 - s) * latents
                    noise_target = noise - latents

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_emb,
                ).sample

                mse = torch.nn.functional.mse_loss(
                    model_pred.float(), noise_target.float(), reduction="none"
                )
                mse = mse.view(mse.size(0), -1).mean(dim=1)
                raw_mse_loss = mse.mean()

                if self.args.use_snr:
                    snr = compute_snr(self.noise_sched, timesteps)
                    gamma = self.args.noise_gamma
                    gamma_tensor = torch.full_like(snr, gamma)
                    weights = torch.minimum(snr, gamma_tensor) / (snr + 1e-8)
                    loss = (weights * mse).mean()
                else:
                    loss = raw_mse_loss

                total_loss += loss.item()
                total_raw_mse += raw_mse_loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        avg_raw_mse = total_raw_mse / max(total_batches, 1)

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("val/loss", avg_loss, ebs_steps)
            self.tb_writer.add_scalar("val/loss_raw", avg_raw_mse, ebs_steps)

        # Simple plateau detection
        if self.val_best_loss is None:
            self.val_best_loss = avg_loss
            self.val_plateau_count = 0
        else:
            rel_improve = (self.val_best_loss - avg_loss) / max(self.val_best_loss, 1e-8)
            if rel_improve > self.val_min_improve:
                self.val_best_loss = avg_loss
                self.val_plateau_count = 0
            else:
                self.val_plateau_count += 1

        if self.val_plateau_count >= self.val_patience:
            print(
                f"[VALIDATION] Plateau detected (avg_loss={avg_loss:.4f}, "
                f"best={self.val_best_loss:.4f})."
            )
            print(f"[VALIDATION] Propose that you lower LR now, at step {ebs_steps}")
            # Avoid spamming: only after next improvement+plateau will it print again
            self.val_plateau_count = 0

        if unet_was_training:
            unet.train()
