"""
hard_region_mining.py

Loss-guided crop mining for VAE reconstruction training.

The idea: after a full forward pass, compute a per-pixel error map between
the reconstruction and the original. Pool that map into patch-level scores,
then sample crop positions weighted by those scores. High-error regions get
sampled more often, giving the model stronger gradient signal exactly where
it's failing.

"""

# -----------------------------
# Tuning notes
# -----------------------------
#
# crop_mining_weight:
#   Start at 0.2-0.4. If global coherence suffers, reduce it.
#   If fine detail isn't improving, increase it.
#
# crop_size:
#   128 is the sweet spot.
#   Smaller = more focus on tiny regions but LPIPS gets unreliable below ~64.
#   Larger = less focused but more context for the VAE.
#
# num_crops:
#   3 adds ~3x the crop forward passes per step. On a 4090 with batch=1
#   and a 512x512 image, 128x128 crop passes are cheap. 4-6 is feasible.
#
# crop_temperature:
#   2.0 = meaningful bias toward hard regions, still samples easy regions sometimes.
#   3.0+ = almost always drills into the worst region. Can cause instability
#          if one pathological image dominates. Don't go above 4.0 early on.
#   1.0 = mild bias, close to uniform sampling. Good for a sanity check run.


import torch
import torch.nn.functional as F
from typing import Optional


class HardRegionMiner:
    def __init__(
        self,
        crop_size: int = 128,
        num_crops: int = 3,
        patch_size: int = 32,
        temperature: float = 2.0,
        min_error_weight: float = 0.1,
    ):
        """
        Args:
            crop_size:
                Side length of each square crop in pixels.
                128 is a good default — large enough for LPIPS to be meaningful,
                small enough that a watch face is a significant fraction of it.

            num_crops:
                How many crops to sample per training step.
                More crops = more signal but more compute.
                2-4 is a reasonable range.

            patch_size:
                Spatial resolution at which error is pooled for sampling weights.
                Smaller = finer-grained sampling. Must divide crop_size evenly.
                32 means we divide the image into 32x32 blocks and score each.

            temperature:
                Controls how sharply we bias toward high-error regions.
                1.0 = mild bias, 3.0+ = very aggressive (almost always picks
                the worst region). 2.0 is a reasonable default.

            min_error_weight:
                Floor weight for any patch, preventing total starvation of
                low-error regions. 0.1 means even easy regions get some chance.
        """
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.patch_size = patch_size
        self.temperature = temperature
        self.min_error_weight = min_error_weight

    @torch.no_grad()
    def _sample_crop_origins(
        self,
        error_map: torch.Tensor,
        image_h: int,
        image_w: int,
    ) -> list[tuple[int, int]]:
        """
        Given a per-pixel error map (1, 1, H, W), compute patch-level scores
        and sample `num_crops` crop top-left corners weighted by those scores.

        Returns list of (top, left) tuples.
        """
        c = self.crop_size
        p = self.patch_size

        # Pool error map into patch-level scores
        # avg_pool2d with kernel=patch_size gives one score per patch
        patch_scores = F.avg_pool2d(
            error_map.float(),
            kernel_size=p,
            stride=p,
            padding=0,
        )  # (1, 1, H//p, W//p)
        patch_scores = patch_scores.squeeze()  # (H//p, W//p)

        ph, pw = patch_scores.shape

        # How many patches fit within a valid crop origin?
        # A crop of size c starting at pixel (top, left) requires:
        #   top + c <= image_h  =>  top <= image_h - c
        # In patch coordinates that's patch row <= (image_h - c) // p
        max_patch_top = (image_h - c) // p
        max_patch_left = (image_w - c) // p

        if max_patch_top <= 0 or max_patch_left <= 0:
            # Image too small for the crop size — fall back to single center crop
            top = (image_h - c) // 2
            left = (image_w - c) // 2
            return [(top, left)] * self.num_crops

        # Restrict scores to valid origin region
        valid_scores = patch_scores[:max_patch_top, :max_patch_left]  # (max_pt, max_pl)
        valid_scores = valid_scores.flatten()  # (N,)

        # Softmax with temperature to get sampling probabilities
        # Higher temperature = sharper focus on worst regions
        probs = F.softmax(valid_scores * self.temperature, dim=0)

        # Apply minimum weight floor to avoid complete starvation
        probs = probs + self.min_error_weight
        probs = probs / probs.sum()

        # Sample crop indices
        indices = torch.multinomial(probs, num_samples=self.num_crops, replacement=True)

        origins = []
        num_cols = max_patch_left
        for idx in indices.tolist():
            patch_row = idx // num_cols
            patch_col = idx % num_cols
            # Convert patch origin to pixel origin
            top = patch_row * p
            left = patch_col * p
            origins.append((top, left))

        return origins

    def _extract_crop(
        self,
        tensor: torch.Tensor,
        top: int,
        left: int,
    ) -> torch.Tensor:
        """Extract a (B, C, crop_size, crop_size) crop from tensor."""
        c = self.crop_size
        return tensor[:, :, top:top + c, left:left + c]

    def crop_loss(
        self,
        vae,
        x: torch.Tensor,
        dec: torch.Tensor,
        l1_weight: float = 1.0,
        lpips_fn=None,
        lpips_weight: float = 0.0,
        use_ddp: bool = False,
    ) -> torch.Tensor:
        """
        Main entry point. Call this after your normal forward pass.

        Args:
            vae:        Your VAE model (used for re-encoding crops).
            x:          Original image batch (B, 3, H, W) in [-1, 1].
            dec:        Reconstructed image from full forward pass (B, 3, H, W).
            l1_weight:  Weight for L1 loss on crops.
            lpips_fn:   LPIPS loss function, or None to skip.
            lpips_weight: Weight for LPIPS loss on crops.
            use_ddp:    Whether model is wrapped in DDP.

        Returns:
            Scalar crop loss tensor (already weighted and averaged over crops).
        """
        B, C, H, W = x.shape
        c = self.crop_size

        if H < c or W < c:
            # Safety: if image is smaller than crop, skip mining
            return torch.tensor(0.0, device=x.device)

        # --- Step 1: Build error map from the already-computed full reconstruction ---
        # No gradient needed here — we're just using this to pick crop locations
        with torch.no_grad():
            # Per-pixel absolute error, averaged over channels: (B, 1, H, W)
            error_map = (dec.detach() - x).abs().mean(dim=1, keepdim=True)

        total_crop_loss = torch.tensor(0.0, device=x.device)
        crops_computed = 0

        for b in range(B):
            origins = self._sample_crop_origins(
                error_map[b:b+1],  # (1, 1, H, W)
                image_h=H,
                image_w=W,
            )

            for (top, left) in origins:
                # Crop the original and the reconstruction at this location
                x_crop = self._extract_crop(x[b:b+1], top, left)      # (1, 3, c, c)
                dec_crop = self._extract_crop(dec[b:b+1], top, left)   # (1, 3, c, c)

                # --- Step 2: Re-encode and re-decode the crop ---
                # This is the key: we do a fresh forward pass on just this crop,
                # so gradients flow through the VAE for this specific region.
                enc_crop = vae.module.encode(x_crop) if use_ddp else vae.encode(x_crop)
                latents_crop = enc_crop.latent_dist.mean
                redec_crop = (
                    vae.module.decode(latents_crop).sample
                    if use_ddp else
                    vae.decode(latents_crop).sample
                )

                # --- Step 3: Compute losses on the crop ---
                crop_l = torch.tensor(0.0, device=x.device)

                if l1_weight > 0:
                    crop_l = crop_l + l1_weight * F.l1_loss(redec_crop, x_crop)

                if lpips_fn is not None and lpips_weight > 0:
                    crop_l = crop_l + lpips_weight * lpips_fn(redec_crop, x_crop).mean()

                total_crop_loss = total_crop_loss + crop_l
                crops_computed += 1

        if crops_computed == 0:
            return torch.tensor(0.0, device=x.device)

        return total_crop_loss / crops_computed
