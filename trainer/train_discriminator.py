"""
train_discriminator.py

Pixel-space PatchGAN discriminator + hinge losses for adversarial UNet
training.

This is the diffusion-trainer sibling of vae/train_discriminator.py.
Same hinge losses, same taming-transformers adaptive weight, same hard
adopt_weight delay, same flag names, and - since the port off latents -
the same RGB input. Two things still differ, and both are forced by where
this one sits in the pipeline:

  1. It discriminates x0_pred, not the raw model output.
     The VAE hands its discriminator a reconstruction directly. There is
     no equivalent here: the UNet emits noise (or velocity), and noise
     has no texture to judge. So we analytically invert the noising step
     to recover the clean latent the model is implying, unscale it, and
     decode it to RGB. See predict_x0(), and the adversarial block in
     train_core.py for the decode itself.

  2. It only fires on low-noise samples. See disc_max_noise below.

REAL SIDE - decoded latent, not the source jpg:
    train_core decodes the cached REAL LATENT for the real half of the
    pair, rather than loading the original image. Both halves then carry
    an identical VAE round trip, which removes the round trip as a
    confounder: the discriminator has no decoder artifact to latch onto
    and has to actually judge texture. (This trainer could not load the
    jpg anyway - CaptionImgDataset only hands out cache paths.)

WHY THIS HELPS AT ALL, given the decoder is frozen:
    MSE on epsilon converges to the CONDITIONAL MEAN latent, and the mean
    of every plausible fine detail is mush. The decoder is then faithfully
    decoding mush. The discriminator is what pushes the prediction off the
    mean and back onto the real manifold, which is where texture lives.
    The creativity gets reintroduced in the UNet, not the decoder.

COST:
    The fake decode carries grad, so it is the expensive part of the step.
    Only the samples passing disc_max_noise get decoded (~25% at the 0.25
    default), and the real side runs under no_grad.
"""

# -----------------------------------------------------------------------
# TUNING NOTES
# -----------------------------------------------------------------------

"""
disc_weight:
    Master enable switch: 0.0 (default) means no discriminator is built
    at all, and nothing in this file runs.
    In the default adaptive mode, g_loss is first rescaled so its gradient
    into the UNet matches the diffusion-loss gradient norm, then
    disc_weight multiplies that. 0.5 is the standard LDM value; 1.0 means
    "as strong as the MSE signal". For a model that is already training
    sanely and just needs detail, start at 0.5.
    With --disc_no_adaptive, disc_weight is a fixed scale instead: start
    at 0.1.
    Signs it's too high: composition/color drift, sample images start
    growing texture that ignores the prompt.
    Signs it's too low: no sharpness improvement after a few thousand
    steps past disc_start.

disc_start:
    Measured in effective-batchsize steps (tstate.batch_count), same unit
    as --save_steps and --max_steps. Before it NOTHING adversarial runs -
    the discriminator is not even trained, it just sits at its random
    init. Note the difference from disc_warmup below.
    Default 0, because the normal use of this is a model that already
    trains sanely and only needs fine detail. If you are combining it with
    --reinit_unet or a heavy --reinit_*, set it to a few thousand so the
    UNet gets basic structure back before the discriminator has an
    opinion. Starting adversarial training against a model that cannot
    yet produce structure is the classic way to collapse it.

disc_warmup:
    Steps after disc_start during which the discriminator trains ALONE:
    it sees real vs fake and updates every step, but g_loss is not added
    to the UNet's loss. Default 500.
    Why this is not optional: at disc_start the critic is at random init,
    and with the adaptive weight on, its opinion is rescaled to carry the
    same gradient norm as the diffusion loss. A random critic's noise
    therefore arrives at FULL strength. The warmup is what buys it an
    opinion worth that weight.
    Why the default is generous rather than minimal: the failure it
    guards against is asymmetric. The first feature a fresh critic finds
    here is a high-frequency energy deficit, which is the right coarse
    signal - but a UNet can satisfy "add high frequencies" with GRAIN
    rather than with structure. Engaging too early buys a model that
    looks sharper without having learned anything, and per this project's
    history the tail MSE will not tell you that happened. Warmup steps are
    cheap (train_core skips the grad-carrying decode and both
    adaptive-weight passes), so overshooting costs little and
    undershooting costs a checkpoint series.
    Tuning it for real: set it absurdly high, say 5000, and watch
    disc/d_loss in tensorboard. At random init it sits at almost exactly
    1.0, since both hinge terms are relu(1 +/- ~0). Take the point where
    it has clearly left 1.0 and the slope has bent - not the point where
    it bottoms out, which just means the critic has solved a fake
    distribution the UNet is about to move off.
    disc/g_loss reads 0 for the whole warmup and jumps when it ends, so
    the transition is visible on the graph.

disc_max_noise:
    THIS IS THE ONE THAT MATTERS for fine detail. Normalized noise level,
    0.0 = clean latent, 1.0 = pure noise; covers both the DDPM timestep
    schedule and the FlowMatch sigma. Only samples at or below this level
    get an adversarial term.
    Rationale: at high noise, x0_pred is a blurry guess at global layout
    and dividing it out of the noised latent amplifies error enormously -
    an adversarial signal there is noise at best and destabilizing at
    worst. Fine detail is decided in the low-noise tail. 0.25 keeps the
    bottom quarter of the schedule.
    Raise toward 0.4 if you want the discriminator influencing midrange
    structure too. Lower toward 0.15 to make it purely a texture pass.
    Note the interaction with --batch_size: at 0.25 only about a quarter
    of each microbatch qualifies, so with batch_size 4 many microbatches
    contribute nothing. Gradient accumulation smooths this out; if you run
    accum 1 and batch_size 2, consider a larger cutoff.

disc_lr:
    Discriminator wants a much higher lr than the UNet (2e-4 vs 1e-5), and
    betas=(0.5, 0.9), which is standard for GAN discriminators and
    different from the UNet optimizer's betas.

disc_layers:
    Receptive field, counted in IMAGE pixels, same as the VAE trainer now
    that this operates on RGB:
      1 -> 16 px. Below the scale of the features we care about.
      2 -> 34 px. DEFAULT. About one eye at mid-distance portrait scale
           (eye ~40px, iris ~20px in a 512 frame), which is the target.
      3 -> 70 px.
    Note this default differs from the VAE trainer's 3. That is the VAE's
    whole-image setting; a 70px patch spans most of a face here and
    averages the eye back into its surroundings - re-diluting exactly the
    detail this whole path exists to sharpen.

BatchNorm vs GroupNorm:
    The VAE version uses BatchNorm2d. This one uses GroupNorm, because
    disc_max_noise means the discriminator often gets a batch of 1 or 2
    surviving samples, and real/fake go through in separate forward passes.
    BatchNorm under those conditions lets the discriminator cheat by
    reading batch statistics instead of texture.

--gradient_checkpointing:
    Nothing to do. The adaptive weight anchors on the UNet's output
    tensor rather than on conv_out's weight, so its two autograd.grad
    calls never enter the checkpointed graph at all. See
    calculate_adaptive_weight() for why that is also the more correct
    place to measure.

Single-process only:
    The discriminator is deliberately NOT passed through
    accelerator.prepare(). It stays a plain module on the device, which
    keeps the requires_grad flips below simple and avoids DDP complaining
    about the two-pass usage. The consequence is that its gradients are
    not all-reduced, so under real multi-GPU each rank would train its own
    copy and they would drift. Same single-process assumption the VAE
    trainer makes.

IMPORTANT - the two requires_grad flips:
    During the generator pass the discriminator's parameters are frozen.
    Gradient still flows THROUGH the discriminator into x0_pred and back
    into the UNet, which is the point, but it never lands on the
    discriminator's own weights. During the discriminator pass both
    inputs are detached, so the UNet gets nothing from the discriminator's
    loss. Both halves are needed; dropping either one trains the wrong
    model.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    """
    Initialize conv and norm weights.
    Call via: discriminator.apply(weights_init)
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1 or classname.find("GroupNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator, operating on decoded RGB.

    Judges overlapping local patches rather than the whole image, so it is
    sensitive to local texture rather than global plausibility - which is
    exactly what we want, since global plausibility is already the MSE
    term's job.

    Args:
        input_nc:   Number of input channels. 3, for RGB.
        ndf:        Base number of filters (64 is standard).
        n_layers:   Number of stride-2 conv layers. See the disc_layers
                    tuning note for the receptive fields; 2 is the
                    default.
    """
    def __init__(self, input_nc: int, ndf: int = 64, n_layers: int = 2):
        super().__init__()

        def norm(ch):
            # GroupNorm, not BatchNorm: see the tuning note. Batches here
            # are frequently 1-2 samples after the noise-level filter.
            return nn.GroupNorm(min(32, ch), ch)

        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=4, stride=2, padding=1, bias=False),
                norm(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=4, stride=1, padding=1, bias=False),
            norm(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final layer: output a patch map (not a single scalar)
        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always fp32, explicitly. The discriminator is tiny so this costs
        # nothing, and it means we never have to care whether accelerate's
        # bf16 autocast happens to be active at the call site - the UNet's
        # bf16 output would otherwise meet fp32 discriminator weights.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.model(x.float())


# -----------------------------
# Recovering the clean latent
# -----------------------------

def noise_level(noise_sched, timesteps, sigmas=None):
    """
    Normalized noise level per sample, 0.0 = clean, 1.0 = pure noise.

    Gives one comparable scale for both branches in train_core, so
    --disc_max_noise means the same thing whichever scheduler is loaded.
    Returns a flat [B] tensor.
    """
    if sigmas is not None:
        # FlowMatch: s is already in [0, 1] with the same orientation.
        return sigmas.reshape(sigmas.size(0)).float()
    steps = float(noise_sched.config.num_train_timesteps)
    return timesteps.float().reshape(timesteps.size(0)) / steps


def predict_x0(noise_sched, noisy_latents, model_pred, timesteps, sigmas=None):
    """
    Recover the clean latent the model is implying, from its raw output.

    The discriminator needs something with texture to look at, and neither
    an epsilon nor a velocity prediction has any. Inverting the forward
    noising step is exact and costs one elementwise expression, so there
    is no reason to approximate it.

    Done in fp32 regardless of compute dtype: the epsilon inversion
    divides by sqrt(alphas_cumprod), which gets small, and bf16 has no
    mantissa to spare there.
    """
    x = noisy_latents.float()
    p = model_pred.float()

    if sigmas is not None:
        # FlowMatch, matching train_core's construction:
        #   noisy = s*noise + (1-s)*x0 ,  target v = noise - x0
        #   noisy - s*v = s*noise + (1-s)*x0 - s*noise + s*x0 = x0
        return x - sigmas.float() * p

    acp = noise_sched.alphas_cumprod.to(device=x.device, dtype=torch.float32)
    acp = acp[timesteps]
    while acp.dim() < x.dim():
        acp = acp.unsqueeze(-1)
    sqrt_acp = acp.sqrt()
    sqrt_one_minus = (1.0 - acp).sqrt()

    ptype = getattr(noise_sched.config, "prediction_type", "epsilon")
    if ptype == "epsilon":
        return (x - sqrt_one_minus * p) / sqrt_acp
    if ptype == "v_prediction":
        return sqrt_acp * x - sqrt_one_minus * p
    if ptype == "sample":
        return p
    raise ValueError(f"predict_x0: unhandled prediction_type '{ptype}'")


# -----------------------------
# Hinge losses
# -----------------------------

def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """
    Discriminator hinge loss.
    Pushes real scores above +1 and fake scores below -1.
    Call this for the discriminator update step.
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def generator_hinge_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """
    Generator (UNet) hinge loss.
    Pushes the discriminator toward believing x0_pred is a real latent.
    Call this during the UNet update step.
    """
    return -torch.mean(logits_fake)


def calculate_adaptive_weight(
    rec_loss: torch.Tensor,
    g_loss: torch.Tensor,
    anchor: torch.Tensor,
    index=None,
    eps: float = 1e-4,
    max_weight: float = 1e4,
) -> torch.Tensor:
    """
    Adaptive generator-loss scale, taming-transformers style.

    Compares the gradient norms of the diffusion loss and the generator
    hinge loss and returns their ratio, so the adversarial gradient is
    scaled to match the diffusion gradient regardless of how small the MSE
    gets. Multiply the result by your base disc_weight:

        d_weight = disc_weight * calculate_adaptive_weight(
            loss, g_loss, model_pred, index=keep)
        loss = loss + d_weight * g_loss

    This matters more here than it does in the VAE trainer: only low-noise
    samples carry an adversarial term, and low-noise samples are exactly
    the ones whose MSE is already tiny. A fixed weight would drift in
    relative strength as training progresses; this does not.

    DIFFERENCE FROM THE VAE VERSION - anchor, not last_layer:
        The VAE version differentiates w.r.t. decoder.conv_out.weight,
        i.e. two extra partial backward passes THROUGH the model. That is
        fine for a VAE trained without gradient checkpointing. It is not
        fine here: with --gradient_checkpointing each of those passes
        re-runs the checkpointed forward, and reentrant checkpointing
        forbids it outright - torch.utils.checkpoint supports
        autograd.backward() but not autograd.grad().

        So we anchor on the UNet's OUTPUT tensor instead. Both losses
        reach the model only through model_pred, which makes that tensor
        the exact interface between loss-land and model-land, and
        balancing there is what the gradient ratio was always trying to
        express. Everything downstream of it - including conv_out's own
        weight gradient - is a linear function of what we measure here.

        Consequences: these two autograd.grad calls never touch the UNet.
        Gradient checkpointing becomes irrelevant rather than
        incompatible, and the balance no longer depends on conv_out being
        unfrozen, so partial --unfreeze_* runs need no special case.

        Not free, though, since the port to pixel space: the g_loss call
        has to traverse the VAE decoder to reach model_pred, so it costs
        one extra decoder backward per microbatch that has surviving
        samples. --disc_no_adaptive skips both calls entirely if that
        turns out to matter more than the balancing does.

    index:
        Optional subset of batch elements that the adversarial term
        actually applies to, since --disc_max_noise filters most of the
        batch out. Both norms are restricted to it, so the ratio measures
        per-sample strength on the samples the discriminator is actually
        judging. Without this the scale would swing around with however
        many samples happened to survive the filter this microbatch.

    Both losses must still be attached to the graph (call this before
    .backward()).
    """
    rec_grads = torch.autograd.grad(rec_loss, anchor, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, anchor, retain_graph=True)[0]
    if index is not None:
        rec_grads = rec_grads[index]
        g_grads = g_grads[index]
    # Norms in fp32: under bf16 these gradients are small enough to lose
    # the ratio to rounding.
    d_weight = torch.norm(rec_grads.float()) / (torch.norm(g_grads.float()) + eps)
    return torch.clamp(d_weight, 0.0, max_weight).detach()


def adopt_weight(weight: float, global_step: int, threshold: int = 0, value: float = 0.0) -> float:
    """
    Zero out a loss weight before a given step threshold.
    Use this to delay adversarial training until the model can produce
    basic structure. Prevents the discriminator from dominating early.

    Example:
        disc_weight = adopt_weight(args.disc_weight, step, threshold=args.disc_start)
    """
    if global_step < threshold:
        return value
    return weight
