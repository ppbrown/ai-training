"""
discriminator.py

PatchGAN discriminator + hinge losses for VAE adversarial training.

Architecture credit: CompVis/taming-transformers (NLayerDiscriminator),
originally from junyanz/pytorch-CycleGAN-and-pix2pix.

Usage:
    from discriminator import NLayerDiscriminator, hinge_d_loss, generator_hinge_loss, weights_init

    # Once, at init time:
    disc = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).apply(weights_init).to(device)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.disc_lr, betas=(0.5, 0.9))

    # In training loop - generator step (VAE update):
    g_loss = generator_hinge_loss(disc(dec))
    loss = loss + args.disc_weight * g_loss

    # In training loop - discriminator step (disc update):
    opt_d.zero_grad(set_to_none=True)
    d_loss = hinge_d_loss(disc(x.detach()), disc(dec.detach()))
    d_loss.backward()
    opt_d.step()
"""

# -----------------------------------------------------------------------
# TUNING NOTES
# -----------------------------------------------------------------------

"""
disc_start:
    Delay adversarial training until your reconstruction losses have
    stabilized. 50k steps is conservative and safe. If you start too
    early the discriminator dominates before the VAE has learned basic
    reconstruction, causing collapse.

disc_weight:
    Start at 0.1. This is intentionally low - the discriminator is a
    nudge toward sharpness, not the primary training signal.
    Your L1 + LPIPS remain the accuracy anchor.
    Signs it's too high: global color/structure accuracy regresses.
    Signs it's too low: no sharpness improvement after 20-30k steps
    past disc_start.

disc_lr:
    Discriminator typically uses a higher lr than the VAE (2e-4 vs 8e-6).
    Also note betas=(0.5, 0.9) which is standard for GAN discriminators
    - different from your VAE optimizer betas=(0.9, 0.999).

disc_layers:
    3 = 70x70 receptive field. Standard, good for general texture.
    2 = 34x34 receptive field. More local, more focused on fine detail
        like skin pores and watch face. Worth trying if 3 layers gives
        the SDXL-style hallucination you're worried about.

IMPORTANT - dec.detach() in discriminator step:
    Notice the discriminator update uses dec.detach().
    This is critical - you do NOT want gradients flowing back through
    the VAE during the discriminator update. The VAE only gets gradient
    from the generator loss (Change A), not from the discriminator loss.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    """
    Initialize conv and batchnorm weights.
    Call via: discriminator.apply(weights_init)
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator.

    Judges overlapping image patches rather than the full image.
    This makes it sensitive to local texture (skin pores, watch face detail)
    rather than global image plausibility — which is exactly what we want.

    Args:
        input_nc:   Number of input channels (3 for RGB).
        ndf:        Base number of filters (64 is standard).
        n_layers:   Number of conv layers. More layers = larger receptive field.
                    3 layers -> ~70x70 patch receptive field (standard).
                    2 layers -> ~34x34 patch (more local, more texture-focused).
    """
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()

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
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final layer: output a patch map (not a single scalar)
        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


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
    Generator (VAE) hinge loss.
    Pushes the discriminator toward believing reconstructions are real.
    Call this during the VAE update step.
    """
    return -torch.mean(logits_fake)


def adopt_weight(weight: float, global_step: int, threshold: int = 0, value: float = 0.0) -> float:
    """
    Zero out a loss weight before a given step threshold.
    Use this to delay adversarial training until reconstruction losses
    have stabilized — prevents the discriminator from dominating early.

    Example:
        disc_weight = adopt_weight(args.disc_weight, step, threshold=args.disc_start)
    """
    if global_step < threshold:
        return value
    return weight
