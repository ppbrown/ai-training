# reinit_qk.py
#
# Re-initialise the UNet *cross-attention* query/key projections only.
# keeps all convolutions, value/out projections, etc., intact
# works with SD 1.5 UNet loaded through diffusers >=0.26
#
# Usage:
#   from diffusers import DiffusionPipeline
#   pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.bfloat16)
#   reinit_qk(pipe.unet)          # hard reset (Xavier)    
#   reinit_qk(pipe.unet, method="noise", sigma=0.1)  # soft reset (Gaussian noise)

import torch
from torch import nn
import torch.nn.init as init
from diffusers.models.attention import Attention as CrossAttention


# Internal util
def make_trainable(module):
    for param in module.parameters():
        param.requires_grad = True

def _xavier(m: nn.Linear) -> None:
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

def _add_noise(m: nn.Linear, sigma: float) -> None:
    with torch.no_grad():
        std = m.weight.std().item()
        m.weight.add_(torch.randn_like(m.weight) * sigma * std)
        if m.bias is not None:
            m.bias.add_(torch.randn_like(m.bias) * sigma * std)


def reinit_qk(unet, *, method: str = "xavier", sigma: float = 0.1) -> None:
    """
    Reset (or perturb) the to_q / to_k projections in every CrossAttention
    layer that is actually used for **text-conditioning**.

    Args:
        unet   - diffusers.models.UNet2DConditionModel
        method - "xavier" (hard reset) or "noise" (additive Gaussian)
        sigma  - stdev factor for the noise method (default 0.1)

    Returns: None (UNet modified in-place)
    """

    if method.lower() == "all":
        print("ERROR: should not be calling reinit_qk with 'all' mode")
        exit(1)
        return

    hard = method.lower() == "xavier"
    affected = 0

    for mod in unet.modules():
        if isinstance(mod, CrossAttention) and getattr(mod, "is_cross_attention", False):
            for proj_name in ("to_q", "to_k"):
                proj: nn.Linear | None = getattr(mod, proj_name, None)
                if proj is None:
                    continue
                if hard:
                    _xavier(proj)
                else:
                    _add_noise(proj, sigma)
                affected += proj.weight.numel()

    print(f"[reinit_qk] updated {affected/1e6:.2f} M params in query/key projections")


def reinit_cross_attention(unet, *, method: str = "xavier", sigma: float = 0.1) -> None:
    """
    Reinitialize q, k, v, and output projections for all cross-attention modules in the UNet.
    """
    hard = method.lower() == "xavier"
    affected = 0

    for mod in unet.modules():
        if isinstance(mod, CrossAttention) and getattr(mod, "is_cross_attention", False):
            for proj_name in ("to_q", "to_k", "to_v"):
                proj: nn.Linear | None = getattr(mod, proj_name, None)
                if proj is None:
                    continue
                if hard:
                    _xavier(proj)
                else:
                    _add_noise(proj, sigma)
                affected += proj.weight.numel()
            # Output projection
            proj = getattr(mod, "to_out", None)
            if proj is not None:
                # If Sequential, usually first is Linear
                if isinstance(proj, nn.Sequential) and isinstance(proj[0], nn.Linear):
                    if hard:
                        _xavier(proj[0])
                    else:
                        _add_noise(proj[0], sigma)
                    affected += proj[0].weight.numel()
                elif isinstance(proj, nn.Linear):
                    if hard:
                        _xavier(proj)
                    else:
                        _add_noise(proj, sigma)
                    affected += proj.weight.numel()

    print(f"[reinit_crossattention] updated {affected/1e6:.2f} M params in q/k/v/out projections")


# Similar to reinit_crossattention but more targetted
#  (attn2.to_out[0])
def reinit_cross_attention_outproj(model):
    """
    SD1.5-only: zero ALL cross-attention output projections:
      '*.attn2.to_out.*.{weight,bias}'
    Call once on a fresh run (NOT when resuming).
    """
    affected = 0
    hits_w = hits_b = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".attn2.to_out." in name and (name.endswith(".weight") or name.endswith(".bias")):
                nn.init.zeros_(p)
                affected += p.numel()
    print(f"[reinit_cross_attention_outproj] zeroed {affected/1e6:.2f} M params ")



# I think "cross attention" is specifically for text emb mapping.
# Whereas self attention is for interpreting latent noisy images
def reinit_all_attention(
    unet,
    method: str = "xavier",
    sigma: float = 0.1,
    cross: bool = True,
    self_attn: bool = True,
):
    """
    Reinitialize q, k, v, and output projections for all attention modules in the UNet.
    By default, both cross-attention and self-attention modules are reinitialized.
    Set `cross` or `self_attn` to False to skip that type.
    """
    hard = method.lower() == "xavier"
    affected = 0

    for mod in unet.modules():
        # Only target modules that have is_cross_attention attribute (typical in SD1.5 code)
        if hasattr(mod, "is_cross_attention"):
            # Decide if this attention should be reset
            if (mod.is_cross_attention and cross) or (not mod.is_cross_attention and self_attn):
                make_trainable(mod)
                # q, k, v
                for proj_name in ("to_q", "to_k", "to_v"):
                    proj = getattr(mod, proj_name, None)
                    if proj is None:
                        continue
                    if hard:
                        torch.nn.init.xavier_uniform_(proj.weight)
                        if proj.bias is not None:
                            torch.nn.init.zeros_(proj.bias)
                    else:
                        proj.weight.data += torch.randn_like(proj.weight) * sigma
                        if proj.bias is not None:
                            proj.bias.data += torch.randn_like(proj.bias) * sigma
                    affected += proj.weight.numel()
                # Output projection
                proj = getattr(mod, "to_out", None)
                if proj is not None:
                    if isinstance(proj, torch.nn.Sequential) and isinstance(proj[0], torch.nn.Linear):
                        if hard:
                            torch.nn.init.xavier_uniform_(proj[0].weight)
                            if proj[0].bias is not None:
                                torch.nn.init.zeros_(proj[0].bias)
                        else:
                            proj[0].weight.data += torch.randn_like(proj[0].weight) * sigma
                            if proj[0].bias is not None:
                                proj[0].bias.data += torch.randn_like(proj[0].bias) * sigma
                        affected += proj[0].weight.numel()
                    elif isinstance(proj, torch.nn.Linear):
                        if hard:
                            torch.nn.init.xavier_uniform_(proj.weight)
                            if proj.bias is not None:
                                torch.nn.init.zeros_(proj.bias)
                        else:
                            proj.weight.data += torch.randn_like(proj.weight) * sigma
                            if proj.bias is not None:
                                proj.bias.data += torch.randn_like(proj.bias) * sigma
                        affected += proj.weight.numel()
    print(f"Reinitialized {affected} attention parameters ({'cross' if cross else ''}{' & ' if cross and self_attn else ''}{'self' if self_attn else ''}-attention).")


# This resets the "first and last", aka in and out, convolution layers,
# AND ALSO CALLS THE ATTENTION LAYER RESET
def reinit_outer_unet(unet):

    reinit_all_attention(unet)
    # Reinit first and last conv layers
    #nninit = torch.nn.init.xavier_uniform_ if method == "xavier" else torch.nn.init.kaiming_uniform_
    nninit = torch.nn.init.kaiming_uniform_
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Conv2d) and ("conv_in" in name or "conv_out" in name):
            nninit(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            print(f"Reinitialized {name}")
            make_trainable(module)

# randomize ALL unet weights not just qk
def reinit_all_unet(unet):
    for m in unet.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        make_trainable(m)
    print("[reinit_all_unet] All weights reinitialized.")

#---------------------------------------------------
# sections below are for trying to train on FlowMatch
# (maybe could be used for other things in future)
def retrain_time(unet, reset=True):
    """Reset & unfreeze time embedding head. If reset=False, just unfreeze."""

    if hasattr(unet, 'time_embedding'):
        for m in unet.time_embedding.modules():
            if hasattr(m, 'reset_parameters') and reset:
                m.reset_parameters()
        make_trainable(unet.time_embedding)
    elif hasattr(unet, 'time_proj'):
        for m in unet.time_proj.modules():
            if hasattr(m, 'reset_parameters') and reset:
                m.reset_parameters()
        make_trainable(unet.time_proj)

def retrain_in(unet, reset=True):
    """Reset & unfreeze input head. If reset=False, just unfreeze."""
    # Reset input conv layer
    if hasattr(unet, 'conv_in'):
        if reset:
            unet.conv_in.reset_parameters()
        make_trainable(unet.conv_in)
    """ Special wierd case. Not worth making this work at the moment.
        We dont need it
    elif hasattr(unet, 'in'):
        if reset:
            unet.in.reset_parameters()
        make_trainable(unet.in)
        """

def retrain_out(unet, reset=True):
    """Reset & unfreeze out head. If reset=False, just unfreeze."""
    # Reset final conv layer
    if hasattr(unet, 'conv_out'):
        if reset:
            unet.conv_out.reset_parameters()
        make_trainable(unet.conv_out)
    # This next bit is probably not needed
    elif hasattr(unet, 'out'):
        if reset:
            unet.out.reset_parameters()
        make_trainable(unet.out)

# train upblock(s)
def unfreeze_up_blocks(unet, blocknum: list[int], reset=False):
    if hasattr(unet, 'up_blocks'):
        for ndx in blocknum:
            upblock = unet.up_blocks[ndx]
            if reset:
                for m in upblock.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
            make_trainable(upblock)

def unfreeze_up_block(unet, n, reset=False):
    if hasattr(unet, 'up_blocks'):
        for upblock in unet.up_blocks[n]:
            if reset:
                for m in upblock.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
            make_trainable(upblock)

def unfreeze_down_blocks(unet, blocknum: list[int], reset=False):
    if hasattr(unet, 'down_blocks'):
        for ndx in blocknum:
            downblock = unet.down_blocks[ndx]
            if reset:
                for m in downblock.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
            make_trainable(downblock)

def unfreeze_mid_block(unet):
    if hasattr(unet, 'mid_block'):
        make_trainable(unet.mid_block)

# ----------------------------------------------------
# Compat defs
# Should probably just remove these
def reinit_deeper_time(unet):
    """
    Unfreeze noise schedule related blocks.
    This involves unfreezing last up/mid block.
    """
    retrain_time(unet, False)
    retrain_out(unet, False)
    unfreeze_up_mid_blocks(unet, 1)

def unfreeze_up_mid_blocks(unet, n, reset=False):
    """
    Unfreezes the last n up_blocks and the mid_block in the UNet.
    No longer unfreezes time embedding and out conv.
    For SD1.5, max n == 4
    """
    unfreeze_up_blocks(unet, n, reset)
    unfreeze_mid_block(unet)

# Norm modules we know, plus a generic "Norm" substring catch for custom classes
_NORM_TYPES = (
    nn.GroupNorm, nn.LayerNorm,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
)

def unfreeze_norms(unet, *, include_bias: bool = True) -> int:
    """
    Set requires_grad flag on all normalization parameters (γ/β) inside `unet
    These parameters are typically used for per-channel scale and shift, so useful
    to focus on them when doing things like a VAE swap

    Args:
        unet: root module to scan (e.g., a Diffusers UNet2DConditionModel).
        include_bias: also unfreeze bias terms if present.

    Returns:
        Count of tensors whose requires_grad was changed.
    """
    def _is_norm(mod: nn.Module) -> bool:
        # Catch both standard nn.*Norm and custom Diffusers norms (class name contains "Norm")
        return isinstance(mod, _NORM_TYPES) or ("Norm" in mod.__class__.__name__)

    toggled = 0
    for name, mod in unet.named_modules():
        if _is_norm(mod):
            if getattr(mod, "weight", None) is not None:
                mod.weight.requires_grad = True
                toggled += 1
            if include_bias and getattr(mod, "bias", None) is not None:
                mod.bias.requires_grad = True
                toggled += 1
    return toggled
def unfreeze_all_attention(unet):
    def is_attn(m):
        n = m.__class__.__name__
        return ("Attention" in n) or ("Transformer" in n)
    # freeze all first
    for p in unet.parameters(): p.requires_grad = False
    # unfreeze attention everywhere
    for _, m in unet.named_modules():
        if is_attn(m):
            for p in m.parameters():
                p.requires_grad = True
    # optional: keep I/O edges adapting a bit
    for m in (unet.conv_in, unet.conv_out):
        for p in m.parameters():
            p.requires_grad = True
