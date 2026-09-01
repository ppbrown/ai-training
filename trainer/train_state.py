
"""
Used to pass values between train_from_cached.py and
train_core.py
"""
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import argparse

@dataclass
class TrainState:
    # Expected init values
    args: argparse.Namespace
    compute_dtype:  Any
    device: torch.device
    latent_scaling: float
    noise_sched:  Any  # torch._inductor.scheduler.SchedulerBuffer



    # step counters
    global_step: int = 0           # micro-batch count
    batch_count: int = 0           # effective-batch-size count (optimizer steps)
    epoch_count: int = 0
    total_epochs: int = 0

    # adversarial (GAN) training. Both None unless --disc_weight > 0.
    disc: Any = None
    opt_d: Any = None

    # Frozen VAE, needed to decode latents to RGB for the pixel-space
    # discriminator. Set by main; unused when --disc_weight is 0.
    vae: Any = None

    # running accumulators (main-process only)
    accum_loss: float = 0.0
    accum_mse: float = 0.0
    accum_qk: float = 0.0
    accum_norm: float = 0.0
    accum_gloss: float = 0.0
    accum_dloss: float = 0.0
    accum_dweight: float = 0.0
    accum_dcount: int = 0

    # Last seen discriminator losses, for the progress bar only.
    # --disc_decode_batch skips any microbatch without enough qualifying
    # samples, so the critic sits idle on a good fraction of steps. Letting
    # the pbar fields appear and vanish on those makes the whole line jump.
    # These are at most a couple of microbatches stale, and deliberately
    # NOT cleared by reset_accums().
    last_gloss: float | None = None
    last_dloss: float | None = None

    # per-checkpoint artifact
    latent_paths: list[str] = field(default_factory=list)

    # data log sinks (set by main)
    pbar: tqdm | None = None
    tb_writer: SummaryWriter | None = None

    def reset_accums(self) -> None:
        self.accum_loss = 0.0
        self.accum_mse = 0.0
        self.accum_qk = 0.0
        self.accum_norm = 0.0
        self.accum_gloss = 0.0
        self.accum_dloss = 0.0
        self.accum_dweight = 0.0
        self.accum_dcount = 0
