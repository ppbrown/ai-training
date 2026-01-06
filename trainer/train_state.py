
"""
Used to pass values between train_from_cached.py and
train_core.py
"""
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


@dataclass
class TrainState:
    # step counters
    global_step: int = 0           # micro-batch count
    batch_count: int = 0           # effective-batch-size count (optimizer steps)
    epoch_count: int = 0
    total_epochs: int = 0

    # running accumulators (main-process only)
    accum_loss: float = 0.0
    accum_mse: float = 0.0
    accum_qk: float = 0.0
    accum_norm: float = 0.0

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
