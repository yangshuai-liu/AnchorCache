from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Model-agnostic trainer settings only. Resolution, data paths, LoRA rank and teacher placement
# belong to the backend entry points.


@dataclass
class TrainCfg:
    out_dir: str
    max_steps: int

    lr: float = 1e-6
    wd: float = 0.03
    eps: float = 1e-10
    warmup: int = 50
    # Under DeepSpeed ZeRO-2, clipping is performed by gradient_clipping in the DS configuration;
    # the framework must not call clip_grad_norm_ again. Dispatch is in Engine.
    clip: float = 0.05

    accum: int = 1
    dist: str = "zero2"
    seed: int = 42
    start_step: int = 0

    log_every: int = 50
    ckpt_every: int = 0
    state_every: int = 0
    val_every: int = 0

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.dist in ("zero2", "ddp"), f"unknown dist={self.dist}"
        assert self.max_steps > 0
        assert self.accum >= 1
