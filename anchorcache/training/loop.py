from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from .config import TrainCfg
from .log import Log, PrintLog

# The sole training loop.
#
# Boundary: the framework does not touch anything beyond the three injection points.
#
#   step_fn(batch) -> dict     Loss and backward for one step. Usually a partial of
#                              Distiller.train_step — the framework does not know how many forwards
#                              there are, which device hosts the teacher, or whether loss is MSE or KL.
#   on_save(path)              Persist weights. save_pretrained / save_lora_adapter /
#                              custom formats are entirely the backend's responsibility.
#   on_val(step)               Validation. Image generation, benchmarks, and SIM-o stay outside the framework.
#
# Data is outside the framework too: loader is an already constructed Iterable, since batch preparation
# (embedding padding, VAE normalization, concatenating multiple references) is backend-specific.


@dataclass
class Trainer:
    cfg: TrainCfg
    engine: Any
    model: Any
    optim: Any
    sched: Any
    step_fn: Callable[[Any], dict]
    log: Log = field(default_factory=PrintLog)
    on_save: Callable[[str], None] | None = None
    on_val: Callable[[int], None] | None = None
    params: Any = None

    def __post_init__(self):
        self.step = self.cfg.start_step
        self._acc: dict[str, float] = {}
        self._n = 0
        self._t0 = time.time()

    def run(self, loader: Iterable[Any]) -> int:
        cfg = self.cfg
        for batch in loader:
            if self.step >= cfg.max_steps:
                break
            with self.engine.accumulate(self.model):
                metrics = self.step_fn(batch)
                loss = metrics["loss"]
                assert torch.isfinite(torch.as_tensor(loss)), (
                    f"non-finite loss at step {self.step}"
                )
                if self.engine.sync_grads:
                    self.engine.clip_grads(self._train_params(), cfg.clip)
                    self.optim.step()
                    self.sched.step()
                    self.optim.zero_grad(set_to_none=True)

            for k, v in metrics.items():
                self._acc[k] = self._acc.get(k, 0.0) + float(v)
            self._n += 1

            if self.engine.sync_grads:
                self.step += 1
                self._after_step()
        self._final()
        return self.step

    def _train_params(self):
        if self.params is not None:
            return self.params
        return [p for p in self.model.parameters() if p.requires_grad]

    def _after_step(self):
        cfg = self.cfg
        if cfg.log_every and self.step % cfg.log_every == 0:
            self._flush_log()
        if cfg.ckpt_every and self.step % cfg.ckpt_every == 0 and self.on_save is not None:
            self.engine.barrier()
            self.on_save(os.path.join(cfg.out_dir, f"ckpt-{self.step:06d}"))
        if cfg.state_every and self.step % cfg.state_every == 0:
            self._save_state()
        if cfg.val_every and self.step % cfg.val_every == 0 and self.on_val is not None:
            self.engine.barrier()
            if self.engine.is_main:
                self.on_val(self.step)
            self.engine.barrier()

    def _flush_log(self):
        n = max(self._n, 1)
        values = {f"train/{k}": v / n for k, v in self._acc.items()}
        values["train/lr"] = self.sched.get_last_lr()[0]
        values["train/sec_per_it"] = (time.time() - self._t0) / self.cfg.log_every
        self.log.log(values, self.step)
        # Reset each window; otherwise this becomes a running average from the start.
        self._acc = {}
        self._n = 0
        self._t0 = time.time()

    def _save_state(self):
        """Strictly continuous state: all ranks participate, and the step number resides with the state.

        Without persisting the step number, the lr phase and data position cannot be restored; the resumed
        curve appears continuous but is actually misaligned.
        """
        path = os.path.join(self.cfg.out_dir, "state-latest")
        self.engine.save_state(path)
        if self.engine.is_main:
            with open(os.path.join(path, "step.txt"), "w") as f:
                f.write(str(self.step))

    def _final(self):
        self.engine.barrier()
        if self.on_save is not None:
            self.on_save(os.path.join(self.cfg.out_dir, "ckpt-final"))
        self.log.close()
        self.engine.close()


def resume_step(path: str) -> int:
    with open(os.path.join(path, "step.txt")) as f:
        return int(f.read().strip())
