from __future__ import annotations

import contextlib
from typing import Any, Protocol, runtime_checkable

from .config import TrainCfg

# The sole abstraction for distributed execution / mixed precision / state persistence.
#
# It guards three mistakes that fail silently:
#
#   1. Duplicate clipping under ZeRO-2. DS already performs clipping internally through gradient_clipping;
#      calling accelerate's clip_grad_norm_ externally clips already-clipped gradients again. It raises no error,
#      but silently reduces the effective lr.
#   2. Forgetting no_sync during per-state backward. Every state triggers an all_reduce; the result
#      is correct but K times slower, with no obvious indication.
#   3. Not all ranks participate in save_state. DS optimizer state is sharded; allowing only rank0
#      to save produces an incomplete checkpoint that fails only upon resume.


@runtime_checkable
class Engine(Protocol):
    device: Any
    is_main: bool
    rank: int
    world: int
    local_rank: int
    local_world: int
    sync_grads: bool

    def prepare(self, model, optim): ...
    def accumulate(self, model): ...
    def backward(self, loss, last: bool = True) -> None: ...
    def clip_grads(self, params, max_norm: float) -> None: ...
    def unwrap(self, model): ...
    def save_state(self, path: str) -> None: ...
    def load_state(self, path: str) -> None: ...
    def barrier(self) -> None: ...


class AccelEngine:
    """accelerate backend with two modes, zero2 / ddp."""

    def __init__(self, cfg: TrainCfg):
        from accelerate import Accelerator
        from accelerate.utils import set_seed

        self.cfg = cfg
        if cfg.dist == "zero2":
            from accelerate.utils import DeepSpeedPlugin, GradientAccumulationPlugin

            ds = DeepSpeedPlugin(
                zero_stage=2,
                gradient_accumulation_steps=cfg.accum,
                gradient_clipping=cfg.clip,
            )
            ds.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
            self.acc = Accelerator(
                mixed_precision="bf16",
                gradient_accumulation_plugin=GradientAccumulationPlugin(
                    num_steps=cfg.accum, sync_each_batch=True
                ),
                deepspeed_plugin=ds,
            )
        else:
            from accelerate.utils import DistributedDataParallelKwargs

            self.acc = Accelerator(
                mixed_precision="bf16",
                gradient_accumulation_steps=cfg.accum,
                kwargs_handlers=[DistributedDataParallelKwargs(gradient_as_bucket_view=True)],
            )
        set_seed(cfg.seed + self.acc.process_index)
        self._model = None

    @property
    def device(self):
        return self.acc.device

    @property
    def is_main(self) -> bool:
        return self.acc.is_main_process

    @property
    def rank(self) -> int:
        return self.acc.process_index

    @property
    def world(self) -> int:
        return self.acc.num_processes

    @property
    def local_rank(self) -> int:
        return self.acc.local_process_index

    @property
    def local_world(self) -> int:
        import os

        return int(os.environ.get("LOCAL_WORLD_SIZE", self.acc.num_processes))

    @property
    def sync_grads(self) -> bool:
        return self.acc.sync_gradients

    def prepare(self, model, optim):
        model, optim = self.acc.prepare(model, optim)
        self._model = model
        return model, optim

    def register(self, obj):
        """Objects such as the lr scheduler that must enter the checkpoint."""
        self.acc.register_for_checkpointing(obj)

    def accumulate(self, model):
        return self.acc.accumulate(model)

    def backward(self, loss, last: bool = True) -> None:
        # Distiller uses last=False to disable intermediate gradient synchronization during per-state backward.
        if self.cfg.dist == "zero2" and self._model is not None:
            # accelerate's DeepSpeed wrapper steps on every backward while sync_gradients is set.
            ds = self.acc.deepspeed_engine_wrapped.engine
            boundary = last and self.acc.sync_gradients
            ds.set_gradient_accumulation_boundary(boundary)
            ds.backward(loss)
            if boundary:
                ds.step()
            return
        if last or self._model is None:
            self.acc.backward(loss)
            return
        with self.acc.no_sync(self._model):
            self.acc.backward(loss)

    def clip_grads(self, params, max_norm: float) -> None:
        if self.cfg.dist == "zero2":
            return  # DS has already performed gradient_clipping; see item 1 in the class docstring
        if not self.acc.sync_gradients:
            return
        self.acc.clip_grad_norm_(params, max_norm)

    def unwrap(self, model):
        return self.acc.unwrap_model(model)

    def state_dict_of(self, model):
        return self.acc.get_state_dict(model)

    def save(self, obj, path: str) -> None:
        self.acc.save(obj, path)

    def save_state(self, path: str) -> None:
        self.acc.wait_for_everyone()
        self.acc.save_state(path, exclude_frozen_parameters=True)

    def load_state(self, path: str) -> None:
        self.acc.load_state(path, load_module_strict=False)

    def barrier(self) -> None:
        self.acc.wait_for_everyone()

    def broadcast_ok(self, ok: bool) -> bool:
        """Other ranks may enter the first backward only after rank0's external tracker is ready.

        Actual failure mode: if rank0's wandb CommError is not broadcast, rank0 raises and exits,
        while the other ranks hang on the first collective until watchdog timeout without seeing
        the root cause.
        """
        import torch

        t = torch.tensor([1 if ok else 0], device=self.device, dtype=torch.uint8)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(t, src=0)
        return bool(t.item())

    def close(self) -> None:
        self.acc.end_training()


class LocalEngine:
    """Single process, without accelerate, with the same sync_grads / accum semantics."""

    def __init__(self, cfg: TrainCfg, device="cpu"):
        self.cfg = cfg
        self._device = device
        self._i = 0
        self._model = None
        self.clip_calls: list[float] = []

    @property
    def device(self):
        return self._device

    @property
    def is_main(self) -> bool:
        return True

    @property
    def rank(self) -> int:
        return 0

    @property
    def world(self) -> int:
        return 1

    @property
    def local_rank(self) -> int:
        return 0

    @property
    def local_world(self) -> int:
        return 1

    @property
    def sync_grads(self) -> bool:
        return self._i % self.cfg.accum == 0

    def prepare(self, model, optim):
        self._model = model
        return model, optim

    def register(self, obj):
        pass

    @contextlib.contextmanager
    def accumulate(self, model):
        self._i += 1
        yield

    def backward(self, loss, last: bool = True) -> None:
        loss.backward()

    def clip_grads(self, params, max_norm: float) -> None:
        if self.cfg.dist == "zero2" or not self.sync_grads:
            return
        import torch

        self.clip_calls.append(float(max_norm))
        torch.nn.utils.clip_grad_norm_(list(params), max_norm)

    def unwrap(self, model):
        return model

    def state_dict_of(self, model):
        return model.state_dict()

    def save(self, obj, path: str) -> None:
        import torch

        torch.save(obj, path)

    def save_state(self, path: str) -> None:
        import os

        os.makedirs(path, exist_ok=True)

    def load_state(self, path: str) -> None:
        pass

    def barrier(self) -> None:
        pass

    def broadcast_ok(self, ok: bool) -> bool:
        return ok

    def close(self) -> None:
        pass
