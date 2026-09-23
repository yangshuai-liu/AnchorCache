from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# Logging. Trainer recognizes only this interface; PrintLog works without any tracker installed.


@runtime_checkable
class Log(Protocol):
    def log(self, values: dict[str, Any], step: int) -> None: ...
    def close(self) -> None: ...


class PrintLog:
    """Log to stdout."""

    def __init__(self, keys: tuple[str, ...] = ("loss",), enabled: bool = True):
        self.keys = keys
        self.enabled = enabled

    def log(self, values: dict[str, Any], step: int) -> None:
        if not self.enabled:
            return
        parts = [f"step {step}"]
        for k in self.keys:
            if k in values:
                parts.append(f"{k.split('/')[-1]} {values[k]:.5f}")
        dt = values.get("train/sec_per_it")
        if dt is not None:
            parts.append(f"{dt:.2f}s/it")
        print(" ".join(parts), flush=True)

    def close(self) -> None:
        pass


class WandbLog:
    """wandb. Initialization failures must be broadcast to all ranks; see Engine.broadcast_ok's docstring."""

    def __init__(self, engine, project: str, name: str, config: dict, mode: str = "offline"):
        self.engine = engine
        self.enabled = engine.is_main
        self.wandb = None
        err = None
        if self.enabled:
            import wandb

            self.wandb = wandb
            try:
                wandb.init(
                    project=project,
                    name=name,
                    config=config,
                    mode=mode,
                    settings=wandb.Settings(x_disable_stats=True),
                )
            except wandb.errors.CommError as exc:
                err = exc
        if not engine.broadcast_ok(err is None):
            if err is not None:
                raise err
            raise RuntimeError("rank 0 tracker initialization failed")

    def log(self, values: dict[str, Any], step: int) -> None:
        if self.enabled:
            self.wandb.log(values, step=step)

    def close(self) -> None:
        pass


class MultiLog:
    def __init__(self, *logs):
        self.logs = [x for x in logs if x is not None]

    def log(self, values: dict[str, Any], step: int) -> None:
        for x in self.logs:
            x.log(values, step)

    def close(self) -> None:
        for x in self.logs:
            x.close()
