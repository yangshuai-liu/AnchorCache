from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Residency(Protocol):
    """Which layers remain resident on the GPU and which are offloaded to the host.

    This is the only runtime concept the model side needs to know, and it only needs to know "whether to prefetch
    the next layer," without knowing that pinned memory / streams / pools exist.
    """

    def on_gpu(self, layer: int, num_layers: int) -> bool: ...

    def prefetch_target(self, layer: int, num_layers: int) -> int | None: ...


@dataclass
class GpuResident:
    """All layers remain resident. FLUX and OmniVoice currently use this tier (they have no offload implementation)."""

    def on_gpu(self, layer: int, num_layers: int) -> bool:
        return True

    def prefetch_target(self, layer: int, num_layers: int) -> int | None:
        return None


@dataclass
class RollingResidency:
    """Offload the last offload_layers layers to the host and prefetch lookahead layers ahead.

    Why the "last N layers" rather than the "first N layers": forward proceeds from layer 0 to L-1, so layers
    nearer the end have more time available for H2D.

    When host_offload=False, the policy degenerates into GpuResident.
    """

    offload_layers: int = 0
    lookahead: int = 1
    host_offload: bool = True

    def on_gpu(self, layer: int, num_layers: int) -> bool:
        if not self.host_offload or self.offload_layers <= 0:
            return True
        off = min(self.offload_layers, num_layers - 1)
        return layer < num_layers - off

    def prefetch_target(self, layer: int, num_layers: int) -> int | None:
        nxt = layer + self.lookahead
        if nxt >= num_layers or self.on_gpu(nxt, num_layers):
            return None
        return nxt


@dataclass
class GpuWindow:
    """Rolling window: only [layer, layer + window) is on the GPU.

    Qwen accounting: with n=10 reference images, GPU KV drops from 56 GiB to ~1.6 GiB, with ~64 GiB of pinned
    host memory. window trades GPU memory against H2D bandwidth.
    """

    window: int = 3

    def on_gpu(self, layer: int, num_layers: int) -> bool:
        return layer < self.window

    def prefetch_target(self, layer: int, num_layers: int) -> int | None:
        nxt = layer + self.window
        return nxt if nxt < num_layers else None
