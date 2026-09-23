from __future__ import annotations

import torch

# Reusable GPU buffers for K/V moved back from the host.


class BufferPool:
    """Reuse GPU buffers by (shape, dtype, device).

    Meaning of cap: at most "one in-flight layer (K+V) + one layer being released (K+V)"
    coexist outside the pool, hence the default of 4. Increasing it only raises resident GPU memory without improving speed.
    """

    def __init__(self, cap: int = 4):
        self.cap = cap
        self._free: list[torch.Tensor] = []
        self.hits = 0
        self.misses = 0

    def acquire(self, shape, dtype: torch.dtype, device) -> torch.Tensor:
        dev = torch.device(device)
        for i, buf in enumerate(self._free):
            if buf.shape == tuple(shape) and buf.dtype == dtype and buf.device == dev:
                self.hits += 1
                return self._free.pop(i)
        self.misses += 1
        return torch.empty(tuple(shape), dtype=dtype, device=dev)

    def acquire_like(self, like: torch.Tensor, device=None) -> torch.Tensor:
        return self.acquire(like.shape, like.dtype, device or like.device)

    def give(self, buf: torch.Tensor | None) -> None:
        if buf is None:
            return
        if len(self._free) < self.cap:
            self._free.append(buf)

    def clear(self) -> None:
        self._free.clear()

    def __len__(self) -> int:
        return len(self._free)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "free": len(self._free),
        }
