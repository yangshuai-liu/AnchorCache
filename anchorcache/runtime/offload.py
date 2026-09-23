from __future__ import annotations

import torch

from ..transformer.kv import KVState

# Host-side copies of offloaded K/V.


class HostStore:
    """KV offload destination in pinned host memory.

    pin_memory is required: non-pinned H2D cannot use DMA or non_blocking transfers,
    causing the entire prefetch overlap to degrade into synchronous copies.
    """

    def __init__(self):
        self._slots: dict = {}

    def put(self, key, kv: KVState) -> None:
        host = self._slots.get(key)
        if host is None or host.key.shape != kv.key.shape or host.key.dtype != kv.key.dtype:
            # pin_memory=True raises without CUDA.
            pinned = torch.cuda.is_available()
            host = KVState(
                torch.empty_like(kv.key, device="cpu", pin_memory=pinned),
                torch.empty_like(kv.value, device="cpu", pin_memory=pinned),
            )
            self._slots[key] = host
        host.key.copy_(kv.key, non_blocking=True)
        host.value.copy_(kv.value, non_blocking=True)

    def get(self, key) -> KVState:
        return self._slots[key]

    def has(self, key) -> bool:
        return key in self._slots

    def drop(self, key) -> None:
        self._slots.pop(key, None)

    def drop_prefix(self, request: str) -> None:
        for key in [k for k in self._slots if getattr(k, "request", None) == request]:
            self._slots.pop(key, None)

    def nbytes(self) -> int:
        return sum(kv.nbytes() for kv in self._slots.values())

    def __len__(self) -> int:
        return len(self._slots)
