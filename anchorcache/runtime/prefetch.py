from __future__ import annotations

import torch

from ..transformer.kv import KVState
from .buffer_pool import BufferPool
from .offload import HostStore

# Asynchronous H2D prefetch of offloaded K/V. Qwen with 10 references moves 480 MiB per layer
# (~9.4 ms at 50 GB/s), which overlaps almost fully with compute.


class Prefetcher:
    """Asynchronous H2D using a dedicated copy stream and per-key events.

    Transfer only the vis segment. The live prefix is overwritten by the current step's Q/K/V at every step,
    so transferring it is entirely wasteful — this is what Qwen's `key_buf[:, live_len:].copy_(key_cpu[:, live_len:])` means.
    """

    def __init__(self, device, host: HostStore, pool: BufferPool | None = None):
        self.device = torch.device(device)
        self.host = host
        self.pool = pool or BufferPool()
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._inflight: dict = {}

    def issue(self, key, live_len: int = 0) -> None:
        if key in self._inflight or not self.host.has(key):
            return
        src = self.host.get(key)
        dst = KVState(
            self.pool.acquire_like(src.key, device=self.device),
            self.pool.acquire_like(src.value, device=self.device),
        )
        if self.stream is None:
            dst.key[:, live_len:].copy_(src.key[:, live_len:])
            dst.value[:, live_len:].copy_(src.value[:, live_len:])
            self._inflight[key] = (dst, None)
            return
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            dst.key[:, live_len:].copy_(src.key[:, live_len:], non_blocking=True)
            dst.value[:, live_len:].copy_(src.value[:, live_len:], non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        self._inflight[key] = (dst, ready)

    def wait(self, key) -> KVState:
        dst, ready = self._inflight.pop(key)
        if ready is not None:
            cur = torch.cuda.current_stream(self.device)
            cur.wait_event(ready)
            # record_stream is required: the buffer was allocated on copy_stream; unless the
            # allocator is told that the compute stream is using it, it may be reused immediately after being returned
            dst.key.record_stream(cur)
            dst.value.record_stream(cur)
        return dst

    def inflight(self) -> int:
        return len(self._inflight)

    def give_back(self, kv: KVState | None) -> None:
        if kv is None:
            return
        self.pool.give(kv.key)
        self.pool.give(kv.value)

    def drain(self) -> None:
        for key in list(self._inflight):
            self.give_back(self.wait(key))
