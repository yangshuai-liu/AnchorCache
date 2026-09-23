from __future__ import annotations

from ..core.cache import CacheKey
from ..transformer.kv import KVState
from .buffer_pool import BufferPool
from .offload import HostStore
from .prefetch import Prefetcher
from .residency import GpuResident, Residency


class CacheManager:
    """Boundary between the logical cache and physical placement.

    The model side only calls put / get / prefetch / release and cannot see the pinned host, copy stream,
    buffer pool, or events. Changing residency_policy requires no changes to any model math code — this is
    the Runtime numerical constraint.
    """

    def __init__(
        self,
        residency: Residency | None = None,
        device=None,
        pool_cap: int = 4,
    ):
        self.residency = residency or GpuResident()
        self.device = device
        self.pool = BufferPool(cap=pool_cap)
        self.host = HostStore()
        self._gpu: dict[CacheKey, KVState] = {}
        self._num_layers: dict[str, int] = {}
        # Two-phase state machine: the first cached pass only moves KV to the host; prefetch is
        # enabled starting with the second pass. During the first pass, the original buffers produced
        # by extract remain on the GPU; enabling prefetch would retain two copies and raise the peak
        # instead.
        self._steady: dict[str, bool] = {}
        self._prefetcher = (
            Prefetcher(device, self.host, self.pool) if device is not None else None
        )

    def begin_request(self, request: str, num_layers: int) -> None:
        self._num_layers[request] = num_layers
        self._steady[request] = False

    def put(self, key: CacheKey, value: KVState) -> None:
        self._gpu[key] = value

    def has(self, key: CacheKey) -> bool:
        return key in self._gpu or self.host.has(key)

    def get(self, key: CacheKey) -> KVState:
        if key in self._gpu:
            return self._gpu[key]
        assert self._prefetcher is not None, f"{key} is not on the GPU and no prefetcher is available"
        if key not in self._prefetcher._inflight:
            self._prefetcher.issue(key)
        kv = self._prefetcher.wait(key)
        self._gpu[key] = kv
        return kv

    def prefetch(self, key: CacheKey, live_len: int = 0) -> None:
        if key in self._gpu or self._prefetcher is None:
            return
        self._prefetcher.issue(key, live_len=live_len)

    def layer_done(self, key: CacheKey) -> None:
        """Call after the attention computation for one layer completes.

        The first pass moves KV to the host; in steady state, it returns GPU buffers to the pool. Both occur only
        when that layer is not resident on the GPU.
        """
        num_layers = self._num_layers.get(key.request)
        if num_layers is None or self.residency.on_gpu(key.layer, num_layers):
            return
        kv = self._gpu.pop(key, None)
        if kv is None:
            return
        if self._steady.get(key.request):
            self.pool.give(kv.key)
            self.pool.give(kv.value)
        else:
            self.host.put(key, kv)

    def step_done(self, request: str) -> None:
        self._steady[request] = True

    def maybe_prefetch_next(self, key: CacheKey, live_len: int = 0) -> None:
        num_layers = self._num_layers.get(key.request)
        if num_layers is None or not self._steady.get(request := key.request):
            return
        nxt = self.residency.prefetch_target(key.layer, num_layers)
        if nxt is None:
            return
        self.prefetch(CacheKey(request=request, layer=nxt, branch=key.branch, slot=key.slot), live_len)

    def release(self, request: str) -> None:
        for key in [k for k in self._gpu if k.request == request]:
            kv = self._gpu.pop(key)
            self.pool.give(kv.key)
            self.pool.give(kv.value)
        self.host.drop_prefix(request)
        self._num_layers.pop(request, None)
        self._steady.pop(request, None)

    def stats(self) -> dict:
        return {
            "gpu_keys": len(self._gpu),
            "host_keys": len(self.host),
            "host_bytes": self.host.nbytes(),
            "gpu_bytes": sum(kv.nbytes() for kv in self._gpu.values()),
            "inflight": self._prefetcher.inflight() if self._prefetcher else 0,
            "pool": self.pool.stats(),
        }
