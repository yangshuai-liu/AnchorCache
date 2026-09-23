from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.state import ReusableState
from .regions import Span

# Two physical layouts with the same logical interface:
#
#   LayeredKV   per-layer [B, S_vis, H, D] tensor pairs, for batched dense SDPA / flash paths.
#
#   FlatKV      one [L, total_slots, H_kv, D] block + slot index table, for flash varlen. Each
#               doc is laid out as [vis|live]; a cached step writes live K/V into its slots in
#               place and passes the whole block to the kernel, with no gathers or concatenation.


@dataclass
class KVState:
    """Post-RoPE K/V.

    Storing pre-RoPE would require the cached step to know the absolute positions of vis segments,
    leaking layout into runtime.
    """

    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self) -> None:
        assert self.key.shape == self.value.shape, f"{self.key.shape} vs {self.value.shape}"

    @property
    def num_tokens(self) -> int:
        return self.key.shape[-3]

    def nbytes(self) -> int:
        return self.key.numel() * self.key.element_size() * 2

    def to(self, *args, **kwargs) -> KVState:
        return KVState(self.key.to(*args, **kwargs), self.value.to(*args, **kwargs))


class LayeredKV(ReusableState):
    """List of per-layer KVState objects."""

    __slots__ = ("layers", "num_vis_tokens", "meta")

    def __init__(self, num_layers: int, num_vis_tokens: int = 0):
        self.layers: list[KVState | None] = [None] * num_layers
        self.num_vis_tokens = num_vis_tokens
        self.meta: dict = {}

    def __len__(self) -> int:
        return len(self.layers)

    def put(self, layer: int, kv: KVState) -> None:
        self.layers[layer] = kv

    def get(self, layer: int) -> KVState:
        kv = self.layers[layer]
        assert kv is not None, f"reusable state for layer {layer} is not populated; extract skipped this layer"
        return kv

    def has(self, layer: int) -> bool:
        return self.layers[layer] is not None

    def populated(self) -> bool:
        return all(kv is not None for kv in self.layers)

    def clear(self) -> None:
        self.layers = [None] * len(self.layers)
        self.meta.clear()

    def nbytes(self) -> int:
        return sum(kv.nbytes() for kv in self.layers if kv is not None)


@dataclass
class KVPlan:
    """Slot table for FlatKV.

    Each doc occupies contiguous [vis | live] slots in the buffer, so cu_main directly
    describes the buffer itself and a cached step requires no gathers.

    hid never enters the buffer: live rows cannot see it, and no one reads its K/V after
    prefill. The prefill input must still include hid (the frozen group is internally
    bidirectional), so vis_row is needed to gather vis rows from the prefill input—the
    input is arranged as [vis_0, hid_0, vis_1, hid_1, ...], and vis rows are necessarily
    noncontiguous when at least two docs have hid>0.
    """

    vis_slot: torch.Tensor
    vis_row: torch.Tensor
    live_slot: torch.Tensor
    cu_vis: torch.Tensor
    cu_live: torch.Tensor
    cu_main: torch.Tensor
    max_vis: int
    max_live: int
    max_main: int
    total: int


def kv_plan(spans: list[Span], device=None) -> KVPlan:
    i32 = dict(dtype=torch.int32, device=device)
    i64 = dict(dtype=torch.long, device=device)

    vis_slot, vis_row, live_slot = [], [], []
    cu_v, cu_l, cu_m = [0], [0], [0]
    off = 0
    row = 0
    for sp in spans:
        if sp.frozen:
            vis_slot.append(torch.arange(off, off + sp.vis, **i64))
            vis_row.append(torch.arange(row, row + sp.vis, **i64))
            cu_v.append(cu_v[-1] + sp.frozen)
            row += sp.frozen
        live_slot.append(torch.arange(off + sp.vis, off + sp.vis + sp.live, **i64))
        cu_l.append(cu_l[-1] + sp.live)
        cu_m.append(cu_m[-1] + sp.vis + sp.live)
        off += sp.vis + sp.live

    empty = torch.zeros(0, **i64)
    return KVPlan(
        vis_slot=torch.cat(vis_slot) if vis_slot else empty,
        vis_row=torch.cat(vis_row) if vis_row else empty,
        live_slot=torch.cat(live_slot) if live_slot else empty,
        cu_vis=torch.tensor(cu_v, **i32),
        cu_live=torch.tensor(cu_l, **i32),
        cu_main=torch.tensor(cu_m, **i32),
        max_vis=max((sp.frozen for sp in spans), default=0),
        max_live=max((sp.live for sp in spans), default=0),
        max_main=max((sp.vis + sp.live for sp in spans), default=0),
        total=off,
    )


class FlatKV(ReusableState):
    """One [L, total, H, D] buffer for the entire model."""

    __slots__ = ("key", "value", "plan", "num_layers")

    def __init__(
        self,
        num_layers: int,
        plan: KVPlan,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device,
        pool=None,
    ):
        shape = (num_layers, plan.total, num_heads, head_dim)
        if pool is not None:
            self.key = pool.acquire(shape, dtype, device)
            self.value = pool.acquire(shape, dtype, device)
        else:
            self.key = torch.zeros(shape, dtype=dtype, device=device)
            self.value = torch.zeros(shape, dtype=dtype, device=device)
        self.plan = plan
        self.num_layers = num_layers

    def store_vis(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        row = self.plan.vis_row
        self.key[layer].index_copy_(0, self.plan.vis_slot, key.index_select(0, row))
        self.value[layer].index_copy_(0, self.plan.vis_slot, value.index_select(0, row))

    def store_live(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.key[layer].index_copy_(0, self.plan.live_slot, key)
        self.value[layer].index_copy_(0, self.plan.live_slot, value)

    def layer_kv(self, layer: int) -> KVState:
        return KVState(self.key[layer], self.value[layer])

    def nbytes(self) -> int:
        return (self.key.numel() + self.value.numel()) * self.key.element_size()

    def release(self, pool=None) -> None:
        if pool is not None:
            pool.give(self.key)
            pool.give(self.value)
        self.key = None
        self.value = None


def kv_bytes_per_token(num_layers: int, num_heads: int, head_dim: int, itemsize: int = 2) -> int:
    """KV footprint of one token, for capacity planning.

    Qwen-Image-Edit-2511 needs 12 KiB/token/layer, 56.25 GiB with 10 reference images;
    CFG with separate positive/negative caches doubles this, which is what runtime offload
    is for.
    """
    return 2 * num_layers * num_heads * head_dim * itemsize
