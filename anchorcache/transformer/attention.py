from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from .kv import KVState

# Tensor layout is standardized as NHD: [B, S, H, D].
# The projection outputs of all three backends have this shape (Qwen/FLUX use
# unflatten(-1, (heads, -1)), OmniVoice uses view(b, s, hq, d)), and flash_attn also
# consumes NHD directly; only SDPA requires a transpose.


def _backend(default: str = "sdpa") -> str:
    return os.environ.get("ANCHORCACHE_ATTN_BACKEND", default)


def group_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    backend: str | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """One dense attention group. q/k/v are all [B, S, H, D]; returns [B, S, H, D].

    The kernel is configurable because the best choice depends on the GPU, on training vs
    inference, and on shape stability: cuDNN is ~3.8x faster than flash for Qwen's short-Q /
    long-K inference shape on B200, yet slower for FLUX training, where varying sequence
    lengths force cuDNN to re-plan for every new shape.
    """
    be = backend or _backend()
    if be == "fa":
        assert mask is None, "the flash_attn path does not support attention bias"
        from flash_attn import flash_attn_func

        return flash_attn_func(q, k, v, causal=False, softmax_scale=scale)
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if be == "cudnn":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        assert mask is None
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(qt, kt, vt, scale=scale)
    elif be == "flash":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        assert mask is None
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(qt, kt, vt, scale=scale)
    else:
        out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask, scale=scale)
    return out.transpose(1, 2)


def varlen_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    max_q: int,
    max_kv: int,
    scale: float | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """Ragged grouping. q/k/v are [total_tokens, H, D].

    The SDPA fallback runs each segment separately; use it with fp32 when comparing prefill
    and full paths, since flash tiling alone introduces ~1e-3 drift.
    """
    be = backend or _backend("flash")
    if be in ("flash", "fa"):
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func(
            q, k, v, cu_q, cu_kv, max_q, max_kv, causal=False, softmax_scale=scale
        )
    lq, lkv = cu_q.tolist(), cu_kv.tolist()
    rep = q.shape[1] // k.shape[1]
    out = torch.empty_like(q)
    for i in range(len(lq) - 1):
        qs, qe = lq[i], lq[i + 1]
        ks, ke = lkv[i], lkv[i + 1]
        if qe == qs:
            continue
        seg = F.scaled_dot_product_attention(
            q[qs:qe].transpose(0, 1).unsqueeze(0),
            k[ks:ke].repeat_interleave(rep, dim=1).transpose(0, 1).unsqueeze(0),
            v[ks:ke].repeat_interleave(rep, dim=1).transpose(0, 1).unsqueeze(0),
            scale=scale,
        )
        out[qs:qe] = seg.squeeze(0).transpose(0, 1)
    return out


@runtime_checkable
class CacheableAttention(Protocol):
    """extract and forward share the same projection / attention implementation.

    One set of weights serves multiple execution modes; separate extract / dynamic classes
    would duplicate the projection.
    """

    def extract(self, hidden, plan, cache) -> tuple: ...

    def forward(self, hidden, plan, reusable: KVState | None = None) -> tuple: ...


def concat_kv(live: KVState, vis: KVState | None) -> KVState:
    """K/V concatenation for a cached step: [live | vis].

    This is the simplest form and pays for a full-length allocation + copy on every
    step. Faster forms preallocate a live_len + vis_len buffer and overwrite only the live
    prefix (QWEN_KV_PREALLOC), or use FlatKV (kv.py).
    """
    if vis is None:
        return live
    return KVState(
        torch.cat([live.key, vis.key], dim=1),
        torch.cat([live.value, vis.value], dim=1),
    )
