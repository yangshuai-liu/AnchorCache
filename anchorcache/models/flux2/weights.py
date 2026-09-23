from __future__ import annotations

from typing import Any

# As for Qwen, the framework manages only grouped attention and K/V. Projections stay in the
# model: an optional anchor-stream LoRA adds its own delta on top of the text projection, and the
# single block fuses projection with the MLP (to_qkv_mlp_proj produces qkv and the MLP hidden;
# to_out consumes cat([attn_out, mlp_act(mlp_h)])).

STREAMS: dict[str, tuple[str, ...]] = {
    "img": ("target", "ref"),
    "txt": ("text", "anchor"),
}
"""Projection ownership for the two streams in a double block
(transformer_blocks).

A single block has only one projection (to_qkv_mlp_proj); text, image, and frz
all use the same weights.
"""


def kv_order() -> list[str]:
    """KV order for the main group: [txt, ref, tgt].

    This differs from Qwen's [txt, tgt, ref]: cached steps must be layout-compatible with the
    official Flux2KVCache, and the ref_bias mask is built from these offsets.
    """
    return ["text", "ref", "target"]


def split_single_qkv(attn: Any, proj: Any):
    """Single block: convert the output of to_qkv_mlp_proj into
    (q, k, v, mlp_hidden).

    QK normalization uses the shared norm_q and norm_k. The caller adds any anchor-stream
    LoRA delta beforehand.
    """
    import torch

    qkv, mlp_h = torch.split(
        proj, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
    )
    q, k, v = (x.unflatten(-1, (attn.heads, -1)) for x in qkv.chunk(3, dim=-1))
    return attn.norm_q(q), attn.norm_k(k), v, mlp_h


def ref_bias_mask(plan: Any, lengths: dict[str, int], bias: float, like: Any):
    """Additive logit bias over visual keys in the main group, constructed in
    plan.main_kv order.

    Offsets come from the plan, so changing kv_order does not misalign the mask. bias=0
    returns None to keep the fast flash path.
    """
    if bias == 0.0:
        return None
    total = sum(lengths[n] for n in plan.main_kv)
    mask = like.new_zeros(1, 1, 1, total)
    off = 0
    for name in plan.main_kv:
        ln = lengths[name]
        if plan.kv_is_vis(name):
            mask[..., off : off + ln] = bias
        off += ln
    return mask
