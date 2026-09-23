from __future__ import annotations

import os

import torch

from ...transformer.cached_attn import cached_attn
from ...transformer.kv import LayeredKV
from ...transformer.topology import anchored_layout, group_plan
from .weights import OUT, PROJ, kv_order

# Qwen attention wiring used by model.py in this directory.
#
# Qwen performs projection -> qk norm -> RoPE -> grouping -> output projection in one function, so this
# shim performs all five steps. It calls the original module's submodules (getattr(attn, ...)) instead
# of copying weights, so LoRA, quantization and FSDP wrappers remain effective.

KERNEL_DTYPE = torch.bfloat16
"""Dtype q/k/v are cast to before the kernel; None disables the cast.

Under the DeepSpeed bf16 engine autocast does not apply, so modulation, RMSNorm and RoPE can leave
q/k/v in fp32 while the flash kernel accepts only fp16/bf16.
"""

_ROPE = None


def _rope_fn():
    global _ROPE
    if _ROPE is None:
        from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

        _ROPE = apply_rotary_emb_qwen
    return _ROPE


def _stream(attn, name: str, hidden, freqs, rope):
    """Projection, qk normalization, and RoPE for one stream.

    v passes through neither normalization nor RoPE. This is intentional because RoPE acts on the q·k inner product.
    """
    q_n, k_n, v_n, nq_n, nk_n = PROJ[name]
    h = attn.heads
    q = getattr(attn, q_n)(hidden).unflatten(-1, (h, -1))
    k = getattr(attn, k_n)(hidden).unflatten(-1, (h, -1))
    v = getattr(attn, v_n)(hidden).unflatten(-1, (h, -1))
    nq, nk = getattr(attn, nq_n), getattr(attn, nk_n)
    if nq is not None:
        q = nq(q)
    if nk is not None:
        k = nk(k)
    return rope(q, freqs, use_real=False), rope(k, freqs, use_real=False), v


def _cast(d: dict, dtype):
    return d if dtype is None else {n: t.to(dtype) for n, t in d.items()}


def _prealloc() -> bool:
    return os.environ.get("QWEN_KV_PREALLOC", "0") == "1"


def store_backend_kv(kv_cache, layer_idx, cond_k, cond_v, live_len, prealloc: bool):
    """Store reference K/V in the backend's dict cache.

    Both formats must be supported: the preallocated variant allocates a long live+ref buffer so cached
    steps can write the live prefix in place, avoiding cat on every step.

    clone is not a defensive copy: cond_k is a view of img_k; without cloning, the entire hidden tensor
    would remain in GPU memory until the request completes.
    """
    if prealloc:
        n_ref = cond_k.shape[1]
        b, _, h, d = cond_k.shape
        k_buf = cond_k.new_empty((b, live_len + n_ref, h, d))
        v_buf = cond_v.new_empty((b, live_len + n_ref, h, d))
        k_buf[:, live_len:].copy_(cond_k)
        v_buf[:, live_len:].copy_(cond_v)
        kv_cache["layers"][layer_idx] = {"k_buf": k_buf, "v_buf": v_buf, "live_len": live_len}
    else:
        kv_cache["layers"][layer_idx] = {"k_ref": cond_k.clone(), "v_ref": cond_v.clone()}


def attn_self_text(
    attn,
    img_modulated,
    txt_modulated,
    frz_modulated,
    image_rotary_emb,
    num_target_tokens,
    layer_idx=None,
    kv_cache=None,
    store_kv_cache=False,
    *,
    rope=None,
    kernel_dtype=KERNEL_DTYPE,
    kv: LayeredKV | None = None,
):
    """Grouped static-anchor attention for one Qwen block.

    Returns (img_attn[target+cond rows], txt_attn, frz_attn | None). When kv is given, reference K/V
    are also written to it (mode="extract").
    """
    rope = rope or _rope_fn()
    img_freqs, txt_freqs = image_rotary_emb

    img_q, img_k, img_v = _stream(attn, "img", img_modulated, img_freqs, rope)
    txt_q, txt_k, txt_v = _stream(attn, "txt", txt_modulated, txt_freqs, rope)

    seq_txt = txt_modulated.shape[1]
    n_cond = img_q.shape[1] - num_target_tokens
    assert n_cond > 0, f"self_text requires condition latents (an attn.py precondition); got {n_cond}"

    q = {"text": txt_q, "target": img_q[:, :num_target_tokens], "ref": img_q[:, num_target_tokens:]}
    k = {"text": txt_k, "target": img_k[:, :num_target_tokens], "ref": img_k[:, num_target_tokens:]}
    v = {"text": txt_v, "target": img_v[:, :num_target_tokens], "ref": img_v[:, num_target_tokens:]}

    n_frz = 0
    if frz_modulated is not None:
        n_frz = frz_modulated.shape[1]
        q["anchor"], k["anchor"], v["anchor"] = _stream(
            attn, "txt", frz_modulated, txt_freqs, rope
        )

    lay = anchored_layout(seq_txt, num_target_tokens, n_cond, anchor=n_frz, order="text_first")
    plan = group_plan(lay, kv_order=kv_order())

    # Store pre-cast K/V in the backend container, then cast for the kernel.
    if store_kv_cache and kv_cache is not None:
        store_backend_kv(
            kv_cache,
            layer_idx,
            k["ref"],
            v["ref"],
            seq_txt + num_target_tokens,
            _prealloc(),
        )

    out = cached_attn(
        _cast(q, kernel_dtype),
        _cast(k, kernel_dtype),
        _cast(v, kernel_dtype),
        plan,
        mode="extract" if kv is not None else "full",
        kv=kv,
        layer=layer_idx or 0,
    )

    # Output projection.
    txt_attn = out["text"].flatten(2, 3)
    tgt_attn = out["target"].flatten(2, 3)
    cond_attn = out["ref"].flatten(2, 3)

    to_out = getattr(attn, OUT["img"])
    add_out = getattr(attn, OUT["txt"])
    img_attn = to_out[0](torch.cat([tgt_attn, cond_attn], dim=1).contiguous())
    if len(to_out) > 1:
        img_attn = to_out[1](img_attn)
    txt_attn = add_out(txt_attn.contiguous())
    frz_out = out.get("anchor")
    if frz_out is not None:
        frz_out = add_out(frz_out.flatten(2, 3).contiguous())
    return img_attn, txt_attn, frz_out
