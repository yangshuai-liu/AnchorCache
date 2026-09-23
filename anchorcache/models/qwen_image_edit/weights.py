from __future__ import annotations

from typing import Any

# The framework manages only grouped attention and K/V (transformer/cached_attn.py). Projection,
# qk normalization, RoPE, and out projection remain on the official module because:
#   - The student may use LoRA; copying weights would disable the adapters.
#   - diffusers RMSNorm explicitly upcasts to fp32; rebuilding it as torch.nn.RMSNorm may change behavior.
#   - The same applies to quantization (svdquant) and FSDP wrappers.
#
# This file defines only the three elements required by the backend: region partitioning, K/V order,
# and the method for arranging official tensors into a region dict.

STREAMS: dict[str, tuple[str, ...]] = {
    "img": ("target", "ref"),
    "txt": ("text", "anchor"),
}
"""Maps each region to its official projection stream.

img stream  to_q / to_k / to_v      norm_q / norm_k              to_out[0]
txt stream  add_q_proj / ...        norm_added_q / norm_added_k  to_add_out

The anchor uses the txt stream: it reuses add_*_proj and txt_freqs, adding no parameters.
"""

PROJ: dict[str, tuple[str, str, str, str, str]] = {
    "img": ("to_q", "to_k", "to_v", "norm_q", "norm_k"),
    "txt": ("add_q_proj", "add_k_proj", "add_v_proj", "norm_added_q", "norm_added_k"),
}
"""Attribute names for each stream on the official Attention module, ordered as (q, k, v, norm_q, norm_k).

These are names, not weights. The shim retrieves and directly invokes the original module's submodules
with getattr, so LoRA, quantization, and FSDP wrappers remain effective.
"""

OUT: dict[str, str] = {"img": "to_out", "txt": "to_add_out"}
"""Out projection attribute names. The img side is a ModuleList: to_out[0] is linear, followed by optional dropout."""



def kv_order() -> list[str]:
    """K/V order for the main group: [txt, tgt, ref].

    This differs from FLUX's [txt, ref, tgt]. The two are mathematically equivalent, but flash reduction
    order introduces approximately 1e-5 drift, so the backend declares the order explicitly rather than
    allowing group_plan to infer it.
    """
    return ["text", "target", "ref"]


def split_img(img_t: Any, num_target: int) -> dict[str, Any]:
    """Split a tensor produced by one img-stream projection into target and ref regions.

    The ref segment immediately follows target, matching the order in which the pipeline
    concatenates latents.
    """
    return {"target": img_t[:, :num_target], "ref": img_t[:, num_target:]}


def freqs_map(img_freqs: Any, txt_freqs: Any, num_target: int, anchor: bool = True) -> dict:
    """Map each region to its RoPE frequencies.

    The anchor uses txt_freqs rather than a separate set, which is why anchored_layout asserts
    anchor == live_text.
    """
    m = {
        "text": txt_freqs,
        "target": img_freqs[:num_target],
        "ref": img_freqs[num_target:],
    }
    if anchor:
        m["anchor"] = txt_freqs
    return m
