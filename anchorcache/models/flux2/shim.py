from __future__ import annotations

import os

from ...transformer.cached_attn import cached_attn
from ...transformer.regions import HID, LIVE, VIS, Region, SeqLayout
from ...transformer.topology import group_plan
from .weights import kv_order, ref_bias_mask

# Routes the external FLUX.2 model's grouped attention through cached_attn. attn_self_text has the
# same signature as the model's own function, so it replaces the name bound in the model module:
#
#     import flux2.model as m
#     import anchorcache.models.flux2.shim as shim
#     m.attn_self_text = shim.attn_self_text


def _layout(n_txt: int, n_ref: int, n_tgt: int, anchor: int) -> SeqLayout:
    """Flux2 sequence layout. The physical order is [txt, ref, target], matching kv_order.

    anchor (frz_txt) has the same length as txt and shares its RoPE positions.
    """
    regions = [Region("text", n_txt, LIVE)]
    if n_ref:
        regions.append(Region("ref", n_ref, VIS))
    regions.append(Region("target", n_tgt, LIVE))
    if anchor:
        regions.append(Region("anchor", anchor, HID))
    return SeqLayout(regions)


def attn_self_text(
    txt_q,
    txt_k,
    txt_v,
    ref_q,
    ref_k,
    ref_v,
    tgt_q,
    tgt_k,
    tgt_v,
    frz_q=None,
    frz_k=None,
    frz_v=None,
    layer_cache=None,
):
    """Drop-in replacement for the FLUX.2 model's attn_self_text.

    All inputs have shape [B,S,H,D] and are post-RoPE; returns
    (txt_out, ref_out, tgt_out, frz_out|None).

    layer_cache continues to use Flux2's own Flux2KVLayerCache.store. The physical K/V container
    remains the backend's responsibility; this function replaces only the grouping logic to keep
    the comparison scope precise.
    """
    if layer_cache is not None:
        layer_cache.store(ref_k.clone(), ref_v.clone())

    n_txt, n_ref, n_tgt = txt_q.shape[1], ref_q.shape[1], tgt_q.shape[1]
    anchor = 0 if frz_q is None else frz_q.shape[1]
    lay = _layout(n_txt, n_ref, n_tgt, anchor)
    plan = group_plan(lay, kv_order=kv_order())

    q = {"text": txt_q, "ref": ref_q, "target": tgt_q}
    k = {"text": txt_k, "ref": ref_k, "target": tgt_k}
    v = {"text": txt_v, "ref": ref_v, "target": tgt_v}
    if anchor:
        q["anchor"], k["anchor"], v["anchor"] = frz_q, frz_k, frz_v

    # Offsets are derived from the plan.
    bias = float(os.environ.get("FLUX2_SELFTEXT_REF_BIAS", "0"))
    mask = ref_bias_mask(plan, {n: lay.len_of(n) for n in plan.main_kv}, bias, txt_k)

    out = cached_attn(q, k, v, plan, mode="full", mask=mask)
    return out["text"], out["ref"], out["target"], out.get("anchor")
