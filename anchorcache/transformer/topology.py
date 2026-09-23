from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .regions import FROZEN, HID, LIVE, VIS, Region, SeqLayout, Span

# There are two ways to consume a topology; do not mix them:
#
#   dense_mask()   Reference specification, not used on production paths: a dense mask
#                  forces SDPA off the flash fast path.
#
#   group_plan()   Production implementation. Compiles the topology into indices for
#                  two dense attention groups: the main group (live rows) and the
#                  frozen group. All three backends work this way.
#
# "Declare with a mask, compile into groups" is the sole design proposition here.


def dense_mask(spans: list[Span], size: int | None = None, device=None) -> torch.Tensor:
    """[S, S] boolean visibility matrix generated directly from role rules."""
    total = size if size is not None else max((s.start + s.total) for s in spans)
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    seen = torch.zeros(total, dtype=torch.bool, device=device)
    for sp in spans:
        fro, vis, live = sp.frozen_slice, sp.vis_slice, sp.live_slice
        mask[fro, fro] = True
        mask[live, vis] = True
        mask[live, live] = True
        seen[sp.start : sp.start + sp.total] = True
    # Give each padding row a diagonal entry; otherwise an all -inf softmax row produces
    # NaN, which propagates through the head all the way to the loss
    idle = (~seen).nonzero(as_tuple=True)[0]
    mask[idle, idle] = True
    return mask


def dense_mask_of(layout: SeqLayout, device=None) -> torch.Tensor:
    """Generate a visibility matrix by region role without requiring physical contiguity.

    Qwen's joint order is [txt(live), target(live), ref(vis)] plus a separate frz(hid),
    which is not contiguous in vis|hid|live order and therefore cannot use dense_mask().
    """
    total = layout.total
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    fro = layout.role_slices(*FROZEN)
    vis = layout.role_slices(VIS)
    live = layout.role_slices(LIVE)
    for a in fro:
        for b in fro:
            mask[a, b] = True
    for a in live:
        for b in vis:
            mask[a, b] = True
        for b in live:
            mask[a, b] = True
    return mask


@dataclass
class GroupPlan:
    """Region lists for grouped attention.

    main_q / main_kv / frozen_q / frozen_kv are all region names; the attention
    implementation torch.cat's the corresponding tensors. Names are used instead of
    slices because Qwen/FLUX place txt and img in different tensors
    (encoder_hidden_states vs hidden_states), which slices cannot express across tensors.

    The **order matters** for main_kv and cannot be inferred from role—the two backends
    differ:
        Qwen  [txt, tgt, cond]   live, live, vis
        FLUX  [txt, ref, tgt]    live, vis,  live
    Attention output is invariant to KV ordering (a weighted sum), but different flash
    reduction orders introduce ~1e-5 drift, and FLUX cached steps must be order-compatible
    with the official Flux2KVCache. Order is therefore part of the backend declaration
    and is specified explicitly through group_plan(kv_order=...).
    """

    main_q: list[str]
    main_kv: list[str]
    frozen_q: list[str]
    frozen_kv: list[str]
    vis: list[str] = field(default_factory=list)

    @property
    def has_frozen_group(self) -> bool:
        return bool(self.frozen_q)

    def kv_is_vis(self, name: str) -> bool:
        """This segment should come from reusable state during a cached step, not be recomputed."""
        return name in self.vis


def group_plan(
    layout: SeqLayout,
    cached: bool = False,
    kv_order: list[str] | None = None,
) -> GroupPlan:
    """Compile a layout into two groups.

    When cached=True, K/V for VIS segments comes from reusable state; their names remain
    in main_kv (the attention implementation replaces them with cache), but the entire
    frozen group disappears—this is precisely why a cached step is cheaper than an
    extract step: MLP/Norm for VIS/HID do not run at all.

    kv_order explicitly specifies the region order in main_kv and must be a permutation
    of live+vis. The default places live before vis (Qwen's order).
    """
    live = [r.name for r in layout.by_role(LIVE)]
    vis = [r.name for r in layout.by_role(VIS)]
    hid = [r.name for r in layout.by_role(HID)]
    assert live, f"{layout} has no LIVE region; there is nothing to do in the iterative stage"

    main_kv = live + vis
    if kv_order is not None:
        assert sorted(kv_order) == sorted(main_kv), (
            f"kv_order must be a permutation of live+vis: got {kv_order}, expected a permutation of {main_kv}"
        )
        main_kv = list(kv_order)

    if cached:
        return GroupPlan(main_q=live, main_kv=main_kv, frozen_q=[], frozen_kv=[], vis=vis)
    return GroupPlan(
        main_q=live,
        main_kv=main_kv,
        frozen_q=vis + hid,
        frozen_kv=vis + hid,
        vis=vis,
    )


@dataclass
class SegIndex:
    """All indices needed for grouped attention over packed sequences (varlen path)."""

    frozen_idx: torch.Tensor
    live_idx: torch.Tensor
    main_idx: torch.Tensor
    put_idx: torch.Tensor
    cu_frozen: torch.Tensor
    cu_live: torch.Tensor
    cu_main: torch.Tensor
    max_frozen: int
    max_live: int
    max_main: int

    def as_dict(self) -> dict:
        return {
            "frozen_idx": self.frozen_idx,
            "live_idx": self.live_idx,
            "main_idx": self.main_idx,
            "put_idx": self.put_idx,
            "cu_frozen": self.cu_frozen,
            "cu_live": self.cu_live,
            "cu_main": self.cu_main,
            "max_frozen": self.max_frozen,
            "max_live": self.max_live,
            "max_main": self.max_main,
        }


def seg_index(spans: list[Span], device=None) -> SegIndex:
    i32 = dict(dtype=torch.int32, device=device)
    i64 = dict(dtype=torch.long, device=device)

    frozen, live, main = [], [], []
    cu_f, cu_l, cu_m = [0], [0], [0]
    for sp in spans:
        if sp.frozen:
            frozen.append(torch.arange(sp.start, sp.start + sp.frozen, **i64))
            cu_f.append(cu_f[-1] + sp.frozen)
        live.append(torch.arange(sp.start + sp.frozen, sp.start + sp.total, **i64))
        cu_l.append(cu_l[-1] + sp.live)
        # live KV skips hid. Per-doc order must be [vis | live], matching the kv_plan
        # buffer layout segment by segment—otherwise flash uses a different reduction
        # order and cached cannot match full within the 1e-5 tolerance.
        main.append(torch.arange(sp.start, sp.start + sp.vis, **i64))
        main.append(torch.arange(sp.start + sp.frozen, sp.start + sp.total, **i64))
        cu_m.append(cu_m[-1] + sp.vis + sp.live)

    empty = torch.zeros(0, **i64)
    frozen_idx = torch.cat(frozen) if frozen else empty
    live_idx = torch.cat(live) if live else empty
    return SegIndex(
        frozen_idx=frozen_idx,
        live_idx=live_idx,
        main_idx=torch.cat(main) if main else empty,
        put_idx=torch.cat([frozen_idx, live_idx]),
        cu_frozen=torch.tensor(cu_f, **i32),
        cu_live=torch.tensor(cu_l, **i32),
        cu_main=torch.tensor(cu_m, **i32),
        max_frozen=max((sp.frozen for sp in spans), default=0),
        max_live=max((sp.live for sp in spans), default=0),
        max_main=max((sp.vis + sp.live for sp in spans), default=0),
    )


def reconstruct_mask(seg: SegIndex, spans: list[Span], size: int) -> torch.Tensor:
    """Reconstruct a visibility matrix from indices, for comparison against dense_mask.

    Incorrect indices neither go out of bounds nor raise errors; they silently alter the topology.
    """
    mask = torch.zeros(size, size, dtype=torch.bool)
    cu_f = seg.cu_frozen.tolist()
    cu_l, cu_m = seg.cu_live.tolist(), seg.cu_main.tolist()
    for i in range(len(cu_f) - 1):
        rows = seg.frozen_idx[cu_f[i] : cu_f[i + 1]]
        mask[rows.unsqueeze(1), rows.unsqueeze(0)] = True
    for i in range(len(cu_l) - 1):
        rows = seg.live_idx[cu_l[i] : cu_l[i + 1]]
        cols = seg.main_idx[cu_m[i] : cu_m[i + 1]]
        if rows.numel():
            mask[rows.unsqueeze(1), cols.unsqueeze(0)] = True
    seen = torch.zeros(size, dtype=torch.bool)
    for sp in spans:
        seen[sp.start : sp.start + sp.total] = True
    idle = (~seen).nonzero(as_tuple=True)[0]
    mask[idle, idle] = True
    return mask


def anchored_layout(
    live_text: int,
    target: int,
    ref: int,
    anchor: int = 0,
    *,
    order: str = "text_first",
) -> SeqLayout:
    """Express the static text anchor topology of Qwen / FLUX as a layout.

    anchor=0 gives the isolated cache; ref=anchor=0 gives dense attention. order selects
    [txt, img] ("text_first") or [img, txt].
    """
    txt = Region("text", live_text, LIVE)
    tgt = Region("target", target, LIVE)
    regions = [txt, tgt] if order == "text_first" else [tgt, txt]
    if ref:
        regions.append(Region("ref", ref, VIS))
    if anchor:
        assert anchor == live_text, (
            f"anchor and live text must have equal lengths (they share txt RoPE): {anchor} vs {live_text}"
        )
        regions.append(Region("anchor", anchor, HID))
    return SeqLayout(regions)


def text_live_layout(style: int, ref: int, text: int, target: int) -> SeqLayout:
    """OmniVoice's text_live layout.

    The text segment appears twice: text_f is the copy in the frozen group (HID), while
    text_l is recomputed on every step (LIVE).
    """
    regions = [Region("style", style, VIS)]
    if ref:
        regions.append(Region("ref", ref, VIS))
    if text:
        regions.append(Region("text_f", text, HID))
        regions.append(Region("text_l", text, LIVE))
    regions.append(Region("target", target, LIVE))
    return SeqLayout(regions)


def dense_layout(total: int) -> SeqLayout:
    """Fully bidirectional baseline, i.e. (vis, hid, live) = (0, 0, total).

    This is a parameter setting, not another code path.
    """
    return SeqLayout([Region("all", total, LIVE)])
