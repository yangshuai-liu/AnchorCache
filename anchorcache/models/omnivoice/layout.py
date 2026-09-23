from __future__ import annotations

from ...transformer.regions import HID, LIVE, VIS, Region, SeqLayout

# OmniVoice's four layouts, each defined by its (vis, hid, live) values.
#
#   layout      vis            hid       live
#   ---------------------------------------------------------
#   bidir       0              0         entire sequence (upstream fully bidirectional baseline)
#   prefix      style+text+ref 0         target
#   text_live   style+ref      text_f    text_l+target
#   ref_only    style+ref      0         text+target
#   uncond doc  0              0         target      (CFG, no special-case branch)
#
# The text segment intentionally appears twice in text_live: text_f enters the
# frozen group (but not the buffer), while text_l is recomputed at every step.
# This lets the target attend to "text representations evolving with state" while
# preserving reusable style+ref K/V. The zero-shot WER improvement from 6.74% to
# 5.23% is attributable to this edge.


def bidir_layout(total: int) -> SeqLayout:
    return SeqLayout([Region("all", total, LIVE)])


def prefix_layout(style: int, text: int, ref: int, target: int) -> SeqLayout:
    regions = [Region("style", style, VIS)]
    if text:
        regions.append(Region("text", text, VIS))
    if ref:
        regions.append(Region("ref", ref, VIS))
    regions.append(Region("target", target, LIVE))
    return SeqLayout(regions)


def text_live_layout(style: int, ref: int, text: int, target: int) -> SeqLayout:
    regions = [Region("style", style, VIS)]
    if ref:
        regions.append(Region("ref", ref, VIS))
    if text:
        regions.append(Region("text_f", text, HID))
        regions.append(Region("text_l", text, LIVE))
    regions.append(Region("target", target, LIVE))
    return SeqLayout(regions)


def ref_only_layout(style: int, ref: int, text: int, target: int) -> SeqLayout:
    regions = [Region("style", style, VIS)]
    if ref:
        regions.append(Region("ref", ref, VIS))
    if text:
        regions.append(Region("text", text, LIVE))
    regions.append(Region("target", target, LIVE))
    return SeqLayout(regions)


def uncond_layout(target: int) -> SeqLayout:
    """The unconditional document used for CFG.

    (vis, hid, live) = (0, 0, T), so it naturally contributes zero frozen slots
    to seg_pack / kv_plan. CFG requires no special-case branch.
    """
    return SeqLayout([Region("target", target, LIVE)])


LAYOUTS = {
    "bidir": bidir_layout,
    "prefix": prefix_layout,
    "text_live": text_live_layout,
    "ref_only": ref_only_layout,
}


def build(layout: str, *, style: int, ref: int, text: int, target: int) -> SeqLayout:
    if layout == "bidir":
        return bidir_layout(style + text + ref + target)
    return LAYOUTS[layout](style=style, ref=ref, text=text, target=target)
