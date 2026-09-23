from __future__ import annotations

from ...transformer.regions import HID, LIVE, VIS, Region, SeqLayout

# FLUX.2-klein token layout.
#
# The sequence layout is [txt, ref, target], but double and single blocks have different
# physical representations:
#   double  hidden_states = [ref, target] and encoder_hidden_states = [txt] are separate tensors
#   single  hidden_states = cat([encoder, hidden]) = [txt, ref, target] is one tensor
# Therefore, layout describes only the logical order; each block handles its physical slicing.
#
# The anchor must have the same length as the txt segment because they share the txt RoPE.
#
# Number of ref tokens = sum_i (H_i//2 * W_i//2); multiple refs are separated along RoPE's T
# dimension (scale=10), as in the diffusers FLUX.2 klein KV pipeline.


def flux2_layout(
    txt_len: int,
    target_len: int,
    ref_lens: list[int] | None = None,
    anchor: bool = False,
) -> SeqLayout:
    regions: list[Region] = [Region("text", txt_len, LIVE)]
    for i, n in enumerate(ref_lens or []):
        regions.append(Region(f"ref_{i}", n, VIS))
    regions.append(Region("target", target_len, LIVE))
    if anchor:
        assert anchor is True
        regions.append(Region("anchor", txt_len, HID))
    return SeqLayout(regions)


def layout_from_packs(
    txt_len: int, ref_pack_len: int, target_len: int, anchor: bool = False
) -> SeqLayout:
    """ref_pack directly concatenates tokens from N reference images.

    Separating individual images is useful only for per-ref statistics; neither the cache nor
    attention distinguishes between them.
    """
    return flux2_layout(txt_len, target_len, [ref_pack_len] if ref_pack_len else [], anchor)
