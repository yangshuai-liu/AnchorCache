from __future__ import annotations

from ...transformer.regions import HID, LIVE, VIS, Region, SeqLayout

# Token layout for Qwen-Image-Edit 2511.
#
# Joint order is [txt, img], while img is internally [target, ref_0..ref_{n-1}] with
# num_target = prod(img_shapes[0][0]).
#
# The static text anchor is a third stream that reuses the txt stream's weights and adds no
# parameters. Its role is HID: visible inside the frozen group, while the main group uses live txt.


def qwen_layout(
    txt_len: int,
    target_len: int,
    ref_lens: list[int] | None = None,
    anchor: bool = False,
    order: str = "text_first",
) -> SeqLayout:
    regions: list[Region] = []
    txt = Region("text", txt_len, LIVE)
    tgt = Region("target", target_len, LIVE)
    regions += [txt, tgt] if order == "text_first" else [tgt, txt]
    for i, n in enumerate(ref_lens or []):
        regions.append(Region(f"ref_{i}", n, VIS))
    if anchor:
        regions.append(Region("anchor", txt_len, HID))
    return SeqLayout(regions)


def layout_from_img_shapes(
    img_shapes: list[list[tuple[int, int, int]]],
    txt_len: int,
    anchor: bool = False,
) -> SeqLayout:
    """Build a layout directly from diffusers img_shapes.

    img_shapes[0][0] is the target shape, while img_shapes[0][1:] contains the reference-image
    shapes.
    """
    shapes = img_shapes[0]

    def prod(s):
        return s[0] * s[1] * s[2]

    return qwen_layout(
        txt_len=txt_len,
        target_len=prod(shapes[0]),
        ref_lens=[prod(s) for s in shapes[1:]],
        anchor=anchor,
    )


def cached_img_shapes(img_shapes: list[list[tuple[int, int, int]]], batch: int):
    """Pass only the target shape during cached steps.

    Condition tokens never enter the network, so MLP and normalization layers do not process them.
    This is the primary source of cached-step savings, not just attention.
    """
    return [[img_shapes[0][0]]] * batch
