from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Every backend's attention topology is the same rule over which columns each row can see:
#
#     LIVE   -> VIS | LIVE
#     FROZEN -> FROZEN            (FROZEN = VIS | HID)
#
#   Qwen-Image-Edit / FLUX.2   vis=ref, hid=static text anchor, live=text+target
#   OmniVoice (text_live)      vis=style+ref, hid=text_f, live=text_l+target
#
# Two degenerate cases need no extra code path:
#   hid == 0            isolated cache
#   vis == hid == 0     dense / bidirectional baseline, e.g. (0, 0, T)



class Role(Enum):
    # Invariant across steps and visible to live rows -> enters reusable state
    VIS = "vis"
    # Invariant across steps but invisible to live rows -> excluded from reusable state,
    # while still entering the network during extract
    HID = "hid"
    # Depends on the current state and is recomputed every step
    LIVE = "live"


VIS = Role.VIS
HID = Role.HID
LIVE = Role.LIVE

FROZEN = (VIS, HID)


@dataclass(frozen=True)
class Region:
    name: str
    length: int
    role: Role

    def __post_init__(self) -> None:
        assert self.length >= 0, f"region {self.name}: negative length {self.length}"


class SeqLayout:
    """Region partitioning for one sequence.

    Physical order is the order in which regions are provided, independent of role:
    Qwen's joint order is [txt, img], while OmniVoice's text_live order is
    [style, ref, text_f, text_l, target].

    For flash varlen paths that require physical [vis|hid|live] contiguity, use spans(),
    which validates this requirement.
    """

    def __init__(self, regions: list[Region]):
        assert regions, "empty layout"
        names = [r.name for r in regions]
        assert len(names) == len(set(names)), f"duplicate region names: {names}"
        self.regions = list(regions)

    @property
    def total(self) -> int:
        return sum(r.length for r in self.regions)

    def offset_of(self, name: str) -> int:
        off = 0
        for r in self.regions:
            if r.name == name:
                return off
            off += r.length
        raise KeyError(name)

    def slice_of(self, name: str) -> slice:
        off = self.offset_of(name)
        for r in self.regions:
            if r.name == name:
                return slice(off, off + r.length)
        raise KeyError(name)

    def len_of(self, name: str) -> int:
        for r in self.regions:
            if r.name == name:
                return r.length
        raise KeyError(name)

    def by_role(self, *roles: Role) -> list[Region]:
        return [r for r in self.regions if r.role in roles]

    def role_len(self, *roles: Role) -> int:
        return sum(r.length for r in self.by_role(*roles))

    @property
    def vis_len(self) -> int:
        return self.role_len(VIS)

    @property
    def hid_len(self) -> int:
        return self.role_len(HID)

    @property
    def live_len(self) -> int:
        return self.role_len(LIVE)

    @property
    def frozen_len(self) -> int:
        return self.role_len(*FROZEN)

    def role_slices(self, *roles: Role) -> list[slice]:
        out, off = [], 0
        for r in self.regions:
            if r.role in roles:
                out.append(slice(off, off + r.length))
            off += r.length
        return out

    def is_contiguous(self) -> bool:
        # The role sequence must be vis* hid* live* for the [vis|hid|live] flash varlen layout
        order = [r.role for r in self.regions if r.length > 0]
        want = [VIS, HID, LIVE]
        pos = 0
        for role in order:
            while pos < len(want) and want[pos] is not role:
                pos += 1
            if pos == len(want):
                return False
        return True

    def spans(self, start: int = 0) -> Span:
        assert self.is_contiguous(), (
            f"the varlen path requires roles to be physically contiguous in vis|hid|live order; got "
            f"{[(r.name, r.role.value) for r in self.regions]}"
        )
        return Span(start=start, vis=self.vis_len, hid=self.hid_len, live=self.live_len)

    def degenerate(self) -> str:
        if self.vis_len == 0 and self.hid_len == 0:
            return "dense"
        if self.hid_len == 0:
            return "self_only"
        return "anchored"

    def __repr__(self) -> str:
        body = ", ".join(f"{r.name}:{r.role.value}={r.length}" for r in self.regions)
        return f"SeqLayout({body})"


@dataclass(frozen=True)
class Span:
    """The three [vis | hid | live] segments of one document in a packed sequence.

    A CFG unconditional doc is (vis, hid, live) = (0, 0, T), so CFG is a parameter setting
    rather than a special-case branch.
    """

    start: int
    vis: int
    hid: int
    live: int

    @property
    def frozen(self) -> int:
        return self.vis + self.hid

    @property
    def total(self) -> int:
        return self.vis + self.hid + self.live

    @property
    def vis_slice(self) -> slice:
        return slice(self.start, self.start + self.vis)

    @property
    def hid_slice(self) -> slice:
        return slice(self.start + self.vis, self.start + self.frozen)

    @property
    def frozen_slice(self) -> slice:
        return slice(self.start, self.start + self.frozen)

    @property
    def live_slice(self) -> slice:
        return slice(self.start + self.frozen, self.start + self.total)


def pack_spans(layouts: list[SeqLayout]) -> list[Span]:
    spans, off = [], 0
    for lay in layouts:
        spans.append(lay.spans(start=off))
        off += lay.total
    return spans
