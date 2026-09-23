from __future__ import annotations

from enum import Enum


class Lifetime(Enum):
    # Valid within one generation request: computed once and reused across all steps
    REQUEST = "request"
    # Depends on the current generation state: recomputed at every step
    STEP = "step"


# For conceptual explanation only; not an attribute of Region. Do not add a lifetime field to Region.
#
# Each category couples two questions: when a value becomes invalid and how long its K/V stays
# resident. Static text anchors are unchanged across steps but their K/V are discarded after
# extraction, so neither category fits them. Region roles imply Lifetime (VIS/HID -> REQUEST,
# LIVE -> STEP), but not the reverse: Lifetime carries no visibility and hence no attention grouping.
#
# CFG branches are a CacheKey field rather than a Lifetime category.
REQUEST = Lifetime.REQUEST
STEP = Lifetime.STEP
