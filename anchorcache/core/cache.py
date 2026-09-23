from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class CacheKey:
    """Logical cache coordinates.

    branch exists for CFG: the positive and negative branches either keep separate reference
    caches (anchored K/V depend on the prompt) or map to the same key (isolated cache).
    """

    request: str
    layer: int
    branch: str = "cond"
    slot: str = "kv"


@runtime_checkable
class CacheStore(Protocol):
    """Logical cache. The model sees only these four operations.

    Placement details such as host copies, pinned buffers, copy streams and GPU pools belong in
    runtime/, never in model code.
    """

    def put(self, key: CacheKey, value: Any) -> None: ...

    def get(self, key: CacheKey) -> Any: ...

    def has(self, key: CacheKey) -> bool: ...

    def release(self, request: str) -> None: ...


class MissTriggeredExtract:
    """Trigger criterion for extract.

    Extraction is triggered by a cache miss rather than by step_index == 0, so the generation
    loop never depends on the first denoising step to build the cache.
    """

    def __init__(self, store: CacheStore):
        self.store = store

    def needs_extract(self, request: str, branch: str = "cond", layer: int = 0) -> bool:
        return not self.store.has(CacheKey(request=request, layer=layer, branch=branch))
