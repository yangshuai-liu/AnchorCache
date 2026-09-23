from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Core recognizes only four concepts: condition / state / reusable state / prediction.
# Their concrete meanings differ substantially across the three backends, so no type constraints are applied here:
#
#   state       Qwen  : packed target latent  [B, G, C]      fp
#               FLUX  : packed target latent  [B, G, C]      fp
#               Omni  : RVQ token tensor      [B, 8, T]      int64, including MASK
#
#   prediction  Qwen  : velocity              [B, G, C]      fp
#               FLUX  : velocity              [B, G, C]      fp
#               Omni  : logits                [B, 8, S, V]   fp
#
# Forcing state into a Tensor would immediately break on OmniVoice: its state consists of discrete tokens,
# and its update semantics are "filled positions cannot be filled again", not numerical addition.


GenerationState = Any


@dataclass
class PreparedCondition:
    """Input that remains fixed within a generation request.

    payload is defined by the adapter; Core does not interpret its fields.
    layout is the only shared part consumed by the transformer/ layer.

    Important: layout is the layout of "the adapter that prepared it", not a shared truth.
    Teacher (full attention) and student (anchored) topologies differ, so an adapter with a
    different topology must derive its layout from payload rather than use this field.
    """

    payload: dict[str, Any] = field(default_factory=dict)
    layout: Any = None

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


class ReusableState:
    """Model state reusable across steps.

    Deliberately not called KVCache: KV is merely the implementation form currently shared by the three backends, not the concept itself.
    Qwen's cond_k/cond_v, FLUX's Flux2KVCache, and OmniVoice's flat kbuf/vbuf
    are all realizations of it.
    """

    __slots__ = ()


@dataclass
class ModelOutput:
    """Result of one predict call.

    Contains only prediction. No velocity / epsilon / logits / parameterization fields:
    the physical meaning is jointly interpreted by the adapter and sampler; Core treats it only as a block of data.
    """

    prediction: Any
