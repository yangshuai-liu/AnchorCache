from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from .state import GenerationState, ModelOutput, PreparedCondition, ReusableState


@runtime_checkable
class Adapter(Protocol):
    """The only interface the three backends need to implement.

    prepare_condition -> extract -> predict is the entire interface.
    extract must be an independently callable stage and must not be hidden in predict's step checks.
    """

    def prepare_condition(self, inputs: Any) -> PreparedCondition: ...

    def extract(self, condition: PreparedCondition, store: Any = None) -> ReusableState: ...

    def predict(
        self,
        state: GenerationState,
        step: Any,
        condition: PreparedCondition,
        reusable: ReusableState | None = None,
    ) -> ModelOutput: ...

    def init_state(self, condition: PreparedCondition, **kwargs: Any) -> GenerationState: ...


@runtime_checkable
class Sampler(Protocol):
    """State update.

    Qwen / FLUX use wrappers around FlowMatchEulerDiscreteScheduler.step;
    OmniVoice uses confidence topk unmasking, which is entirely unrelated to numerical integration.
    Their signature is the only commonality.
    """

    def schedule(self, condition: PreparedCondition, **kwargs: Any) -> Iterable[Any]: ...

    def step(
        self,
        state: GenerationState,
        prediction: Any,
        step: Any,
        condition: PreparedCondition,
    ) -> GenerationState: ...


@runtime_checkable
class Guidance(Protocol):
    """Encapsulate "multi-branch predict + merge" so it does not leak into adapter.predict.

    The merge methods of the three backends cannot be unified:
      Qwen  v = v_neg + s*(v_cond - v_neg), then rescale by |v_cond|/|v|
      FLUX  v = v_neg + s*(v_cond - v_neg); ref-CFG additionally extrapolates from a no-ref branch
      Omni  merge in log-prob space, then renormalize with log_softmax (discrete distributions)
    Therefore, Guidance is a strategy object, not a first-class Core concept.
    """

    def predict(
        self,
        adapter: Adapter,
        state: GenerationState,
        step: Any,
        condition: PreparedCondition,
        reusable: ReusableState | None,
    ) -> ModelOutput: ...
