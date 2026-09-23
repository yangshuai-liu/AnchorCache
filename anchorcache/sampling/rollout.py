from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from ..core.execution import Adapter, Guidance, Sampler
from ..core.state import GenerationState, PreparedCondition, ReusableState


@dataclass
class Trajectory:
    """Artifacts retained along the rollout path.

    captured contains (step_index, state, step) triples. The training side retrieves supervision
    points from here, while the inference side only needs final.
    """

    final: GenerationState
    captured: list[tuple[int, GenerationState, Any]] = field(default_factory=list)
    predictions: list[Any] = field(default_factory=list)

    def states(self) -> list[GenerationState]:
        return [s for _, s, _ in self.captured]

    def steps(self) -> list[Any]:
        return [t for _, _, t in self.captured]


def rollout(
    adapter: Adapter,
    condition: PreparedCondition,
    sampler: Sampler,
    *,
    reusable: ReusableState | None = None,
    guidance: Guidance | None = None,
    state: GenerationState | None = None,
    schedule: Iterable[Any] | None = None,
    capture: Sequence[int] = (),
    grad_steps: Sequence[int] = (),
    max_step: int | None = None,
    keep_predictions: bool = False,
) -> Trajectory:
    """The sole generation loop.

    normal inference / student rollout / evaluation / guided rollout ablation all go through
    this path: if they diverge, training optimizes a trajectory that inference never follows.

    capture     Step indices whose states should be retained.
    grad_steps  Steps whose predictions carry gradients; the rest use inference_mode.
    max_step    Capture the state at this step and stop before predicting there. On-policy
                distillation uses max(sample_idx) so no forward is spent past the last
                supervised state. When None, every step is predicted and applied, so final
                is the fully denoised state.
    """
    if state is None:
        state = adapter.init_state(condition)
    steps = list(schedule) if schedule is not None else list(sampler.schedule(condition))
    cap = set(int(i) for i in capture)
    grad = set(int(i) for i in grad_steps)
    stop = len(steps) if max_step is None else int(max_step)

    traj = Trajectory(final=state)
    for i, step in enumerate(steps):
        if i in cap:
            traj.captured.append((i, _detach_state(state), step))
        if i >= stop:
            break

        if i in grad:
            out = _predict(adapter, state, step, condition, reusable, guidance)
            state = sampler.step(state, out.prediction, step, condition)
        else:
            with torch.inference_mode():
                out = _predict(adapter, state, step, condition, reusable, guidance)
                state = sampler.step(state, out.prediction, step, condition)
            state = _detach_state(state)
        if keep_predictions:
            traj.predictions.append(out.prediction)

    traj.final = state
    return traj


def _predict(adapter, state, step, condition, reusable, guidance):
    if guidance is None:
        return adapter.predict(state, step, condition, reusable)
    return guidance.predict(adapter, state, step, condition, reusable)


def _detach_state(state: GenerationState) -> GenerationState:
    # clone is not for saving memory, but for escaping inference_mode: tensors created inside
    # inference_mode cannot participate in autograd; using them directly in a student forward raises
    # "Inference tensors cannot be saved for backward".
    if isinstance(state, torch.Tensor):
        return state.detach().clone()
    if isinstance(state, (list, tuple)):
        return type(state)(_detach_state(s) for s in state)
    if isinstance(state, dict):
        return {k: _detach_state(v) for k, v in state.items()}
    return state


def generate(
    adapter: Adapter,
    inputs: Any,
    sampler: Sampler,
    *,
    guidance: Guidance | None = None,
    cache: bool = True,
    store: Any = None,
) -> GenerationState:
    """condition -> extract once -> iterative prediction -> state update.

    extract is an explicit line, not hidden inside a step check in the loop. This is an inference loop constraint:
    the generic generation loop must not depend on the first denoising step to decide whether to build the cache.
    """
    condition = adapter.prepare_condition(inputs)
    reusable = adapter.extract(condition, store) if cache else None
    state = adapter.init_state(condition)
    return rollout(
        adapter, condition, sampler, reusable=reusable, guidance=guidance, state=state
    ).final
