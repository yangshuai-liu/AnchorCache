from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ..core.execution import Adapter
from ..core.state import PreparedCondition, ReusableState
from .matching import Match, mse
from .states import StateProvider

# The entirety of distillation:
#
#           StateProvider
#                 │
#         ┌───────┴───────┐
#   teacher-forced   student rollout
#         └───────┬───────┘
#                 ▼
#              state_t
#                 │
#         ┌───────┴───────┐
#         ▼               ▼
#      Teacher         Student
#         └───────┬───────┘
#                 ▼
#          output matching
#
# The only branch is at the top. Stage I and Stage II share the same distill_step; this is a training consistency constraint: there must not be two prediction matching implementations.


def distill_step(
    teacher: Adapter,
    student: Adapter,
    state: Any,
    step: Any,
    condition: PreparedCondition,
    *,
    student_reusable: ReusableState | None = None,
    teacher_reusable: ReusableState | None = None,
    match: Match = mse,
    select_fn: Callable[[Any, PreparedCondition], torch.Tensor] | None = None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """same state, same step, same condition -> two predictions -> matching.

    teacher_reusable defaults to None: the teacher usually has a full-attention topology and no reusable state
    available: it is the frozen base model with full attention, independent of the student's
    warm-start weights.

    select_fn restricts matching to the positions inference reads. OmniVoice uses it for positions that
    are still masked and belong to audio rows; Qwen/FLUX supervise all target tokens and pass None.
    """
    with torch.no_grad():
        t_out = teacher.predict(state, step, condition, teacher_reusable)
    s_out = student.predict(state, step, condition, student_reusable)
    select = select_fn(state, condition) if select_fn is not None else None
    return match(s_out.prediction, t_out.prediction, select=select, weights=weights)


@dataclass
class Distiller:
    """Trainer shared by both stages.

    The backward strategy in train_step is not optional, but a memory constraint: backward each state immediately,
    discard its computation graph, and reduce peak memory to the single-state level. The alternative
    (fold_batch=True) concatenates K states into batch=K for one forward with gradient checkpointing.

    no_sync is the corresponding mechanism: when backward is performed state by state, gradients should
    synchronize only on the final call; otherwise every state triggers an all_reduce.
    """

    teacher: Adapter
    student: Adapter
    match: Match = mse
    select_fn: Callable | None = None
    weights: torch.Tensor | None = None
    fold_batch: bool = False
    backward: Callable[[torch.Tensor, bool], None] | None = None

    def train_step(self, provider: StateProvider, batch: Any) -> dict:
        condition = self.student.prepare_condition(batch)
        pairs = provider.states(self.student, batch)
        assert pairs, "StateProvider returned no states; every rank must supervise the same number of states"

        if self.fold_batch:
            state, step = self.student.fold(pairs)
            loss = distill_step(
                self.teacher,
                self.student,
                state,
                step,
                condition,
                match=self.match,
                select_fn=self.select_fn,
                weights=self.weights,
            )
            if self.backward is not None:
                self.backward(loss, True)
            return {"loss": float(loss.detach()), "n_states": len(pairs)}

        total = 0.0
        n = len(pairs)
        for i, (state, step) in enumerate(pairs):
            loss = distill_step(
                self.teacher,
                self.student,
                state,
                step,
                condition,
                match=self.match,
                select_fn=self.select_fn,
                weights=self.weights,
            ) / n
            if self.backward is not None:
                self.backward(loss, i == n - 1)
            total += float(loss.detach())
        return {"loss": total, "n_states": n}
