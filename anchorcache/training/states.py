from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from ..core.execution import Adapter, Guidance, Sampler
from ..sampling.rollout import rollout

# The only difference between the two training stages is where the state comes from.
#
#   Stage I  teacher-forced   state = noise added to the data distribution / random masking
#   Stage II on-policy        state = trajectory points rolled out by the student itself
#
#   Qwen / FLUX  Stage I (1-σ)x0 + σε; Stage II rollout from pure noise
#   OmniVoice    Stage I randomly masks ground-truth tokens; Stage II rollout from all-MASK
#
# Output matching is identical across stages; only the state source differs, hence StateProvider.


@runtime_checkable
class StateProvider(Protocol):
    def states(self, adapter: Adapter, batch: Any) -> list[tuple[Any, Any]]:
        """Return [(state, step), ...]. The type of step is determined by the adapter/sampler."""
        ...


@dataclass
class TeacherForced:
    """Stage I. The state is constructed directly from data by the adapter.

    interpolate is provided by the adapter because it represents model semantics:
      Qwen/FLUX  z_t = (1-σ)x0 + σε   (continuous interpolation)
      OmniVoice  randomly replace ground-truth tokens with MASK according to mask_ratio (discrete)
    Forcing these into one mathematical expression immediately fails for OmniVoice.
    """

    num_states: int = 1

    def states(self, adapter: Adapter, batch: Any) -> list[tuple[Any, Any]]:
        return [adapter.sample_train_state(batch) for _ in range(self.num_states)]


@dataclass
class StudentRollout:
    """Stage II. The state comes from the student's own trajectory.

    It goes through sampling.rollout, the same path as inference, so the schedule, the sampling
    rule and masking of the mask token all match deployment.
    """

    sampler: Sampler
    num_states: int = 4
    guidance: Guidance | None = None
    lo: int = 1
    hi: int | None = None
    strata: Sequence[float] | None = None
    strata_fn: Any = None
    seed: int = 0
    rank: int = 0
    shared_across_ranks: bool = True
    step_counter: int = 0

    def states(self, adapter: Adapter, batch: Any) -> list[tuple[Any, Any]]:
        condition = adapter.prepare_condition(batch)
        schedule = list(self.sampler.schedule(condition))
        n_steps = len(schedule)
        hi = (n_steps - 1) if self.hi is None else self.hi
        # strata_fn must be obtained after schedule: schedule constructs the grid, while stratification must
        # use noise level rather than step index. The sampler's sigma grid is not
        # in the Sampler protocol — it belongs to the backend, so inject it instead of calling it directly.
        strata = self.strata if self.strata_fn is None else self.strata_fn()
        idx = sample_steps(
            n_steps,
            self.num_states,
            seed=self._seed(),
            lo=self.lo,
            hi=hi,
            strata=strata,
        )
        self.step_counter += 1
        traj = rollout(
            adapter,
            condition,
            self.sampler,
            guidance=self.guidance,
            schedule=schedule,
            capture=idx,
            max_step=max(idx),
        )
        return [(s, t) for _, s, t in traj.captured]

    def _seed(self) -> int:
        base = self.seed + self.step_counter * 100003
        return base if self.shared_across_ranks else base + self.rank


def sample_steps(
    step_count: int,
    sample_count: int,
    *,
    seed: int,
    lo: int = 0,
    hi: int | None = None,
    strata: Sequence[float] | None = None,
) -> tuple[int, ...]:
    """Choose which state steps to supervise.

    Two strict requirements:

    1. Always return min(sample_count, number of available steps) step indices.
       Each state requires one forward with gradients, hence one collective communication. If ranks receive
       different counts, the numbers of FSDP _all_gather_flat_param / DDP all_reduce calls become mismatched,
       causing an immediate watchdog timeout or NCCL SIGABRT without a Python traceback.
       A rank-shared seed (identical indices) or rank-dependent seeds with an identical count are
       both valid.

    2. When strata is provided, stratify by "fill ratio/noise level," not by step index.
       With OmniVoice's t_shift=0.1, 23/32 steps have a fill ratio below 20%; uniform sampling by step index
       has a 72% probability of selecting only almost-entirely-masked early states, leaving the later refinement
       stage effectively unsupervised. Qwen's sigma grid is likewise nonuniform (dynamic mu from calculate_shift).
    """
    hi = (step_count - 1) if hi is None else hi
    avail = list(range(max(lo, 0), min(hi, step_count - 1) + 1))
    assert avail, f"no steps available for sampling: lo={lo} hi={hi} step_count={step_count}"
    n = min(sample_count, len(avail))
    g = torch.Generator().manual_seed(int(seed))

    if strata is None:
        perm = torch.randperm(len(avail), generator=g)[:n].tolist()
        return tuple(sorted(avail[i] for i in perm))

    f = torch.tensor([float(strata[i]) for i in avail])
    # Divide [0,1) into n segments, randomly choose a target level within each segment, and assign each
    # the nearest unclaimed step. Simple deduplication would reduce the returned count (requirement 1).
    u = (torch.arange(n) + torch.rand(n, generator=g)) / n
    taken: list[int] = []
    for t in u.tolist():
        d = (f - t).abs()
        if taken:
            d[torch.tensor(taken)] = float("inf")
        taken.append(int(d.argmin()))
    return tuple(sorted(avail[i] for i in taken))
