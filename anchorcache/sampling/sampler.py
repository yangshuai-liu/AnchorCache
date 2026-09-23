from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# Only the Sampler signature is shared; numerical algorithms are not.
#
#   Qwen / FLUX  Wrappers around diffusers FlowMatchEulerDiscreteScheduler.step with
#                linspace(1, 1/N, N) sigmas and a dynamic mu shift.
#   OmniVoice    Confidence top-k unmasking; its "timestep" only sets how many masked slots to
#                reveal per step (models/omnivoice/sampler.py).
#
# Scheduler math is delegated to diffusers rather than reimplemented, to avoid numerical drift.


class SchedulerSampler:
    """Minimal wrapper around a diffusers scheduler.

    deepcopy on every schedule() call: the scheduler is stateful (_step_index), and reuse across
    samples causes contamination.
    """

    def __init__(
        self,
        scheduler,
        num_steps: int,
        shift_fn=None,
        retrieve_fn=None,
        seq_len_fn=None,
        device_fn=None,
    ):
        self.base = scheduler
        self.num_steps = num_steps
        self.shift_fn = shift_fn
        self.retrieve_fn = retrieve_fn
        # StateProvider calls schedule(condition) without seq_len/device. Without seq_len the mu shift is
        # lost and the rollout schedule differs from inference, so backends must inject these hooks.
        self.seq_len_fn = seq_len_fn
        self.device_fn = device_fn
        self._active = None

    def schedule(self, condition, *, seq_len: int | None = None, device=None) -> Iterable[Any]:
        import copy

        import numpy as np

        if seq_len is None and self.seq_len_fn is not None:
            seq_len = self.seq_len_fn(condition)
        if device is None and self.device_fn is not None:
            device = self.device_fn(condition)

        sched = copy.deepcopy(self.base)
        sigmas = np.linspace(1.0, 1.0 / self.num_steps, self.num_steps)
        kwargs = {"sigmas": sigmas}
        if self.shift_fn is not None:
            assert seq_len is not None, "shift_fn requires seq_len; pass it or provide seq_len_fn"
            kwargs["mu"] = self.shift_fn(seq_len, self.num_steps)
        if self.retrieve_fn is not None:
            timesteps, _ = self.retrieve_fn(sched, self.num_steps, device, **kwargs)
        else:
            sched.set_timesteps(device=device, **kwargs)
            timesteps = sched.timesteps
        sched.set_begin_index(0)
        self._active = sched
        return timesteps

    def sigmas(self) -> list[float]:
        """Sigma sequence of the current grid, used by sample_steps(strata=...).

        Stratification must be based on noise level rather than step index: the shifted grid is nonuniform,
        so sampling by step index places most samples in the same noise interval.
        """
        assert self._active is not None, "call schedule() first"
        return [float(s) for s in self._active.sigmas[: self.num_steps]]

    def step(self, state, prediction, step, condition=None):
        assert self._active is not None, "call schedule() first to establish the timestep grid"
        return self._active.step(prediction, step, state, return_dict=False)[0]
