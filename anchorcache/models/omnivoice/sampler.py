from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# OmniVoice's state update is unrelated to numerical integration; it uses
# confidence-based iterative unmasking (MaskGIT / masked discrete diffusion), following the
# upstream OmniVoice generation loop.
#
# This sampler must reside under models/ rather than sampling/ because each rule
# is part of the model semantics:
#   t_shift rational transform  Extremely slow initially and extremely fast
#                               near the end; unrelated to the sigma shift in
#                               flow matching
#   layer_penalty               Fill lower RVQ codebooks first so confidence
#                               scores are comparable across codebooks
#   single-assignment commit    Filled positions cannot be overwritten; this is
#                               not a numerical update


def time_steps(num_step: int, t_shift: float = 0.1, device=None) -> torch.Tensor:
    """Möbius transform t_shift*t / (1 + (t_shift-1)*t).

    With t_shift=0.1, the fill rate remains below 20% for 23 of 32 steps. This is
    why training must stratify samples by fill rate rather than step index
    (see training/states.sample_steps).
    """
    ts = torch.linspace(0.0, 1.0, num_step + 1, device=device)
    return t_shift * ts / (1 + (t_shift - 1) * ts)


def quota(total_slots: int, num_step: int, t_shift: float = 0.1) -> list[int]:
    """Number of slots to unmask per step. The final step consumes all remaining slots."""
    ts = time_steps(num_step, t_shift).tolist()
    rem = total_slots
    out = []
    for s in range(num_step):
        n = rem if s == num_step - 1 else min(math.ceil(total_slots * (ts[s + 1] - ts[s])), rem)
        out.append(int(n))
        rem -= int(n)
    return out


def fill_rates(total_slots: int, num_step: int, t_shift: float = 0.1) -> list[float]:
    """Filled proportion at the start of each step, used by sample_steps strata."""
    q = quota(total_slots, num_step, t_shift)
    out, done = [], 0
    for n in q:
        out.append(done / total_slots if total_slots else 0.0)
        done += n
    return out


@dataclass
class UnmaskSampler:
    mask_id: int = 1024
    num_step: int = 32
    t_shift: float = 0.1
    layer_penalty: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    total_slots: int | None = None

    def schedule(self, condition, **kwargs):
        return list(range(self.num_step))

    def step(self, state, prediction, step, condition=None):
        """`prediction` contains [B, C, T, V] logits or guided log-probabilities.

        `state` is an [B, C, T] int64 tensor containing mask_id. Returns the new state.
        When condition carries `target_segments` (per-document target lengths of a packed
        state along T), each document gets its own quota, matching per-item inference.
        """
        b, c, t = state.shape
        segs = condition.get("target_segments") if condition is not None else None
        if segs is None:
            spans = [(0, t, self.total_slots if self.total_slots is not None else c * t)]
        else:
            assert sum(segs) == t, f"target_segments {segs} do not cover state length {t}"
            starts = [sum(segs[:j]) for j in range(len(segs))]
            spans = [(s, n, c * n) for s, n in zip(starts, segs, strict=True)]
        ks = [quota(total, self.num_step, self.t_shift)[int(step)] for _, _, total in spans]
        if max(ks) <= 0:
            return state

        lp = prediction.float().log_softmax(-1)
        # As in upstream inference, predicting the mask token itself is forbidden.
        lp[..., self.mask_id] = -float("inf")
        if self.class_temperature > 0.0:
            pred = _gumbel(_top_k(lp, 0.1), self.class_temperature).argmax(-1)
        else:
            pred = lp.argmax(-1)
        scores = lp.max(-1).values

        layer_ids = torch.arange(c, device=state.device).view(1, c, 1)
        scores = scores - layer_ids * self.layer_penalty
        if self.position_temperature > 0.0:
            scores = _gumbel(scores, self.position_temperature)
        # Filled positions cannot be overwritten. Omitting this step raises no error;
        # it only causes the 32 steps to overwrite one another.
        scores = scores.masked_fill(state != self.mask_id, -float("inf"))

        out = state.clone()
        for i in range(b):
            for (s, n, _), k in zip(spans, ks, strict=True):
                blk = state[i, :, s : s + n]
                avail = int((blk == self.mask_id).sum())
                if k <= 0 or avail == 0:
                    continue
                _, top = scores[i, :, s : s + n].flatten().topk(min(k, avail))
                flat = blk.flatten().clone()
                flat[top] = pred[i, :, s : s + n].flatten()[top]
                out[i, :, s : s + n] = flat.view(c, n)
        return out

    def strata(self, total_slots: int) -> list[float]:
        return fill_rates(total_slots, self.num_step, self.t_shift)


def _top_k(logits: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    k = math.ceil(ratio * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    out = torch.full_like(logits, float("-inf"))
    return out.scatter_(-1, ind, val)


def _gumbel(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled = logits / temperature
    u = torch.rand_like(scaled)
    return scaled + (-torch.log(-torch.log(u + 1e-10) + 1e-10))
