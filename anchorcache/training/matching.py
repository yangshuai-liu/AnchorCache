from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F

# Output matching is a replaceable callable with a backend-specific default rather than a hard-coded
# MSE. OmniVoice predicts [B, 8, T, 1025] logits, not velocities; among the divergences below,
# reverse KL preserved speaker similarity while forward KL was the only clearly worse option (its
# mode covering spreads mass over teacher modes the student cannot reach).



@runtime_checkable
class Match(Protocol):
    def __call__(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        *,
        select: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


def mse(student, teacher, *, select=None, weights=None):
    """Velocity MSE for Qwen / FLUX, computed in fp32 (bf16 summation loses precision)."""
    s, t = student.float(), teacher.float()
    if select is not None:
        s, t = s[select], t[select]
    return F.mse_loss(s, t, reduction="mean")


def logit_mse(student, teacher, *, select=None, weights=None, dim: int = -1):
    """L2 on centered logits.

    Centering is required: logits are invariant to a global shift (invisible to softmax); without
    subtracting the mean, an irrelevant degree of freedom is fitted.
    """
    s, t = _gather(student, teacher, select)
    d = (s - s.mean(dim, keepdim=True)) - (t - t.mean(dim, keepdim=True))
    return _reduce(d.pow(2).mean(dim), select, weights, student)


def reverse_kl(student, teacher, *, select=None, weights=None, temp: float = 1.0, dim: int = -1):
    """KL(student || teacher), mode-seeking. The default choice for OmniVoice."""
    s, t = _gather(student, teacher, select, temp)
    lp_s, lp_t = F.log_softmax(s, dim), F.log_softmax(t, dim)
    return _reduce((lp_s.exp() * (lp_s - lp_t)).sum(dim), select, weights, student)


def forward_kl(student, teacher, *, select=None, weights=None, temp: float = 1.0, dim: int = -1):
    """KL(teacher || student), mode-covering. Empirically the only significantly worse option on OmniVoice."""
    s, t = _gather(student, teacher, select, temp)
    lp_s, lp_t = F.log_softmax(s, dim), F.log_softmax(t, dim)
    return _reduce((lp_t.exp() * (lp_t - lp_s)).sum(dim), select, weights, student)


def js_div(student, teacher, *, select=None, weights=None, temp: float = 1.0, dim: int = -1):
    s, t = _gather(student, teacher, select, temp)
    lp_s, lp_t = F.log_softmax(s, dim), F.log_softmax(t, dim)
    lm = torch.logaddexp(lp_t, lp_s) - math.log(2.0)
    per = 0.5 * (lp_t.exp() * (lp_t - lm)).sum(dim) + 0.5 * (lp_s.exp() * (lp_s - lm)).sum(dim)
    return _reduce(per, select, weights, student)


def topk_kl(student, teacher, *, select=None, weights=None, temp: float = 1.0, k: int = 64, dim: int = -1):
    s, t = _gather(student, teacher, select, temp)
    lp_t = F.log_softmax(t, dim)
    val, idx = lp_t.topk(min(k, lp_t.shape[dim]), dim=dim)
    p_t = F.softmax(val, dim=dim)
    lp_s = F.log_softmax(s.gather(dim, idx), dim)
    per = (p_t * (p_t.clamp_min(1e-9).log() - lp_s)).sum(dim)
    return _reduce(per, select, weights, student)


def _gather(student, teacher, select, temp: float = 1.0):
    """Gather by select into [N, V] before computing.

    This is not cosmetic: computing over the entire [B, C, T, V] and then multiplying by the mask
    materializes three fp32 copies (537 MB each when T≈16384), the primary cause of OOM during
    multi-state accumulation. It is numerically identical.
    """
    if select is None:
        s, t = student.float(), teacher.float()
    else:
        idx = select.nonzero(as_tuple=True)
        s, t = student[idx].float(), teacher[idx].float()
    if temp != 1.0:
        s, t = s / temp, t / temp
    return s, t


def _reduce(per, select, weights, ref):
    """Per-position values -> scalar.

    When weights are provided, compute means by channel and then take a weighted sum, matching OmniVoice's
    CE loss structure (per-codebook mean, then normalized_audio_codebook_weights).
    """
    if per.numel() == 0:
        return ref.sum() * 0.0
    if weights is None:
        return per.mean()
    assert select is not None, "weights requires select to locate the channel dimension"
    ci = select.nonzero(as_tuple=True)[1]
    n_c = weights.numel()
    num = per.new_zeros(n_c).index_add_(0, ci, per)
    den = per.new_zeros(n_c).index_add_(0, ci, torch.ones_like(per))
    return ((num / den.clamp(min=1.0)) * weights.to(per)).sum()


MATCHES = {
    "mse": mse,
    "logit_mse": logit_mse,
    "reverse_kl": reverse_kl,
    "forward_kl": forward_kl,
    "js_div": js_div,
    "topk_kl": topk_kl,
}
