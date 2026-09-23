from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.state import ModelOutput

# Guidance is a strategy object rather than part of adapter.predict, so training and inference can
# configure it independently. Both recovery stages are CFG-free; CFG is applied only at inference.



class NoGuidance:
    def predict(self, adapter, state, step, condition, reusable):
        return adapter.predict(state, step, condition, reusable)


@dataclass
class CfgGuidance:
    """Linear-extrapolation CFG. Use this for continuous predictions (velocity / epsilon).

    When rescale=True, scale the magnitude back to the conditional branch according to
    |cond| / |combined| (Qwen). FLUX does not rescale.

    The negative branch runs under no_grad: gradients flow only through the conditional forward.
    """

    scale: float = 4.0
    negative: str = "negative"
    rescale: bool = False
    # Sharing is exact only for the isolated cache; anchored K/V depend on the prompt.
    share_reusable: bool = True
    neg_reusable: object = None

    def predict(self, adapter, state, step, condition, reusable):
        cond_out = adapter.predict(state, step, condition, reusable)
        if self.scale <= 1.0:
            return cond_out
        neg_cond = condition.get(self.negative)
        assert neg_cond is not None, f"CFG requires condition['{self.negative}']"
        if self.share_reusable:
            neg_reusable = reusable
        elif self.neg_reusable is not None:
            neg_reusable = self.neg_reusable
        else:
            neg_reusable = adapter.extract(neg_cond)
        with torch.no_grad():
            neg_out = adapter.predict(state, step, neg_cond, neg_reusable)
        c, n = cond_out.prediction, neg_out.prediction
        merged = n + self.scale * (c - n)
        if self.rescale:
            cond_norm = torch.norm(c, dim=-1, keepdim=True)
            merged_norm = torch.norm(merged, dim=-1, keepdim=True).clamp_min(1e-12)
            merged = merged * (cond_norm / merged_norm)
        return ModelOutput(prediction=merged)


@dataclass
class LogProbCfgGuidance:
    """CFG for a discrete distribution: combine in log-probability space, then renormalize.

    The continuous version cannot be copied directly: after linearly combining logits, the
    distribution is no longer normalized, making the softmax output meaningless. OmniVoice uses
        log_softmax(c_lp + s*(c_lp - u_lp))
    """

    scale: float = 2.0
    negative: str = "uncond"
    dim: int = -1

    def predict(self, adapter, state, step, condition, reusable):
        import torch.nn.functional as F

        cond_out = adapter.predict(state, step, condition, reusable)
        if self.scale == 0.0:
            return cond_out
        neg_cond = condition.get(self.negative)
        assert neg_cond is not None, f"CFG requires condition['{self.negative}']"
        with torch.no_grad():
            neg_out = adapter.predict(state, step, neg_cond, None)
        c = F.log_softmax(cond_out.prediction.float(), dim=self.dim)
        u = F.log_softmax(neg_out.prediction.float(), dim=self.dim)
        return ModelOutput(prediction=F.log_softmax(c + self.scale * (c - u), dim=self.dim))


@dataclass
class AblationGuidance:
    """Recompute without reusable state and extrapolate using the difference.

    FLUX ref-CFG: v = v_cond + w*(v_cond - v_noref), where the no-ref branch is the model's
    forward without kv_cache.
    lo/hi restricts it to only some steps because it adds one forward per step.
    """

    weight: float = 0.5
    lo: int = 0
    hi: int = 10**9
    _calls: int = 0

    def predict(self, adapter, state, step, condition, reusable):
        cond_out = adapter.predict(state, step, condition, reusable)
        # Steps are counted per instance: use one AblationGuidance per generation.
        idx = self._calls
        self._calls += 1
        if self.weight <= 0.0 or not (self.lo <= idx < self.hi):
            return cond_out
        with torch.no_grad():
            bare = adapter.predict(state, step, condition, None)
        c, b = cond_out.prediction, bare.prediction
        return ModelOutput(prediction=c + self.weight * (c - b))
