from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...core.state import ModelOutput, PreparedCondition
from ...transformer.kv import FlatKV, kv_plan
from ...transformer.regions import Span
from . import layout as layout_mod

# Adapter over an OmniVoice build with SelfText attention:
#   model.use_selftext(True, backend=..., layout=...)
#   model(..., seg=plan, mode="prefill"|"cached"|"full")
#
# extract is an explicit mode="prefill" forward before the denoising loop; skip_head=True lets
# prefill bypass the 8x1025-dimensional output head.
#
# HF use_cache must be disabled; otherwise DynamicCache.update() prepends cached keys before the
# attention interface and corrupts every segment index.


@dataclass
class OmniVoiceAdapter:
    model: Any
    layout_name: str = "text_live"
    mask_id: int = 1024
    num_codebook: int = 8

    def prepare_condition(self, inputs: Any) -> PreparedCondition:
        if isinstance(inputs, PreparedCondition):
            return inputs
        payload = dict(inputs) if isinstance(inputs, dict) else {"raw": inputs}
        for k in ("input_ids", "audio_mask", "position_ids"):
            assert k in payload, f"prepare_condition is missing {k}"
        lay = layout_mod.build(
            self.layout_name,
            style=payload.get("n_style", 0),
            ref=payload.get("n_ref", 0),
            text=payload.get("n_text", 0),
            target=payload["n_target"],
        )
        payload["spans"] = payload.get("spans") or [lay.spans()]
        return PreparedCondition(payload=payload, layout=lay)

    def extract(self, condition: PreparedCondition, store=None) -> FlatKV:
        """Perform an explicit prefill pass."""
        spans: list[Span] = condition["spans"]
        plan = kv_plan(spans, device=condition["input_ids"].device)
        cfg = self.model.config
        kv = FlatKV(
            num_layers=cfg.num_hidden_layers,
            plan=plan,
            num_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            dtype=self.model.dtype,
            device=condition["input_ids"].device,
            pool=store.pool if store is not None else None,
        )
        sel = _frozen_rows(spans, condition["input_ids"].device)
        ctx = self.model._omni_ctx
        ctx["kbuf"], ctx["vbuf"] = kv.key, kv.value
        self.model(
            input_ids=condition["input_ids"][:, :, sel],
            audio_mask=condition["audio_mask"][:, sel],
            position_ids=condition["position_ids"][:, sel],
            seg=plan,
            mode="prefill",
            skip_head=True,
        )
        return kv

    def predict(self, state, step, condition, reusable=None) -> ModelOutput:
        seg = condition["step_seg"] if reusable is not None else condition["full_seg"]
        mode = "cached" if reusable is not None else "full"
        ids = self._write_state(condition, state, cached=reusable is not None)
        logits = self.model(
            input_ids=ids,
            audio_mask=condition["step_audio_mask" if reusable is not None else "audio_mask"],
            position_ids=condition["step_position_ids" if reusable is not None else "position_ids"],
            seg=seg,
            mode=mode,
        ).logits
        return ModelOutput(prediction=self._slice_target(logits, condition))

    def init_state(self, condition: PreparedCondition, **kwargs):
        """Initialize every position to MASK."""
        n = condition["n_target"]
        b = condition.get("batch", 1)
        return torch.full(
            (b, self.num_codebook, n),
            self.mask_id,
            dtype=torch.long,
            device=condition["input_ids"].device,
        )

    def sample_train_state(self, batch):
        """Stage I: randomly mask ground-truth tokens.

        mask_ratio ~ U(0,1) is the continuous noise level for discrete diffusion.
        """
        ids = batch["input_ids"].clone()
        rows = batch["target_rows"]
        ratio = float(torch.rand(()))
        sel = torch.rand(ids[:, :, rows].shape, device=ids.device) < ratio
        seg = ids[:, :, rows]
        seg[sel] = self.mask_id
        ids[:, :, rows] = seg
        return ids, batch.get("step", 0)

    def select(self, state, condition):
        """Align with the teacher only at positions that remain MASK and are audio rows.

        `labels != -100` cannot be used: because mask_ratio is random, unmasked target
        positions also have label -100 and would be conflated with the reference prompt.
        """
        am = condition["audio_mask"]
        return (state == self.mask_id) & am.unsqueeze(1)

    def fold(self, pairs):
        raise NotImplementedError(
            "OmniVoice performs backward propagation per state and does not fold: keeping the "
            "computation graphs of four states alive simultaneously runs out of memory"
        )

    def _write_state(self, condition, state, cached: bool):
        key = "step_ids" if cached else "input_ids"
        ids = condition[key]
        out_at = condition["out_at_cached" if cached else "out_at_full"]
        t = state.shape[-1]
        # audio_mask and out_at come from different sources; a wrong offset stays in bounds and
        # would silently write audio tokens into text rows.
        am = condition["step_audio_mask" if cached else "audio_mask"]
        for i, at in enumerate(out_at):
            assert bool(am[0, at : at + t].all()), f"doc {i}: out_at falls on a non-audio row"
            ids[0, :, at : at + t] = state[min(i, state.shape[0] - 1)]
        return ids

    def _slice_target(self, logits, condition):
        out_at = condition["out_at_cached"]
        t = condition["n_target"]
        return logits[:, :, out_at[0] : out_at[0] + t, :]


def _frozen_rows(spans: list[Span], device) -> torch.Tensor:
    """Input rows for prefill. Includes hid because the frozen group is bidirectional;
    omitting it would change the attention topology.

    However, only vis enters the buffer; see the documentation for kv.KVPlan.
    """
    parts = [
        torch.arange(sp.start, sp.start + sp.frozen, device=device)
        for sp in spans
        if sp.frozen
    ]
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long, device=device)
