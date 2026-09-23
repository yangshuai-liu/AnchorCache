from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...core.state import ModelOutput, PreparedCondition

# Teacher-side adapter for distill_step. The teacher:
#
#   1. runs the official full-attention forward, Flux2Transformer2DModel.forward. enable_self_text
#      patches forward on the instance only, so the class method stays the full-attention forward;
#   2. may live on a different device (paired mode) and moves inputs/outputs accordingly;
#   3. slices the output with [:, n_ref:]: the official forward returns ref + target tokens while
#      the student returns target tokens only. Without the slice, MSE would still broadcast
#      silently when n_ref happens to be zero.


@dataclass
class FullAttnTeacher:
    """Paired mode: an independent copy of frozen base weights."""

    transformer: Any
    device: Any = None
    out_device: Any = None
    # Must equal the student's Flux2Adapter.ts_scale.
    ts_scale: float = 1.0

    def prepare_condition(self, inputs: Any) -> PreparedCondition:
        raise AssertionError("The teacher does not prepare conditions; use student.prepare_condition")

    def extract(self, condition, store=None):
        raise AssertionError("The teacher uses full-attention topology and has no reusable state")

    def predict(self, state, step, condition, reusable=None) -> ModelOutput:
        assert reusable is None, "The teacher does not accept reusable state"
        dev = self.device or state.device
        emb = condition["emb"]
        ref = condition["ref_pack"]
        n_ref = ref.shape[1]

        ts = step
        if not isinstance(ts, torch.Tensor):
            ts = torch.tensor(ts)
        if ts.ndim == 0:
            ts = ts.expand(state.shape[0])
        if self.ts_scale != 1.0:
            ts = ts * self.ts_scale

        with torch.inference_mode():
            pred = self._forward(
                hidden_states=torch.cat([ref, state], dim=1).to(dev),
                timestep=ts.to(dev, emb.dtype),
                encoder_hidden_states=emb.to(dev),
                img_ids=torch.cat([condition["ref_ids"], condition["tgt_ids"]], dim=1).to(dev),
                txt_ids=condition["txt_ids"].to(dev),
                return_dict=False,
            )[0][:, n_ref:]
        # clone is required: tensors created in inference_mode cannot participate in
        # autograd. Using one directly in the loss raises
        # "Inference tensors cannot be saved for backward".
        return ModelOutput(prediction=pred.to(self.out_device or state.device).clone())

    def _forward(self, **kw):
        return self.transformer(**kw)


@dataclass
class SharedTeacher(FullAttnTeacher):
    """Shared mode: shares base weights with the student and invokes the official
    forward pass with LoRA disabled.

    This is valid only when train_mode=lora. Full mode has no adapter to disable,
    so the student weights are the teacher weights and the distillation target
    degenerates into self-distillation.
    """

    def _forward(self, **kw):
        from diffusers import Flux2Transformer2DModel

        self.transformer.disable_adapters()
        try:
            return Flux2Transformer2DModel.forward(self.transformer, **kw)
        finally:
            self.transformer.enable_adapters()
