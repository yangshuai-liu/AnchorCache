from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


def load_model(path: str, device: str, dtype: str = "float16", *, train: bool = False):
    import torch
    from omnivoice.models.omnivoice import OmniVoice

    torch_dtype = getattr(torch, dtype)
    kwargs = {"dtype": torch_dtype, "train": train, "attn_implementation": "sdpa"}
    if not train:
        kwargs["device_map"] = device
    model = OmniVoice.from_pretrained(path, **kwargs)
    if train:
        model.to(device)
    return model


def enable_selftext(model, backend: str = "flash", layout: str = "text_live"):
    model.use_selftext(True, backend=backend, layout=layout)
    assert model.llm.config.use_cache is False, (
        "SelfText requires HF use_cache to be disabled; otherwise, DynamicCache corrupts segment indices"
    )
    return model


def _seg(plan) -> dict:
    if isinstance(plan, dict):
        return plan
    return {
        "pre_slot": plan.vis_slot,
        "pre_row": plan.vis_row,
        "tgt_slot": plan.live_slot,
        "cu_pre": plan.cu_vis,
        "cu_tgt": plan.cu_live,
        "cu_main": plan.cu_main,
        "max_pre": plan.max_vis,
        "max_tgt": plan.max_live,
        "max_main": plan.max_main,
        "total": plan.total,
    }


class ModelView:
    """Convert only the AnchorCache KVPlan boundary to the upstream SelfText segment dict."""

    def __init__(self, model):
        lc = model.config.llm_config
        self.model = model
        self.config = SimpleNamespace(
            num_hidden_layers=lc.num_hidden_layers,
            num_key_value_heads=lc.num_key_value_heads,
            head_dim=lc.head_dim,
        )

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def _omni_ctx(self):
        return self.model._omni_ctx

    def __call__(self, *args, **kwargs):
        if "seg" in kwargs:
            kwargs["seg"] = _seg(kwargs["seg"])
        return self.model(*args, **kwargs)


@dataclass
class PackedAdapter:
    model: Any
    teacher: bool = False
    mask_id: int = 1024

    def prepare_condition(self, batch):
        from anchorcache.core.state import PreparedCondition

        if isinstance(batch, PreparedCondition):
            return batch
        return PreparedCondition(payload=batch)

    def sample_train_state(self, batch):
        ids = batch["input_ids"].index_select(2, batch["target_rows"])
        return ids, batch.get("step", 0)

    def init_state(self, condition, **kwargs):
        import torch

        ids = condition["input_ids"]
        n = condition["target_rows"].numel()
        return torch.full(
            (ids.shape[0], ids.shape[1], n),
            self.mask_id,
            dtype=torch.long,
            device=ids.device,
        )

    def predict(self, state, step, condition, reusable=None):
        from anchorcache.core.state import ModelOutput

        rows = condition["target_rows"]
        ids = condition["input_ids"].clone()
        ids.index_copy_(2, rows, state)
        if self.teacher:
            idx = condition["main_rows"]
            ids = ids.index_select(2, idx)
            am = condition["audio_mask"].index_select(1, idx)
            pos = condition["position_ids"].index_select(1, idx)
            seg = condition["teacher_seg"]
            out_rows = condition["teacher_rows"]
        else:
            am = condition["audio_mask"]
            pos = condition["position_ids"]
            seg = condition["seg"]
            out_rows = rows
        logits = self.model(
            input_ids=ids,
            audio_mask=am,
            position_ids=pos,
            seg=seg,
            mode="full",
        ).logits
        return ModelOutput(prediction=logits.index_select(2, out_rows))

    def select(self, state, condition):
        return state == self.mask_id

    def extract(self, condition, store=None):
        return None

    def fold(self, pairs):
        raise NotImplementedError("OmniVoice must backpropagate each state separately; batch folding is unsupported")


def prepare_batch(batch: dict, device) -> dict:
    import torch
    from omnivoice.data.seg import seg_pack

    def move(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, dict):
            return {k: move(v) for k, v in x.items()}
        return x

    batch = move(batch)
    seg = batch["seg"]
    live = seg["tgt_idx"]
    is_audio = batch["audio_mask"][0, live]
    rows = live[is_audio]
    assert rows.numel(), "batch contains no target audio rows; cannot construct an OmniVoice distillation state"
    # Per-document target lengths: each packed document gets its own unmask quota.
    cu = seg["cu_tgt"].tolist()
    sizes = [int(is_audio[cu[i] : cu[i + 1]].sum()) for i in range(len(cu) - 1)]
    batch["target_segments"] = [n for n in sizes if n]

    main = seg["main_idx"]
    cu = seg["cu_main"].tolist()
    spans = [(cu[i], 0, 0, cu[i + 1] - cu[i]) for i in range(len(cu) - 1)]
    teacher_seg = seg_pack(spans, device=device)
    row_map = torch.full(
        (batch["input_ids"].shape[-1],), -1, dtype=torch.long, device=device
    )
    row_map[main] = torch.arange(main.numel(), device=device)
    teacher_rows = row_map[rows]
    assert bool((teacher_rows >= 0).all()), "target rows are absent from main_idx; the SelfText segment is corrupted"

    batch["target_rows"] = rows
    batch["main_rows"] = main
    batch["teacher_seg"] = teacher_seg
    batch["teacher_rows"] = teacher_rows
    return batch
