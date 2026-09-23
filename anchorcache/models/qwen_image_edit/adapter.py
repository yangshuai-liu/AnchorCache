from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ...core.state import ModelOutput, PreparedCondition
from ...transformer.kv import KVState, LayeredKV
from .layout import cached_img_shapes, layout_from_img_shapes

# Unified execution surface for the Qwen backend.
#
# Required local symbols:
#   transformer.forward_kv_extract / forward_kv_cached  processor.py
#   Qwen self-text forward                            model.py


@dataclass
class QwenAdapter:
    transformer: Any
    encode_prompt: Callable[..., Any] | None = None
    encode_vae: Callable[..., Any] | None = None
    prepare_latents: Callable[..., Any] | None = None
    anchor: bool = False
    num_layers: int | None = None
    _store_extract: bool = True

    def prepare_condition(self, inputs: Any) -> PreparedCondition:
        """Convert a text instruction and reference images into prompt embeddings and source latents.

        Each reference image follows two paths: condition images are resized to an area of 384² for
        the VLM, and VAE images to an area of 1024².
        """
        if isinstance(inputs, PreparedCondition):
            return inputs
        payload = dict(inputs) if isinstance(inputs, dict) else {"raw": inputs}
        assert "prompt_embeds" in payload and "source_latents" in payload, (
            "prepare_condition requires prompt_embeds and source_latents; "
            "provide them directly in inputs for offline pre-extraction, or inject "
            "encode_prompt / encode_vae for online extraction"
        )
        mask = payload.get("prompt_embeds_mask")
        # The self-text forward and the cached processor attend without a text mask.
        assert mask is None or bool(mask.all()), "padded prompt embeddings are not supported"
        img_shapes = payload["img_shapes"]
        txt_len = payload["prompt_embeds"].shape[1]
        layout = layout_from_img_shapes(img_shapes, txt_len, anchor=self.anchor)
        return PreparedCondition(payload=payload, layout=layout)

    def extract(self, condition: PreparedCondition, store=None) -> LayeredKV:
        """Extract explicitly, without relying on step == 0.

        CFG calls it once per branch. condition["frz_encoder_hs"] optionally overrides the anchor
        text embeddings (passed as attention_kwargs["qwen_frz_encoder_hs"]).
        """
        latents = self.init_state(condition)
        hidden = torch.cat([latents, condition["source_latents"]], dim=1)
        timestep = condition.get("extract_timestep")
        if timestep is None:
            # References and anchors use fixed timesteps; this only affects discarded live outputs.
            timestep = latents.new_ones(latents.shape[0])
        ak = dict(condition.get("attention_kwargs") or {})
        frz = condition.get("frz_encoder_hs")
        if frz is not None:
            ak["qwen_frz_encoder_hs"] = frz
        out, raw = self.transformer.forward_kv_extract(
            hidden_states=hidden,
            timestep=timestep,
            encoder_hidden_states=condition["prompt_embeds"],
            encoder_hidden_states_mask=condition.get("prompt_embeds_mask"),
            img_shapes=condition["img_shapes"],
            attention_kwargs=ak if ak else None,
            return_dict=False,
            num_target_tokens=latents.shape[1],
        )
        return _wrap_qwen_cache(raw)

    def predict(self, state, step, condition, reusable=None) -> ModelOutput:
        if reusable is None:
            # The training path deliberately uses the extraction topology without persistence:
            # cached is mathematically equivalent to extraction. Training uses the full-graph
            # self-text forward pass throughout, without persisting a mutable cache container.
            hidden = torch.cat([state, condition["source_latents"]], dim=1)
            pred = self.transformer(
                hidden_states=hidden,
                timestep=step,
                encoder_hidden_states=condition["prompt_embeds"],
                encoder_hidden_states_mask=condition.get("prompt_embeds_mask"),
                img_shapes=condition["img_shapes"],
                attention_kwargs=condition.get("extract_attention_kwargs"),
                return_dict=False,
            )[0][:, : state.shape[1]]
            return ModelOutput(prediction=pred)

        pred = self.transformer.forward_kv_cached(
            hidden_states=state,
            timestep=step,
            encoder_hidden_states=condition["prompt_embeds"],
            encoder_hidden_states_mask=condition.get("prompt_embeds_mask"),
            img_shapes=cached_img_shapes(condition["img_shapes"], state.shape[0]),
            attention_kwargs=condition.get("cached_attention_kwargs"),
            return_dict=False,
            kv_cache=_unwrap_qwen_cache(reusable),
        )[0]
        return ModelOutput(prediction=pred)

    def init_state(self, condition: PreparedCondition, **kwargs):
        state = condition.get("latents")
        if state is not None:
            return state
        layout = condition.layout
        shape = (
            condition["source_latents"].shape[0],
            layout.len_of("target"),
            condition["source_latents"].shape[-1],
        )
        gen = condition.get("generator")
        return torch.randn(
            shape,
            device=condition["source_latents"].device,
            dtype=condition["source_latents"].dtype,
            generator=gen,
        )

    def sample_train_state(self, batch):
        """Stage I: z_t = (1-σ)x0 + σε, with x0, ε, σ and the timestep supplied by the batch.

        Raw mode samples σ in data.prepare (--weighting); tensor mode stores it with the sample.
        """
        x0, noise, sigma = batch["latents"], batch["noise"], batch["sigmas"]
        while sigma.ndim < x0.ndim:
            sigma = sigma.unsqueeze(-1)
        return (1.0 - sigma) * x0 + sigma * noise, batch["timesteps"]

    def fold(self, pairs):
        """Combine K supervision points into a single batch=K forward pass.

        With gradient checkpointing, activations from a single-chain backward pass retain only one
        layer, so peak memory is comparable to batch=1.
        """
        states = torch.cat([s for s, _ in pairs], dim=0)
        steps = torch.cat([t for _, t in pairs], dim=0)
        return states, steps


def _wrap_qwen_cache(raw: dict) -> LayeredKV:
    """Convert the processor's dict cache into LayeredKV.

    The dict has two per-layer formats, selected by QWEN_KV_PREALLOC:
      {"k_ref", "v_ref"}                 concatenate the full length at each step
      {"k_buf", "v_buf", "live_len"}     preallocated; overwrite only the live prefix
    """
    layers = raw["layers"]
    out = LayeredKV(len(layers), num_vis_tokens=0)
    for i, lc in enumerate(layers):
        if lc is None:
            continue
        if "k_ref" in lc:
            out.put(i, KVState(lc["k_ref"], lc["v_ref"]))
        else:
            live = lc["live_len"]
            out.put(i, KVState(lc["k_buf"][:, live:], lc["v_buf"][:, live:]))
            out.meta.setdefault("live_len", live)
    out.num_vis_tokens = out.layers[0].num_tokens if out.layers[0] is not None else 0
    out.meta["num_target_tokens"] = raw.get("num_target_tokens")
    # Retain ownership of the backend's original container. Preallocated k_buf/v_buf must be reused
    # in place across cached steps. Saving only reference views and rebuilding them in _unwrap would
    # instead reallocate the full-length buffer at every step, negating the optimization.
    out.meta["qwen_cache"] = raw
    return out


def _unwrap_qwen_cache(kv: LayeredKV) -> dict:
    raw = kv.meta.get("qwen_cache")
    if raw is not None:
        return raw

    live_len = kv.meta.get("live_len")
    layers = []
    for s in kv.layers:
        if s is None:
            layers.append(None)
        elif live_len is not None:
            # Rebuild k_buf as [live_len empty slots | visual K], matching _extract_attn's format.
            B, S_vis, H, D = s.key.shape
            k_buf = s.key.new_empty((B, live_len + S_vis, H, D))
            v_buf = s.value.new_empty((B, live_len + S_vis, H, D))
            k_buf[:, live_len:].copy_(s.key)
            v_buf[:, live_len:].copy_(s.value)
            layers.append({"k_buf": k_buf, "v_buf": v_buf, "live_len": live_len})
        else:
            layers.append({"k_ref": s.key, "v_ref": s.value})
    return {
        "layers": layers,
        "num_target_tokens": kv.meta.get("num_target_tokens"),
    }

