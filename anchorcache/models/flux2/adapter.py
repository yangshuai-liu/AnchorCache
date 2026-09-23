from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...core.state import ModelOutput, PreparedCondition
from ...transformer.kv import KVState, LayeredKV
from .layout import layout_from_packs

# Adapter over the external FLUX.2 AnchorCache model. After enable_self_text:
#   transformer(...)              anchored full forward (training and extraction)
#   kv_cache_mode="extract"       returns (output, Flux2KVCache)
#   kv_cache_mode="cached"        falls back to the official diffusers forward with the cache


@dataclass
class Flux2Adapter:
    transformer: Any
    topology: str = "self_text"
    ref_timestep: float = 0.0
    frz_timestep: float = 0.0
    ts_scale: float = 1.0
    sigma_fn: Any = None
    noise_fn: Any = None

    def _ts(self, step, bsz: int, dtype):
        """Convert step to the transformer's timestep argument.

        Teacher-forced steps are sigma in [0, 1]; rollout steps are scheduler timesteps in
        [0, 1000], which sched.step also consumes. ts_scale is therefore explicit: 1.0 for teacher
        forcing, 1/1000 for rollout. An fp32 timestep would take a different precision path through
        the time embedding, so it is cast to dtype.
        """
        if not isinstance(step, torch.Tensor):
            step = torch.tensor(step)
        if step.ndim == 0:
            step = step.expand(bsz)
        if self.ts_scale != 1.0:
            step = step * self.ts_scale
        return step.to(dtype)

    def prepare_condition(self, inputs: Any) -> PreparedCondition:
        if isinstance(inputs, PreparedCondition):
            return inputs
        payload = dict(inputs) if isinstance(inputs, dict) else {"raw": inputs}
        for k in ("emb", "txt_ids", "ref_pack", "ref_ids", "tgt_ids"):
            assert k in payload, f"prepare_condition missing {k}"
        layout = layout_from_packs(
            txt_len=payload["emb"].shape[1],
            ref_pack_len=payload["ref_pack"].shape[1],
            target_len=payload["tgt_ids"].shape[1],
            anchor=self.topology == "self_text",
        )
        return PreparedCondition(payload=payload, layout=layout)

    def extract(self, condition: PreparedCondition, store=None) -> LayeredKV:
        """Build the reference cache explicitly, before the denoising loop."""
        latents = self.init_state(condition)
        ref = condition["ref_pack"]
        _, cache = self.transformer(
            hidden_states=torch.cat([ref, latents], dim=1),
            timestep=condition.get("extract_timestep"),
            encoder_hidden_states=condition["emb"],
            img_ids=torch.cat([condition["ref_ids"], condition["tgt_ids"]], dim=1),
            txt_ids=condition["txt_ids"],
            kv_cache_mode="extract",
            num_ref_tokens=ref.shape[1],
            return_dict=False,
        )
        return _wrap_flux_cache(cache)

    def predict(self, state, step, condition, reusable=None) -> ModelOutput:
        emb = condition["emb"]
        ts = self._ts(step, state.shape[0], emb.dtype)
        if reusable is None:
            ref = condition["ref_pack"]
            pred = self.transformer(
                hidden_states=torch.cat([ref, state], dim=1),
                timestep=ts,
                encoder_hidden_states=emb,
                img_ids=torch.cat([condition["ref_ids"], condition["tgt_ids"]], dim=1),
                txt_ids=condition["txt_ids"],
                num_ref_tokens=ref.shape[1],
                frz_hidden_states=condition.get("frz_emb"),
                return_dict=False,
            )[0]
            return ModelOutput(prediction=pred)

        pred = self.transformer(
            hidden_states=state,
            timestep=ts,
            encoder_hidden_states=emb,
            img_ids=condition["tgt_ids"],
            txt_ids=condition["txt_ids"],
            kv_cache=_unwrap_flux_cache(reusable),
            kv_cache_mode="cached",
            return_dict=False,
        )[0]
        return ModelOutput(prediction=pred)

    def init_state(self, condition: PreparedCondition, **kwargs):
        state = condition.get("latents")
        if state is not None:
            return state
        x0 = condition.get("x0")
        assert x0 is not None, "init_state requires condition['x0'] or condition['latents'] to define the shape"
        return torch.randn_like(x0)

    def sample_train_state(self, batch):
        """Stage I: noisy = (1-sigma)x0 + sigma*noise.

        Keep sigma in bf16 rather than promoting it to fp32: promotion would make all hidden states
        fp32 and double memory usage. sigma is sampled before noise: grid sigma consumes the CPU
        generator and randn_like the CUDA generator. sigma_fn / noise_fn override either draw.
        """
        x0 = batch["x0"]
        sigma = batch["sigma"] if self.sigma_fn is None else self.sigma_fn(batch)
        sig = sigma.view(-1, 1, 1).to(x0.dtype)
        noise = torch.randn_like(x0) if self.noise_fn is None else self.noise_fn(x0)
        return (1 - sig) * x0 + sig * noise, sigma

    def fold(self, pairs):
        steps = [t.reshape(-1).expand(s.shape[0]) for s, t in pairs]
        return torch.cat([s for s, _ in pairs], 0), torch.cat(steps, 0)


def _wrap_flux_cache(cache) -> LayeredKV:
    """Convert Flux2KVCache to LayeredKV.

    The official structure contains two lists, double_block_caches and single_block_caches
    in diffusers; flatten them here with double blocks first.
    """
    doubles = list(cache.double_block_caches)
    singles = list(cache.single_block_caches)
    out = LayeredKV(len(doubles) + len(singles), num_vis_tokens=cache.num_ref_tokens)
    for i, lc in enumerate(doubles + singles):
        if lc.k_ref is not None:
            out.put(i, KVState(lc.k_ref, lc.v_ref))
    out.meta["num_double"] = len(doubles)
    return out


def _unwrap_flux_cache(kv: LayeredKV):
    from diffusers.models.transformers.transformer_flux2 import Flux2KVCache

    n_double = kv.meta["num_double"]
    cache = Flux2KVCache(n_double, len(kv) - n_double)
    cache.num_ref_tokens = kv.num_vis_tokens
    for i, s in enumerate(kv.layers):
        if s is None:
            continue
        target = (
            cache.get_double(i) if i < n_double else cache.get_single(i - n_double)
        )
        target.store(s.key, s.value)
    return cache
