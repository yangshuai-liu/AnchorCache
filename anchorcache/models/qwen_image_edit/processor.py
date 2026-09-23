# Adapted from Hugging Face Diffusers (transformer_qwenimage.py), Copyright The HuggingFace Team,
# licensed under the Apache License, Version 2.0.
import os
import types
import weakref

import torch
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers import transformer_qwenimage as qwen_mod
from diffusers.models.transformers.transformer_qwenimage import Attention
from torch.nn.attention import SDPBackend, sdpa_kernel

# QWEN_KV_PREALLOC=1: cached inference uses preallocated KV buffers (the reference segment is written once
# during extraction, and only the live segment is updated in place at each step), eliminating full-length KV
# torch.cat allocation and copying at every layer and step. Measurements on B200 show that cat is not a bottleneck,
# so the default of 0 preserves the previous behavior; enable this for batch sizes above 1 or very long conditions.
# Results are bitwise identical to the previous path.
_KV_PREALLOC = os.environ.get("QWEN_KV_PREALLOC", "0") == "1"

# QWEN_KV_SDP_BACKEND=flash|cudnn|sage: attention backend for this processor.
# B200 microbenchmark (Q=4.6k, KV=37.4k, cross-attention shape): cuDNN 1484 TFLOPS versus Flash 389 TFLOPS (3.8x).
# sage requires the sageattention package.
_BACKEND = os.environ.get("QWEN_KV_SDP_BACKEND", "flash")
_SDP_BACKEND = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}.get(_BACKEND)
assert _BACKEND == "sage" or _SDP_BACKEND is not None, f"unknown QWEN_KV_SDP_BACKEND={_BACKEND}"


def _dispatch_attention(query, key, value, attention_mask=None, backend=None, parallel_config=None):
    if _BACKEND == "sage":
        from sageattention import sageattn

        return sageattn(query, key, value, tensor_layout="NHD", is_causal=False)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    with sdpa_kernel(_SDP_BACKEND):
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
    return hidden_states.transpose(1, 2)


def _slice_query_mask(attention_mask, query_slice):
    if attention_mask is None:
        return None
    if attention_mask.dim() >= 4 and attention_mask.shape[-2] != 1:
        return attention_mask[..., query_slice, :]
    return attention_mask


class QwenImageEditKVAttnProcessor2_0:
    """Qwen-Image-Edit attention processor with reference-image KV cache support."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, layer_idx):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("QwenImageEditKVAttnProcessor2_0 requires PyTorch 2.0+.")
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        qwen_kv_cache_mode: str | None = None,
        qwen_kv_cache: dict | None = None,
        qwen_num_target_tokens: int | None = None,
        qwen_store_kv_cache: bool = True,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenImageEditKVAttnProcessor2_0 requires encoder_hidden_states.")

        # This project does not require an attention mask: offline VLM embeddings align prompt lengths,
        # training uses batch size 1, and inference does not use CFG padding. Forcing the mask to None lets SDPA
        # use the FlashAttention v2 fast path; otherwise, the upper diffusers layer converts None into an all-ones
        # mask and falls back to the memory-efficient or math path.
        attention_mask = None

        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = qwen_mod.apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = qwen_mod.apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = qwen_mod.apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = qwen_mod.apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        if qwen_kv_cache_mode is None:
            return self._vanilla_attn(
                attn, seq_txt, txt_query, txt_key, txt_value, img_query, img_key, img_value, attention_mask
            )

        if qwen_kv_cache is None:
            raise ValueError("qwen_kv_cache must be provided when qwen_kv_cache_mode is set.")
        if qwen_num_target_tokens is None:
            raise ValueError("qwen_num_target_tokens must be provided for Qwen KV cache modes.")

        if qwen_kv_cache_mode == "extract":
            return self._extract_attn(
                attn,
                seq_txt,
                txt_query,
                txt_key,
                txt_value,
                img_query,
                img_key,
                img_value,
                attention_mask,
                qwen_kv_cache,
                qwen_num_target_tokens,
                qwen_store_kv_cache,
            )
        if qwen_kv_cache_mode == "cached":
            return self._cached_attn(
                attn,
                seq_txt,
                txt_query,
                txt_key,
                txt_value,
                img_query,
                img_key,
                img_value,
                attention_mask,
                qwen_kv_cache,
            )
        raise ValueError(f"Unsupported qwen_kv_cache_mode: {qwen_kv_cache_mode}")

    def _project_outputs(self, attn, seq_txt, joint_hidden_states):
        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_hidden_states.dtype)
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]
        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())
        return img_attn_output, txt_attn_output

    def _vanilla_attn(self, attn, seq_txt, txt_q, txt_k, txt_v, img_q, img_k, img_v, attention_mask):
        joint_query = torch.cat([txt_q, img_q], dim=1)
        joint_key = torch.cat([txt_k, img_k], dim=1)
        joint_value = torch.cat([txt_v, img_v], dim=1)
        joint_hidden_states = _dispatch_attention(
            joint_query,
            joint_key,
            joint_value,
            attention_mask,
            self._attention_backend,
            self._parallel_config,
        )
        return self._project_outputs(attn, seq_txt, joint_hidden_states)

    def _extract_attn(
        self,
        attn,
        seq_txt,
        txt_q,
        txt_k,
        txt_v,
        img_q,
        img_k,
        img_v,
        attention_mask,
        cache,
        num_target_tokens,
        store_kv_cache=True,
    ):
        target_q = img_q[:, :num_target_tokens]
        target_k = img_k[:, :num_target_tokens]
        target_v = img_v[:, :num_target_tokens]
        ref_q = img_q[:, num_target_tokens:]
        ref_k = img_k[:, num_target_tokens:]
        ref_v = img_v[:, num_target_tokens:]
        num_ref_tokens = ref_k.shape[1]

        if store_kv_cache:
            if _KV_PREALLOC:
                live_len = seq_txt + num_target_tokens
                total_len = live_len + num_ref_tokens
                k_buf = ref_k.new_empty((ref_k.shape[0], total_len, ref_k.shape[2], ref_k.shape[3]))
                v_buf = ref_v.new_empty((ref_v.shape[0], total_len, ref_v.shape[2], ref_v.shape[3]))
                k_buf[:, live_len:].copy_(ref_k)
                v_buf[:, live_len:].copy_(ref_v)
                cache["layers"][self.layer_idx] = {"k_buf": k_buf, "v_buf": v_buf, "live_len": live_len}
            else:
                cache["layers"][self.layer_idx] = {
                    "k_ref": ref_k.clone(),
                    "v_ref": ref_v.clone(),
                }

        main_query = torch.cat([txt_q, target_q], dim=1)
        main_key = torch.cat([txt_k, target_k, ref_k], dim=1)
        main_value = torch.cat([txt_v, target_v, ref_v], dim=1)
        main_hidden_states = _dispatch_attention(
            main_query,
            main_key,
            main_value,
            _slice_query_mask(attention_mask, slice(0, main_query.shape[1])),
            self._attention_backend,
            self._parallel_config,
        )

        if num_ref_tokens > 0:
            ref_hidden_states = _dispatch_attention(
                ref_q,
                ref_k,
                ref_v,
                None,
                self._attention_backend,
                self._parallel_config,
            )
            joint_hidden_states = torch.cat(
                [
                    main_hidden_states[:, :seq_txt],
                    main_hidden_states[:, seq_txt:],
                    ref_hidden_states,
                ],
                dim=1,
            )
        else:
            joint_hidden_states = main_hidden_states
        return self._project_outputs(attn, seq_txt, joint_hidden_states)

    def _cached_attn(self, attn, seq_txt, txt_q, txt_k, txt_v, img_q, img_k, img_v, attention_mask, cache):
        layer_cache = cache["layers"][self.layer_idx]
        if layer_cache is None:
            raise ValueError(f"Missing Qwen KV cache for layer {self.layer_idx}.")

        joint_query = torch.cat([txt_q, img_q], dim=1)

        if "k_buf" in layer_cache:
            # Preallocated-buffer path: the reference segment is populated during extraction, and only the live
            # segment (approximately 4.6k tokens) is updated in place each step, avoiding full-length KV
            # (live + reference) reallocation and copying.
            k_buf, v_buf = layer_cache["k_buf"], layer_cache["v_buf"]
            live_len = layer_cache["live_len"]
            n_txt = txt_k.shape[1]
            seq_live = n_txt + img_k.shape[1]
            if seq_live != live_len:
                raise ValueError(f"live length mismatch: extract={live_len} cached={seq_live}")
            k_buf[:, :n_txt].copy_(txt_k)
            k_buf[:, n_txt:live_len].copy_(img_k)
            v_buf[:, :n_txt].copy_(txt_v)
            v_buf[:, n_txt:live_len].copy_(img_v)
            joint_key, joint_value = k_buf, v_buf
            mask_all = None
        else:
            ref_k = layer_cache["k_ref"]
            ref_v = layer_cache["v_ref"]
            joint_key = torch.cat([txt_k, img_k, ref_k], dim=1)
            joint_value = torch.cat([txt_v, img_v, ref_v], dim=1)
            mask_all = None
            if attention_mask is not None:
                ref_mask = torch.ones(
                    attention_mask.shape[:-1] + (ref_k.shape[1],),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                mask_all = torch.cat([attention_mask, ref_mask], dim=-1)

        joint_hidden_states = _dispatch_attention(
            joint_query,
            joint_key,
            joint_value,
            mask_all,
            self._attention_backend,
            self._parallel_config,
        )
        return self._project_outputs(attn, seq_txt, joint_hidden_states)


def enable_qwen_image_edit_kv_cache(transformer):
    """Attach KV-cache-aware processors and forward helpers to a QwenImageTransformer2DModel."""

    if getattr(transformer, "_qwen_image_edit_kv_cache_enabled", False):
        return transformer

    core = _get_qwen_transformer_core(transformer)
    for layer_idx, block in enumerate(core.transformer_blocks):
        block.attn.processor = QwenImageEditKVAttnProcessor2_0(layer_idx)

    transformer.forward_kv_extract = types.MethodType(_forward_kv_extract, transformer)
    transformer.forward_kv_cached = types.MethodType(_forward_kv_cached, transformer)
    _set_non_module_attr(transformer, "_qwen_image_edit_kv_cache_core_ref", weakref.ref(core))
    _set_non_module_attr(transformer, "_qwen_image_edit_kv_cache_enabled", True)
    if core is not transformer:
        core.forward_kv_extract = types.MethodType(_forward_kv_extract, core)
        core.forward_kv_cached = types.MethodType(_forward_kv_cached, core)
        _set_non_module_attr(core, "_qwen_image_edit_kv_cache_core_ref", weakref.ref(core))
        _set_non_module_attr(core, "_qwen_image_edit_kv_cache_enabled", True)
    return transformer


def _set_non_module_attr(module, name, value):
    modules = getattr(module, "_modules", None)
    if modules is not None and name in modules:
        del modules[name]
    object.__setattr__(module, name, value)


def _get_cached_core_ref(module):
    ref = getattr(module, "_qwen_image_edit_kv_cache_core_ref", None)
    if ref is None:
        return None
    return ref()


def _get_qwen_transformer_core(transformer):
    if hasattr(transformer, "transformer_blocks"):
        return transformer
    if hasattr(transformer, "get_base_model"):
        base_model = transformer.get_base_model()
        if hasattr(base_model, "transformer_blocks"):
            return base_model
    base_model = getattr(transformer, "base_model", None)
    if base_model is not None:
        model = getattr(base_model, "model", base_model)
        if hasattr(model, "transformer_blocks"):
            return model
    raise TypeError("Could not find QwenImageTransformer2DModel core with transformer_blocks.")


def _forward_kv_extract(
    self,
    hidden_states,
    encoder_hidden_states=None,
    encoder_hidden_states_mask=None,
    timestep=None,
    img_shapes=None,
    txt_seq_lens=None,
    guidance=None,
    attention_kwargs=None,
    controlnet_block_samples=None,
    additional_t_cond=None,
    return_dict=True,
    num_target_tokens=None,
):
    if num_target_tokens is None:
        raise ValueError("num_target_tokens is required for forward_kv_extract.")
    core = _get_cached_core_ref(self) or self
    kv_cache = {"layers": [None] * len(core.transformer_blocks), "num_target_tokens": num_target_tokens}
    attention_kwargs = dict(attention_kwargs or {})
    attention_kwargs.update(
        {
            "qwen_kv_cache_mode": "extract",
            "qwen_kv_cache": kv_cache,
            "qwen_num_target_tokens": num_target_tokens,
        }
    )
    output = self.forward(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        timestep=timestep,
        img_shapes=img_shapes,
        guidance=guidance,
        attention_kwargs=attention_kwargs,
        controlnet_block_samples=controlnet_block_samples,
        additional_t_cond=additional_t_cond,
        return_dict=False,
    )[0]
    output = output[:, :num_target_tokens]
    if not return_dict:
        return output, kv_cache
    return Transformer2DModelOutput(sample=output), kv_cache


def _forward_kv_cached(
    self,
    hidden_states,
    encoder_hidden_states=None,
    encoder_hidden_states_mask=None,
    timestep=None,
    img_shapes=None,
    txt_seq_lens=None,
    guidance=None,
    attention_kwargs=None,
    controlnet_block_samples=None,
    additional_t_cond=None,
    return_dict=True,
    kv_cache=None,
):
    if kv_cache is None:
        raise ValueError("kv_cache is required for forward_kv_cached.")
    attention_kwargs = dict(attention_kwargs or {})
    attention_kwargs.update(
        {
            "qwen_kv_cache_mode": "cached",
            "qwen_kv_cache": kv_cache,
            "qwen_num_target_tokens": hidden_states.shape[1],
        }
    )
    return self.forward(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        timestep=timestep,
        img_shapes=img_shapes,
        guidance=guidance,
        attention_kwargs=attention_kwargs,
        controlnet_block_samples=controlnet_block_samples,
        additional_t_cond=additional_t_cond,
        return_dict=return_dict,
    )
