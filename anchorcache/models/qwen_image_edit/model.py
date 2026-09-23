# Adapted from Hugging Face Diffusers (transformer_qwenimage.py), Copyright The HuggingFace Team,
# licensed under the Apache License, Version 2.0.
"""Three-stream self_text forward pass: txt(live t) / img=[target, cond] / frozen_txt(fixed t).

Reuses the existing weights from diffusers 0.37 QwenImageTransformer2DModel(zero_cond_t=True), with no new parameters:
- temb = time_text_embed([t, 0]) uses a doubled batch; cond rows use t=0 through modulate_index (native 2511 behavior)
- The frozen_txt stream reuses txt_mod/txt_norm/add_*proj/to_add_out/txt_mlp and takes modulation from the t=0 half of temb
- Training and extraction share a single full-image forward pass; cached mode follows the original processor path (reference KV already contains text information)
"""
import os
import types
from math import prod

import torch

from .processor import (
    _get_qwen_transformer_core,
    _set_non_module_attr,
    enable_qwen_image_edit_kv_cache,
)
from .shim import attn_self_text

# Segmented activation recomputation: retain activations for the first N layers without recomputation
# (approximately 1.7 GB per layer at the 1024 bucket); checkpoint all remaining layers as usual.
# On B200, setting this to 40 saves approximately two-thirds of recomputation time;
# the default of 0 recomputes all layers, matching the existing behavior.
_GC_SKIP_LAYERS = int(os.environ.get("QWEN_GC_SKIP_LAYERS", "0"))


def extend_rope(pos_embed, img_shapes, text_len):
    sample_shapes = img_shapes[0] if isinstance(img_shapes, list) else img_shapes
    max_img = max(max(shape[1], shape[2]) for shape in sample_shapes)
    need = max_img + int(text_len)
    if pos_embed.pos_freqs.shape[0] >= need:
        return
    index = torch.arange(need)
    pos_embed.pos_freqs = torch.cat(
        [
            pos_embed.rope_params(index, pos_embed.axes_dim[0], pos_embed.theta),
            pos_embed.rope_params(index, pos_embed.axes_dim[1], pos_embed.theta),
            pos_embed.rope_params(index, pos_embed.axes_dim[2], pos_embed.theta),
        ],
        dim=1,
    )


def block_self_text(
    block,
    hidden_states,
    encoder_hidden_states,
    frozen_hidden,
    temb,
    frz_temb,
    image_rotary_emb,
    modulate_index,
    num_target_tokens,
    layer_idx,
    kv_cache,
    store_kv_cache,
):
    temb_live = temb.chunk(2, dim=0)[0]
    img_mod1, img_mod2 = block.img_mod(temb).chunk(2, dim=-1)
    txt_mod1, txt_mod2 = block.txt_mod(temb_live).chunk(2, dim=-1)

    img_modulated, img_gate1 = block._modulate(block.img_norm1(hidden_states), img_mod1, modulate_index)
    txt_modulated, txt_gate1 = block._modulate(block.txt_norm1(encoder_hidden_states), txt_mod1)

    frz_modulated = None
    if frozen_hidden is not None:
        frz_mod1, frz_mod2 = block.txt_mod(frz_temb).chunk(2, dim=-1)
        frz_modulated, frz_gate1 = block._modulate(block.txt_norm1(frozen_hidden), frz_mod1)

    img_attn, txt_attn, frz_attn = attn_self_text(
        block.attn,
        img_modulated,
        txt_modulated,
        frz_modulated,
        image_rotary_emb,
        num_target_tokens,
        layer_idx=layer_idx,
        kv_cache=kv_cache,
        store_kv_cache=store_kv_cache,
        kernel_dtype=hidden_states.dtype,
    )

    hidden_states = hidden_states + img_gate1 * img_attn
    encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn

    img_modulated2, img_gate2 = block._modulate(block.img_norm2(hidden_states), img_mod2, modulate_index)
    hidden_states = hidden_states + img_gate2 * block.img_mlp(img_modulated2)

    txt_modulated2, txt_gate2 = block._modulate(block.txt_norm2(encoder_hidden_states), txt_mod2)
    encoder_hidden_states = encoder_hidden_states + txt_gate2 * block.txt_mlp(txt_modulated2)

    if frz_attn is not None:
        frozen_hidden = frozen_hidden + frz_gate1 * frz_attn
        frz_modulated2, frz_gate2 = block._modulate(block.txt_norm2(frozen_hidden), frz_mod2)
        frozen_hidden = frozen_hidden + frz_gate2 * block.txt_mlp(frz_modulated2)

    return encoder_hidden_states, hidden_states, frozen_hidden


def forward_self_text(
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
    frz_timestep=0.0,
    anchor=True,
    return_dict=True,
):
    assert self.config.zero_cond_t, "self_text forward only supports zero_cond_t models (Qwen-Image-Edit-2511)"
    assert guidance is None and controlnet_block_samples is None and additional_t_cond is None

    ak = dict(attention_kwargs or {})
    ak.pop("qwen_kv_cache_mode", None)
    kv_cache = ak.pop("qwen_kv_cache", None)
    ak.pop("qwen_num_target_tokens", None)
    store_kv_cache = ak.pop("qwen_store_kv_cache", True) and kv_cache is not None
    # Raw encoder embeddings for the independent static anchor (frozen_txt), in the same space as pipeline prompt_embeds.
    # When omitted, frozen is identical to live (the correct anchor); experiments may provide embeddings from different
    # text for empty/shuffled conditions.
    frz_raw = ak.pop("qwen_frz_encoder_hs", None)

    # The target/cond partition and modulate_index share the same source and are both determined by img_shapes.
    num_target_tokens = prod(img_shapes[0][0])
    num_cond_tokens = hidden_states.shape[1] - num_target_tokens
    assert num_cond_tokens > 0, "self_text requires condition latents"

    hidden_states = self.img_in(hidden_states)
    timestep = timestep.to(hidden_states.dtype)
    live_timestep = timestep
    timestep = torch.cat([live_timestep, torch.zeros_like(live_timestep)], dim=0)
    modulate_index = torch.tensor(
        [[0] * prod(sample[0]) + [1] * sum(prod(s) for s in sample[1:]) for sample in img_shapes],
        device=timestep.device,
        dtype=torch.int,
    )

    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)
    if not anchor:
        # Isolated cache: references attend only to themselves.
        frozen_hidden = None
    elif frz_raw is not None:
        # Clone to prevent in-place overwrites of storage shared with subsequent pipeline encode_prompt calls.
        frz_raw = frz_raw.detach().to(
            device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype
        ).clone()
        # Pad or truncate to the live text length so RoPE txt_freqs covers the full sequence.
        live_len = encoder_hidden_states.shape[1]
        if frz_raw.shape[1] < live_len:
            pad = frz_raw.new_zeros(frz_raw.shape[0], live_len - frz_raw.shape[1], frz_raw.shape[2])
            frz_raw = torch.cat([frz_raw, pad], dim=1)
        elif frz_raw.shape[1] > live_len:
            frz_raw = frz_raw[:, :live_len]
        frozen_hidden = self.txt_in(self.txt_norm(frz_raw))
    else:
        frozen_hidden = encoder_hidden_states

    temb = self.time_text_embed(timestep, hidden_states)
    frz_temb = self.time_text_embed(torch.full_like(live_timestep, frz_timestep), hidden_states)
    text_seq_len = encoder_hidden_states.shape[1]
    extend_rope(self.pos_embed, img_shapes, text_seq_len)
    image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

    for layer_idx, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing and layer_idx >= _GC_SKIP_LAYERS:
            encoder_hidden_states, hidden_states, frozen_hidden = self._gradient_checkpointing_func(
                block_self_text,
                block,
                hidden_states,
                encoder_hidden_states,
                frozen_hidden,
                temb,
                frz_temb,
                image_rotary_emb,
                modulate_index,
                num_target_tokens,
                layer_idx,
                kv_cache,
                store_kv_cache,
            )
        else:
            encoder_hidden_states, hidden_states, frozen_hidden = block_self_text(
                block,
                hidden_states,
                encoder_hidden_states,
                frozen_hidden,
                temb,
                frz_temb,
                image_rotary_emb,
                modulate_index,
                num_target_tokens,
                layer_idx,
                kv_cache,
                store_kv_cache,
            )

    temb_live = temb.chunk(2, dim=0)[0]
    hidden_states = self.norm_out(hidden_states, temb_live)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    return Transformer2DModelOutput(sample=output)


def enable_self_text(transformer, frz_timestep=0.0, anchor=True):
    """Replace the extraction/training forward pass with the three-stream self_text variant; cached mode uses the original processor path.

    enable_qwen_image_edit_kv_cache provides forward_kv_extract/forward_kv_cached.
    This patch intercepts their internal self.forward calls, so extraction and validation require no changes.
    anchor=False gives the isolated-cache baseline (no static text anchors).
    """
    transformer = enable_qwen_image_edit_kv_cache(transformer)
    core = _get_qwen_transformer_core(transformer)
    _set_non_module_attr(core, "_qwen_frz_timestep", float(frz_timestep))
    _set_non_module_attr(core, "_qwen_anchor", bool(anchor))
    if getattr(core, "_qwen_self_text_enabled", False):
        return transformer

    orig_forward = core.forward

    def routed(self, *args, **kwargs):
        mode = (kwargs.get("attention_kwargs") or {}).get("qwen_kv_cache_mode")
        if mode == "cached":
            return orig_forward(*args, **kwargs)
        return forward_self_text(
            self,
            *args,
            frz_timestep=getattr(self, "_qwen_frz_timestep", 0.0),
            anchor=getattr(self, "_qwen_anchor", True),
            **kwargs,
        )

    core.forward = types.MethodType(routed, core)
    _set_non_module_attr(core, "_qwen_self_text_enabled", True)
    return transformer
