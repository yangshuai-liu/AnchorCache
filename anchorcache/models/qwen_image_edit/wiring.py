from __future__ import annotations

from typing import Any


def enable_kv(transformer):
    from .processor import enable_qwen_image_edit_kv_cache

    return enable_qwen_image_edit_kv_cache(transformer)


def enable_selftext(transformer, frz_timestep: float = 0.0, anchor: bool = True):
    from .model import enable_self_text

    return enable_self_text(transformer, frz_timestep=frz_timestep, anchor=anchor)


def load_transformer(
    model_path: str,
    *,
    transformer_path: str | None = None,
    device=None,
    dtype=None,
    frz_timestep: float = 0.0,
    anchor: bool = True,
):
    import torch
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    dtype = dtype or torch.bfloat16
    path = transformer_path or model_path
    subfolder = None if transformer_path else "transformer"
    transformer = QwenImageTransformer2DModel.from_pretrained(
        path,
        subfolder=subfolder,
        torch_dtype=dtype,
    )
    transformer = enable_selftext(transformer, frz_timestep, anchor)
    transformer.requires_grad_(False)
    transformer.eval()
    if device is not None:
        transformer.to(device=device, dtype=dtype)
    return transformer


def load_train_student(
    model_path: str,
    *,
    student_path: str | None,
    device,
    dtype,
    train_mode: str,
    frz_timestep: float,
    lora_rank: int,
    lora_alpha: int,
    lora_targets: str,
    anchor: bool = True,
):
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    path = student_path or model_path
    subfolder = None if student_path else "transformer"
    transformer = QwenImageTransformer2DModel.from_pretrained(
        path,
        subfolder=subfolder,
        torch_dtype=dtype,
    )
    transformer = enable_selftext(transformer, frz_timestep, anchor)
    if train_mode == "full":
        transformer.requires_grad_(True)
    else:
        from peft import LoraConfig

        transformer.requires_grad_(False)
        transformer.add_adapter(
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
                init_lora_weights="gaussian",
                target_modules=[x for x in lora_targets.split(",") if x],
            )
        )
    transformer.enable_gradient_checkpointing()
    return transformer.to(device=device, dtype=dtype)


def load_teacher(model_path: str, *, teacher_path: str | None, device, dtype):
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    path = teacher_path or model_path
    subfolder = None if teacher_path else "transformer"
    teacher = QwenImageTransformer2DModel.from_pretrained(
        path,
        subfolder=subfolder,
        torch_dtype=dtype,
    ).to(device=device, dtype=dtype)
    teacher.requires_grad_(False).eval()
    return teacher


def make_pipeline(
    transformer,
    pretrained_path: str,
    weight_dtype=None,
    *,
    load_text: bool = True,
    **kwargs,
):
    import torch
    from diffusers import QwenImageEditPlusPipeline

    dtype = weight_dtype or torch.bfloat16
    if not load_text:
        kwargs.update(text_encoder=None, tokenizer=None, processor=None)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        pretrained_path,
        transformer=transformer,
        torch_dtype=dtype,
        **kwargs,
    )
    for name in ("vae", "text_encoder"):
        module = getattr(pipe, name, None)
        if module is not None:
            module.requires_grad_(False)
            module.eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_adapter(
    transformer,
    frz_timestep: float = 0.0,
    anchor: bool = False,
    **adapter_kwargs: Any,
):
    from .adapter import QwenAdapter

    transformer = enable_selftext(transformer, frz_timestep=frz_timestep, anchor=anchor)
    return QwenAdapter(transformer=transformer, anchor=anchor, **adapter_kwargs)
