from __future__ import annotations

from typing import Any


def build_adapter(model, backend: str = "flash", layout: str = "text_live"):
    from .adapter import OmniVoiceAdapter
    from .wiring import ModelView, enable_selftext

    enable_selftext(model, backend=backend, layout=layout)
    return OmniVoiceAdapter(
        model=ModelView(model),
        layout_name=layout,
        mask_id=model.config.audio_mask_id,
        num_codebook=model.config.num_audio_codebook,
    )


def prepare_inputs(
    model,
    *,
    text: str,
    language: str | None = None,
    ref_text: str | None = None,
    ref_audio: str | None = None,
    instruct: str | None = None,
    duration: float | None = None,
    speed: float = 1.0,
    denoise: bool = True,
    layout: str = "text_live",
    preprocess_prompt: bool = True,
) -> tuple[dict[str, Any], Any, float | None]:
    import torch
    from omnivoice.data.seg import seg_pack

    task = model._preprocess_all(
        text=text,
        language=language,
        ref_text=ref_text,
        ref_audio=ref_audio,
        instruct=instruct,
        preprocess_prompt=preprocess_prompt,
        speed=speed,
        duration=duration,
    )
    assert task.batch_size == 1, "The production entry point processes one audio sample at a time"
    src = model._prepare_inference_inputs(
        task.texts[0],
        task.target_lens[0],
        task.ref_texts[0],
        task.ref_audio_tokens[0],
        task.langs[0],
        task.instructs[0],
        denoise,
    )
    ids, am = src["input_ids"], src["audio_mask"]
    n_style, n_text, n_ref = src["n_style"], src["n_text"], src["n_ref"]
    n_target = task.target_lens[0]
    dev = ids.device

    sl_style = slice(0, n_style)
    sl_text = slice(n_style, n_style + n_text)
    sl_ref = slice(n_style + n_text, n_style + n_text + n_ref)
    sl_target = slice(n_style + n_text + n_ref, None)
    if layout == "text_live":
        slices = [sl_style, sl_ref, sl_text, sl_text, sl_target]
        vis, hid, text_live = n_style + n_ref, n_text, n_text
        pos_parts = [
            torch.arange(0, n_style, device=dev),
            torch.arange(n_style + n_text, n_style + n_text + n_ref, device=dev),
            torch.arange(n_style, n_style + n_text, device=dev),
            torch.arange(n_style, n_style + n_text, device=dev),
            torch.arange(n_style + n_text + n_ref, n_style + n_text + n_ref + n_target, device=dev),
        ]
    elif layout == "ref_only":
        slices = [sl_style, sl_ref, sl_text, sl_target]
        vis, hid, text_live = n_style + n_ref, 0, n_text
        pos_parts = [
            torch.arange(0, n_style, device=dev),
            torch.arange(n_style + n_text, n_style + n_text + n_ref, device=dev),
            torch.arange(n_style, n_style + n_text, device=dev),
            torch.arange(n_style + n_text + n_ref, n_style + n_text + n_ref + n_target, device=dev),
        ]
    else:
        assert layout == "prefix", f"Unsupported layout={layout}"
        slices = [sl_style, sl_text, sl_ref, sl_target]
        vis, hid, text_live = n_style + n_text + n_ref, 0, 0
        pos_parts = [torch.arange(ids.shape[-1], device=dev)]

    full_ids = torch.cat([ids[..., s] for s in slices], dim=2)
    full_am = torch.cat([am[:, s] for s in slices], dim=1)
    full_pos = torch.cat(pos_parts).unsqueeze(0)
    live_start = vis + hid
    step_ids = full_ids[:, :, live_start:].contiguous()
    step_am = full_am[:, live_start:].contiguous()
    step_pos = full_pos[:, live_start:].contiguous()
    out_cached = text_live
    out_full = vis + hid + text_live

    from anchorcache.transformer.regions import Span

    span = Span(0, vis, hid, text_live + n_target)
    payload = {
        "input_ids": full_ids,
        "audio_mask": full_am,
        "position_ids": full_pos,
        "n_style": n_style,
        "n_ref": n_ref,
        "n_text": n_text,
        "n_target": n_target,
        "batch": 1,
        "spans": [span],
        "step_ids": step_ids,
        "step_audio_mask": step_am,
        "step_position_ids": step_pos,
        "step_seg": None,
        "full_seg": seg_pack([(0, vis, hid, text_live + n_target)], device=dev),
        "out_at_cached": [out_cached],
        "out_at_full": [out_full],
    }
    uncond_ids = ids[..., -n_target:].clone()
    uncond_am = torch.ones(1, n_target, dtype=torch.bool, device=dev)
    uncond_pos = torch.arange(n_target, device=dev).unsqueeze(0)
    payload["uncond"] = {
        "input_ids": uncond_ids,
        "audio_mask": uncond_am,
        "position_ids": uncond_pos,
        "n_style": 0,
        "n_ref": 0,
        "n_text": 0,
        "n_target": n_target,
        "batch": 1,
        "spans": [Span(0, 0, 0, n_target)],
        "step_ids": uncond_ids,
        "step_audio_mask": uncond_am,
        "step_position_ids": uncond_pos,
        "step_seg": None,
        "full_seg": seg_pack([(0, 0, 0, n_target)], device=dev),
        "out_at_cached": [0],
        "out_at_full": [0],
    }
    return payload, task, task.ref_rms[0]


def run_tokens(
    model,
    inputs: dict[str, Any],
    *,
    backend: str = "flash",
    layout: str = "text_live",
    num_step: int = 32,
    guidance_scale: float = 2.0,
    t_shift: float = 0.1,
    layer_penalty: float = 5.0,
    position_temp: float = 5.0,
    class_temp: float = 0.0,
):
    from anchorcache.models.omnivoice.sampler import UnmaskSampler
    from anchorcache.sampling import LogProbCfgGuidance, rollout
    from anchorcache.transformer.kv import kv_plan

    adapter = build_adapter(model, backend=backend, layout=layout)
    condition = adapter.prepare_condition(inputs)
    condition.payload["step_seg"] = kv_plan(
        condition["spans"], device=condition["input_ids"].device
    )
    condition.payload["uncond"] = adapter.prepare_condition(inputs["uncond"])
    reusable = adapter.extract(condition)
    sampler = UnmaskSampler(
        mask_id=model.config.audio_mask_id,
        num_step=num_step,
        t_shift=t_shift,
        layer_penalty=layer_penalty,
        position_temperature=position_temp,
        class_temperature=class_temp,
    )
    guidance = LogProbCfgGuidance(scale=guidance_scale) if guidance_scale else None
    return rollout(
        adapter,
        condition,
        sampler,
        reusable=reusable,
        guidance=guidance,
        schedule=list(range(num_step)),
    ).final


def decode(model, tokens, *, ref_rms=None, postprocess: bool = True):
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    cfg = OmniVoiceGenerationConfig(postprocess_output=postprocess)
    return model._decode_and_post_process(tokens[0], ref_rms, cfg)


def infer_parser():
    import argparse

    p = argparse.ArgumentParser(description="OmniVoice AnchorCache inference")
    p.add_argument("--model", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--inputs")
    p.add_argument("--output", required=True)
    p.add_argument("--ref_audio")
    p.add_argument("--ref_text")
    p.add_argument("--instruct")
    p.add_argument("--language")
    p.add_argument("--duration", type=float)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    p.add_argument("--layout", choices=("prefix", "text_live", "ref_only"), default="text_live")
    p.add_argument("--backend", choices=("flash", "sdpa"), default="flash")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--t_shift", type=float, default=0.1)
    p.add_argument("--layer_penalty", type=float, default=5.0)
    p.add_argument("--position_temp", type=float, default=5.0)
    p.add_argument("--class_temp", type=float, default=0.0)
    p.add_argument("--no_postprocess", action="store_true")
    return p


def infer_main() -> None:
    args = infer_parser().parse_args()

    import soundfile as sf
    import torch

    from .wiring import load_model

    model = load_model(args.model, args.device, args.dtype)
    model.eval().requires_grad_(False)
    if args.inputs:
        data = torch.load(args.inputs, map_location=args.device, weights_only=False)
        inputs = data["inputs"] if "inputs" in data else data
        ref_rms = data.get("ref_rms") if isinstance(data, dict) else None
    else:
        inputs, _, ref_rms = prepare_inputs(
            model,
            text=args.text,
            language=args.language,
            ref_text=args.ref_text,
            ref_audio=args.ref_audio,
            instruct=args.instruct,
            duration=args.duration,
            speed=args.speed,
            layout=args.layout,
        )
    with torch.inference_mode():
        tokens = run_tokens(
            model,
            inputs,
            backend=args.backend,
            layout=args.layout,
            num_step=args.steps,
            guidance_scale=args.cfg,
            t_shift=args.t_shift,
            layer_penalty=args.layer_penalty,
            position_temp=args.position_temp,
            class_temp=args.class_temp,
        )
        audio = decode(
            model,
            tokens,
            ref_rms=ref_rms,
            postprocess=not args.no_postprocess,
        )
    sf.write(args.output, audio, model.sampling_rate)


def train_parser():
    import argparse

    p = argparse.ArgumentParser(description="OmniVoice AnchorCache training")
    p.add_argument("--train_config", required=True)
    p.add_argument("--data_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stage", choices=("teacher_forced", "opd"), default="teacher_forced")
    p.add_argument("--teacher")
    p.add_argument("--layout", choices=("prefix", "text_live", "ref_only"), default="text_live")
    p.add_argument("--backend", choices=("flash", "sdpa"), default="flash")
    p.add_argument("--dist", choices=("ddp", "zero2"), default="ddp")
    p.add_argument("--num_states", type=int, default=4)
    p.add_argument("--steps", type=int, default=32)
    # Rollout sampling must match the inference settings.
    p.add_argument("--t_shift", type=float, default=0.1)
    p.add_argument("--layer_penalty", type=float, default=5.0)
    p.add_argument("--position_temp", type=float, default=5.0)
    p.add_argument("--class_temp", type=float, default=0.0)
    return p


def train_main() -> None:
    args = train_parser().parse_args()

    import os

    import torch
    from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
    from omnivoice.training.config import TrainingConfig

    from anchorcache.training import (
        AccelEngine,
        Distiller,
        StudentRollout,
        TeacherForced,
        TrainCfg,
        Trainer,
        build_optim,
        reverse_kl,
    )

    from .sampler import UnmaskSampler, time_steps
    from .wiring import PackedAdapter, enable_selftext, load_model, prepare_batch

    src = TrainingConfig.from_json(args.train_config)
    src.output_dir = args.output
    src.data_config = args.data_config
    src.attn_implementation = "omni_selftext"
    src.selftext_backend = args.backend
    src.selftext_layout = args.layout
    src.distill_stage = args.stage
    warmup = int(src.steps * src.warmup_ratio) if src.warmup_type == "ratio" else src.warmup_steps
    cfg = TrainCfg(
        out_dir=args.output,
        max_steps=src.steps,
        lr=src.learning_rate,
        wd=src.weight_decay,
        warmup=warmup,
        clip=src.max_grad_norm,
        accum=src.gradient_accumulation_steps,
        dist=args.dist,
        seed=src.seed,
        log_every=src.logging_steps,
        ckpt_every=src.save_steps,
    )
    engine = AccelEngine(cfg)
    if engine.is_main:
        os.makedirs(args.output, exist_ok=True)
    student_model, tokenizer = build_model_and_tokenizer(src)
    enable_selftext(student_model, backend=args.backend, layout=args.layout)
    student_model.to(engine.device)
    student = PackedAdapter(student_model, mask_id=student_model.config.audio_mask_id)
    teacher_path = args.teacher or src.teacher_path
    assert teacher_path, "Distillation training requires a teacher checkpoint (--teacher or teacher_path)"
    teacher_model = load_model(teacher_path, str(engine.device), "bfloat16", train=True)
    enable_selftext(teacher_model, backend=args.backend, layout="prefix")
    teacher_model.eval().requires_grad_(False)
    teacher = PackedAdapter(teacher_model, teacher=True, mask_id=teacher_model.config.audio_mask_id)
    loader, _ = build_dataloaders(src, tokenizer)
    params = [p for p in student_model.parameters() if p.requires_grad]
    optim, sched = build_optim(params, cfg)
    student_model, optim = engine.prepare(student_model, optim)
    engine.register(sched)
    student.model = student_model
    if args.stage == "teacher_forced":
        provider = TeacherForced(num_states=1)
    else:
        sampler = UnmaskSampler(
            mask_id=student.mask_id,
            num_step=args.steps,
            t_shift=args.t_shift,
            layer_penalty=args.layer_penalty,
            position_temperature=args.position_temp,
            class_temperature=args.class_temp,
        )
        provider = StudentRollout(
            sampler=sampler,
            num_states=args.num_states,
            strata=time_steps(args.steps, args.t_shift).tolist()[:-1],
            lo=1,
            seed=src.seed,
            rank=engine.rank,
            shared_across_ranks=False,
        )
    raw = engine.unwrap(student_model)
    weights = torch.tensor(raw.normalized_audio_codebook_weights, device=engine.device)
    dist = Distiller(
        teacher=teacher,
        student=student,
        match=reverse_kl,
        select_fn=student.select,
        weights=weights,
        fold_batch=False,
        backward=engine.backward,
    )

    def step_fn(batch):
        return dist.train_step(provider, prepare_batch(batch, engine.device))

    def save(path: str) -> None:
        model = engine.unwrap(student_model)
        model.save_pretrained(
            path,
            is_main_process=engine.is_main,
            save_function=engine.save,
            state_dict=engine.state_dict_of(student_model),
        )
        if engine.is_main:
            tokenizer.save_pretrained(path)

    student_model.train()
    Trainer(
        cfg=cfg,
        engine=engine,
        model=student_model,
        optim=optim,
        sched=sched,
        step_fn=step_fn,
        on_save=save,
        params=params,
    ).run(loader)
