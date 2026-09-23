from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


def _dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def infer_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen-Image-Edit AnchorCache inference")
    p.add_argument("--model", required=True, help="diffusers base model providing the frozen VAE, VLM, and scheduler")
    p.add_argument("--transformer", "--student", dest="transformer", help="student DiT weights directory")
    p.add_argument("--image", action="append", required=True, help="reference image; may be specified multiple times")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default=" ")
    p.add_argument("--output", default="qwen_edit.png")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frz_timestep", type=float, default=0.0)
    _add_topology(p)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return p


def _add_topology(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--topology",
        default="anchor",
        choices=["anchor", "isolated"],
        help="anchor: AnchorCache static text anchors; isolated: isolated-cache baseline",
    )


def train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen-Image-Edit AnchorCache training")
    p.add_argument("--model", required=True, help="diffusers base model providing the frozen VAE, VLM, and scheduler")
    p.add_argument("--input_mode", choices=["raw", "tensor"], default="raw")
    p.add_argument(
        "--stage",
        choices=["teacher_forced", "opd"],
        default="teacher_forced",
        help="teacher_forced: Stage 1; opd: Stage 2 on-policy distillation (start from the Stage 1 checkpoint)",
    )
    p.add_argument("--data_root", help="raw JSONL: image_refs/images, target, prompt")
    p.add_argument("--tensor_data", help="preprocessed DiT input .pt file or directory")
    p.add_argument("--output", required=True)
    p.add_argument("--transformer", "--student", dest="transformer", help="student DiT weights directory")
    p.add_argument("--teacher", help="teacher DiT weights directory; defaults to the base transformer")
    _add_topology(p)
    p.add_argument("--image_size", type=int, default=1024, help="raw mode: target/reference area is image_size**2")
    p.add_argument("--max_steps", type=int, help="default: 30000 (teacher_forced), 500 (opd)")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--wd", type=float, help="default: 3e-2 (teacher_forced), 1e-2 (opd)")
    p.add_argument("--eps", type=float, default=1e-10)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--clip", type=float, default=0.05)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--dist", choices=["zero2", "ddp"], default="zero2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--state_every", type=int, default=500)
    p.add_argument("--resume_state", help="state directory written during training (OUTPUT/state-latest)")
    p.add_argument("--frz_timestep", type=float, default=0.0)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument(
        "--lora_targets",
        default="to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out",
    )
    p.add_argument("--train_mode", choices=["lora", "full"], default="full")
    p.add_argument("--weighting", choices=["logit_normal", "uniform"], default="logit_normal", help="raw mode: sigma density")
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--num_states", type=int, default=4, help="opd: states supervised per rollout (N_q)")
    p.add_argument("--rollout_steps", type=int, default=40, help="opd: student rollout length; states k in [1, N-1]")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return p


class QwenSampler:
    def __init__(self, scheduler, steps: int):
        self.base = scheduler
        self.steps = steps
        self.active = None
        self.raw = None

    def schedule(self, condition):
        import copy

        import numpy as np
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_shift

        self.active = copy.deepcopy(self.base)
        n = condition.layout.len_of("target")
        cfg = self.active.config
        mu = calculate_shift(
            n,
            cfg.get("base_image_seq_len", 256),
            cfg.get("max_image_seq_len", 4096),
            cfg.get("base_shift", 0.5),
            cfg.get("max_shift", 1.15),
        )
        sigmas = np.linspace(1.0, 1.0 / self.steps, self.steps)
        self.active.set_timesteps(sigmas=sigmas, device=condition["latents"].device, mu=mu)
        self.active.set_begin_index(0)
        self.raw = list(self.active.timesteps)
        batch = condition["latents"].shape[0]
        dtype = condition["latents"].dtype
        return [t.expand(batch).to(dtype) / 1000 for t in self.raw]

    def step(self, state, prediction, step, condition=None):
        raw = self.raw[self.active.step_index or 0]
        return self.active.step(prediction, raw, state, return_dict=False)[0]


@dataclass
class QwenCfg:
    scale: float
    cond_cache: object
    neg_cache: object
    negative: str = "negative"

    def predict(self, adapter, state, step, condition, reusable):
        import torch

        from anchorcache.core.state import ModelOutput

        cond = adapter.predict(state, step, condition, self.cond_cache).prediction
        neg_cond = condition[self.negative]
        neg = adapter.predict(state, step, neg_cond, self.neg_cache).prediction
        merged = neg + self.scale * (cond - neg)
        norm = torch.norm(cond, dim=-1, keepdim=True)
        merged = merged * (norm / torch.norm(merged, dim=-1, keepdim=True).clamp_min(1e-12))
        return ModelOutput(prediction=merged)


def _images(paths):
    from PIL import Image

    return [Image.open(path).convert("RGB") for path in paths]


def _size(pipe, images, height, width):
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        VAE_IMAGE_SIZE,
        calculate_dimensions,
    )

    calc_w, calc_h = calculate_dimensions(VAE_IMAGE_SIZE, images[-1].width / images[-1].height)
    height = height or calc_h
    width = width or calc_w
    multiple = pipe.vae_scale_factor * 2
    return height // multiple * multiple, width // multiple * multiple


def _prepare(pipe, images, prompt, negative, height, width, generator):
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        VAE_IMAGE_SIZE,
        calculate_dimensions,
    )

    cond_images = []
    vae_images = []
    vae_sizes = []
    for image in images:
        cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, image.width / image.height)
        vw, vh = calculate_dimensions(VAE_IMAGE_SIZE, image.width / image.height)
        cond_images.append(pipe.image_processor.resize(image, ch, cw))
        vae_images.append(pipe.image_processor.preprocess(image, vh, vw).unsqueeze(2))
        vae_sizes.append((vw, vh))
    device = pipe._execution_device
    pos, pos_mask = pipe.encode_prompt(image=cond_images, prompt=prompt, device=device, num_images_per_prompt=1)
    neg, neg_mask = pipe.encode_prompt(image=cond_images, prompt=negative, device=device, num_images_per_prompt=1)
    channels = pipe.transformer.config.in_channels // 4
    latents, source = pipe.prepare_latents(
        vae_images, 1, channels, height, width, pos.dtype, device, generator, None
    )
    target = (1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2)
    shapes = [[target, *[(1, h // pipe.vae_scale_factor // 2, w // pipe.vae_scale_factor // 2) for w, h in vae_sizes]]]
    base = {
        "source_latents": source,
        "img_shapes": shapes,
        "latents": latents,
        "generator": generator,
    }
    pos_data = {**base, "prompt_embeds": pos, "prompt_embeds_mask": pos_mask}
    neg_data = {**base, "prompt_embeds": neg, "prompt_embeds_mask": neg_mask}
    return pos_data, neg_data


def _decode(pipe, latents, height, width):
    import torch

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor).to(pipe.vae.dtype)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    latents = latents / std.to(latents) + mean.to(latents)
    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def infer(args) -> None:
    import torch

    from anchorcache.models.qwen_image_edit.adapter import QwenAdapter
    from anchorcache.models.qwen_image_edit.wiring import load_transformer, make_pipeline
    from anchorcache.sampling import rollout

    dtype = _dtype(args.dtype)
    device = torch.device(args.device)
    anchor = args.topology == "anchor"
    transformer = load_transformer(
        args.model,
        transformer_path=args.transformer,
        device=device,
        dtype=dtype,
        frz_timestep=args.frz_timestep,
        anchor=anchor,
    )
    pipe = make_pipeline(transformer, args.model, weight_dtype=dtype).to(device)
    images = _images(args.image)
    height, width = _size(pipe, images, args.height, args.width)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        pos_data, neg_data = _prepare(
            pipe, images, args.prompt, args.negative, height, width, generator
        )
        adapter = QwenAdapter(transformer=transformer, anchor=anchor)
        cond = adapter.prepare_condition(pos_data)
        neg = adapter.prepare_condition(neg_data)
        cond.payload["negative"] = neg
        sampler = QwenSampler(pipe.scheduler, args.steps)
        schedule = list(sampler.schedule(cond))
        cond.payload["extract_timestep"] = schedule[0]
        neg.payload["extract_timestep"] = schedule[0]
        cond_cache = adapter.extract(cond)
        neg_cache = adapter.extract(neg)
        guidance = QwenCfg(args.cfg, cond_cache, neg_cache)
        final = rollout(
            adapter,
            cond,
            sampler,
            reusable=cond_cache,
            guidance=guidance,
            state=cond["latents"],
            schedule=schedule,
        ).final
        image = _decode(pipe, final, height, width)
    image.save(args.output)


def infer_main() -> None:
    infer(infer_parser().parse_args())


def _tensor_loader(path: str, rank: int = 0, world: int = 1):
    import glob

    import torch

    files = sorted(glob.glob(os.path.join(path, "*.pt"))) if os.path.isdir(path) else [path]
    assert files and all(os.path.isfile(p) for p in files), f"tensor_data not found: {path}"
    while True:
        i = 0
        for file in files:
            data = torch.load(file, map_location="cpu", weights_only=False)
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                i += 1
                if (i - 1) % world != rank:
                    continue
                required = {
                    "latents",
                    "noise",
                    "sigmas",
                    "timesteps",
                    "prompt_embeds",
                    "source_latents",
                    "img_shapes",
                }
                missing = required - row.keys()
                assert not missing, f"{file} is missing required DiT input fields: {sorted(missing)}"
                yield row


def _to_device(data, device, dtype):
    import torch

    if isinstance(data, torch.Tensor):
        out = data.to(device)
        return out.to(dtype) if out.is_floating_point() else out
    if isinstance(data, dict):
        return {k: _to_device(v, device, dtype) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_device(v, device, dtype) for v in data]
    return data


def _run_train(args) -> None:
    import itertools

    import torch

    from anchorcache.models.qwen_image_edit.adapter import QwenAdapter
    from anchorcache.models.qwen_image_edit.wiring import load_teacher, load_train_student
    from anchorcache.training import (
        AccelEngine,
        Distiller,
        StudentRollout,
        TeacherForced,
        TrainCfg,
        Trainer,
        build_optim,
        mse,
        resume_step,
    )

    opd = args.stage == "opd"
    if args.max_steps is None:
        args.max_steps = 500 if opd else 30000
    if args.wd is None:
        args.wd = 1e-2 if opd else 3e-2
    cfg = TrainCfg(
        out_dir=args.output,
        max_steps=args.max_steps,
        lr=args.lr,
        wd=args.wd,
        eps=args.eps,
        warmup=args.warmup,
        clip=args.clip,
        accum=args.accum,
        dist=args.dist,
        seed=args.seed,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        state_every=args.state_every,
    )
    engine = AccelEngine(cfg)
    dtype = _dtype(args.dtype)
    student_tr = load_train_student(
        args.model,
        student_path=args.transformer,
        device=engine.device,
        dtype=dtype,
        train_mode=args.train_mode,
        frz_timestep=args.frz_timestep,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
        anchor=args.topology == "anchor",
    )
    pipe = None
    generator = None
    if args.input_mode == "raw":
        from anchorcache.models.qwen_image_edit.wiring import make_pipeline

        pipe = make_pipeline(student_tr, args.model, weight_dtype=dtype).to(engine.device)
        generator = torch.Generator(device=engine.device).manual_seed(args.seed + engine.rank)

    teacher_tr = load_teacher(
        args.model,
        teacher_path=args.teacher,
        device=engine.device,
        dtype=dtype,
    )

    params = [p for p in student_tr.parameters() if p.requires_grad]
    optim, sched = build_optim(params, cfg)
    student_tr, optim = engine.prepare(student_tr, optim)
    engine.register(sched)
    if args.resume_state:
        engine.load_state(args.resume_state)
        cfg.start_step = resume_step(args.resume_state)
    student = QwenAdapter(transformer=student_tr, anchor=args.topology == "anchor")
    teacher = QwenAdapter(transformer=teacher_tr, anchor=False)
    dist = Distiller(
        teacher=teacher,
        student=student,
        match=mse,
        fold_batch=False,
        backward=engine.backward,
    )
    if opd:
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
        provider = StudentRollout(
            sampler=QwenSampler(scheduler, args.rollout_steps),
            num_states=args.num_states,
            lo=1,
            seed=args.seed,
            rank=engine.rank,
            shared_across_ranks=True,
        )
    else:
        provider = TeacherForced(num_states=1)
    if args.input_mode == "tensor":
        assert args.tensor_data, "tensor mode requires --tensor_data"
        loader = _tensor_loader(args.tensor_data, engine.rank, engine.world)
    else:
        assert args.data_root, "raw mode requires --data_root JSONL"
        from anchorcache.models.qwen_image_edit.data import loader as raw_loader

        loader = raw_loader(args.data_root, engine.rank, engine.world)
    # Skip the samples this rank consumed before the resumed step.
    loader = itertools.islice(loader, cfg.start_step * cfg.accum, None)

    def step_fn(batch):
        if args.input_mode == "tensor":
            prepared = _to_device(batch, engine.device, dtype)
        else:
            from anchorcache.models.qwen_image_edit.data import prepare

            prepared = prepare(
                batch,
                pipe,
                engine.device,
                dtype,
                args.image_size,
                generator,
                weighting=args.weighting,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
            )
        if opd:
            # The rollout starts from noise; init_state reads condition["latents"].
            prepared = {**prepared, "latents": prepared["noise"]}
        return dist.train_step(provider, prepared)

    def save(path):
        unw = engine.unwrap(student_tr)
        unw.save_pretrained(
            path,
            is_main_process=engine.is_main,
            save_function=engine.save,
            state_dict=engine.state_dict_of(student_tr),
        )

    student_tr.train()
    Trainer(
        cfg=cfg,
        engine=engine,
        model=student_tr,
        optim=optim,
        sched=sched,
        step_fn=step_fn,
        on_save=save,
        params=params,
    ).run(loader)


def train_main() -> None:
    args = train_parser().parse_args()
    _run_train(args)
