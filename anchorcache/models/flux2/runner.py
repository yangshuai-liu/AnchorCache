from __future__ import annotations

import argparse
import importlib
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

FLUX_ROOT = os.environ.get("FLUX2_ROOT", "")


def _wiring():
    return importlib.import_module("anchorcache.models.flux2.wiring")


def _topology(args) -> str:
    # The external flux2 model reads the anchor switch from this env var.
    isolated = args.topology == "isolated"
    os.environ["FLUX2_SELFTEXT_DISABLE_EDGE"] = "1" if isolated else "0"
    return "self_only" if isolated else "self_text"


def _add_topology(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--topology",
        default="anchor",
        choices=["anchor", "isolated"],
        help="anchor: AnchorCache static text anchors; isolated: isolated-cache baseline",
    )


def train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train FLUX.2 with AnchorCache Trainer/Distiller")
    p.add_argument("--config", action="append", default=[], help="Merge YAML files in order, with later files overriding earlier ones")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--teacher_dir", default=None)
    p.add_argument("--data_root", required=True)
    p.add_argument("--embeds_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--train_mode", default="full", choices=["lora", "full"])
    p.add_argument("--lora_r", type=int, default=256)
    p.add_argument("--lora_alpha", type=int, default=256)
    p.add_argument(
        "--lora_target_modules",
        default="to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,"
        "to_qkv_mlp_proj,to_out,linear_out,double_stream_modulation_img.linear,"
        "double_stream_modulation_txt.linear,single_stream_modulation.linear",
    )
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--lr_warmup_steps", type=int, default=200)
    p.add_argument("--max_train_steps", type=int, default=30000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_grad_norm", type=float, default=0.05)
    p.add_argument("--adam_weight_decay", type=float, default=3e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-10)
    p.add_argument("--logit_mean", type=float, default=1.5)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--sigma_dist", default="logit_normal", choices=["logit_normal", "uniform", "grid"])
    p.add_argument("--grid_steps", type=int, default=4)
    p.add_argument(
        "--frz_lora_r",
        type=int,
        default=0,
        help="rank of a separate LoRA on the anchor stream (adds parameters)",
    )
    p.add_argument("--frz_timestep", type=float, default=0.0)
    p.add_argument("--ref_timestep", type=float, default=0.0)
    p.add_argument("--rollout_steps", type=int, default=0, help="opd: supervise states k in [1, rollout_steps - 1]")
    p.add_argument("--rollout_infer_steps", type=int, default=4, help="opd: student rollout length")
    p.add_argument("--dist_mode", default="zero2", choices=["zero2", "ddp"])
    p.add_argument("--teacher_mode", default="auto", choices=["auto", "paired", "shared"])
    p.add_argument("--checkpointing_steps", type=int, default=250)
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_lora", default="")
    p.add_argument("--resume_model", default="")
    p.add_argument("--start_step", type=int, default=0)
    p.add_argument("--resume_state", default="")
    p.add_argument("--state_steps", type=int, default=1000)
    p.add_argument("--tracker_project", default="Flux2-AnchorCache-Training")
    p.add_argument("--run_name", default="run")
    p.add_argument("--stage", default="teacher_forced", choices=["teacher_forced", "opd"])
    p.add_argument("--det", action="store_true", help="Enable deterministic algorithms for bitwise comparison")
    p.add_argument("--num_states", type=int, default=4, help="Number of trajectory points supervised per OPD step")
    p.add_argument("--no_tracker", action="store_true")
    p.add_argument("--attn", default="anchorcache", choices=["anchorcache", "backend"])
    p.add_argument("--flux_root", default=FLUX_ROOT)
    _add_topology(p)
    return p


def infer_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run FLUX.2 inference with Flux2Adapter and AnchorCache rollout")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--image", action="append", required=True, help="Reference image; may be specified multiple times")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lora", default="")
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--frz_timestep", type=float, default=0.0)
    p.add_argument("--ref_timestep", type=float, default=0.0)
    p.add_argument("--frz_lora", action="store_true", help="load frz_lora.pt from --lora (ablation checkpoints only)")
    p.add_argument("--flux_root", default=FLUX_ROOT)
    _add_topology(p)
    return p


def _parse_cfg(parser: argparse.ArgumentParser, argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", action="append", default=[])
    known, _ = probe.parse_known_args(argv)
    defaults = {}
    if known.config:
        import yaml

        for path in known.config:
            with open(path) as f:
                values = yaml.safe_load(f) or {}
            if not isinstance(values, dict):
                parser.error(f"YAML top level must be an argument mapping: {path}")
            defaults.update(values)
        valid = {a.dest for a in parser._actions}
        unknown = sorted(defaults.keys() - valid)
        if unknown:
            parser.error(f"YAML contains unknown arguments: {', '.join(unknown)}")
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


class AttnGuard:
    def __init__(self, on: bool):
        self.on = on
        self.n = 0
        self.armed = on

    def install(self):
        if not self.on:
            return
        from flux2 import model as fm

        from . import shim

        base = shim.attn_self_text

        def counted(*args, **kwargs):
            if self.armed:
                self.n += 1
            return base(*args, **kwargs)

        fm.attn_self_text = counted

    def check_once(self, n_layers: int):
        if not self.armed:
            return
        self.armed = False
        assert self.n > 0, "attn=anchorcache, but the shim was never called; training is still using backend attention"
        assert self.n % n_layers == 0, f"The shim was called {self.n} times, which is not a multiple of {n_layers}"


def _check_train(args):
    resumes = sum(bool(x) for x in (args.resume_state, args.resume_lora, args.resume_model))
    assert resumes <= 1, "Specify only one of resume_state, resume_lora, and resume_model"
    if args.stage == "opd":
        assert 1 < args.rollout_steps <= args.rollout_infer_steps, (
            "--stage opd requires 1 < rollout_steps <= rollout_infer_steps"
        )
    else:
        assert args.rollout_steps == 0, "teacher_forced cannot be combined with rollout_steps"


def train_main(argv=None):
    args = _parse_cfg(train_parser(), argv)
    _check_train(args)
    topology = _topology(args)
    wiring = _wiring()
    wiring.add_root(args.flux_root)

    import torch
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler

    from anchorcache.models.flux2 import Flux2Adapter, FullAttnTeacher, SharedTeacher
    from anchorcache.sampling import SchedulerSampler
    from anchorcache.training import (
        AccelEngine,
        Distiller,
        MultiLog,
        PrintLog,
        StudentRollout,
        TeacherForced,
        TrainCfg,
        Trainer,
        WandbLog,
        build_optim,
        mse,
        resume_step,
    )

    if args.det:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    cfg = TrainCfg(
        out_dir=args.output_dir,
        max_steps=args.max_train_steps,
        lr=args.learning_rate,
        wd=args.adam_weight_decay,
        eps=args.adam_epsilon,
        warmup=args.lr_warmup_steps,
        clip=args.max_grad_norm,
        accum=args.gradient_accumulation_steps,
        dist=args.dist_mode,
        seed=args.seed,
        start_step=args.start_step,
        log_every=args.log_steps,
        ckpt_every=args.checkpointing_steps,
        state_every=args.state_steps if args.dist_mode == "zero2" else 0,
        val_every=0,
    )
    engine = AccelEngine(cfg)
    dev, dtype = engine.device, torch.bfloat16
    if engine.is_main:
        os.makedirs(cfg.out_dir, exist_ok=True)

    student_tr = wiring.build_student(
        args.model_dir,
        dev,
        dtype,
        train_mode=args.train_mode,
        frz_timestep=args.frz_timestep,
        ref_timestep=args.ref_timestep,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_targets_csv=args.lora_target_modules,
        frz_lora_r=args.frz_lora_r,
        resume_model=args.resume_model,
        resume_lora=args.resume_lora,
        grad_ckpt=args.gradient_checkpointing,
    )
    params = [p for p in student_tr.parameters() if p.requires_grad]

    tmode = args.teacher_mode
    if tmode == "auto":
        spare = torch.cuda.device_count() >= 2 * engine.local_world
        tmode = "paired" if args.train_mode == "full" or spare else "shared"
    assert args.train_mode == "lora" or tmode != "shared", "Full mode requires a separately frozen teacher"
    if tmode == "paired":
        tdev = wiring.teacher_dev(engine.local_rank, engine.local_world)
        teacher_tr = wiring.build_teacher(args.teacher_dir or args.model_dir, tdev, dtype)
    else:
        assert not args.teacher_dir, "teacher_dir is effective only in paired mode"
        tdev, teacher_tr = dev, None

    vae = AutoencoderKLFlux2.from_pretrained(
        args.model_dir, subfolder="vae", torch_dtype=dtype
    ).to(dev).eval()
    vae.requires_grad_(False)
    sched_base = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_dir, subfolder="scheduler"
    )
    # Teacher-forced steps are sigma in [0, 1]; rollout steps are scheduler timesteps in [0, 1000].
    ts_scale = 1.0 if args.stage == "teacher_forced" else 1.0 / 1000.0
    student = Flux2Adapter(
        transformer=student_tr,
        topology=topology,
        ref_timestep=args.ref_timestep,
        frz_timestep=args.frz_timestep,
        ts_scale=ts_scale,
        sigma_fn=wiring.build_sigma_fn(
            args.sigma_dist, args.grid_steps, args.logit_mean, args.logit_std
        ),
    )
    teacher = (
        FullAttnTeacher(transformer=teacher_tr, device=tdev, out_device=dev, ts_scale=ts_scale)
        if tmode == "paired"
        else SharedTeacher(
            transformer=engine.unwrap(student_tr), device=dev, out_device=dev, ts_scale=ts_scale
        )
    )

    if args.stage == "teacher_forced":
        provider = TeacherForced(num_states=1)
    else:
        from diffusers.pipelines.flux2.pipeline_flux2_klein_kv import (
            compute_empirical_mu,
            retrieve_timesteps,
        )

        sampler = SchedulerSampler(
            sched_base,
            num_steps=args.rollout_infer_steps,
            shift_fn=lambda n, k: compute_empirical_mu(image_seq_len=n, num_steps=k),
            retrieve_fn=retrieve_timesteps,
            seq_len_fn=lambda c: c["tgt_ids"].shape[1],
            device_fn=lambda c: c["tgt_ids"].device,
        )
        provider = StudentRollout(
            sampler=sampler,
            num_states=args.num_states,
            lo=1,
            hi=args.rollout_steps - 1,
            strata_fn=sampler.sigmas,
            seed=args.seed,
            rank=engine.rank,
            shared_across_ranks=True,
        )

    optim, sched = build_optim(params, cfg)
    student_tr, optim = engine.prepare(student_tr, optim)
    engine.register(sched)
    student.transformer = student_tr
    if tmode == "shared":
        teacher.transformer = engine.unwrap(student_tr)
    if args.resume_state:
        engine.load_state(args.resume_state)
        cfg.start_step = resume_step(args.resume_state)

    loader = wiring.build_loader(
        args.data_root,
        args.embeds_root,
        args.image_size,
        engine.rank,
        engine.world,
        args.seed,
        workers=args.dataloader_num_workers,
    )
    prep = wiring.build_prep(vae, dev, dtype)
    dist = Distiller(
        teacher=teacher,
        student=student,
        match=mse,
        fold_batch=False,
        backward=engine.backward,
    )
    log = MultiLog(
        PrintLog(keys=("train/loss",), enabled=engine.is_main),
        WandbLog(
            engine,
            args.tracker_project,
            args.run_name,
            vars(args),
            mode=os.environ.get("WANDB_MODE", "offline"),
        ) if args.tracker_project and not args.no_tracker else None,
    )

    unw = engine.unwrap(student_tr)
    n_layers = len(unw.transformer_blocks) + len(unw.single_transformer_blocks)
    guard = AttnGuard(args.attn == "anchorcache")
    guard.install()

    def step_fn(batch):
        out = dist.train_step(provider, prep(batch))
        guard.check_once(n_layers)
        return out

    if engine.is_main:
        print(
            f"stage={args.stage} train_mode={args.train_mode} teacher_mode={tmode} "
            f"trainable={sum(p.numel() for p in params):,} tdev={tdev} attn={args.attn}",
            flush=True,
        )
    student_tr.train()
    Trainer(
        cfg=cfg,
        engine=engine,
        model=student_tr,
        optim=optim,
        sched=sched,
        step_fn=step_fn,
        log=log,
        on_save=wiring.build_save_fn(engine, student_tr, args.train_mode, args.frz_lora_r),
        params=params,
    ).run(loader)


def infer_main(argv=None):
    args = infer_parser().parse_args(argv)
    topology = _topology(args)
    wiring = _wiring()
    wiring.add_root(args.flux_root)

    import torch
    from diffusers.pipelines.flux2.pipeline_flux2_klein_kv import (
        compute_empirical_mu,
        retrieve_timesteps,
    )

    from anchorcache.models.flux2 import Flux2Adapter
    from anchorcache.sampling import CfgGuidance, SchedulerSampler, rollout

    device = torch.device(args.device)
    dtype = torch.bfloat16
    with torch.inference_mode():
        transformer = wiring.build_infer(
            args.model_dir,
            device,
            dtype,
            lora=args.lora,
            frz_timestep=args.frz_timestep,
            ref_timestep=args.ref_timestep,
            frz_lora=args.frz_lora,
        )
        pipe = wiring.build_pipe(args.model_dir, transformer, device, dtype)
        generator = torch.Generator(device=device.type).manual_seed(args.seed)

        emb, txt_ids = pipe.encode_prompt(args.prompt, device=device)
        ref_pack, ref_ids, ref_hw = wiring.prep_images(
            pipe, args.image, device, dtype, generator
        )
        height = args.height or ref_hw[0]
        width = args.width or ref_hw[1]
        latents, tgt_ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=transformer.config.in_channels // 4,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        inputs = {
            "emb": emb,
            "txt_ids": txt_ids,
            "ref_pack": ref_pack,
            "ref_ids": ref_ids,
            "tgt_ids": tgt_ids,
            "latents": latents,
        }
        if args.guidance > 1.0:
            neg_emb, neg_ids = pipe.encode_prompt(args.negative_prompt, device=device)
            inputs["negative"] = {
                **inputs,
                "emb": neg_emb,
                "txt_ids": neg_ids,
            }

        adapter = Flux2Adapter(
            transformer=transformer,
            topology=topology,
            ref_timestep=args.ref_timestep,
            frz_timestep=args.frz_timestep,
            ts_scale=1.0 / 1000.0,
        )
        condition = adapter.prepare_condition(inputs)
        sampler = SchedulerSampler(
            pipe.scheduler,
            num_steps=args.steps,
            shift_fn=lambda n, k: compute_empirical_mu(image_seq_len=n, num_steps=k),
            retrieve_fn=retrieve_timesteps,
            seq_len_fn=lambda c: c["tgt_ids"].shape[1],
            device_fn=lambda c: c["tgt_ids"].device,
        )
        schedule = list(sampler.schedule(condition))
        extract_ts = (schedule[0].expand(latents.shape[0]) / 1000).to(dtype)
        condition.payload["extract_timestep"] = extract_ts
        reusable = adapter.extract(condition)
        guidance = None
        if args.guidance > 1.0:
            # Anchored reference K/V depend on the prompt.
            neg = condition["negative"]
            neg["extract_timestep"] = extract_ts
            share = topology == "self_only"
            guidance = CfgGuidance(
                scale=args.guidance,
                rescale=False,
                share_reusable=share,
                neg_reusable=None if share else adapter.extract(neg),
            )
        final = rollout(
            adapter,
            condition,
            sampler,
            reusable=reusable,
            guidance=guidance,
            state=latents,
            schedule=schedule,
        ).final
        image = wiring.decode(pipe, final, tgt_ids)[0]
    image.save(args.output)


__all__ = ["infer_main", "infer_parser", "train_main", "train_parser"]
