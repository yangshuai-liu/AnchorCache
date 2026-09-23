from __future__ import annotations

import math
import os
import sys

FLUX_ROOT = os.environ.get("FLUX2_ROOT", "")


def add_root(flux_root: str = FLUX_ROOT) -> str:
    assert flux_root, "Specify the FLUX.2 backend via --flux_root or FLUX2_ROOT"
    path = os.path.join(flux_root, "training")
    assert os.path.isdir(path), f"FLUX.2 training directory not found: {path}"
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def build_student(
    model_dir: str,
    device,
    dtype,
    *,
    train_mode: str = "full",
    frz_timestep: float = 0.0,
    ref_timestep: float = 0.0,
    lora_r: int = 256,
    lora_alpha: int = 256,
    lora_targets_csv: str = "",
    frz_lora_r: int = 0,
    resume_model: str = "",
    resume_lora: str = "",
    grad_ckpt: bool = True,
):
    import torch
    from diffusers import Flux2Transformer2DModel
    from flux2 import enable_frz, enable_self_text, load_frz, lora_targets
    from peft import LoraConfig

    path = resume_model or model_dir
    sub = None if resume_model else "transformer"
    tr = Flux2Transformer2DModel.from_pretrained(path, subfolder=sub, torch_dtype=dtype).to(device)
    # enable_self_text must precede LoRA injection.
    tr = enable_self_text(tr, frz_timestep, ref_timestep=ref_timestep)

    if train_mode == "full":
        assert not resume_lora, "Full mode cannot load LoRA weights"
        assert frz_lora_r == 0, "Full mode cannot create an frz LoRA"
        tr.requires_grad_(True)
    else:
        assert not resume_model, "LoRA mode cannot use resume_model"
        tr.requires_grad_(False)
        if resume_lora:
            tr.load_lora_adapter(
                resume_lora,
                prefix=None,
                adapter_name="default",
                weight_name="pytorch_lora_weights.safetensors",
            )
        else:
            tr.add_adapter(
                LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    init_lora_weights="gaussian",
                    target_modules=lora_targets(tr, lora_targets_csv.split(",")),
                )
            )
        if frz_lora_r > 0:
            enable_frz(tr, frz_lora_r)
            frz_path = os.path.join(resume_lora, "frz_lora.pt") if resume_lora else ""
            if frz_path and os.path.exists(frz_path):
                load_frz(tr, torch.load(frz_path, map_location="cpu"))

    if grad_ckpt:
        tr.enable_gradient_checkpointing()
    return tr


def build_infer(
    model_dir: str,
    device,
    dtype,
    *,
    lora: str = "",
    frz_timestep: float = 0.0,
    ref_timestep: float = 0.0,
    frz_lora: bool = False,
):
    import torch
    from diffusers import Flux2Transformer2DModel
    from flux2 import enable_self_text

    tr = Flux2Transformer2DModel.from_pretrained(
        model_dir, subfolder="transformer", torch_dtype=dtype
    ).to(device)
    tr = enable_self_text(tr, frz_timestep, ref_timestep=ref_timestep)
    if lora:
        tr.load_lora_adapter(
            lora,
            prefix=None,
            adapter_name="default",
            weight_name="pytorch_lora_weights.safetensors",
        )
        frz_path = os.path.join(lora, "frz_lora.pt")
        assert frz_lora or not os.path.exists(frz_path), (
            f"{frz_path} exists: this checkpoint was trained with --frz_lora_r; pass --frz_lora to load it"
        )
        if frz_lora:
            from flux2 import load_frz

            load_frz(tr, torch.load(frz_path, map_location="cpu"))
    tr.requires_grad_(False)
    tr.eval()
    return tr


def teacher_dev(local_rank: int, local_world: int, offset: int | None = None):
    import torch

    if offset is None:
        offset = int(os.environ.get("FLUX2_TEACHER_DEVICE_OFFSET", "-1"))
    if offset < 0:
        offset = local_world if torch.cuda.device_count() >= 2 * local_world else 0
    return torch.device("cuda", local_rank + offset)


def build_teacher(model_dir: str, device, dtype):
    from diffusers import Flux2Transformer2DModel

    tr = Flux2Transformer2DModel.from_pretrained(
        model_dir, subfolder="transformer", torch_dtype=dtype
    ).to(device)
    tr.requires_grad_(False)
    tr.eval()
    return tr


def build_save_fn(engine, model, train_mode: str, frz_lora_r: int = 0):
    def save(path: str) -> None:
        unw = engine.unwrap(model)
        if train_mode == "full":
            unw.save_pretrained(
                path,
                is_main_process=engine.is_main,
                save_function=engine.save,
                state_dict=engine.state_dict_of(model),
            )
        elif engine.is_main:
            unw.save_lora_adapter(path)
            if frz_lora_r > 0:
                import torch
                from flux2 import frz_sd

                torch.save(frz_sd(unw), os.path.join(path, "frz_lora.pt"))
        if engine.is_main:
            print(f"saved {path}", flush=True)

    return save


def build_loader(data_root, embeds_root, size, rank, world, seed, workers=0):
    from data import collate_one, omni_stream
    from torch.utils.data import DataLoader

    ds = omni_stream(data_root, embeds_root, size, rank=rank, world=world, seed=seed)
    return DataLoader(
        ds,
        batch_size=1,
        collate_fn=collate_one,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=2 if workers else None,
    )


def build_prep(vae, device, dtype):
    from flux2.latent import prep_batch

    def prep(batch):
        emb, txt_ids, ref_pack, ref_ids, x0, tgt_ids, _hw = prep_batch(
            batch, vae, device, dtype
        )
        return {
            "emb": emb,
            "txt_ids": txt_ids,
            "ref_pack": ref_pack,
            "ref_ids": ref_ids,
            "tgt_ids": tgt_ids,
            "x0": x0,
        }

    return prep


def build_sigma_fn(
    dist: str,
    grid_steps: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
):
    import torch

    def sigma_fn(batch):
        x0 = batch["x0"]
        bsz = x0.shape[0]
        if dist == "uniform":
            return torch.rand(bsz, device=x0.device)
        if dist == "grid":
            from diffusers.pipelines.flux2.pipeline_flux2_klein_kv import compute_empirical_mu

            mu = compute_empirical_mu(image_seq_len=x0.shape[1], num_steps=grid_steps)
            raw = torch.linspace(1.0, 1e-3, 1000, dtype=torch.float64)
            shifted = math.exp(mu) / (math.exp(mu) + (1.0 / raw - 1.0))
            return shifted[torch.randint(1000, (bsz,))].float().to(x0.device)
        u = torch.randn(bsz, device=x0.device) * logit_std + logit_mean
        return torch.sigmoid(u)

    return sigma_fn


def build_pipe(model_dir: str, transformer, device, dtype):
    from diffusers.pipelines.flux2.pipeline_flux2_klein_kv import Flux2KleinKVPipeline

    pipe = Flux2KleinKVPipeline.from_pretrained(
        model_dir, transformer=transformer, torch_dtype=dtype
    ).to(device)
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def prep_images(pipe, paths: list[str], device, dtype, generator):
    from PIL import Image

    first_hw = None
    images = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        width, height = img.size
        if width * height > 1024 * 1024:
            img = pipe.image_processor._resize_to_target_area(img, 1024 * 1024)
            width, height = img.size
        multiple = pipe.vae_scale_factor * 2
        width = width // multiple * multiple
        height = height // multiple * multiple
        if first_hw is None:
            first_hw = (height, width)
        images.append(
            pipe.image_processor.preprocess(
                img, height=height, width=width, resize_mode="crop"
            )
        )
    ref_pack, ref_ids = pipe.prepare_image_latents(
        images=images,
        batch_size=1,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return ref_pack, ref_ids, first_hw


def decode(pipe, latents, lat_ids):
    import torch

    lat = pipe._unpack_latents_with_ids(latents, lat_ids)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
    std = torch.sqrt(
        pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
    ).to(lat.device, lat.dtype)
    lat = pipe._unpatchify_latents(lat * std + mean)
    image = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")
