from __future__ import annotations

import json
import os


def rows(path: str):
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            refs = row.get("image_refs") or row.get("images")
            assert refs and row.get("target") and row.get("prompt"), (
                "raw JSONL requires image_refs/images, target, and prompt"
            )
            row["image_refs"] = [p if os.path.isabs(p) else os.path.join(base, p) for p in refs]
            target = row["target"]
            row["target"] = target if os.path.isabs(target) else os.path.join(base, target)
            yield row


def loader(path: str, rank: int = 0, world: int = 1):
    while True:
        for i, row in enumerate(rows(path)):
            if i % world == rank:
                yield row


def prepare(
    row,
    pipe,
    device,
    dtype,
    image_size: int,
    generator,
    weighting: str = "logit_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
):
    import torch
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        calculate_dimensions,
    )
    from diffusers.training_utils import compute_density_for_timestep_sampling
    from PIL import Image

    def hw(image, area):
        # CONDITION_IMAGE_SIZE and image_size**2 are areas.
        w, h = calculate_dimensions(area, image.width / image.height)
        return h, w

    refs = [Image.open(path).convert("RGB") for path in row["image_refs"]]
    target = Image.open(row["target"]).convert("RGB")
    cond = [pipe.image_processor.resize(image, *hw(image, CONDITION_IMAGE_SIZE)) for image in refs]
    embeds, mask = pipe.encode_prompt(
        prompt=row["prompt"],
        image=cond,
        device=device,
        num_images_per_prompt=1,
    )

    def latent(image):
        tensor = pipe.image_processor.preprocess(image, *hw(image, image_size * image_size)).unsqueeze(2)
        raw = pipe._encode_vae_image(tensor.to(device=device, dtype=dtype), generator)
        h, w = raw.shape[3:]
        packed = pipe._pack_latents(raw, 1, raw.shape[2], h, w)
        return packed, (1, h // 2, w // 2)

    x0, target_shape = latent(target)
    source = []
    ref_shapes = []
    for image in refs:
        packed, shape = latent(image)
        source.append(packed)
        ref_shapes.append(shape)
    source = torch.cat(source, dim=1)
    noise = torch.randn(x0.shape, generator=generator, device=device, dtype=dtype)
    sigma = compute_density_for_timestep_sampling(
        weighting_scheme=weighting,
        batch_size=1,
        logit_mean=logit_mean,
        logit_std=logit_std,
        device=device,
        generator=generator,
    ).float()
    return {
        "latents": x0,
        "noise": noise,
        "sigmas": sigma,
        "timesteps": sigma.to(dtype),
        "prompt_embeds": embeds,
        "prompt_embeds_mask": mask,
        "source_latents": source,
        "img_shapes": [[target_shape, *ref_shapes]],
    }
