# AnchorCache

Official implementation of *Beyond Attention Masks: Instruction Anchoring for Efficient In-Context Diffusion Generation*. The framework separates fixed-condition computation from the evolving generation state, computes it once, and reuses it exactly across denoising steps.

AnchorCache inserts static text anchors (a copy of the instruction with a fixed timestep of 0) that condition the reference tokens during cache construction. The reference K/V are then instruction-aware yet independent of the denoising target, so they are computed once and reused at every step. Quality lost in the architectural conversion is recovered with teacher-forced velocity distillation (Stage 1) followed by a short on-policy stage (Stage 2).

| Backend | Status |
| --- | --- |
| Qwen-Image-Edit-2511 | Self-contained: inference, Stage 1 and Stage 2 training |
| FLUX.2 | Requires the external FLUX.2 AnchorCache model code (`--flux_root` / `FLUX2_ROOT`) |
| OmniVoice | Requires an OmniVoice build with SelfText attention support |

## Table of Contents

- [Overview](#mega-overview)
- [Installation](#hammer-installation)
- [Model Weights](#model-weights)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Inference and Training Protocols](#inference-and-training-protocols)
- [Runtime](#runtime)
- [Acknowledgement](#hearts-acknowledgement)
- [Citation](#black_nib-citation)

## :mega: Overview

### Core Abstractions

```text
Condition       Fixed input within a request
State           Generation state that evolves step by step
ReusableState   Model state reusable across steps
Execution       prepare_condition -> extract -> predict
```

Transformer sequences are partitioned into regions by role:

| Region | Meaning | Cached | Paper notation |
| --- | --- | --- | --- |
| `LIVE` | Depends on the current generation state; recomputed every step | No | instruction `LT`, target `X_t` |
| `VIS` | Invariant across steps and visible to live queries | Yes | references `R` |
| `HID` | Invariant across steps and visible only inside the static group | No (discarded after extraction) | static text anchors `ST` |

`FROZEN` denotes `VIS | HID`. Each row attends to these columns:

```text
LIVE   -> VIS | LIVE          # LT, X_t attend to LT, X_t, R
FROZEN -> FROZEN              # ST, R attend only to each other
```

Setting `HID` to empty gives the isolated-cache baseline; setting both `VIS` and `HID` to empty gives full attention.

### Repository Structure

```text
anchorcache/
├── core/          # Condition / State / ReusableState / Execution
├── transformer/   # regions / topology / cached attention / KV
├── runtime/       # residency / offload / prefetch / buffer pool
├── sampling/      # sampler / rollout / guidance
├── training/      # state provider / matching / distillation / trainer
└── models/
    ├── qwen_image_edit/
    ├── flux2/
    └── omnivoice/
```

`core/` does not depend on any specific model.

## :hammer: Installation

```bash
pip install -e .
pip install -e '.[qwen]'
pip install -e '.[flux2]'
pip install -e '.[omnivoice]'
```

The dependency stacks of the three backends may conflict, so separate virtual environments are recommended. Compiled dependencies such as FlashAttention must be installed for the target PyTorch/CUDA environment. DeepSpeed is required for the default ZeRO-2 training mode.

**Hardware.** Qwen-Image-Edit-2511 has a 20B-parameter DiT (about 40 GB in bf16), so inference needs a GPU with at least 48 GB. Training keeps a frozen teacher and a fully trained student on every rank (ZeRO-2 shards optimizer state and gradients, not parameters), which is about 80 GB of bf16 weights per GPU before optimizer state and activations; use GPUs with well over 80 GB of memory. FLUX.2 9B training fits on 80 GB GPUs.

## Model Weights

## Data Preparation

## Usage

```text
examples/
├── qwen_image_edit/{train.py,infer.py}
├── flux2/{train.py,infer.py}
└── omnivoice/{train.py,infer.py}
```

After installation, the following commands are available:

```text
anchorcache-qwen-train / anchorcache-qwen-infer
anchorcache-flux2-train / anchorcache-flux2-infer
anchorcache-voice-train / anchorcache-voice-infer
```

All backends accept `--topology anchor` (AnchorCache, default) or `--topology isolated` (isolated-cache baseline). The same value must be used for training and inference.

### Qwen-Image-Edit

The Qwen backend is built into `anchorcache/models/qwen_image_edit/` and depends only on diffusers and the model weights.

```bash
# Inference: 40 steps, CFG 4, blank negative prompt, seed 42 by default
scripts/qwen.sh infer \
  --model /path/to/Qwen-Image-Edit-2511 \
  --student /path/to/student \
  --image ref1.png --image ref2.png \
  --prompt "edit instruction" \
  --output out.png

# Stage 1: teacher-forced velocity distillation on 8 GPUs (global batch 8)
torchrun --nproc_per_node 8 examples/qwen_image_edit/train.py \
  --input_mode raw \
  --model /path/to/Qwen-Image-Edit-2511 \
  --data_root train.jsonl \
  --output stage1

# Stage 2: on-policy distillation from the Stage 1 checkpoint
torchrun --nproc_per_node 8 examples/qwen_image_edit/train.py \
  --stage opd \
  --input_mode raw \
  --model /path/to/Qwen-Image-Edit-2511 \
  --student stage1/ckpt-final \
  --data_root train.jsonl \
  --output stage2
```

The training defaults are the paper's recovery settings: full-parameter bf16 training with ZeRO-2 and gradient checkpointing, AdamW (betas 0.9/0.999, eps 1e-10), learning rate 5e-6 with cosine decay and 200 warmup steps, gradient clipping at 0.05, 30k steps with weight decay 3e-2 for Stage 1 and 500 steps with weight decay 1e-2 for Stage 2. Each GPU processes one sample per step, so the global batch size is the number of GPUs times `--accum`; the paper uses 8. Stage 2 rolls out the student for 40 steps without CFG and supervises `N_q = 4` states sampled from steps `k ∈ [1, 39]`.

Training state is saved to `OUTPUT/state-latest` every `--state_every` steps; pass `--resume_state OUTPUT/state-latest` to continue from it.

Each line of the raw JSONL has the following format:

```json
{"image_refs": ["ref1.jpg", "ref2.jpg"], "target": "target.jpg", "prompt": "instruction"}
```

With `--input_mode tensor --tensor_data /path/to/dit_inputs`, each preprocessed `.pt` sample contains:

```text
latents, noise, sigmas, timesteps, prompt_embeds,
source_latents, img_shapes
```

`prompt_embeds_mask` is optional. Tensor mode loads only the teacher/student DiT; raw mode uses the frozen diffusers VAE/VLM for preprocessing.

### FLUX.2

The FLUX.2 backend wraps an external AnchorCache model implementation. Point `--flux_root` (or `FLUX2_ROOT`) at a checkout that provides `training/flux2`. Run `anchorcache-flux2-train --help` and `anchorcache-flux2-infer --help` for the full argument list; Stage 2 uses `--stage opd --rollout_steps 4`.

### OmniVoice

The OmniVoice backend needs an OmniVoice installation with SelfText attention support. It uses discrete masked-token prediction, so distillation matches logits with reverse KL instead of velocity MSE; Stage 2 is `--stage opd`.

### Environment Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `QWEN_KV_SDP_BACKEND` | `flash` | Qwen attention backend: `flash`, `cudnn`, or `sage` (SageAttention2) |
| `QWEN_KV_PREALLOC` | `0` (`1` in `scripts/qwen.sh`) | Preallocate cached K/V buffers and update the live segment in place |
| `QWEN_GC_SKIP_LAYERS` | `0` | Skip activation checkpointing for the first N layers |
| `ANCHORCACHE_ATTN_BACKEND` | `sdpa` | Backend for the framework's grouped attention: `sdpa`, `flash`, `cudnn`, or `fa` |
| `FLUX2_ROOT` | unset | Location of the external FLUX.2 model code |

## Inference and Training Protocols

### Inference

```python
condition = adapter.prepare_condition(inputs)
reusable = adapter.extract(condition)
state = adapter.init_state(condition)

for step in sampler.schedule(condition):
    output = adapter.predict(state, step, condition, reusable)
    state = sampler.step(state, output.prediction, step, condition)
```

`anchorcache.sampling.generate` implements exactly this loop.

### Training

Training stages differ only in where states come from:

```text
TeacherForced   Stage 1: data-derived states (noise interpolation or masking)
StudentRollout  Stage 2: states visited by the student's own unguided rollout
```

Both stages query the frozen full-attention teacher at the same states. Qwen/FLUX.2 use velocity MSE; OmniVoice uses reverse KL over logits.

## Runtime

Logical cache management is separated from physical placement:

```python
from anchorcache.runtime import CacheManager, RollingResidency

manager = CacheManager(
    residency=RollingResidency(offload_layers=40, lookahead=1, host_offload=True),
    device="cuda",
)
```

## :hearts: Acknowledgement

### :hugs: Open-Sourced Repositories

This project would not be possible without the following open-source repositories and communities:

- [Qwen-Image](https://github.com/QwenLM/Qwen-Image), which provides the Qwen image generation and editing model family.
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers), which provides the image model implementations, pipelines, schedulers, VAEs, and ecosystem integrations used by the image backends.
- [FireRed-Image-Edit](https://github.com/FireRedTeam/FireRed-Image-Edit), whose open-source image-editing training code informed the original data, model-loading, and training workflow used for the Qwen-Image-Edit backend.
- [FLUX](https://github.com/black-forest-labs/flux) and Black Forest Labs, which provide the FLUX model family.
- [OmniVoice](https://github.com/k2-fsa/OmniVoice) and the k2-fsa community, which provide the multilingual TTS model, tokenizer, codec, and data pipeline.
- [FastVideo](https://github.com/hao-ai-lab/FastVideo), which provides a unified framework for accelerated video generation inference and post-training.
- [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3), which provides an open omni-modal generative system and practical references for multimodal generation workflows.
- [PyTorch](https://github.com/pytorch/pytorch), [Transformers](https://github.com/huggingface/transformers), [Accelerate](https://github.com/huggingface/accelerate), and [PEFT](https://github.com/huggingface/peft), which form the training and inference foundation.

More details and third-party license notes are available in [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).

AnchorCache is not affiliated with or endorsed by these projects. Model weights, datasets, and third-party components remain subject to their respective licenses and terms of use.

## :black_nib: Citation

If you find this work useful, please cite:

```bibtex
@article{liu2026anchorcache,
  title   = {Beyond Attention Masks: Instruction Anchoring for Efficient In-Context Diffusion Generation},
  author  = {Liu, Yangshuai and Li, Zheming and Li, Jiaao and He, Kang and Lai, Ziliang and Liu, Zhitai and Song, Chengru},
  year    = {2026}
}
```
