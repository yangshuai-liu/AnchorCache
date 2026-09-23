#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python3}

usage() {
  cat <<'EOF'
Usage:
  scripts/qwen.sh infer --model BASE --student DIT --image IMG [--image IMG] --prompt TEXT --output OUT [additional arguments]
  scripts/qwen.sh train --input_mode raw --model BASE --student DIT --teacher TEA \
    --data_root DATA_JSONL --output OUT [additional arguments]
  scripts/qwen.sh train --input_mode tensor --model BASE --student DIT --teacher TEA \
    --tensor_data FILE_OR_DIR --output OUT [additional arguments]

Arguments:
  --model       Diffusers base model providing the frozen VAE, VLM, and scheduler
  --student     AnchorCache student DiT weights; defaults to --model/transformer when omitted
  --teacher     Full-attention teacher DiT weights; defaults to --model/transformer when omitted
  --tensor_data Preprocessed DiT input .pt file or directory
  --stage       teacher_forced (Stage 1, default) or opd (Stage 2)
  --topology    anchor (AnchorCache, default) or isolated (isolated-cache baseline)

Fields required in each preprocessed .pt sample:
  latents, noise, sigmas, timesteps, prompt_embeds,
  source_latents, img_shapes; prompt_embeds_mask is optional.
EOF
}

[[ $# -gt 0 ]] || { usage; exit 2; }
mode=$1
shift
case "$mode" in
  infer) entry="$ROOT/examples/qwen_image_edit/infer.py" ;;
  train) entry="$ROOT/examples/qwen_image_edit/train.py" ;;
  -h|--help|help) usage; exit 0 ;;
  *) echo "Unknown mode: $mode" >&2; usage; exit 2 ;;
esac

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export QWEN_KV_PREALLOC=${QWEN_KV_PREALLOC:-1}

exec "$PY" "$entry" "$@"
