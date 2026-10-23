#!/usr/bin/env bash
# DINO / DINO+iBOT pretraining on ImageNet-1K (300 epochs).
# Run from the repository root. Extra arguments are OmegaConf key=value overrides.
#   bash scripts/train.sh dino      b
#   bash scripts/train.sh dino_ibot b train.output_dir=/checkpoints
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/train.sh {dino|dino_ibot} {s|b|l|h} [key=value ...]" >&2
  exit 2
fi
method="$1"; size="$2"; shift 2
[[ "$method" == dino || "$method" == dino_ibot ]] || { echo "Unknown method: $method" >&2; exit 2; }
[[ "$size" =~ ^[sblh]$ ]] || { echo "Unknown model size: $size" >&2; exit 2; }
gpu="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra devices <<< "$gpu"
export CUDA_VISIBLE_DEVICES="$gpu"
export WANDB_MODE="${WANDB_MODE:-disabled}"
torchrun --nnodes=1 --nproc_per_node="${#devices[@]}" --master_port="${MASTER_PORT:-29500}" \
  -m jea.train.train --config-file "jea/configs/train/${method}/vit_${size}16.yaml" "$@"
