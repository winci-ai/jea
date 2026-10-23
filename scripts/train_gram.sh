#!/usr/bin/env bash
# Gram anchoring: continue a DINO+iBOT run from its epoch-260 FSDP2 checkpoint
# (student, EMA teacher, optimizer and epoch counter are restored) with a Gram
# anchor initialized from an epoch-250 EMA teacher export.
#   gram      : anchor = DINO+iBOT epoch-250 teacher (standard Gram anchoring)
#   dino_gram : anchor = DINO epoch-250 teacher      (DINO-Gram)
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/train_gram.sh {b|l} {gram|dino_gram} BASELINE_EPOCH260_DIR ANCHOR_EPOCH250_PTH [key=value ...]" >&2
  exit 2
fi
size="$1"; anchor="$2"; resume="$3"; teacher="$4"; shift 4
[[ "$size" =~ ^[bl]$ ]] || { echo 'Gram experiments use B/16 or L/16.' >&2; exit 2; }
[[ "$anchor" == gram || "$anchor" == dino_gram ]] || { echo 'Unknown anchor type.' >&2; exit 2; }
[[ -f "$resume/.metadata" && -f "$teacher" ]] || { echo 'Missing FSDP2 checkpoint or full anchor weights.' >&2; exit 2; }
export GRAM_TEACHER_CHECKPOINT="$teacher"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
torchrun --nnodes=1 --nproc_per_node="${#devices[@]}" --master_port="${MASTER_PORT:-29500}" \
  -m jea.train.train --config-file "jea/configs/train/dino_ibot/vit_${size}16_${anchor}.yaml" \
  "MODEL.WEIGHTS=$resume" "$@"
