#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/local_global_correspondence/run.sh \
    --config-file /path/to/config.yaml \
    --pretrained-weights /path/to/teacher_checkpoint.pth \
    --output-dir /path/to/output_dir \
    --data-path /path/to/copydays \
    [--eval-module jea.diagnostics.local_global_correspondence] \
    [--gpu 0,1,2,3] \
    [--port 29501] \
    [--layer final] \
    [--crop-ratios 0.1,0.2,0.3,0.5] \
    [--run-seeds 0] \
    [--selected-crop-specs-json /path/to/fixed_crop_specs.json] \
    [--write-crop-specs-json /path/to/crop_specs.json] \
    [--topk 5] \
    [--save-visualizations] \
    [--visualization-samples 16] \
    [--visualization-images 16] \
    [--visualization-max-match-lines 48] \
    [--visualization-max-per-image 16] \
    [-- ...extra local_global_correspondence args...]
EOF
}

CONFIG_FILE=""
PRETRAINED_WEIGHTS=""
OUTPUT_DIR=""
DATA_PATH=""
EVAL_MODULE="jea.diagnostics.local_global_correspondence"
GPU=""
PORT=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    --pretrained-weights) PRETRAINED_WEIGHTS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --eval-module) EVAL_MODULE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -n "$CONFIG_FILE" && -n "$PRETRAINED_WEIGHTS" && -n "$OUTPUT_DIR" && -n "$DATA_PATH" ]] || { echo "Missing required arguments." >&2; usage; exit 1; }
[[ -f "$CONFIG_FILE" ]] || { echo "Config file not found: $CONFIG_FILE" >&2; exit 1; }
[[ -f "$PRETRAINED_WEIGHTS" ]] || { echo "Weights not found: $PRETRAINED_WEIGHTS" >&2; exit 1; }
[[ -d "$DATA_PATH" ]] || { echo "Data path not found: $DATA_PATH" >&2; exit 1; }

infer_nproc_from_gpu() {
  local gpu_list="${1// /}"
  if [[ -n "$gpu_list" && "$gpu_list" == *,* ]]; then
    awk -F',' '{print NF}' <<< "$gpu_list"
  fi
}

RUN_NPROC="${TORCHRUN_NPROC_PER_NODE:-}"
if [[ -z "$RUN_NPROC" ]]; then
  RUN_NPROC="$(infer_nproc_from_gpu "$GPU")"
fi
if [[ "$RUN_NPROC" =~ ^[0-9]+$ && "$RUN_NPROC" -gt 1 ]]; then
  RUN_CMD=(torchrun --nproc_per_node="$RUN_NPROC")
  [[ -n "$PORT" ]] && RUN_CMD+=(--master_port="$PORT")
  RUN_CMD+=(-m "$EVAL_MODULE")
else
  RUN_CMD=(python -m "$EVAL_MODULE")
fi

CMD=(
  "${RUN_CMD[@]}"
  --config-file "$CONFIG_FILE"
  --pretrained-weights "$PRETRAINED_WEIGHTS"
  --output-dir "$OUTPUT_DIR"
  --data-path "$DATA_PATH"
  "${EXTRA_ARGS[@]}"
)

if [[ -n "$GPU" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
else
  "${CMD[@]}"
fi
