#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/image_retrieval/run_all.sh \
    --config-file /path/to/config.yaml \
    --eval-root /path/to/eval \
    --output-root /path/to/output_dir \
    --csv-out /path/to/summary.csv \
    --data-path /path/to/revisited_paris_oxford \
    [--gpu 0] \
    [--port 29501] \
    [--dataset roxford5k] \
    [--imsize 224] \
    [--num-workers 4] \
    [--gather-on-cpu]
EOF
}

CONFIG_FILE=""
EVAL_ROOT=""
OUTPUT_ROOT=""
CSV_OUT=""
DATA_PATH=""
GPU=""
PORT=""
DATASET="roxford5k"
IMSIZE=224
NUM_WORKERS=4
GATHER_ON_CPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --csv-out) CSV_OUT="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --imsize) IMSIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --gather-on-cpu) GATHER_ON_CPU=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$CONFIG_FILE" && -n "$EVAL_ROOT" && -n "$OUTPUT_ROOT" && -n "$CSV_OUT" && -n "$DATA_PATH" ]] || { echo "Missing required arguments." >&2; usage; exit 1; }
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$(dirname "$CSV_OUT")"

if [[ -n "${TORCHRUN_NPROC_PER_NODE:-}" ]]; then
  RUN_CMD=(torchrun --nproc_per_node="$TORCHRUN_NPROC_PER_NODE")
  [[ -n "$PORT" ]] && RUN_CMD+=(--master_port="$PORT")
  RUN_CMD+=(jea/eval/eval_image_retrieval.py)
else
  RUN_CMD=(python jea/eval/eval_image_retrieval.py)
fi

TMP_JSONL="$(mktemp)"
trap 'rm -f "$TMP_JSONL"' EXIT

append_metrics_record() {
  local metrics_json="$1"
  local run_name="$2"
  local ckpt_path="$3"

  python - "$metrics_json" "$run_name" "$ckpt_path" "$DATASET" "$TMP_JSONL" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
run_name = sys.argv[2]
ckpt_path = sys.argv[3]
dataset = sys.argv[4]
jsonl_path = Path(sys.argv[5])

record = json.loads(metrics_path.read_text())
record["run_name"] = run_name
record["checkpoint_path"] = ckpt_path
record["dataset"] = dataset

with jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
}

mapfile -t TRAIN_DIRS < <(find "$EVAL_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'training_*' | sort -V)
[[ ${#TRAIN_DIRS[@]} -gt 0 ]] || { echo "No training_* directories found under $EVAL_ROOT" >&2; exit 1; }

for TRAIN_DIR in "${TRAIN_DIRS[@]}"; do
  CKPT_PATH="$TRAIN_DIR/teacher_checkpoint.pth"
  [[ -f "$CKPT_PATH" ]] || { echo "Skipping $(basename "$TRAIN_DIR"): missing teacher_checkpoint.pth" >&2; continue; }

  RUN_NAME="$(basename "$TRAIN_DIR")"
  RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
  mkdir -p "$RUN_OUTPUT_DIR"
  METRICS_JSON="$RUN_OUTPUT_DIR/results_eval_image_retrieval.json"

  if [[ -f "$METRICS_JSON" ]]; then
    echo "==== Skipping $RUN_NAME: existing metrics found at $METRICS_JSON ===="
    append_metrics_record "$METRICS_JSON" "$RUN_NAME" "$CKPT_PATH"
    continue
  fi

  CMD=(
    "${RUN_CMD[@]}"
    --config-file "$CONFIG_FILE"
    --pretrained-weights "$CKPT_PATH"
    --output-dir "$RUN_OUTPUT_DIR"
    --data_path "$DATA_PATH"
    --dataset "$DATASET"
    --imsize "$IMSIZE"
    --num_workers "$NUM_WORKERS"
  )

  [[ "$GATHER_ON_CPU" -eq 1 ]] && CMD+=(--gather_on_cpu)

  echo "==== Evaluating $RUN_NAME ===="
  if [[ -n "$GPU" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
  else
    "${CMD[@]}"
  fi

  [[ -f "$METRICS_JSON" ]] || { echo "Missing metrics file: $METRICS_JSON" >&2; exit 1; }

  append_metrics_record "$METRICS_JSON" "$RUN_NAME" "$CKPT_PATH"
done

python - "$TMP_JSONL" "$CSV_OUT" <<'PY'
import csv
import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])

rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
all_keys = set().union(*(row.keys() for row in rows)) if rows else set()
fieldnames = ["run_name", "checkpoint_path", "dataset"] + sorted(
    k for k in all_keys if k not in {"run_name", "checkpoint_path", "dataset"}
)

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
PY

echo "Saved aggregated CSV to: $CSV_OUT"
