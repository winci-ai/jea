#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/knn/run_all.sh \
    --config-file /path/to/config.yaml \
    --eval-root /path/to/eval \
    --output-root /path/to/output_dir \
    --csv-out /path/to/summary.csv \
    [--gpu 0] \
    [--port 29501] \
    [--train-dataset ImageNet:split=TRAIN] \
    [--val-dataset ImageNet:split=VAL] \
    [--batch-size 256] \
    [--avg-pool] \
    [--gather-on-cpu] \
    [-- ...extra knn args...]
EOF
}

CONFIG_FILE=""
EVAL_ROOT=""
OUTPUT_ROOT=""
CSV_OUT=""
GPU=""
PORT=""
TRAIN_DATASET="ImageNet:split=TRAIN"
VAL_DATASET="ImageNet:split=VAL"
BATCH_SIZE=256
AVG_POOL=0
GATHER_ON_CPU=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --csv-out) CSV_OUT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --train-dataset) TRAIN_DATASET="$2"; shift 2 ;;
    --val-dataset) VAL_DATASET="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --avg-pool) AVG_POOL=1; shift ;;
    --gather-on-cpu) GATHER_ON_CPU=1; shift ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$CONFIG_FILE" && -n "$EVAL_ROOT" && -n "$OUTPUT_ROOT" && -n "$CSV_OUT" ]] || { echo "Missing required arguments." >&2; usage; exit 1; }
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$(dirname "$CSV_OUT")"

if [[ -n "${TORCHRUN_NPROC_PER_NODE:-}" ]]; then
  RUN_CMD=(torchrun --nproc_per_node="$TORCHRUN_NPROC_PER_NODE")
  [[ -n "$PORT" ]] && RUN_CMD+=(--master_port="$PORT")
  RUN_CMD+=(-m jea.eval.knn)
else
  RUN_CMD=(python -m jea.eval.knn)
fi

TMP_JSONL="$(mktemp)"
trap 'rm -f "$TMP_JSONL"' EXIT

append_metrics_record() {
  local metrics_json="$1"
  local run_name="$2"
  local ckpt_path="$3"

  python - "$metrics_json" "$run_name" "$ckpt_path" "$TMP_JSONL" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
run_name = sys.argv[2]
ckpt_path = sys.argv[3]
jsonl_path = Path(sys.argv[4])

record = json.loads(metrics_path.read_text())
record["run_name"] = run_name
record["checkpoint_path"] = ckpt_path

with jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
}

mapfile -t TRAIN_DIRS < <(find "$EVAL_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'training_*' | sort -V)

for TRAIN_DIR in "${TRAIN_DIRS[@]}"; do
  CKPT_PATH="$TRAIN_DIR/teacher_checkpoint.pth"
  [[ -f "$CKPT_PATH" ]] || continue

  RUN_NAME="$(basename "$TRAIN_DIR")"
  RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
  mkdir -p "$RUN_OUTPUT_DIR"
  METRICS_JSON="$RUN_OUTPUT_DIR/results_eval_knn_summary.json"

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
    --train-dataset "$TRAIN_DATASET"
    --val-dataset "$VAL_DATASET"
    --batch-size "$BATCH_SIZE"
    "${EXTRA_ARGS[@]}"
  )

  [[ "$AVG_POOL" -eq 1 ]] && CMD+=(--avg-pool)
  [[ "$GATHER_ON_CPU" -eq 1 ]] && CMD+=(--gather-on-cpu)

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
fieldnames = ["run_name", "checkpoint_path"] + sorted(k for k in all_keys if k not in {"run_name", "checkpoint_path"})

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
PY

echo "Saved aggregated CSV to: $CSV_OUT"
