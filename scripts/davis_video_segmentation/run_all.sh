#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/davis_video_segmentation/run_all.sh \
    --config-file /path/to/config.yaml \
    --eval-root /path/to/eval \
    --output-root /path/to/output_dir \
    --csv-out /path/to/summary.csv \
    --data-path /path/to/davis \
    [--gpu 0] \
    [--port 29501] \
    [--task semi-supervised] \
    [--set val] \
    [--n-last-frames 7] \
    [--size-mask-neighborhood 12] \
    [--topk 5] \
    [--bs 6] \
    [--layers "3 6 9 11"]
EOF
}

CONFIG_FILE=""
EVAL_ROOT=""
OUTPUT_ROOT=""
CSV_OUT=""
DATA_PATH=""
GPU=""
PORT=""
TASK="semi-supervised"
SET_NAME="val"
N_LAST_FRAMES=7
SIZE_MASK_NEIGHBORHOOD=12
TOPK=5
BS=6
LAYERS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --csv-out) CSV_OUT="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --set) SET_NAME="$2"; shift 2 ;;
    --n-last-frames) N_LAST_FRAMES="$2"; shift 2 ;;
    --size-mask-neighborhood) SIZE_MASK_NEIGHBORHOOD="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --bs) BS="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$CONFIG_FILE" && -n "$EVAL_ROOT" && -n "$OUTPUT_ROOT" && -n "$CSV_OUT" && -n "$DATA_PATH" ]] || { echo "Missing required arguments." >&2; usage; exit 1; }
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$(dirname "$CSV_OUT")"

if [[ -n "${TORCHRUN_NPROC_PER_NODE:-}" ]]; then
  SEG_CMD=(torchrun --nproc_per_node="$TORCHRUN_NPROC_PER_NODE")
  [[ -n "$PORT" ]] && SEG_CMD+=(--master_port="$PORT")
  SEG_CMD+=(jea/eval/davis2017_evaluation/eval_video_segmentation.py)
else
  SEG_CMD=(python jea/eval/davis2017_evaluation/eval_video_segmentation.py)
fi

TMP_JSONL="$(mktemp)"
trap 'rm -f "$TMP_JSONL"' EXIT

append_davis_record() {
  local global_csv="$1"
  local run_name="$2"
  local ckpt_path="$3"

  python - "$global_csv" "$run_name" "$ckpt_path" "$TASK" "$SET_NAME" "$TMP_JSONL" <<'PY'
import csv
import json
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
run_name = sys.argv[2]
ckpt_path = sys.argv[3]
task = sys.argv[4]
set_name = sys.argv[5]
jsonl_path = Path(sys.argv[6])

with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 1:
    raise RuntimeError(f"Expected exactly one row in {csv_path}, got {len(rows)}")

record = rows[0]
record["run_name"] = run_name
record["checkpoint_path"] = ckpt_path
record["task"] = task
record["set"] = set_name

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
  GLOBAL_CSV="$RUN_OUTPUT_DIR/global_results-${SET_NAME}.csv"

  if [[ -f "$GLOBAL_CSV" ]]; then
    echo "==== Skipping $RUN_NAME: existing DAVIS results found at $GLOBAL_CSV ===="
    append_davis_record "$GLOBAL_CSV" "$RUN_NAME" "$CKPT_PATH"
    continue
  fi

  CMD=(
    "${SEG_CMD[@]}"
    --config-file "$CONFIG_FILE"
    --pretrained-weights "$CKPT_PATH"
    --output-dir "$RUN_OUTPUT_DIR"
    --data_path "$DATA_PATH"
    --n_last_frames "$N_LAST_FRAMES"
    --size_mask_neighborhood "$SIZE_MASK_NEIGHBORHOOD"
    --topk "$TOPK"
    --bs "$BS"
  )
  # shellcheck disable=SC2206
  [[ -n "$LAYERS" ]] && CMD+=(--layers $LAYERS)

  echo "==== Evaluating $RUN_NAME ===="
  if [[ -n "$GPU" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
  else
    "${CMD[@]}"
  fi

  python jea/eval/davis2017_evaluation/evaluation_method.py \
    --task "$TASK" \
    --set "$SET_NAME" \
    --davis_path "$DATA_PATH" \
    --results_path "$RUN_OUTPUT_DIR"

  [[ -f "$GLOBAL_CSV" ]] || { echo "Missing DAVIS global results file: $GLOBAL_CSV" >&2; exit 1; }

  append_davis_record "$GLOBAL_CSV" "$RUN_NAME" "$CKPT_PATH"
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
fieldnames = ["run_name", "checkpoint_path", "task", "set"] + sorted(
    k for k in all_keys if k not in {"run_name", "checkpoint_path", "task", "set"}
)

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
PY

echo "Saved aggregated CSV to: $CSV_OUT"
