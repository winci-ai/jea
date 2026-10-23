#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/optical_flow/run_all.sh \
    --train-config /path/to/config.py \
    --eval-root /path/to/eval \
    --output-root /path/to/output_dir \
    --csv-out /path/to/summary.csv \
    [--gpu 0] \
    [--port 29501] \
    [--train-batch-size 2] \
    [--train-num-workers 6] \
    [--lr 1e-4] \
    [--eval-metrics EPE] \
    [-- ...extra cfg-options...]
EOF
}

TRAIN_CONFIG=""
EVAL_ROOT=""
OUTPUT_ROOT=""
CSV_OUT=""
GPU=""
PORT="29501"
TRAIN_BATCH_SIZE=4
TRAIN_NUM_WORKERS=6
LR="1e-4"
EVAL_METRICS="EPE"
EXTRA_CFG_OPTIONS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-config) TRAIN_CONFIG="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --csv-out) CSV_OUT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --train-num-workers) TRAIN_NUM_WORKERS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --eval-metrics) EVAL_METRICS="$2"; shift 2 ;;
    --) shift; EXTRA_CFG_OPTIONS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$TRAIN_CONFIG" && -n "$EVAL_ROOT" && -n "$OUTPUT_ROOT" && -n "$CSV_OUT" ]] || { echo "Missing required arguments." >&2; usage; exit 1; }
[[ -f "$TRAIN_CONFIG" ]] || { echo "Train config not found: $TRAIN_CONFIG" >&2; exit 1; }
[[ -d "$EVAL_ROOT" ]] || { echo "Eval root not found: $EVAL_ROOT" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$(dirname "$CSV_OUT")"

if [[ -n "${TORCHRUN_NPROC_PER_NODE:-}" ]]; then
  TRAIN_CMD=(torchrun --nproc_per_node="$TORCHRUN_NPROC_PER_NODE" --master_port="$PORT" jea/eval/optical_flow/train.py)
  TEST_PORT=$((PORT + 1))
  TEST_CMD=(torchrun --nproc_per_node="$TORCHRUN_NPROC_PER_NODE" --master_port="$TEST_PORT" jea/eval/optical_flow/test.py)
  LAUNCHER="pytorch"
else
  TRAIN_CMD=(python jea/eval/optical_flow/train.py)
  TEST_CMD=(python jea/eval/optical_flow/test.py)
  LAUNCHER="none"
fi

IFS=',' read -r -a EVAL_METRICS_ARR <<< "$EVAL_METRICS"

TMP_JSONL="$(mktemp)"
trap 'rm -f "$TMP_JSONL"' EXIT

append_metrics_record() {
  local metrics_json="$1"
  local run_name="$2"
  local pretrain_ckpt="$3"
  local eval_ckpt="$4"

  python - "$metrics_json" "$run_name" "$pretrain_ckpt" "$eval_ckpt" "$TMP_JSONL" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
run_name = sys.argv[2]
pretrain_ckpt = sys.argv[3]
eval_ckpt = sys.argv[4]
jsonl_path = Path(sys.argv[5])

record = json.loads(metrics_path.read_text(encoding='utf-8'))
record["run_name"] = run_name
record["pretrain_checkpoint"] = pretrain_ckpt
record["eval_checkpoint"] = eval_ckpt

with jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
}

find_existing_eval_checkpoint() {
  local work_dir="$1"

  python - "$work_dir" <<'PY'
import sys
from pathlib import Path

work_dir = Path(sys.argv[1])
latest = work_dir / 'latest.pth'
if latest.is_file():
    print(latest)
    raise SystemExit(0)

meta_last = work_dir / 'last_checkpoint'
if meta_last.is_file():
    content = meta_last.read_text(encoding='utf-8').strip()
    if content:
        ckpt_path = Path(content)
        if not ckpt_path.is_absolute():
            ckpt_path = work_dir / ckpt_path
        print(ckpt_path)
        raise SystemExit(0)

candidates = list(work_dir.glob('epoch_*.pth')) + list(work_dir.glob('iter_*.pth'))
if candidates:
    def score(path: Path):
        stem = path.stem
        try:
            return int(stem.split('_')[-1])
        except ValueError:
            return -1
    candidates.sort(key=lambda p: (score(p), p.stat().st_mtime))
    print(candidates[-1])
    raise SystemExit(0)

print("")
PY
}

mapfile -t TRAIN_DIRS < <(find "$EVAL_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'training_*' | sort -V)
[[ ${#TRAIN_DIRS[@]} -gt 0 ]] || { echo "No training_* directories found under $EVAL_ROOT" >&2; exit 1; }

for TRAIN_DIR in "${TRAIN_DIRS[@]}"; do
  PRETRAIN_CKPT="$TRAIN_DIR/teacher_checkpoint.pth"
  [[ -f "$PRETRAIN_CKPT" ]] || { echo "Skipping $(basename "$TRAIN_DIR"): missing teacher_checkpoint.pth" >&2; continue; }

  RUN_NAME="$(basename "$TRAIN_DIR")"
  RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
  TRAIN_WORK_DIR="$RUN_OUTPUT_DIR/train"
  TEST_WORK_DIR="$RUN_OUTPUT_DIR/test"
  mkdir -p "$TRAIN_WORK_DIR" "$TEST_WORK_DIR"
  METRICS_JSON="$TEST_WORK_DIR/results_eval_optical_flow.json"

  if [[ -f "$METRICS_JSON" ]]; then
    echo "==== Skipping $RUN_NAME: existing metrics found at $METRICS_JSON ===="
    EXISTING_EVAL_CKPT="$(find_existing_eval_checkpoint "$TRAIN_WORK_DIR")"
    append_metrics_record "$METRICS_JSON" "$RUN_NAME" "$PRETRAIN_CKPT" "$EXISTING_EVAL_CKPT"
    continue
  fi

  COMMON_CFG_OPTIONS=(
    "model.encoder.init_cfg.checkpoint=$PRETRAIN_CKPT"
    "data.train_dataloader.samples_per_gpu=$TRAIN_BATCH_SIZE"
    "data.train_dataloader.workers_per_gpu=$TRAIN_NUM_WORKERS"
    "data.val_dataloader.workers_per_gpu=$TRAIN_NUM_WORKERS"
    "data.test_dataloader.workers_per_gpu=$TRAIN_NUM_WORKERS"
    "optim_wrapper.optimizer.lr=$LR"
  )
  if [[ ${#EXTRA_CFG_OPTIONS[@]} -gt 0 ]]; then
    COMMON_CFG_OPTIONS+=("${EXTRA_CFG_OPTIONS[@]}")
  fi

  TRAIN_ARGS=(
    "${TRAIN_CMD[@]}"
    "$TRAIN_CONFIG"
    --launcher "$LAUNCHER"
    --work-dir "$TRAIN_WORK_DIR"
    --cfg-options
    "${COMMON_CFG_OPTIONS[@]}"
  )

  echo "==== Training $RUN_NAME ===="
  if [[ -n "$GPU" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "${TRAIN_ARGS[@]}"
  else
    "${TRAIN_ARGS[@]}"
  fi

  EVAL_CKPT="$(python - "$TRAIN_WORK_DIR" <<'PY'
import sys
from pathlib import Path

work_dir = Path(sys.argv[1])
last_checkpoint = work_dir / 'latest.pth'
if last_checkpoint.is_file():
    print(last_checkpoint)
    raise SystemExit(0)

meta_last = work_dir / 'last_checkpoint'
if meta_last.is_file():
    content = meta_last.read_text(encoding='utf-8').strip()
    if content:
        ckpt_path = Path(content)
        if not ckpt_path.is_absolute():
            ckpt_path = work_dir / ckpt_path
        print(ckpt_path)
        raise SystemExit(0)

candidates = list(work_dir.glob('epoch_*.pth')) + list(work_dir.glob('iter_*.pth'))
if candidates:
    def score(path: Path):
        stem = path.stem
        try:
            return int(stem.split('_')[-1])
        except ValueError:
            return -1
    candidates.sort(key=lambda p: (score(p), p.stat().st_mtime))
    print(candidates[-1])
    raise SystemExit(0)

raise SystemExit('Could not locate a checkpoint to evaluate.')
PY
)"

  TEST_ARGS=(
    "${TEST_CMD[@]}"
    "$TRAIN_CONFIG"
    "$EVAL_CKPT"
    --launcher "$LAUNCHER"
    --work-dir "$TEST_WORK_DIR"
    --eval
    "${EVAL_METRICS_ARR[@]}"
    --cfg-options
    "${COMMON_CFG_OPTIONS[@]}"
  )

  echo "==== Testing $RUN_NAME ===="
  if [[ -n "$GPU" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "${TEST_ARGS[@]}"
  else
    "${TEST_ARGS[@]}"
  fi

  [[ -f "$METRICS_JSON" ]] || { echo "Missing metrics file: $METRICS_JSON" >&2; exit 1; }

  append_metrics_record "$METRICS_JSON" "$RUN_NAME" "$PRETRAIN_CKPT" "$EVAL_CKPT"
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
fieldnames = ["run_name", "pretrain_checkpoint", "eval_checkpoint"] + sorted(
    k for k in all_keys if k not in {"run_name", "pretrain_checkpoint", "eval_checkpoint"}
)

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
PY

echo "Saved aggregated CSV to: $CSV_OUT"
