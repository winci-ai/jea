#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/local_detail_decodability/run_all.sh \
    --config-file /path/to/config.yaml \
    --eval-root /path/to/eval \
    --output-root /path/to/output_dir \
    --csv-out /path/to/summary.csv \
    [--gpu 0] \
    [--port 29501] \
    [--checkpoint-steps 312749,375299] \
    [--dataset ImageNet:split=TRAIN] \
    [--train-images 10000] \
    [--val-images 10000] \
    [--subset-seed 0] \
    [--run-seeds 0] \
    [--batch-size 64] \
    [--num-workers 5] \
    [--patches-per-image 0] \
    [--layer final] \
    [--layers final,11,8] \
    [--targets rgb_mean,rgb_hist,hog,dct_energy,lbp_hist,raw_pca] \
    [--ridge-alpha 1e-3] \
    [--save-probe] \
    [--save-visualizations] \
    [--visualization-samples 8] \
    [--visualization-targets rgb_mean,hog,dct_energy,raw_pca] \
    [--visualization-dir /path/to/visualizations] \
    [-- ...extra local_detail_decodability args...]
EOF
}

CONFIG_FILE=""
EVAL_ROOT=""
OUTPUT_ROOT=""
CSV_OUT=""
GPU=""
PORT=""
CHECKPOINT_STEPS=""
DATASET="ImageNet:split=TRAIN"
TRAIN_IMAGES=10000
VAL_IMAGES=10000
SUBSET_SEED=0
RUN_SEEDS="0"
BATCH_SIZE=64
NUM_WORKERS=5
PATCHES_PER_IMAGE=0
LAYERS="final"
TARGETS="rgb_mean,rgb_hist,hog,dct_energy,lbp_hist,raw_pca"
RIDGE_ALPHA="1e-3"
SAVE_PROBE=0
SAVE_VISUALIZATIONS=0
VISUALIZATION_SAMPLES=8
VISUALIZATION_TARGETS="rgb_mean,hog,dct_energy,raw_pca"
VISUALIZATION_DIR=""
EXTRA_ARGS=()

layer_output_name() {
  if [[ "$1" == "final" ]]; then
    echo "layer_final"
  else
    echo "layer_$1"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --csv-out) CSV_OUT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --checkpoint-steps) CHECKPOINT_STEPS="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --train-images) TRAIN_IMAGES="$2"; shift 2 ;;
    --val-images) VAL_IMAGES="$2"; shift 2 ;;
    --subset-seed) SUBSET_SEED="$2"; shift 2 ;;
    --run-seeds) RUN_SEEDS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --patches-per-image) PATCHES_PER_IMAGE="$2"; shift 2 ;;
    --layer) LAYERS="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --targets) TARGETS="$2"; shift 2 ;;
    --ridge-alpha) RIDGE_ALPHA="$2"; shift 2 ;;
    --save-probe) SAVE_PROBE=1; shift ;;
    --save-visualizations) SAVE_VISUALIZATIONS=1; shift ;;
    --visualization-samples) VISUALIZATION_SAMPLES="$2"; shift 2 ;;
    --visualization-targets) VISUALIZATION_TARGETS="$2"; shift 2 ;;
    --visualization-dir) VISUALIZATION_DIR="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$CONFIG_FILE" && -n "$EVAL_ROOT" && -n "$OUTPUT_ROOT" && -n "$CSV_OUT" ]] || { echo "Missing required arguments." >&2; usage; exit 1; }
[[ -f "$CONFIG_FILE" ]] || { echo "Config file not found: $CONFIG_FILE" >&2; exit 1; }
[[ -d "$EVAL_ROOT" ]] || { echo "Eval root not found: $EVAL_ROOT" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$(dirname "$CSV_OUT")"
IFS=',' read -r -a LAYER_ARRAY <<< "$LAYERS"

if [[ -n "${TORCHRUN_NPROC_PER_NODE:-}" ]]; then
  RUN_CMD=(torchrun --nproc_per_node="$TORCHRUN_NPROC_PER_NODE")
  [[ -n "$PORT" ]] && RUN_CMD+=(--master_port="$PORT")
  RUN_CMD+=(-m jea.diagnostics.local_detail_decodability)
else
  RUN_CMD=(python -m jea.diagnostics.local_detail_decodability)
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
flat = {
    "run_name": run_name,
    "checkpoint_path": ckpt_path,
}
for key, value in record.items():
    if key in ("metrics", "run_metrics"):
        continue
    if key == "target_slices":
        flat["target_slices_json"] = json.dumps(value, sort_keys=True)
    elif isinstance(value, list):
        flat[key] = ",".join(str(v) for v in value)
    elif isinstance(value, dict):
        flat[f"{key}_json"] = json.dumps(value, sort_keys=True)
    else:
        flat[key] = value

for key, value in record.get("metrics", {}).items():
    flat[key.replace("/", "_")] = value

with jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(flat, ensure_ascii=False) + "\n")
PY
}

metrics_has_requested_runs() {
  local metrics_json="$1"
  local run_seeds="$2"
  local patches_per_image="$3"

  python - "$metrics_json" "$run_seeds" "$patches_per_image" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
requested = [int(item.strip()) for item in sys.argv[2].split(",") if item.strip()]
patches_per_image = int(sys.argv[3])
try:
    record = json.loads(metrics_path.read_text(encoding="utf-8"))
    actual = [int(value) for value in record.get("run_seeds", [])]
except Exception:
    sys.exit(1)

if actual != requested:
    sys.exit(1)
if int(record.get("patches_per_image", -1)) != patches_per_image:
    sys.exit(1)
if record.get("subset_seed_mode") != "per_run":
    sys.exit(1)
metrics = record.get("metrics", {})
if len(requested) > 1 and not any(str(key).endswith("/run_std") for key in metrics):
    sys.exit(1)
PY
}

mapfile -t TRAIN_DIRS < <(find "$EVAL_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'training_*' | sort -V)
[[ ${#TRAIN_DIRS[@]} -gt 0 ]] || { echo "No training_* directories found under $EVAL_ROOT" >&2; exit 1; }
if [[ -n "$CHECKPOINT_STEPS" ]]; then
  IFS=',' read -r -a CHECKPOINT_STEP_ARRAY <<< "$CHECKPOINT_STEPS"
  FILTERED_TRAIN_DIRS=()
  for STEP in "${CHECKPOINT_STEP_ARRAY[@]}"; do
    STEP="$(echo "$STEP" | xargs)"
    [[ -n "$STEP" ]] || continue
    TRAIN_DIR="$EVAL_ROOT/training_$STEP"
    [[ -d "$TRAIN_DIR" ]] || { echo "Requested checkpoint directory not found: $TRAIN_DIR" >&2; exit 1; }
    FILTERED_TRAIN_DIRS+=("$TRAIN_DIR")
  done
  TRAIN_DIRS=("${FILTERED_TRAIN_DIRS[@]}")
fi

for TRAIN_DIR in "${TRAIN_DIRS[@]}"; do
  CKPT_PATH="$TRAIN_DIR/teacher_checkpoint.pth"
  [[ -f "$CKPT_PATH" ]] || { echo "Skipping $(basename "$TRAIN_DIR"): missing teacher_checkpoint.pth" >&2; continue; }

  RUN_NAME="$(basename "$TRAIN_DIR")"
  RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
  mkdir -p "$RUN_OUTPUT_DIR"

  for LAYER in "${LAYER_ARRAY[@]}"; do
    LAYER="$(echo "$LAYER" | xargs)"
    [[ -n "$LAYER" ]] || continue

    RUN_LAYER_OUTPUT_DIR="$RUN_OUTPUT_DIR/$(layer_output_name "$LAYER")"
    METRICS_JSON="$RUN_LAYER_OUTPUT_DIR/results_local_detail_decodability.json"
    if [[ -n "$VISUALIZATION_DIR" ]]; then
      VIS_MANIFEST="$VISUALIZATION_DIR/manifest_pdf.csv"
    else
      VIS_MANIFEST="$RUN_LAYER_OUTPUT_DIR/visualizations/local_detail/manifest_pdf.csv"
    fi

    if [[ -f "$METRICS_JSON" ]]; then
      METRICS_READY=0
      if metrics_has_requested_runs "$METRICS_JSON" "$RUN_SEEDS" "$PATCHES_PER_IMAGE"; then
        METRICS_READY=1
      fi
      if [[ "$METRICS_READY" -eq 0 ]]; then
        echo "==== Re-running $RUN_NAME layer=$LAYER: existing metrics do not match requested run settings ===="
      elif [[ "$SAVE_VISUALIZATIONS" -eq 1 && ! -f "$VIS_MANIFEST" ]]; then
        echo "==== Re-running $RUN_NAME layer=$LAYER: metrics exist but visualizations are missing ===="
      else
        echo "==== Skipping $RUN_NAME layer=$LAYER: existing metrics found at $METRICS_JSON ===="
        append_metrics_record "$METRICS_JSON" "$RUN_NAME" "$CKPT_PATH"
        continue
      fi
    fi

    echo "==== Evaluating $RUN_NAME layer=$LAYER ===="

    CMD=(
      "${RUN_CMD[@]}"
      --config-file "$CONFIG_FILE"
      --pretrained-weights "$CKPT_PATH"
      --output-dir "$RUN_OUTPUT_DIR"
      --dataset "$DATASET"
      --train-images "$TRAIN_IMAGES"
      --val-images "$VAL_IMAGES"
      --subset-seed "$SUBSET_SEED"
      --run-seeds "$RUN_SEEDS"
      --batch-size "$BATCH_SIZE"
      --num-workers "$NUM_WORKERS"
      --patches-per-image "$PATCHES_PER_IMAGE"
      --layer "$LAYER"
      --targets "$TARGETS"
      --ridge-alpha "$RIDGE_ALPHA"
      "${EXTRA_ARGS[@]}"
    )
    [[ "$SAVE_PROBE" -eq 1 ]] && CMD+=(--save-probe)
    if [[ "$SAVE_VISUALIZATIONS" -eq 1 ]]; then
      CMD+=(
        --save-visualizations
        --visualization-samples "$VISUALIZATION_SAMPLES"
        --visualization-targets "$VISUALIZATION_TARGETS"
      )
      [[ -n "$VISUALIZATION_DIR" ]] && CMD+=(--visualization-dir "$VISUALIZATION_DIR")
    fi

    if [[ -n "$GPU" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
    else
      "${CMD[@]}"
    fi

    [[ -f "$METRICS_JSON" ]] || { echo "Missing metrics file after evaluation: $METRICS_JSON" >&2; exit 1; }

    append_metrics_record "$METRICS_JSON" "$RUN_NAME" "$CKPT_PATH"
  done
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
front = [
    "run_name",
    "checkpoint_path",
    "run_seeds",
    "num_runs",
    "layer",
    "targets",
    "train_images",
    "val_images",
    "patches_per_image",
    "all_r2_variance_weighted",
]
fieldnames = front + sorted(k for k in all_keys if k not in set(front))

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
PY

echo "Saved aggregated CSV to: $CSV_OUT"
