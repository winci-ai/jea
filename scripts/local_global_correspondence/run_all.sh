#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/local_global_correspondence/run_all.sh \
    --config-file /path/to/config.yaml \
    --eval-root /path/to/eval \
    --output-root /path/to/output_dir \
    --csv-out /path/to/summary.csv \
    --data-path /path/to/copydays \
    [--eval-module jea.diagnostics.local_global_correspondence] \
    [--gpu 0,1,2,3] \
    [--port 29501] \
    [--checkpoint-steps 312749,375299] \
    [--image-dir /path/to/source_images] \
    [--max-images 157] \
    [--crop-ratios 0.1,0.2,0.3,0.5] \
    [--crops-per-image 4] \
    [--subset-seed 0] \
    [--run-seeds 0] \
    [--crop-seed 0] \
    [--aspect-jitter 0.25] \
    [--selected-crop-specs-json /path/to/fixed_crop_specs.json] \
    [--write-crop-specs-json /path/to/crop_specs.json] \
    [--imsize 224] \
    [--batch-size 16] \
    [--num-workers 4] \
    [--layers final] \
    [--topk 5] \
    [--ransac-iters 128] \
    [--geometric-tolerance-patches 1.5] \
    [--save-visualizations] \
    [--visualization-samples 16] \
    [--visualization-images 16] \
    [--visualization-dir /path/to/visualizations] \
    [--visualization-max-match-lines 48] \
    [--visualization-max-per-image 16] \
    [--refresh-visualizations] \
    [-- ...extra local_global_correspondence args...]
EOF
}

CONFIG_FILE=""
EVAL_ROOT=""
OUTPUT_ROOT=""
CSV_OUT=""
DATA_PATH=""
EVAL_MODULE="jea.diagnostics.local_global_correspondence"
IMAGE_DIR=""
GPU=""
PORT=""
CHECKPOINT_STEPS=""
MAX_IMAGES=157
CROP_RATIOS="0.1,0.2,0.3,0.5"
CROPS_PER_IMAGE=4
SUBSET_SEED=0
RUN_SEEDS="0"
CROP_SEED=0
ASPECT_JITTER="0.25"
SELECTED_CROP_SPECS_JSON=""
SELECTED_CROP_SPECS_JSON_EXPLICIT=0
WRITE_CROP_SPECS_JSON=""
IMSIZE=224
BATCH_SIZE=16
NUM_WORKERS=4
LAYERS="final"
TOPK=5
RANSAC_ITERS=128
GEOMETRIC_TOLERANCE_PATCHES="1.5"
SAVE_VISUALIZATIONS=0
VISUALIZATION_SAMPLES=16
VISUALIZATION_IMAGES=0
VISUALIZATION_DIR=""
VISUALIZATION_MAX_MATCH_LINES=48
VISUALIZATION_MAX_PER_IMAGE=16
REFRESH_VISUALIZATIONS=0
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
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --eval-module) EVAL_MODULE="$2"; shift 2 ;;
    --image-dir) IMAGE_DIR="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --checkpoint-steps) CHECKPOINT_STEPS="$2"; shift 2 ;;
    --max-images) MAX_IMAGES="$2"; shift 2 ;;
    --crop-ratios) CROP_RATIOS="$2"; shift 2 ;;
    --crops-per-image) CROPS_PER_IMAGE="$2"; shift 2 ;;
    --subset-seed) SUBSET_SEED="$2"; shift 2 ;;
    --run-seeds) RUN_SEEDS="$2"; shift 2 ;;
    --crop-seed) CROP_SEED="$2"; shift 2 ;;
    --aspect-jitter) ASPECT_JITTER="$2"; shift 2 ;;
    --selected-crop-specs-json) SELECTED_CROP_SPECS_JSON="$2"; SELECTED_CROP_SPECS_JSON_EXPLICIT=1; shift 2 ;;
    --write-crop-specs-json) WRITE_CROP_SPECS_JSON="$2"; shift 2 ;;
    --imsize) IMSIZE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --layer) LAYERS="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --ransac-iters) RANSAC_ITERS="$2"; shift 2 ;;
    --geometric-tolerance-patches) GEOMETRIC_TOLERANCE_PATCHES="$2"; shift 2 ;;
    --save-visualizations) SAVE_VISUALIZATIONS=1; shift ;;
    --visualization-samples) VISUALIZATION_SAMPLES="$2"; shift 2 ;;
    --visualization-images) VISUALIZATION_IMAGES="$2"; shift 2 ;;
    --visualization-dir) VISUALIZATION_DIR="$2"; shift 2 ;;
    --visualization-max-match-lines) VISUALIZATION_MAX_MATCH_LINES="$2"; shift 2 ;;
    --visualization-max-per-image) VISUALIZATION_MAX_PER_IMAGE="$2"; shift 2 ;;
    --refresh-visualizations) REFRESH_VISUALIZATIONS=1; shift ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$CONFIG_FILE" && -n "$EVAL_ROOT" && -n "$OUTPUT_ROOT" && -n "$CSV_OUT" && -n "$DATA_PATH" ]] || { echo "Missing required arguments." >&2; usage; exit 1; }
[[ -f "$CONFIG_FILE" ]] || { echo "Config file not found: $CONFIG_FILE" >&2; exit 1; }
[[ -d "$EVAL_ROOT" ]] || { echo "Eval root not found: $EVAL_ROOT" >&2; exit 1; }
[[ -d "$DATA_PATH" ]] || { echo "Data path not found: $DATA_PATH" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$(dirname "$CSV_OUT")"
IFS=',' read -r -a LAYER_ARRAY <<< "$LAYERS"

infer_nproc_from_gpu() {
  local gpu_list="${1// /}"
  if [[ -n "$gpu_list" && "$gpu_list" == *,* ]]; then
    awk -F',' '{print NF}' <<< "$gpu_list"
  fi
}

count_csv_items() {
  awk -F',' '{
    count = 0
    for (i = 1; i <= NF; i++) {
      gsub(/^[ \t]+|[ \t]+$/, "", $i)
      if ($i != "") count++
    }
    print count
  }' <<< "$1"
}

RUN_NPROC="${TORCHRUN_NPROC_PER_NODE:-}"
if [[ -z "$RUN_NPROC" ]]; then
  RUN_NPROC="$(infer_nproc_from_gpu "$GPU")"
fi
if [[ "$RUN_NPROC" =~ ^[0-9]+$ && "$RUN_NPROC" -gt 1 ]]; then
  EXPECTED_WORLD_SIZE="$RUN_NPROC"
  RUN_CMD=(torchrun --nproc_per_node="$RUN_NPROC")
  [[ -n "$PORT" ]] && RUN_CMD+=(--master_port="$PORT")
  RUN_CMD+=(-m "$EVAL_MODULE")
else
  EXPECTED_WORLD_SIZE=1
  RUN_CMD=(python -m "$EVAL_MODULE")
fi

RUN_SEED_COUNT="$(count_csv_items "$RUN_SEEDS")"
SAFE_CROP_RATIOS="${CROP_RATIOS//,/p}"
SAFE_CROP_RATIOS="${SAFE_CROP_RATIOS//./d}"
SAFE_ASPECT_JITTER="${ASPECT_JITTER//./d}"
CROP_SPEC_TAG="seed${SUBSET_SEED}_crop${CROP_SEED}_max${MAX_IMAGES}_ratios${SAFE_CROP_RATIOS}_n${CROPS_PER_IMAGE}_jitter${SAFE_ASPECT_JITTER}"
CROP_SPEC_DIR="$DATA_PATH/jea_local_global_correspondence/${CROP_SPEC_TAG}"
if [[ -z "$SELECTED_CROP_SPECS_JSON" && -z "$WRITE_CROP_SPECS_JSON" ]]; then
  if [[ "$RUN_SEED_COUNT" -gt 1 ]]; then
    WRITE_CROP_SPECS_JSON="$CROP_SPEC_DIR/crop_specs.json"
  else
    SELECTED_CROP_SPECS_JSON="$CROP_SPEC_DIR/crop_specs.json"
    WRITE_CROP_SPECS_JSON="$SELECTED_CROP_SPECS_JSON"
  fi
fi
if [[ "$SELECTED_CROP_SPECS_JSON_EXPLICIT" -eq 1 && ! -f "$SELECTED_CROP_SPECS_JSON" ]]; then
  echo "Selected crop specs json not found: $SELECTED_CROP_SPECS_JSON" >&2
  exit 1
elif [[ "$SELECTED_CROP_SPECS_JSON_EXPLICIT" -eq 1 ]]; then
  echo "Using explicitly fixed crop specs: $SELECTED_CROP_SPECS_JSON"
elif [[ "$RUN_SEED_COUNT" -gt 1 ]]; then
  echo "Using cached per-run crop specs for run_seeds=$RUN_SEEDS; missing specs will be created under base: $WRITE_CROP_SPECS_JSON"
elif [[ -f "$SELECTED_CROP_SPECS_JSON" ]]; then
  echo "Using existing crop specs: $SELECTED_CROP_SPECS_JSON"
else
  echo "No crop specs found; the first run will create: $WRITE_CROP_SPECS_JSON"
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

record = json.loads(metrics_path.read_text(encoding="utf-8"))
flat = {
    "run_name": run_name,
    "checkpoint_path": ckpt_path,
}
for key, value in record.items():
    if key in ("metrics", "run_metrics"):
        continue
    if isinstance(value, list):
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
  local expected_world_size="$3"
  local subset_seed="$4"
  local crop_seed="$5"

  python - "$metrics_json" "$run_seeds" "$expected_world_size" "$subset_seed" "$crop_seed" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
requested = [int(item.strip()) for item in sys.argv[2].split(",") if item.strip()]
expected_world_size = int(sys.argv[3])
subset_seed = int(sys.argv[4])
crop_seed = int(sys.argv[5])
expected_subset_seeds = [subset_seed + seed for seed in requested]
expected_crop_seeds = [crop_seed + seed for seed in requested]
try:
    record = json.loads(metrics_path.read_text(encoding="utf-8"))
    actual = [int(value) for value in record.get("run_seeds", [])]
    actual_subset_seeds = [int(value) for value in record.get("effective_subset_seeds", [])]
    actual_crop_seeds = [int(value) for value in record.get("effective_crop_seeds", [])]
except Exception:
    sys.exit(1)

if actual != requested:
    sys.exit(1)
if actual_subset_seeds != expected_subset_seeds:
    sys.exit(1)
if actual_crop_seeds != expected_crop_seeds:
    sys.exit(1)
if int(record.get("distributed_world_size", -1)) != expected_world_size:
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
    METRICS_JSON="$RUN_LAYER_OUTPUT_DIR/results_local_global_correspondence.json"
    if [[ -n "$VISUALIZATION_DIR" ]]; then
      VIS_MANIFEST="$VISUALIZATION_DIR/manifest_pdf.csv"
    else
      VIS_MANIFEST="$RUN_LAYER_OUTPUT_DIR/visualizations/local_global_correspondence/manifest_pdf.csv"
    fi

    if [[ -f "$METRICS_JSON" ]]; then
      METRICS_READY=0
      if metrics_has_requested_runs "$METRICS_JSON" "$RUN_SEEDS" "$EXPECTED_WORLD_SIZE" "$SUBSET_SEED" "$CROP_SEED"; then
        METRICS_READY=1
      fi
      if [[ "$METRICS_READY" -eq 0 ]]; then
        echo "==== Re-running $RUN_NAME layer=$LAYER: existing metrics do not match requested run settings ===="
      elif [[ "$SAVE_VISUALIZATIONS" -eq 1 && ( "$REFRESH_VISUALIZATIONS" -eq 1 || ! -f "$VIS_MANIFEST" ) ]]; then
        echo "==== Re-running $RUN_NAME layer=$LAYER: metrics exist; refreshing or completing visualizations ===="
      else
        echo "==== Skipping $RUN_NAME layer=$LAYER: existing metrics found at $METRICS_JSON ===="
        append_metrics_record "$METRICS_JSON" "$RUN_NAME" "$CKPT_PATH"
        continue
      fi
    fi

    CMD=(
      "${RUN_CMD[@]}"
      --config-file "$CONFIG_FILE"
      --pretrained-weights "$CKPT_PATH"
      --output-dir "$RUN_OUTPUT_DIR"
      --data-path "$DATA_PATH"
      --max-images "$MAX_IMAGES"
      --crop-ratios "$CROP_RATIOS"
      --crops-per-image "$CROPS_PER_IMAGE"
      --subset-seed "$SUBSET_SEED"
      --run-seeds "$RUN_SEEDS"
      --crop-seed "$CROP_SEED"
      --aspect-jitter "$ASPECT_JITTER"
      --imsize "$IMSIZE"
      --batch-size "$BATCH_SIZE"
      --num-workers "$NUM_WORKERS"
      --layer "$LAYER"
      --topk "$TOPK"
      --ransac-iters "$RANSAC_ITERS"
      --geometric-tolerance-patches "$GEOMETRIC_TOLERANCE_PATCHES"
      "${EXTRA_ARGS[@]}"
    )
    if [[ -n "$SELECTED_CROP_SPECS_JSON" && -f "$SELECTED_CROP_SPECS_JSON" ]]; then
      CMD+=(--selected-crop-specs-json "$SELECTED_CROP_SPECS_JSON")
    elif [[ -n "$WRITE_CROP_SPECS_JSON" ]]; then
      CMD+=(--write-crop-specs-json "$WRITE_CROP_SPECS_JSON")
    fi
    if [[ "$SAVE_VISUALIZATIONS" -eq 1 ]]; then
      CMD+=(
        --save-visualizations
        --visualization-samples "$VISUALIZATION_SAMPLES"
        --visualization-images "$VISUALIZATION_IMAGES"
        --visualization-max-match-lines "$VISUALIZATION_MAX_MATCH_LINES"
        --visualization-max-per-image "$VISUALIZATION_MAX_PER_IMAGE"
      )
      [[ -n "$VISUALIZATION_DIR" ]] && CMD+=(--visualization-dir "$VISUALIZATION_DIR")
    fi
    [[ -n "$IMAGE_DIR" ]] && CMD+=(--image-dir "$IMAGE_DIR")

    echo "==== Evaluating $RUN_NAME layer=$LAYER local-to-global correspondence ===="
    if [[ -n "$GPU" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
    else
      "${CMD[@]}"
    fi

    [[ -f "$METRICS_JSON" ]] || { echo "Missing metrics file: $METRICS_JSON" >&2; exit 1; }
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
    "num_source_images",
    "num_crop_queries",
    "crop_ratios",
    "all_patch_match_top1_in_gt_rate_mean",
    "all_patch_match_topk_in_gt_rate_mean",
    "all_localization_top1_in_gt_mean",
    "all_localization_topk_hit_mean",
    "all_localization_iou_top_gt_count_mean",
    "all_geometric_ransac_inlier_ratio_mean",
    "all_geometric_ransac_box_iou_mean",
    "all_geometric_success_iou50_mean",
]
fieldnames = front + sorted(k for k in all_keys if k not in set(front))

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
PY

echo "Saved aggregated CSV to: $CSV_OUT"
