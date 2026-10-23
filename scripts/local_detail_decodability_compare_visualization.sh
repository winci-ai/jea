#!/usr/bin/env bash
set -euo pipefail

# Names are optional; model_roots alone determines the model count and order.
model_names="${model_names:-}"
model_roots="${model_roots:-}"
checkpoint_step="${checkpoint_step:-375299}"
model_weights="${model_weights:-}"
model_probe_files="${model_probe_files:-}"
gpu="${gpu:-0}"

dataset="${dataset:-ImageNet:split=TRAIN:root=datasets/ImageNet:extra=datasets/ImageNet}"
train_images="${train_images:-10000}"
val_images="${val_images:-10000}"
subset_seed="${subset_seed:-0}"
batch_size="${batch_size:-64}"
num_workers="${num_workers:-5}"
patches_per_image="${patches_per_image:-0}"
layer="${layer:-final}"
targets="${targets:-rgb_mean,rgb_hist,hog,dct_energy,lbp_hist,raw_pca}"
ridge_alpha="${ridge_alpha:-1e-3}"
visualization_samples="${visualization_samples:-8}"
visualization_targets="${visualization_targets:-rgb_mean,hog,dct_energy,raw_pca}"
visualization_indices_file="${visualization_indices_file:-}"
write_visualization_indices_file="${write_visualization_indices_file:-}"
left_probe_file="${left_probe_file:-}"
right_probe_file="${right_probe_file:-}"
save_probes="${save_probes:-1}"
gap="${gap:-8}"
sample_gap="${sample_gap:-18}"
metric_label_width="${metric_label_width:-120}"
output_dir="${output_dir:-}"

IFS=',' read -r -a MODEL_ROOT_ARRAY <<< "${model_roots}"
[[ "${#MODEL_ROOT_ARRAY[@]}" -gt 0 ]] || {
  echo "Expected at least one directory in model_roots." >&2
  exit 1
}
name_args=()
if [[ -n "${model_names}" ]]; then
  IFS=',' read -r -a MODEL_NAME_ARRAY <<< "${model_names}"
  [[ "${#MODEL_NAME_ARRAY[@]}" -eq "${#MODEL_ROOT_ARRAY[@]}" ]] || {
    echo "Provide one model_names entry per root, or omit model_names to use directory names." >&2
    exit 1
  }
  name_args+=(--model-names "${MODEL_NAME_ARRAY[@]}")
fi
MODEL_WEIGHT_ARRAY=()
MODEL_PROBE_ARRAY=()
if [[ -n "${model_weights}" ]]; then
  IFS=',' read -r -a MODEL_WEIGHT_ARRAY <<< "${model_weights}"
  [[ "${#MODEL_WEIGHT_ARRAY[@]}" -eq "${#MODEL_ROOT_ARRAY[@]}" ]] || {
    echo "Provide one model_weights entry per root." >&2; exit 1;
  }
fi
if [[ -n "${model_probe_files}" ]]; then
  # Preserve empty entries, including the last one, for models needing fitted probes.
  IFS=',' read -r -a MODEL_PROBE_ARRAY <<< "${model_probe_files},_"
  unset 'MODEL_PROBE_ARRAY[${#MODEL_PROBE_ARRAY[@]}-1]'
  [[ "${#MODEL_PROBE_ARRAY[@]}" -eq "${#MODEL_ROOT_ARRAY[@]}" ]] || {
    echo "Provide one model_probe_files entry per root (empty entries fit a new probe)." >&2; exit 1;
  }
fi

resolve_model_root() {
  local model_root="$1"
  local guess
  model_root="$(echo "${model_root}" | xargs)"
  guess="$(basename "${model_root}")"
  if [[ ! -f "${model_root}/config.yaml" && -f "../${guess}/config.yaml" ]]; then
    model_root="../${guess}"
  fi
  if [[ ! -f "${model_root}/config.yaml" && -f "${guess}/config.yaml" ]]; then
    model_root="${guess}"
  fi
  if [[ ! -f "${model_root}/config.yaml" && -f "../${model_root}/config.yaml" ]]; then
    model_root="../${model_root}"
  fi
  echo "${model_root}"
}

for i in "${!MODEL_ROOT_ARRAY[@]}"; do
  root="$(resolve_model_root "${MODEL_ROOT_ARRAY[i]}")"
  [[ -n "${root}" && -f "${root}/config.yaml" ]] || { echo "Config not found for model root: ${root}" >&2; exit 1; }
  MODEL_ROOT_ARRAY[i]="${root}"
  weights="${root}/eval/training_${checkpoint_step}/teacher_checkpoint.pth"
  if [[ -n "${model_weights}" ]]; then
    weights="${MODEL_WEIGHT_ARRAY[i]}"
  elif [[ "${i}" -eq 0 && -n "${left_weights:-}" ]]; then
    weights="${left_weights}"
  elif [[ "${i}" -eq 1 && -n "${right_weights:-}" ]]; then
    weights="${right_weights}"
  fi
  [[ -f "${weights}" ]] || { echo "Checkpoint not found: ${weights}" >&2; exit 1; }
  MODEL_WEIGHT_ARRAY[i]="${weights}"
  probe=""
  if [[ -n "${model_probe_files}" ]]; then
    probe="${MODEL_PROBE_ARRAY[i]}"
  elif [[ "${i}" -eq 0 ]]; then
    probe="${left_probe_file}"
  elif [[ "${i}" -eq 1 ]]; then
    probe="${right_probe_file}"
  fi
  [[ -z "${probe}" || -f "${probe}" ]] || { echo "Probe not found: ${probe}" >&2; exit 1; }
  MODEL_PROBE_ARRAY[i]="${probe}"
done

if [[ -z "${output_dir}" ]]; then
  output_dir="${MODEL_ROOT_ARRAY[0]}/local_detail_decodability_compare_visualization_${#MODEL_ROOT_ARRAY[@]}_models"
fi

probe_args=(--model-probe-files "${MODEL_PROBE_ARRAY[@]}")
if [[ "${save_probes}" == "1" ]]; then
  probe_args+=(--save-probes)
fi

selection_args=()
if [[ -n "${visualization_indices_file}" ]]; then
  [[ -f "${visualization_indices_file}" ]] || {
    echo "Visualization indices file not found: ${visualization_indices_file}" >&2
    exit 1
  }
  selection_args+=(--visualization-indices-file "${visualization_indices_file}")
fi
if [[ -n "${write_visualization_indices_file}" ]]; then
  selection_args+=(--write-visualization-indices-file "${write_visualization_indices_file}")
fi

CMD=(
  python -m jea.diagnostics.local_detail_decodability_compare_visualization
  --model-roots "${MODEL_ROOT_ARRAY[@]}"
  --checkpoint-step "${checkpoint_step}"
  --model-weights "${MODEL_WEIGHT_ARRAY[@]}"
  "${name_args[@]}"
  --output-dir "${output_dir}"
  --dataset "${dataset}"
  --train-images "${train_images}"
  --val-images "${val_images}"
  --subset-seed "${subset_seed}"
  --batch-size "${batch_size}"
  --num-workers "${num_workers}"
  --patches-per-image "${patches_per_image}"
  --layer "${layer}"
  --targets "${targets}"
  --ridge-alpha "${ridge_alpha}"
  --visualization-samples "${visualization_samples}"
  --visualization-targets "${visualization_targets}"
  --gap "${gap}"
  --sample-gap "${sample_gap}"
  --metric-label-width "${metric_label_width}"
  "${probe_args[@]}"
  "${selection_args[@]}"
)

echo "==== Local-detail decodability comparison visualization ===="
echo "Model roots: ${MODEL_ROOT_ARRAY[*]}"
for i in "${!MODEL_ROOT_ARRAY[@]}"; do
  echo "Model $((i + 1)) checkpoint: ${MODEL_WEIGHT_ARRAY[i]}"
done
echo "Dataset    : ${dataset}"
echo "Output dir : ${output_dir}"
echo "Layer      : ${layer}"
echo "Targets    : ${visualization_targets}"

if [[ -n "${gpu}" ]]; then
  CUDA_VISIBLE_DEVICES="${gpu}" "${CMD[@]}"
else
  "${CMD[@]}"
fi
