#!/usr/bin/env bash
set -euo pipefail

# Optional display names; when omitted, derive them from model_roots.
model_names="${model_names:-}"
# Add any number of comma-separated model directories here.
model_roots="${model_roots:-out/dino_vit_b16,out/dino_ibot_vit_b16}"
# Optional comma-separated mask roots, one per model, overriding model_roots/checkpoint_step.
model_results_paths="${model_results_paths:-}"
checkpoint_step="${checkpoint_step:-375299}"
davis_path="${davis_path:-datasets/davis}"
set_name="${set_name:-val}"
video="${video:-}"
frame_indices="${frame_indices:-}"
num_frames="${num_frames:-5}"
frame_stride="${frame_stride:-1}"
center_frame="${center_frame:-}"
gap="${gap:-4}"
save_png="${save_png:-0}"
output_dir="${output_dir:-}"

[[ -d "${davis_path}" ]] || {
  echo "DAVIS root not found: ${davis_path}. Set davis_path=/path/to/davis." >&2
  exit 1
}
[[ -n "${video}" ]] || {
  echo "Set video=<DAVIS video folder name>, e.g. video=dogs-jump or video=pigs." >&2
  exit 1
}
[[ -n "${frame_indices}" || -n "${center_frame}" ]] || {
  echo "Set frame_indices=0,5,10,15,20 or center_frame=<0-based frame index>." >&2
  exit 1
}

IFS=',' read -r -a MODEL_ROOT_ARRAY <<< "${model_roots}"
[[ "${#MODEL_ROOT_ARRAY[@]}" -gt 0 ]] || {
  echo "Expected at least one model directory in model_roots." >&2
  exit 1
}
MODEL_NAME_ARRAY=()
if [[ -n "${model_names}" ]]; then
  IFS=',' read -r -a MODEL_NAME_ARRAY <<< "${model_names}"
  [[ "${#MODEL_NAME_ARRAY[@]}" -eq "${#MODEL_ROOT_ARRAY[@]}" ]] || {
    echo "When model_names is set, provide one name per model root; otherwise omit it to use directory names." >&2
    exit 1
  }
fi
MODEL_RESULTS_ARRAY=()
if [[ -n "${model_results_paths}" ]]; then
  IFS=',' read -r -a MODEL_RESULTS_ARRAY <<< "${model_results_paths}"
  [[ "${#MODEL_RESULTS_ARRAY[@]}" -eq "${#MODEL_ROOT_ARRAY[@]}" ]] || {
    echo "Expected one model_results_paths entry per model." >&2
    exit 1
  }
fi

resolve_model_root() {
  local model_root="$1"
  local guess
  model_root="$(echo "${model_root}" | xargs)"
  guess="$(basename "${model_root}")"
  if [[ ! -d "${model_root}" && -d "../${guess}" ]]; then
    model_root="../${guess}"
  fi
  if [[ ! -d "${model_root}" && -d "${guess}" ]]; then
    model_root="${guess}"
  fi
  if [[ ! -d "${model_root}" && -d "../${model_root}" ]]; then
    model_root="../${model_root}"
  fi
  echo "${model_root}"
}

model_args=()
comparison_name=""
used_names=(frames groundtruth)
for i in "${!MODEL_ROOT_ARRAY[@]}"; do
  root="$(resolve_model_root "${MODEL_ROOT_ARRAY[i]}")"
  if [[ -n "${model_names}" ]]; then
    name="$(echo "${MODEL_NAME_ARRAY[i]}" | xargs)"
  else
    base_name="$(basename "${root}")"
    name="${base_name}"
    suffix=1
    while true; do
      duplicate=0
      for used_name in "${used_names[@]}"; do
        if [[ "${name}" == "${used_name}" ]]; then
          duplicate=1
          break
        fi
      done
      [[ "${duplicate}" -eq 1 ]] || break
      suffix=$((suffix + 1))
      name="${base_name}_${suffix}"
    done
  fi
  [[ -n "${name}" && -n "${root}" ]] || {
    echo "Model names and roots must not be empty (index ${i})." >&2
    exit 1
  }
  MODEL_NAME_ARRAY[i]="${name}"
  used_names+=("${name}")
  MODEL_ROOT_ARRAY[i]="${root}"
  results_path="${root}/davis_video_segmentation_all/training_${checkpoint_step}"
  if [[ -n "${model_results_paths}" ]]; then
    results_path="${MODEL_RESULTS_ARRAY[i]}"
  elif [[ "${i}" -eq 0 && -n "${left_results_path:-}" ]]; then
    results_path="${left_results_path}"
  elif [[ "${i}" -eq 1 && -n "${right_results_path:-}" ]]; then
    results_path="${right_results_path}"
  fi
  [[ -d "${results_path}" ]] || {
    echo "DAVIS results path not found for ${name}: ${results_path}" >&2
    echo "Set model_results_paths=/path/one,/path/two,... or checkpoint_step=<step>." >&2
    exit 1
  }
  MODEL_RESULTS_ARRAY[i]="${results_path}"
  model_args+=(--model "${name}" "${results_path}")
  comparison_name+="${comparison_name:+_vs_}${name}"
done

if [[ -z "${output_dir}" ]]; then
  # Full checkpoint directory names can exceed a filesystem component's limit.
  if [[ "${#comparison_name}" -gt 120 ]]; then
    comparison_name="${#MODEL_ROOT_ARRAY[@]}_models"
  fi
  output_dir="${MODEL_ROOT_ARRAY[0]}/davis_video_segmentation_mask_comparison_${comparison_name}"
fi

png_args=()
if [[ "${save_png}" == "1" ]]; then
  png_args+=(--save-png)
fi

frame_selection_args=()
if [[ -n "${frame_indices}" ]]; then
  frame_selection_args+=(--frame-indices "${frame_indices}")
else
  frame_selection_args+=(
    --num-frames "${num_frames}"
    --frame-stride "${frame_stride}"
    --center-frame "${center_frame}"
  )
fi

CMD=(
  python -m jea.diagnostics.davis_video_segmentation_mask_comparison
  --davis-path "${davis_path}"
  "${model_args[@]}"
  --video "${video}"
  --set "${set_name}"
  --output-dir "${output_dir}"
  --gap "${gap}"
  "${frame_selection_args[@]}"
  "${png_args[@]}"
)

echo "==== DAVIS video segmentation mask comparison ===="
echo "Models      : ${MODEL_NAME_ARRAY[*]}"
for i in "${!MODEL_NAME_ARRAY[@]}"; do
  echo "${MODEL_NAME_ARRAY[i]} masks: ${MODEL_RESULTS_ARRAY[i]}"
done
echo "DAVIS root  : ${davis_path}"
echo "Video       : ${video}"
if [[ -n "${frame_indices}" ]]; then
  echo "Frames      : custom (${frame_indices})"
else
  echo "Frames      : ${num_frames} (center_frame=${center_frame}, frame_stride=${frame_stride})"
fi
echo "Output dir  : ${output_dir}"

"${CMD[@]}"
