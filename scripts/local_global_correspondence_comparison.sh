#!/usr/bin/env bash
set -euo pipefail

# Compare saved results; missing visualizations use inference on selected crops only.
# Example: model_roots="out/model1,out/model2,out/model3" bash scripts/local_global_correspondence_comparison.sh
model_roots="${model_roots:-}"
checkpoint_steps="${checkpoint_steps-375299}"
layers="${layers:-11}"
gap="${gap:-4}"
output_dir="${output_dir:-${comparison_output_dir:-}}"
gpu="${gpu:-0}"
visualization_images="${visualization_images:-16}"
visualization_samples="${visualization_samples:-16}"
visualization_max_per_image="${visualization_max_per_image:-1}"
visualization_max_match_lines="${visualization_max_match_lines:-48}"
batch_size="${batch_size:-16}"
num_workers="${num_workers:-0}"
image_dir="${image_dir:-}"

[[ -n "${model_roots}" ]] || {
  echo "Set model_roots=/path/to/model1,/path/to/model2,... to compare saved results." >&2
  exit 1
}
IFS=',' read -r -a MODEL_ROOT_ARRAY <<< "${model_roots}"
[[ "${#MODEL_ROOT_ARRAY[@]}" -gt 0 ]] || { echo "Set at least one model_roots directory." >&2; exit 1; }
for i in "${!MODEL_ROOT_ARRAY[@]}"; do
  root="${MODEL_ROOT_ARRAY[i]}"
  # Trim surrounding whitespace while preserving spaces and quotes inside paths.
  root="${root#"${root%%[![:space:]]*}"}"
  root="${root%"${root##*[![:space:]]}"}"
  [[ -n "${root}" ]] || { echo "Empty model root at index ${i}." >&2; exit 1; }
  name="$(basename "${root}")"
  if [[ ! -d "${root}/local_global_correspondence_all" && -d "../${name}/local_global_correspondence_all" ]]; then
    root="../${name}"
  fi
  if [[ ! -d "${root}/local_global_correspondence_all" && -d "${name}/local_global_correspondence_all" ]]; then
    root="${name}"
  fi
  [[ -d "${root}/local_global_correspondence_all" ]] || {
    echo "Saved results not found: ${root}/local_global_correspondence_all. Comparison does not run evaluation." >&2
    exit 1
  }
  MODEL_ROOT_ARRAY[i]="${root}"
done

if [[ -z "${output_dir}" ]]; then
  output_dir="${MODEL_ROOT_ARRAY[0]}/local_global_correspondence_comparison"
fi

CUDA_VISIBLE_DEVICES="${gpu}" python -m jea.diagnostics.local_global_correspondence_comparison \
  --model-roots "${MODEL_ROOT_ARRAY[@]}" \
  --checkpoint-steps "${checkpoint_steps}" \
  --layers "${layers}" \
  --gap "${gap}" \
  --visualization-images "${visualization_images}" \
  --visualization-samples "${visualization_samples}" \
  --visualization-max-per-image "${visualization_max_per_image}" \
  --visualization-max-match-lines "${visualization_max_match_lines}" \
  --batch-size "${batch_size}" \
  --num-workers "${num_workers}" \
  --image-dir "${image_dir}" \
  --output-dir "${output_dir}"
