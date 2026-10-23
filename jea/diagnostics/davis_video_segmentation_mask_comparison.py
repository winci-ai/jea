# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

from jea.diagnostics.visualization_utils import ensure_dir, make_unlabeled_panel_grid, save_visualization_pdf


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Visualize DAVIS frame/GT/multiple model mask rows for one video sequence."
    )
    parser.add_argument("--davis-path", default="datasets/davis", type=str)
    parser.add_argument(
        "--model", action="append", nargs=2, metavar=("NAME", "RESULTS_PATH"),
        help="Model name and mask results root. Repeat for each model, in display order.",
    )
    parser.add_argument("--left-results-path", type=str, help="Legacy two-model input; use with --right-results-path.")
    parser.add_argument("--right-results-path", type=str, help="Legacy two-model input; use with --left-results-path.")
    parser.add_argument("--left-name", default="dino", type=str)
    parser.add_argument("--right-name", default="dino_ibot", type=str)
    parser.add_argument("--video", required=True, type=str, help="DAVIS video folder name, e.g. dogs-jump or pigs.")
    parser.add_argument("--set", default="val", type=str)
    parser.add_argument(
        "--frame-indices",
        default="",
        type=str,
        help="Comma-separated 0-based frame indices to visualize, e.g. 0,5,10,15,20. Overrides centered selection.",
    )
    parser.add_argument(
        "--num-frames",
        default=5,
        type=int,
        help="Odd number of frames around --center-frame when --frame-indices is not set.",
    )
    parser.add_argument("--frame-stride", default=1, type=int, help="Frame stride around the selected center frame.")
    parser.add_argument("--center-frame", default=-1, type=int, help="0-based center frame index.")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--output-prefix", default="", type=str)
    parser.add_argument("--gap", default=4, type=int)
    parser.add_argument("--save-png", action="store_true")
    return parser


def resolve_models(args) -> List[Tuple[str, Path]]:
    models = getattr(args, "model", None)
    left_path = args.left_results_path
    right_path = args.right_results_path
    if models:
        if left_path or right_path:
            raise ValueError("Use either repeated --model arguments or legacy left/right paths, not both.")
    else:
        if not left_path or not right_path:
            raise ValueError("Provide at least one --model NAME RESULTS_PATH, or both legacy left/right paths.")
        models = [(args.left_name, left_path), (args.right_name, right_path)]
    resolved = []
    names = {"frames", "groundtruth"}
    for name, path in models:
        name = name.strip()
        if not name or name in names:
            raise ValueError(f"Model names must be non-empty, unique, and not frames/groundtruth: {name!r}")
        if not str(path).strip():
            raise ValueError(f"Results path must not be empty for model {name!r}")
        names.add(name)
        resolved.append((name, Path(path)))
    return resolved


def load_palette(mask_path: Path):
    image = Image.open(mask_path)
    palette = image.getpalette()
    image.close()
    return palette


def mask_to_rgb(mask_path: Path, palette=None) -> Image.Image:
    image = Image.open(mask_path)
    if palette is not None:
        image = image.convert("P")
        image.putpalette(palette)
    return image.convert("RGB")


def read_mask_array(mask_path: Path) -> np.ndarray:
    return np.asarray(Image.open(mask_path))


def frame_stem(path: Path) -> str:
    return path.stem


def list_frames(davis_path: Path, video: str) -> List[Path]:
    frame_dir = davis_path / "JPEGImages" / "480p" / video
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"DAVIS frame directory not found: {frame_dir}")
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No DAVIS frames found under {frame_dir}")
    return frames


def mask_path_for(root: Path, video: str, frame: Path) -> Path:
    return root / video / f"{frame_stem(frame)}.png"


def list_set_videos(davis_path: Path, set_name: str) -> List[str]:
    candidates = [
        davis_path / "ImageSets" / "2017" / f"{set_name}.txt",
        davis_path / "ImageSets" / f"{set_name}.txt",
    ]
    for path in candidates:
        if path.is_file():
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame_root = davis_path / "JPEGImages" / "480p"
    if frame_root.is_dir():
        return sorted(path.name for path in frame_root.iterdir() if path.is_dir())
    raise FileNotFoundError(f"Could not find DAVIS ImageSets file for set={set_name} under {davis_path}")


def video_mean_error(davis_path: Path, right_results_path: Path, video: str) -> float:
    frames = list_frames(davis_path, video)
    errors = []
    for frame in frames:
        gt_path = davis_path / "Annotations" / "480p" / video / f"{frame_stem(frame)}.png"
        pred_path = mask_path_for(right_results_path, video, frame)
        if gt_path.is_file() and pred_path.is_file():
            errors.append(frame_error(gt_path, pred_path))
    if not errors:
        return -1.0
    return float(sum(errors) / len(errors))


def choose_video_from_mask_difference(davis_path: Path, right_results_path: Path, set_name: str) -> Tuple[str, float]:
    videos = list_set_videos(davis_path, set_name)
    if not videos:
        raise RuntimeError(f"No videos found for DAVIS set={set_name}")
    scored = [(video, video_mean_error(davis_path, right_results_path, video)) for video in videos]
    scored = [(video, score) for video, score in scored if score >= 0.0]
    if not scored:
        raise RuntimeError(f"No videos had matching GT/predicted masks under {right_results_path}")
    return max(scored, key=lambda item: item[1])


def frame_error(gt_path: Path, pred_path: Path) -> float:
    gt = read_mask_array(gt_path)
    pred = read_mask_array(pred_path)
    if gt.shape != pred.shape:
        pred = np.asarray(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST))
    return float(np.mean(gt != pred))


def parse_frame_indices(value: str) -> List[int]:
    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        indices.append(int(item))
    if not indices:
        raise ValueError("--frame-indices was provided but no indices were parsed.")
    return indices


def choose_custom_frames(
    frames: Sequence[Path],
    davis_path: Path,
    right_results_path: Path,
    video: str,
    frame_indices: Sequence[int],
) -> Tuple[int, List[int], List[Path], List[float]]:
    if any(index < 0 or index >= len(frames) for index in frame_indices):
        raise RuntimeError(
            f"--frame-indices contains out-of-range frame ids for video {video}: "
            f"indices={list(frame_indices)}, num_video_frames={len(frames)}."
        )
    chosen = [frames[index] for index in frame_indices]
    errors = []
    for frame in chosen:
        gt_path = davis_path / "Annotations" / "480p" / video / f"{frame_stem(frame)}.png"
        pred_path = mask_path_for(right_results_path, video, frame)
        errors.append(frame_error(gt_path, pred_path) if gt_path.is_file() and pred_path.is_file() else 0.0)
    center = int(frame_indices[len(frame_indices) // 2])
    return center, list(frame_indices), chosen, errors


def choose_centered_frames(
    frames: Sequence[Path],
    davis_path: Path,
    right_results_path: Path,
    video: str,
    num_frames: int,
    frame_stride: int,
    center_frame: int,
) -> Tuple[int, List[int], List[Path], List[float]]:
    if frame_stride <= 0:
        raise ValueError(f"--frame-stride must be positive, got {frame_stride}")
    if num_frames <= 0 or num_frames % 2 == 0:
        raise ValueError(f"--num-frames must be a positive odd number, got {num_frames}")
    center = int(center_frame)
    if center < 0:
        raise ValueError("Set --center-frame when --frame-indices is not provided.")
    if center < 0 or center >= len(frames):
        raise RuntimeError(
            f"--center-frame is outside video {video}: center_frame={center}, "
            f"num_video_frames={len(frames)}."
        )

    indices = {center}
    step = 1
    while len(indices) < num_frames:
        before = center - step * frame_stride
        after = center + step * frame_stride
        added = False
        if before >= 0 and len(indices) < num_frames:
            indices.add(before)
            added = True
        if after < len(frames) and len(indices) < num_frames:
            indices.add(after)
            added = True
        if not added:
            break
        step += 1
    if len(indices) < num_frames:
        raise RuntimeError(
            f"Could only select {len(indices)} frames around center_frame={center} "
            f"with stride={frame_stride} in video {video}; requested {num_frames}."
        )
    indices = sorted(indices)
    chosen = [frames[index] for index in indices]
    errors = []
    for frame in chosen:
        gt_path = davis_path / "Annotations" / "480p" / video / f"{frame_stem(frame)}.png"
        pred_path = mask_path_for(right_results_path, video, frame)
        errors.append(frame_error(gt_path, pred_path) if gt_path.is_file() and pred_path.is_file() else 0.0)
    return center, indices, chosen, errors


def resize_to_reference(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image.convert("RGB")
    return image.convert("RGB").resize(size, Image.NEAREST)


def build_grid(args):
    davis_path = Path(args.davis_path)
    models = resolve_models(args)
    # Keep legacy summary errors tied to the final (formerly right-hand) model.
    right_results_path = models[-1][1]
    if not davis_path.is_dir():
        raise FileNotFoundError(f"DAVIS path not found: {davis_path}")
    for name, results_path in models:
        if not results_path.is_dir():
            raise FileNotFoundError(f"Results path not found for model {name!r}: {results_path}")

    video = args.video.strip()
    if not video:
        raise ValueError("--video must be a DAVIS video folder name, e.g. dogs-jump or pigs.")
    frames = list_frames(davis_path, video)
    if args.frame_indices:
        center, frame_indices, chosen_frames, errors = choose_custom_frames(
            frames=frames,
            davis_path=davis_path,
            right_results_path=right_results_path,
            video=video,
            frame_indices=parse_frame_indices(args.frame_indices),
        )
    else:
        center, frame_indices, chosen_frames, errors = choose_centered_frames(
            frames=frames,
            davis_path=davis_path,
            right_results_path=right_results_path,
            video=video,
            num_frames=args.num_frames,
            frame_stride=args.frame_stride,
            center_frame=args.center_frame,
        )

    first_gt = davis_path / "Annotations" / "480p" / video / f"{frame_stem(chosen_frames[0])}.png"
    palette = load_palette(first_gt)

    rows = []
    row_sources = [("frames", None), ("groundtruth", davis_path / "Annotations" / "480p")] + models
    for kind, results_path in row_sources:
        row_images = []
        for frame in chosen_frames:
            if kind == "frames":
                image = Image.open(frame).convert("RGB")
                ref_size = image.size
            else:
                mask_path = mask_path_for(results_path, video, frame)
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Mask not found for row {kind!r}: {mask_path}")
                image = mask_to_rgb(mask_path, palette=palette)
                ref_size = Image.open(frame).size
            row_images.append(resize_to_reference(image, ref_size))
        rows.append(make_unlabeled_panel_grid(row_images, columns=len(row_images), gap=args.gap))
    grid = make_unlabeled_panel_grid(rows, columns=1, gap=args.gap)
    model_metrics = []
    for name, results_path in models:
        model_metrics.append(OrderedDict([
            ("name", name),
            ("results_path", str(results_path)),
            ("video_mean_pixel_error", video_mean_error(davis_path, results_path, video)),
            ("mask_gt_pixel_error", errors if name == models[-1][0] else [
                frame_error(
                    davis_path / "Annotations" / "480p" / video / f"{frame_stem(frame)}.png",
                    mask_path_for(results_path, video, frame),
                ) for frame in chosen_frames
            ]),
        ]))
    return video, model_metrics, center, frame_indices, chosen_frames, grid


def main(args):
    output_dir = ensure_dir(args.output_dir)
    video, model_metrics, center, frame_indices, chosen_frames, grid = build_grid(args)
    if args.frame_indices:
        prefix = args.output_prefix or f"{video}_frames_{'_'.join(str(index) for index in frame_indices)}"
    else:
        prefix = args.output_prefix or f"{video}_center{center:05d}_stride{args.frame_stride}"
    pdf_path = save_visualization_pdf(grid, output_dir / f"{prefix}.pdf")
    png_path = ""
    if args.save_png:
        png_path = output_dir / f"{prefix}.png"
        grid.save(png_path)

    manifest = OrderedDict()
    manifest["video"] = video
    # These legacy summary fields describe the last model in row_order.
    manifest["video_mean_pixel_error"] = model_metrics[-1]["video_mean_pixel_error"]
    manifest["center_frame"] = center
    manifest["num_frames"] = len(chosen_frames)
    manifest["frame_stride"] = args.frame_stride
    manifest["custom_frame_indices"] = args.frame_indices
    manifest["frame_indices"] = frame_indices
    manifest["frame_names"] = [frame.name for frame in chosen_frames]
    manifest["right_mask_gt_pixel_error"] = model_metrics[-1]["mask_gt_pixel_error"]
    manifest["models"] = model_metrics
    manifest["output_pdf"] = str(pdf_path)
    manifest["output_png"] = str(png_path)
    manifest["row_order"] = ["frames", "groundtruth"] + [model["name"] for model in model_metrics]
    with (output_dir / f"{prefix}.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote DAVIS mask comparison to {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(get_args_parser().parse_args()))
