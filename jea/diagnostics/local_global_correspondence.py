# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import functional as TVF

import jea.distributed as distributed
from jea.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from jea.eval.setup import get_args_parser as get_setup_args_parser
from jea.eval.setup import setup_and_build_model
from jea.diagnostics.local_detail_decodability import (
    append_layer_to_output_dir,
    extract_patch_tokens,
    get_patch_size,
    parse_layer,
)
from jea.diagnostics.visualization_utils import (
    draw_norm_box,
    ensure_dir,
    overlay_heatmap,
    patch_grid_point,
    safe_name,
    save_visualization_manifest,
    save_visualization_pdf,
    tensor_to_pil_image,
)


logger = logging.getLogger("JEA")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parser = argparse.ArgumentParser(
        description=description,
        parents=[setup_args_parser],
        add_help=add_help,
    )
    parser.add_argument("--data-path", default="datasets/copydays", type=str)
    parser.add_argument(
        "--image-dir",
        default="",
        type=str,
        help="Overrides --data-path/original as the source-image directory.",
    )
    parser.add_argument("--max-images", default=157, type=int)
    parser.add_argument("--subset-seed", default=0, type=int)
    parser.add_argument(
        "--run-seeds",
        default="0",
        type=str,
        help="Comma-separated run seeds. Defaults to one run; pass multiple seeds to report mean/std across repeats.",
    )
    parser.add_argument(
        "--crop-ratios",
        default="0.1,0.2,0.3,0.5",
        type=str,
        help="Comma-separated crop area ratios relative to the source image.",
    )
    parser.add_argument("--crops-per-image", default=4, type=int)
    parser.add_argument("--crop-seed", default=0, type=int)
    parser.add_argument(
        "--selected-crop-specs-json",
        default="",
        type=str,
        help="Use fixed crop specs JSON instead of selecting images/crops in this run.",
    )
    parser.add_argument(
        "--write-crop-specs-json",
        default="",
        type=str,
        help=(
            "Write selected crop specs to this path. With multiple run seeds, "
            "per-run files are created by appending run/seed suffixes."
        ),
    )
    parser.add_argument("--aspect-jitter", default=0.25, type=float)
    parser.add_argument("--imsize", default=224, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--layer",
        default="final",
        type=str,
        help="Patch-token layer to evaluate: 'final' or a 0-based transformer block index.",
    )
    parser.add_argument("--topk", default=5, type=int)
    parser.add_argument("--ransac-iters", default=128, type=int)
    parser.add_argument("--geometric-tolerance-patches", default=1.5, type=float)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--visualization-samples", default=16, type=int)
    parser.add_argument(
        "--visualization-images",
        default=0,
        type=int,
        help=(
            "Save visualizations from this many distinct source images. "
            "When >0, the total visualization budget is "
            "--visualization-images * max(1, --visualization-max-per-image)."
        ),
    )
    parser.add_argument("--visualization-dir", default="", type=str)
    parser.add_argument("--visualization-max-match-lines", default=48, type=int)
    parser.add_argument(
        "--visualization-max-per-image",
        default=1,
        type=int,
        help="Maximum saved visualization crops per source image. <=0 disables this limit.",
    )
    return parser


def parse_float_list(value: str) -> List[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one crop ratio.")
    for ratio in parsed:
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError(f"Crop ratios must be in (0, 1), got {ratio}.")
    return parsed


def parse_int_list(value: str) -> List[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one run seed.")
    return parsed


def set_run_seed(seed: int):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def summarize_metric_runs(run_metrics: Sequence[Dict[str, float]]) -> OrderedDict:
    metrics = OrderedDict()
    keys = OrderedDict()
    for metric_dict in run_metrics:
        for key in metric_dict.keys():
            keys[key] = None

    for key in keys.keys():
        values = []
        for metric_dict in run_metrics:
            value = metric_dict.get(key)
            if value is None:
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                values.append(numeric)
        if not values:
            metrics[key] = None
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        metrics[key] = mean
        if len(values) > 1:
            metrics[f"{key}/run_std"] = variance ** 0.5
    return metrics


def rows_with_run_metadata(
    rows: Sequence[Dict],
    run_index: int,
    run_seed: int,
    subset_seed: int,
    crop_seed: int,
    crop_specs_json: str,
) -> List[OrderedDict]:
    annotated = []
    for row in rows:
        out = OrderedDict()
        out["run_index"] = run_index
        out["run_seed"] = run_seed
        out["effective_subset_seed"] = subset_seed
        out["effective_crop_seed"] = crop_seed
        out["crop_specs_json"] = crop_specs_json
        out.update(row)
        annotated.append(out)
    return annotated


def distributed_all_gather_object(value):
    if not distributed.is_enabled():
        return [value]
    gathered = [None for _ in range(distributed.get_global_size())]
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def broadcast_object_from_main(value):
    if not distributed.is_enabled():
        return value
    values = [value if distributed.is_main_process() else None]
    torch.distributed.broadcast_object_list(values, src=0)
    return values[0]


def crop_specs_output_path(base_path: str, run_index: int, subset_seed: int, crop_seed: int, num_runs: int) -> str:
    if not base_path:
        return ""
    path = Path(base_path)
    if num_runs <= 1:
        return str(path)
    suffix = path.suffix or ".json"
    stem = path.stem if path.suffix else path.name
    return str(path.with_name(f"{stem}_run{run_index:02d}_subset{subset_seed}_crop{crop_seed}{suffix}"))


def local_shard_indices(num_items: int) -> List[int]:
    if not distributed.is_enabled():
        return list(range(num_items))
    rank = distributed.get_global_rank()
    world_size = distributed.get_global_size()
    return list(range(rank, num_items, world_size))


def accumulators_to_plain(accumulators) -> Dict[str, Dict[str, float]]:
    return {
        str(key): {
            "sum": float(value.get("sum", 0.0)),
            "count": int(value.get("count", 0)),
        }
        for key, value in accumulators.items()
    }


def merge_plain_accumulators(dst, src: Dict[str, Dict[str, float]]):
    for key, value in src.items():
        dst[key]["sum"] += float(value.get("sum", 0.0))
        dst[key]["count"] += int(value.get("count", 0))


def gather_crop_eval_outputs(global_acc, ratio_acc, rows: Sequence[Dict]):
    payload = {
        "global_acc": accumulators_to_plain(global_acc),
        "ratio_acc": {
            str(ratio_key): accumulators_to_plain(accumulators)
            for ratio_key, accumulators in ratio_acc.items()
        },
        "rows": list(rows),
    }
    gathered = distributed_all_gather_object(payload)
    if not distributed.is_enabled():
        return global_acc, ratio_acc, list(rows)

    merged_global = defaultdict(empty_accumulator)
    merged_ratio = defaultdict(lambda: defaultdict(empty_accumulator))
    merged_rows = []
    for item in gathered:
        if not item:
            continue
        merge_plain_accumulators(merged_global, item.get("global_acc", {}))
        for ratio_key, accumulators in item.get("ratio_acc", {}).items():
            merge_plain_accumulators(merged_ratio[ratio_key], accumulators)
        merged_rows.extend(item.get("rows", []))
    merged_rows.sort(key=lambda row: int(row.get("query_index", 0)))
    return merged_global, merged_ratio, merged_rows


def list_images(image_dir: Path) -> List[Path]:
    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise RuntimeError(f"No source images found in {image_dir}.")
    return images


def choose_subset(paths: Sequence[Path], max_images: int, seed: int) -> List[Path]:
    if max_images <= 0 or max_images >= len(paths):
        return list(paths)
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(paths), generator=generator)[:max_images].tolist()
    return [paths[index] for index in sorted(indices)]


def make_crop_box(width: int, height: int, ratio: float, aspect_jitter: float, generator: torch.Generator):
    area = float(width * height) * ratio
    jitter = float(aspect_jitter)
    log_aspect = 0.0
    if jitter > 0.0:
        low = math.log(max(1e-3, 1.0 - jitter))
        high = math.log(1.0 + jitter)
        log_aspect = low + (high - low) * float(torch.rand((), generator=generator).item())
    aspect = math.exp(log_aspect)
    crop_w = int(round(math.sqrt(area * aspect)))
    crop_h = int(round(math.sqrt(area / aspect)))
    crop_w = max(1, min(crop_w, width))
    crop_h = max(1, min(crop_h, height))
    if crop_w >= width:
        x0 = 0
    else:
        x0 = int(torch.randint(0, width - crop_w + 1, (1,), generator=generator).item())
    if crop_h >= height:
        y0 = 0
    else:
        y0 = int(torch.randint(0, height - crop_h + 1, (1,), generator=generator).item())
    return x0, y0, x0 + crop_w, y0 + crop_h


def build_crop_specs(
    image_paths: Sequence[Path],
    crop_ratios: Sequence[float],
    crops_per_image: int,
    crop_seed: int,
    aspect_jitter: float,
) -> List[Dict]:
    specs = []
    generator = torch.Generator()
    generator.manual_seed(crop_seed)
    for image_index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            width, height = image.size
        for ratio in crop_ratios:
            for crop_index in range(crops_per_image):
                x0, y0, x1, y1 = make_crop_box(width, height, ratio, aspect_jitter, generator)
                specs.append(
                    {
                        "image_index": image_index,
                        "image_id": image_path.stem,
                        "image_path": str(image_path),
                        "crop_ratio": float(ratio),
                        "crop_index": crop_index,
                        "source_width": width,
                        "source_height": height,
                        "box_xyxy": [x0, y0, x1, y1],
                        "box_norm_xyxy": [x0 / width, y0 / height, x1 / width, y1 / height],
                    }
                )
    return specs


def load_crop_specs(path: str) -> Tuple[List[Dict], Dict]:
    specs_path = Path(path)
    with specs_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        specs = payload
        metadata = {}
    elif isinstance(payload, dict) and isinstance(payload.get("specs"), list):
        specs = payload["specs"]
        metadata = {key: value for key, value in payload.items() if key != "specs"}
    else:
        raise RuntimeError(f"Crop specs JSON must be a list or a dict with a 'specs' list: {specs_path}")
    required = (
        "image_index",
        "image_id",
        "image_path",
        "crop_ratio",
        "crop_index",
        "source_width",
        "source_height",
        "box_xyxy",
        "box_norm_xyxy",
    )
    for index, spec in enumerate(specs):
        missing = [key for key in required if key not in spec]
        if missing:
            raise RuntimeError(f"Crop spec {index} from {specs_path} is missing keys: {missing}")
    return list(specs), metadata


def write_crop_specs(path: str, specs: Sequence[Dict], metadata: Dict):
    if not distributed.is_main_process() or not path:
        return
    specs_path = Path(path)
    specs_path.parent.mkdir(parents=True, exist_ok=True)
    payload = OrderedDict(metadata)
    payload["specs"] = list(specs)
    with specs_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote selected correspondence crop specs to {specs_path}")


class LocalGlobalCorrespondenceDataset(Dataset):
    def __init__(
        self,
        specs: Sequence[Dict],
        imsize: int,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
    ):
        self.specs = list(specs)
        self.imsize = imsize
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __len__(self):
        return len(self.specs)

    def _prepare(self, image: Image.Image):
        image = image.resize((self.imsize, self.imsize), Image.BICUBIC)
        tensor = TVF.to_tensor(image)
        return self.normalize(tensor)

    def __getitem__(self, index):
        spec = self.specs[index]
        image = Image.open(spec["image_path"]).convert("RGB")
        x0, y0, x1, y1 = spec["box_xyxy"]
        crop = image.crop((x0, y0, x1, y1))
        return {
            "source": self._prepare(image),
            "crop": self._prepare(crop),
            "index": index,
            "box_norm_xyxy": torch.tensor(spec["box_norm_xyxy"], dtype=torch.float32),
            "crop_ratio": torch.tensor(spec["crop_ratio"], dtype=torch.float32),
        }


def write_csv(path: Path, rows: Sequence[Dict]):
    if not rows:
        return
    keys = list(OrderedDict((key, None) for row in rows for key in row.keys()).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def empty_accumulator():
    return {"sum": 0.0, "count": 0}


def update_accumulator(accumulators, key: str, value: float):
    if value is None or not math.isfinite(float(value)):
        return
    accumulators[key]["sum"] += float(value)
    accumulators[key]["count"] += 1


def finalize_accumulators(accumulators) -> OrderedDict:
    metrics = OrderedDict()
    for key in sorted(accumulators.keys()):
        count = accumulators[key]["count"]
        if count > 0:
            metrics[f"{key}/mean"] = accumulators[key]["sum"] / count
            metrics[f"{key}/count"] = count
    return metrics


def patch_centers(grid_h: int, grid_w: int, device) -> torch.Tensor:
    rows, cols = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=torch.float32),
        torch.arange(grid_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x = (cols.reshape(-1) + 0.5) / float(grid_w)
    y = (rows.reshape(-1) + 0.5) / float(grid_h)
    return torch.stack([x, y], dim=1)


def bbox_iou_xyxy(left: torch.Tensor, right: torch.Tensor) -> float:
    ix0 = torch.maximum(left[0], right[0])
    iy0 = torch.maximum(left[1], right[1])
    ix1 = torch.minimum(left[2], right[2])
    iy1 = torch.minimum(left[3], right[3])
    inter = (ix1 - ix0).clamp_min(0) * (iy1 - iy0).clamp_min(0)
    left_area = (left[2] - left[0]).clamp_min(0) * (left[3] - left[1]).clamp_min(0)
    right_area = (right[2] - right[0]).clamp_min(0) * (right[3] - right[1]).clamp_min(0)
    union = left_area + right_area - inter
    if float(union.item()) <= 0.0:
        return 0.0
    return float((inter / union).item())


def evaluate_one_crop(
    source_tokens: torch.Tensor,
    crop_tokens: torch.Tensor,
    box: torch.Tensor,
    grid_h: int,
    grid_w: int,
    topk: int,
    ransac_iters: int,
    geometric_tolerance_patches: float,
    return_artifacts: bool = False,
):
    source_tokens = F.normalize(source_tokens.float(), dim=1)
    crop_tokens = F.normalize(crop_tokens.float(), dim=1)
    sim = crop_tokens.matmul(source_tokens.t())

    source_coords = patch_centers(grid_h, grid_w, source_tokens.device)
    crop_coords = patch_centers(grid_h, grid_w, crop_tokens.device)
    x0, y0, x1, y1 = box.to(source_tokens.device)
    gt_mask = (
        (source_coords[:, 0] >= x0)
        & (source_coords[:, 0] < x1)
        & (source_coords[:, 1] >= y0)
        & (source_coords[:, 1] < y1)
    )
    gt_count = int(gt_mask.sum().item())
    if gt_count == 0:
        nearest = torch.argmin((source_coords[:, 0] - (x0 + x1) * 0.5).abs() + (source_coords[:, 1] - (y0 + y1) * 0.5).abs())
        gt_mask[nearest] = True
        gt_count = 1

    k = min(max(1, int(topk)), source_tokens.shape[0])
    source_score = sim.max(dim=0).values
    top_source = torch.topk(source_score, k=k, largest=True).indices
    top1_source = int(top_source[0].item())

    pred_count = min(gt_count, source_tokens.shape[0])
    pred_indices = torch.topk(source_score, k=pred_count, largest=True).indices
    pred_mask = torch.zeros_like(gt_mask)
    pred_mask[pred_indices] = True
    intersection = (pred_mask & gt_mask).sum().float()
    union = (pred_mask | gt_mask).sum().float().clamp_min(1.0)

    match_topk = torch.topk(sim, k=k, dim=1, largest=True).indices
    top1_matches = match_topk[:, 0]
    top1_in_gt = gt_mask.index_select(0, top1_matches)
    topk_in_gt = gt_mask.index_select(0, match_topk.reshape(-1)).view(match_topk.shape).any(dim=1)

    matched_source_coords = source_coords.index_select(0, top1_matches)
    box_wh = torch.stack([(x1 - x0).clamp_min(1e-6), (y1 - y0).clamp_min(1e-6)])
    top_left_votes = matched_source_coords - crop_coords * box_wh.view(1, 2)
    if ransac_iters > 0 and top_left_votes.shape[0] > ransac_iters:
        sample_ids = torch.linspace(0, top_left_votes.shape[0] - 1, steps=ransac_iters, device=top_left_votes.device).long()
        candidates = top_left_votes.index_select(0, sample_ids)
    else:
        candidates = top_left_votes
    tolerance = geometric_tolerance_patches / float(min(grid_h, grid_w))
    residual = (top_left_votes.unsqueeze(0) - candidates.unsqueeze(1)).abs().amax(dim=2)
    inlier_counts = (residual <= tolerance).sum(dim=1)
    best_pos = int(torch.argmax(inlier_counts).item())
    best_top_left = candidates[best_pos]
    best_inlier_ratio = float((inlier_counts[best_pos].float() / top_left_votes.shape[0]).item())
    pred_box = torch.stack(
        [
            best_top_left[0],
            best_top_left[1],
            best_top_left[0] + box_wh[0],
            best_top_left[1] + box_wh[1],
        ]
    ).clamp(0.0, 1.0)
    true_box = torch.stack([x0, y0, x1, y1])
    geom_iou = bbox_iou_xyxy(pred_box, true_box)

    row = OrderedDict()
    row["localization_top1_in_gt"] = float(gt_mask[top1_source].item())
    row["localization_topk_hit"] = float(gt_mask.index_select(0, top_source).any().item())
    row["localization_topk_precision"] = float(gt_mask.index_select(0, top_source).float().mean().item())
    row["localization_iou_top_gt_count"] = float((intersection / union).item())
    row["patch_match_top1_in_gt_rate"] = float(top1_in_gt.float().mean().item())
    row["patch_match_topk_in_gt_rate"] = float(topk_in_gt.float().mean().item())
    row["geometric_ransac_inlier_ratio"] = best_inlier_ratio
    row["geometric_ransac_box_iou"] = geom_iou
    row["geometric_success_iou50"] = float(geom_iou >= 0.5)
    row["gt_patch_count"] = gt_count
    if not return_artifacts:
        return row

    artifacts = {
        "source_score": source_score.detach().cpu().view(grid_h, grid_w),
        "gt_mask": gt_mask.detach().cpu().view(grid_h, grid_w),
        "pred_mask": pred_mask.detach().cpu().view(grid_h, grid_w),
        "top_source": top_source.detach().cpu(),
        "match_topk": match_topk.detach().cpu(),
        "top1_matches": top1_matches.detach().cpu(),
        "pred_box": pred_box.detach().cpu(),
        "true_box": true_box.detach().cpu(),
    }
    return row, artifacts


def draw_patch_matches(
    crop_image: Image.Image,
    source_image: Image.Image,
    top1_matches: torch.Tensor,
    gt_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    max_lines: int,
) -> Image.Image:
    gap = 4
    crop_image = crop_image.convert("RGB")
    source_image = source_image.convert("RGB")
    canvas = Image.new(
        "RGB",
        (crop_image.width + gap + source_image.width, max(crop_image.height, source_image.height)),
        (255, 255, 255),
    )
    canvas.paste(crop_image, (0, 0))
    canvas.paste(source_image, (crop_image.width + gap, 0))
    draw = ImageDraw.Draw(canvas)

    top1_matches = top1_matches.detach().cpu().long()
    gt_flat = gt_mask.detach().cpu().reshape(-1).bool()
    num_queries = top1_matches.numel()
    if max_lines > 0 and num_queries > max_lines:
        query_indices = torch.linspace(0, num_queries - 1, steps=max_lines).long()
    else:
        query_indices = torch.arange(num_queries).long()

    for query_index in query_indices.tolist():
        source_index = int(top1_matches[query_index].item())
        crop_x, crop_y = patch_grid_point(query_index, grid_w, crop_image.size, grid_h=grid_h)
        source_x, source_y = patch_grid_point(source_index, grid_w, source_image.size, grid_h=grid_h)
        source_x += crop_image.width + gap
        hit = bool(gt_flat[source_index].item())
        color = (30, 170, 70) if hit else (220, 45, 35)
        draw.line((crop_x, crop_y, source_x, source_y), fill=color, width=1)
        radius = 2
        draw.ellipse((crop_x - radius, crop_y - radius, crop_x + radius, crop_y + radius), fill=color)
        draw.ellipse((source_x - radius, source_y - radius, source_x + radius, source_y + radius), fill=color)
    return canvas


def make_compact_panel_row(images: Sequence[Image.Image], gap: int = 4) -> Image.Image:
    if not images:
        raise ValueError("No images were provided.")
    converted = [image.convert("RGB") for image in images]
    width = sum(image.width for image in converted) + gap * (len(converted) - 1)
    height = max(image.height for image in converted)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for image in converted:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    return canvas


def save_crop_visualizations(
    output_dir: str,
    payloads: Sequence[Dict],
    args,
    grid_h: int,
    grid_w: int,
):
    if not distributed.is_main_process() or not args.save_visualizations or not payloads:
        return
    payloads = select_visualization_payloads(payloads, args)
    if not payloads:
        return
    vis_dir = ensure_dir(args.visualization_dir or Path(output_dir) / "visualizations" / "local_global_correspondence")
    manifest_rows = []
    for rank, payload in enumerate(payloads):
        row = payload["row"]
        artifacts = payload["artifacts"]
        source_image = tensor_to_pil_image(payload["source_tensor"], denormalize=True)
        crop_image = tensor_to_pil_image(payload["crop_tensor"], denormalize=True)
        true_box = [float(value) for value in artifacts["true_box"].tolist()]
        pred_box = [float(value) for value in artifacts["pred_box"].tolist()]

        box_image = draw_norm_box(source_image, true_box, color=(20, 180, 70), width=3)
        box_image = draw_norm_box(box_image, pred_box, color=(225, 40, 35), width=3)
        heat_image = overlay_heatmap(source_image, artifacts["source_score"], alpha=0.45)
        heat_image = draw_norm_box(heat_image, true_box, color=(20, 180, 70), width=3)
        heat_image = draw_norm_box(heat_image, pred_box, color=(225, 40, 35), width=3)
        match_image = draw_patch_matches(
            crop_image=crop_image,
            source_image=source_image,
            top1_matches=artifacts["top1_matches"],
            gt_mask=artifacts["gt_mask"],
            grid_h=grid_h,
            grid_w=grid_w,
            max_lines=args.visualization_max_match_lines,
        )

        panels = [source_image, crop_image, box_image, heat_image, match_image]
        out_name = (
            f"{rank:04d}_{safe_name(row['image_id'])}_ratio{safe_name(row['crop_ratio'])}"
            f"_crop{row['crop_index']}.pdf"
        )
        out_path = vis_dir / out_name
        panel_row = make_compact_panel_row(panels)
        save_visualization_pdf(panel_row, out_path)
        # Keep a lossless row for comparisons without rasterizing the PDF.
        png_path = out_path.with_suffix(".png")
        panel_row.save(png_path)

        manifest_row = OrderedDict(row)
        manifest_row["visualization_path"] = str(out_path)
        manifest_row["visualization_png"] = str(png_path)
        manifest_rows.append(manifest_row)

    save_visualization_manifest(vis_dir, manifest_rows)
    logger.info(f"Wrote {len(manifest_rows)} correspondence visualizations to {vis_dir}")


def payload_query_index(payload: Dict) -> int:
    try:
        return int(payload.get("row", {}).get("query_index", 0))
    except (TypeError, ValueError):
        return 0


def sorted_visualization_payloads(payloads: Sequence[Dict]) -> List[Dict]:
    return sorted(list(payloads), key=payload_query_index)


def gather_visualization_payloads(payloads: Sequence[Dict]) -> List[Dict]:
    payloads = sorted_visualization_payloads(payloads)
    if not distributed.is_enabled():
        return payloads
    gathered = distributed_all_gather_object(payloads)
    merged = []
    for rank_payloads in gathered:
        if rank_payloads:
            merged.extend(rank_payloads)
    return sorted_visualization_payloads(merged)


def select_visualization_payloads(payloads: Sequence[Dict], args) -> List[Dict]:
    selected = []
    counts_by_image = defaultdict(int)
    seen_images = set()
    payloads = sorted_visualization_payloads(payloads)

    if args.visualization_images > 0:
        max_images = int(args.visualization_images)
        max_per_image = max(1, int(args.visualization_max_per_image))
        for payload in payloads:
            image_id = payload.get("row", {}).get("image_id")
            if image_id not in seen_images and len(seen_images) >= max_images:
                continue
            if counts_by_image[image_id] >= max_per_image:
                continue
            selected.append(payload)
            counts_by_image[image_id] += 1
            seen_images.add(image_id)
        return selected

    sample_budget = max(0, int(args.visualization_samples))
    max_per_image = int(args.visualization_max_per_image)
    for payload in payloads:
        if len(selected) >= sample_budget:
            break
        image_id = payload.get("row", {}).get("image_id")
        if max_per_image > 0 and counts_by_image[image_id] >= max_per_image:
            continue
        selected.append(payload)
        counts_by_image[image_id] += 1
    return selected


@torch.inference_mode()
def evaluate(model, autocast_dtype, dataset: LocalGlobalCorrespondenceDataset, specs: Sequence[Dict], args, layer):
    local_indices = local_shard_indices(len(dataset))
    loader = DataLoader(
        Subset(dataset, local_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    patch_size = get_patch_size(model)
    if args.imsize % patch_size != 0:
        raise ValueError(f"--imsize ({args.imsize}) must be divisible by patch size ({patch_size}).")
    grid_h = args.imsize // patch_size
    grid_w = args.imsize // patch_size

    rows = []
    global_acc = defaultdict(empty_accumulator)
    ratio_acc = defaultdict(lambda: defaultdict(empty_accumulator))
    vis_payloads = []
    vis_counts_by_image = defaultdict(int)
    if args.visualization_images > 0:
        visualization_max_per_image = max(1, int(args.visualization_max_per_image))
        visualization_budget = args.visualization_images * visualization_max_per_image
    else:
        visualization_budget = args.visualization_samples
        visualization_max_per_image = args.visualization_max_per_image
    vis_seen_images = set()
    processed = 0
    for batch in loader:
        source = batch["source"].cuda(non_blocking=True)
        crop = batch["crop"].cuda(non_blocking=True)
        images = torch.cat([source, crop], dim=0)
        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            tokens = extract_patch_tokens(model, images, layer)
        source_tokens = tokens[: source.shape[0]]
        crop_tokens = tokens[source.shape[0] :]

        for batch_pos, spec_index in enumerate(batch["index"].tolist()):
            spec = specs[spec_index]
            image_id = spec["image_id"]
            image_vis_allowed = (
                visualization_max_per_image <= 0
                or vis_counts_by_image[image_id] < visualization_max_per_image
            )
            image_budget_allowed = (
                args.visualization_images <= 0
                or image_id in vis_seen_images
                or len(vis_seen_images) < args.visualization_images
            )
            want_artifacts = (
                distributed.is_main_process()
                and args.save_visualizations
                and len(vis_payloads) < visualization_budget
                and image_vis_allowed
                and image_budget_allowed
            )
            eval_output = evaluate_one_crop(
                source_tokens=source_tokens[batch_pos],
                crop_tokens=crop_tokens[batch_pos],
                box=batch["box_norm_xyxy"][batch_pos],
                grid_h=grid_h,
                grid_w=grid_w,
                topk=args.topk,
                ransac_iters=args.ransac_iters,
                geometric_tolerance_patches=args.geometric_tolerance_patches,
                return_artifacts=want_artifacts,
            )
            if want_artifacts:
                metric_row, artifacts = eval_output
            else:
                metric_row = eval_output
                artifacts = None
            row = OrderedDict()
            row["query_index"] = int(spec_index)
            row["image_id"] = spec["image_id"]
            row["image_path"] = spec["image_path"]
            row["crop_ratio"] = f"{spec['crop_ratio']:.4f}"
            row["crop_index"] = spec["crop_index"]
            row["box_xyxy"] = json.dumps(spec["box_xyxy"])
            row["box_norm_xyxy"] = json.dumps(spec["box_norm_xyxy"])
            row.update(metric_row)
            rows.append(row)
            if artifacts is not None:
                vis_payloads.append(
                    {
                        "row": row.copy(),
                        "source_tensor": source[batch_pos].detach().cpu(),
                        "crop_tensor": crop[batch_pos].detach().cpu(),
                        "artifacts": artifacts,
                    }
                )
                vis_counts_by_image[image_id] += 1
                vis_seen_images.add(image_id)
            ratio_key = f"{spec['crop_ratio']:.4f}"
            for key, value in metric_row.items():
                if key == "gt_patch_count":
                    continue
                update_accumulator(global_acc, key, value)
                update_accumulator(ratio_acc[ratio_key], key, value)
        processed += source.shape[0]
        logger.info(
            f"Rank {distributed.get_global_rank()} processed "
            f"{processed}/{len(local_indices)} local synthetic crop queries "
            f"({len(dataset)} global)"
        )

    global_acc, ratio_acc, rows = gather_crop_eval_outputs(global_acc, ratio_acc, rows)
    vis_payloads = gather_visualization_payloads(vis_payloads)

    metrics = OrderedDict()
    for key, value in finalize_accumulators(global_acc).items():
        metrics[f"all/{key}"] = value
    for ratio_key in sorted(ratio_acc.keys(), key=lambda item: float(item)):
        for key, value in finalize_accumulators(ratio_acc[ratio_key]).items():
            metrics[f"ratio_{ratio_key}/{key}"] = value
    return metrics, rows, patch_size, grid_h, grid_w, vis_payloads


def main(args):
    layer = parse_layer(args.layer)
    args.output_dir = append_layer_to_output_dir(args.output_dir, args.layer)
    run_seeds = parse_int_list(args.run_seeds)
    model, autocast_dtype = setup_and_build_model(args)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    data_path = Path(args.data_path)
    image_dir = Path(args.image_dir) if args.image_dir else data_path / "original"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Source image directory not found: {image_dir}")
    crop_ratios = parse_float_list(args.crop_ratios)
    all_image_paths = list_images(image_dir)

    run_records = []
    run_metrics = []
    all_rows = []
    all_vis_payloads = []
    patch_size = None
    grid_h = None
    grid_w = None
    num_crop_queries = None
    num_source_images_per_run = []
    num_crop_queries_per_run = []
    crop_specs_sources = []
    effective_subset_seeds = []
    effective_crop_seeds = []

    for run_index, run_seed in enumerate(run_seeds):
        effective_subset_seed = int(args.subset_seed) + int(run_seed)
        effective_crop_seed = int(args.crop_seed) + int(run_seed)
        effective_subset_seeds.append(effective_subset_seed)
        effective_crop_seeds.append(effective_crop_seed)
        set_run_seed(run_seed)

        if args.selected_crop_specs_json:
            logger.info(f"Loading fixed correspondence specs from {args.selected_crop_specs_json}.")
            specs, spec_metadata = load_crop_specs(args.selected_crop_specs_json)
            crop_specs_source = args.selected_crop_specs_json
        else:
            write_specs_path = crop_specs_output_path(
                args.write_crop_specs_json,
                run_index,
                effective_subset_seed,
                effective_crop_seed,
                len(run_seeds),
            )
            if distributed.is_main_process():
                if write_specs_path and Path(write_specs_path).is_file():
                    logger.info(
                        "Loading cached per-run correspondence specs "
                        f"for effective_subset_seed={effective_subset_seed}, "
                        f"effective_crop_seed={effective_crop_seed} from {write_specs_path}."
                    )
                    specs, spec_metadata = load_crop_specs(write_specs_path)
                    crop_specs_source = write_specs_path
                else:
                    image_paths = choose_subset(all_image_paths, args.max_images, effective_subset_seed)
                    logger.info(
                        "Selecting correspondence images/crops "
                        f"with effective_subset_seed={effective_subset_seed}, "
                        f"effective_crop_seed={effective_crop_seed}."
                    )
                    specs = build_crop_specs(
                        image_paths=image_paths,
                        crop_ratios=crop_ratios,
                        crops_per_image=args.crops_per_image,
                        crop_seed=effective_crop_seed,
                        aspect_jitter=args.aspect_jitter,
                    )
                    spec_metadata = OrderedDict(
                        [
                            ("data_path", str(data_path)),
                            ("image_dir", str(image_dir)),
                            ("max_images", args.max_images),
                            ("subset_seed", args.subset_seed),
                            ("crop_seed", args.crop_seed),
                            ("run_index", run_index),
                            ("run_seed", int(run_seed)),
                            ("effective_subset_seed", effective_subset_seed),
                            ("effective_crop_seed", effective_crop_seed),
                            ("crop_ratios", crop_ratios),
                            ("crops_per_image", args.crops_per_image),
                            ("aspect_jitter", args.aspect_jitter),
                        ]
                    )
                    write_crop_specs(write_specs_path, specs, spec_metadata)
                    crop_specs_source = write_specs_path or "current_run_selection"
            else:
                specs = []
                spec_metadata = {}
                crop_specs_source = write_specs_path or "current_run_selection"
            specs, spec_metadata, crop_specs_source = broadcast_object_from_main(
                (specs, spec_metadata, crop_specs_source)
            )

        num_source_images_i = len(OrderedDict((spec["image_path"], None) for spec in specs))
        num_source_images_per_run.append(num_source_images_i)
        num_crop_queries_per_run.append(len(specs))
        crop_specs_sources.append(crop_specs_source)
        logger.info(
            f"Starting correspondence repeat {run_index + 1}/{len(run_seeds)} "
            f"with run_seed={run_seed}, effective_subset_seed={effective_subset_seed}, "
            f"effective_crop_seed={effective_crop_seed}; "
            f"num_source_images={num_source_images_i}, num_crop_queries={len(specs)}"
        )
        dataset = LocalGlobalCorrespondenceDataset(specs, imsize=args.imsize)
        metrics_i, rows_i, patch_size_i, grid_h_i, grid_w_i, vis_payloads_i = evaluate(
            model,
            autocast_dtype,
            dataset,
            specs,
            args,
            layer,
        )
        if patch_size is None:
            patch_size = patch_size_i
            grid_h = grid_h_i
            grid_w = grid_w_i
            num_crop_queries = len(specs)
        run_metrics.append(metrics_i)
        run_records.append(
            OrderedDict(
                [
                    ("run_index", run_index),
                    ("run_seed", int(run_seed)),
                    ("effective_subset_seed", effective_subset_seed),
                    ("effective_crop_seed", effective_crop_seed),
                    ("num_source_images", num_source_images_i),
                    ("num_crop_queries", len(specs)),
                    ("crop_specs_json", crop_specs_source),
                    ("metrics", metrics_i),
                ]
            )
        )
        all_rows.extend(
            rows_with_run_metadata(
                rows_i,
                run_index,
                run_seed,
                effective_subset_seed,
                effective_crop_seed,
                crop_specs_source,
            )
        )
        if run_index == 0:
            all_vis_payloads.extend(vis_payloads_i)

    metrics = summarize_metric_runs(run_metrics)

    results = OrderedDict()
    results["data_path"] = str(data_path)
    results["image_dir"] = str(image_dir)
    results["num_source_images"] = (
        num_source_images_per_run[0]
        if num_source_images_per_run and len(set(num_source_images_per_run)) == 1
        else None
    )
    results["num_source_images_per_run"] = num_source_images_per_run
    results["num_crop_queries"] = (
        num_crop_queries_per_run[0]
        if num_crop_queries_per_run and len(set(num_crop_queries_per_run)) == 1
        else num_crop_queries
    )
    results["num_crop_queries_per_run"] = num_crop_queries_per_run
    results["crop_ratios"] = crop_ratios
    results["crops_per_image"] = args.crops_per_image
    results["subset_seed"] = args.subset_seed
    results["subset_seed_mode"] = "fixed_crop_specs" if args.selected_crop_specs_json else "per_run"
    results["effective_subset_seeds"] = effective_subset_seeds
    results["crop_seed"] = args.crop_seed
    results["run_seeds"] = run_seeds
    results["effective_crop_seeds"] = effective_crop_seeds
    results["crop_specs_json"] = (
        crop_specs_sources[0]
        if crop_specs_sources and len(set(str(value) for value in crop_specs_sources)) == 1
        else crop_specs_sources
    )
    results["uses_fixed_crop_specs"] = bool(args.selected_crop_specs_json)
    results["num_runs"] = len(run_seeds)
    results["distributed_world_size"] = distributed.get_global_size()
    results["aspect_jitter"] = args.aspect_jitter
    results["imsize"] = args.imsize
    results["patch_size"] = patch_size
    results["grid_h"] = grid_h
    results["grid_w"] = grid_w
    results["layer"] = args.layer
    results["topk"] = args.topk
    results["ransac_iters"] = args.ransac_iters
    results["geometric_tolerance_patches"] = args.geometric_tolerance_patches
    results["metrics"] = metrics
    results["run_metrics"] = run_records

    if distributed.is_main_process():
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "results_local_global_correspondence.json"
        per_crop_path = output_dir / "per_local_global_correspondence.csv"
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        write_csv(per_crop_path, all_rows)
        save_crop_visualizations(args.output_dir, all_vis_payloads, args, grid_h, grid_w)
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
        logger.info(f"Wrote local-to-global correspondence results to {results_path}")

    if distributed.is_enabled():
        torch.distributed.barrier()
    return 0


if __name__ == "__main__":
    description = "Local-to-global correspondence: localize crops inside their source images"
    args_parser = get_args_parser(description=description)
    parsed_args = args_parser.parse_args()
    sys.exit(main(parsed_args))
