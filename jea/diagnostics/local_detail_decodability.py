# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import math
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import transforms

import jea.distributed as distributed
from jea.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader, make_dataset
from jea.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, MaybeToTensor
from jea.eval.setup import get_args_parser as get_setup_args_parser
from jea.eval.setup import setup_and_build_model
from jea.diagnostics.visualization_utils import (
    ensure_dir,
    load_visualization_indices,
    make_panel_grid,
    patch_values_to_image,
    safe_name,
    save_visualization_manifest,
    save_visualization_indices,
    save_visualization_pdf,
    tensor_to_pil_image,
)


logger = logging.getLogger("JEA")


VALID_TARGETS = ("rgb_mean", "rgb_hist", "hog", "dct_energy", "lbp_hist", "raw_pca")


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
    parser.add_argument("--dataset", dest="dataset_str", default="ImageNet:split=TRAIN", type=str)
    parser.add_argument("--train-images", default=10000, type=int)
    parser.add_argument("--val-images", default=10000, type=int)
    parser.add_argument("--subset-seed", default=0, type=int)
    parser.add_argument(
        "--run-seeds",
        default="0",
        type=str,
        help=(
            "Comma-separated run seeds. Defaults to one run; pass multiple seeds "
            "to report mean/std across different train/val image subsets."
        ),
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=5, type=int)
    parser.add_argument(
        "--patches-per-image",
        default=0,
        type=int,
        help="Use at most this many patch tokens per image. <=0 uses all patches.",
    )
    parser.add_argument(
        "--layer",
        default="final",
        type=str,
        help="Patch-token layer to probe: 'final' or a 0-based transformer block index.",
    )
    parser.add_argument(
        "--targets",
        default="rgb_mean,rgb_hist,hog,dct_energy,lbp_hist,raw_pca",
        type=str,
        help="Comma-separated subset of: " + ",".join(VALID_TARGETS),
    )
    parser.add_argument("--resize-size", default=256, type=int)
    parser.add_argument("--crop-size", default=224, type=int)
    parser.add_argument("--rgb-hist-bins", default=8, type=int)
    parser.add_argument("--hog-bins", default=9, type=int)
    parser.add_argument("--dct-bands", default=6, type=int)
    parser.add_argument("--lbp-bins", default=16, type=int)
    parser.add_argument("--raw-pca-dim", default=64, type=int)
    parser.add_argument(
        "--ridge-alpha",
        default=1e-3,
        type=float,
        help="Ridge penalty for standardized features, solved on mean squared loss.",
    )
    parser.add_argument("--standardize-eps", default=1e-6, type=float)
    parser.add_argument("--log-every", default=20, type=int)
    parser.add_argument("--save-probe", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--visualization-samples", default=8, type=int)
    parser.add_argument("--visualization-targets", default="rgb_mean,hog,dct_energy,raw_pca", type=str)
    parser.add_argument("--visualization-dir", default="", type=str)
    parser.add_argument(
        "--visualization-indices-file",
        default="",
        type=str,
        help=(
            "Optional locked visualization sample-index file. Accepts txt, csv, "
            "json, or pt. CSV may be a previous manifest with `sample_index`."
        ),
    )
    parser.add_argument(
        "--write-visualization-indices-file",
        default="",
        type=str,
        help="Write the actually visualized sample indices for reuse by other eval scripts.",
    )
    return parser


def parse_targets(targets: str) -> List[str]:
    parsed = [target.strip() for target in targets.split(",") if target.strip()]
    if not parsed:
        raise ValueError("At least one target must be requested.")
    unknown = sorted(set(parsed) - set(VALID_TARGETS))
    if unknown:
        raise ValueError(f"Unknown local-detail targets: {unknown}. Valid targets: {VALID_TARGETS}")
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


def parse_layer(layer: str):
    if layer == "final":
        return "final"
    try:
        value = int(layer)
    except ValueError as exc:
        raise ValueError("--layer must be 'final' or a 0-based integer block index.") from exc
    if value < 0:
        raise ValueError("--layer block index must be non-negative.")
    return value


def layer_output_name(layer: str) -> str:
    if layer == "final":
        return "layer_final"
    return f"layer_{layer}"


def append_layer_to_output_dir(output_dir: str, layer: str) -> str:
    layer_dirname = layer_output_name(layer)
    normalized = os.path.normpath(output_dir)
    if os.path.basename(normalized) == layer_dirname:
        return output_dir
    return os.path.join(output_dir, layer_dirname)


class LocalDetailEvalTransform:
    """Return both normalized model input and raw [0, 1] pixels after the same crop."""

    def __init__(
        self,
        resize_size: int = 256,
        crop_size: int = 224,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
    ):
        self.spatial = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(crop_size),
                MaybeToTensor(),
            ]
        )
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image):
        raw = self.spatial(image)
        return {
            "image": self.normalize(raw.clone()),
            "raw": raw,
        }


def build_train_val_subsets(dataset, train_images: int, val_images: int, seed: int):
    total = train_images + val_images
    if train_images <= 0 or val_images <= 0:
        raise ValueError("--train-images and --val-images must be positive.")
    world_size = distributed.get_global_size()
    if train_images % world_size != 0 or val_images % world_size != 0:
        raise ValueError(
            "--train-images and --val-images must be divisible by the distributed world size "
            f"({world_size}) to avoid duplicated samples from DistributedSampler."
        )
    if len(dataset) < total:
        raise ValueError(f"Requested {total} images, but dataset only has {len(dataset)} samples.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:total]
    train_indices = indices[:train_images].tolist()
    val_indices = indices[train_images:].tolist()
    return Subset(dataset, train_indices), Subset(dataset, val_indices), indices


def make_indexed_loader(dataset, batch_size: int, num_workers: int):
    indexed_dataset = DatasetWithEnumeratedTargets(dataset)
    return make_data_loader(
        dataset=indexed_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=num_workers > 0,
    )


def get_patch_size(model) -> int:
    patch_size = getattr(model, "patch_size", None)
    if isinstance(patch_size, tuple):
        if patch_size[0] != patch_size[1]:
            raise ValueError(f"Only square patches are supported, got patch_size={patch_size}.")
        return int(patch_size[0])
    if patch_size is None:
        patch_embed = getattr(model, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", None)
        if isinstance(patch_size, tuple):
            if patch_size[0] != patch_size[1]:
                raise ValueError(f"Only square patches are supported, got patch_size={patch_size}.")
            return int(patch_size[0])
    return int(patch_size)


def distributed_sum_(tensor: torch.Tensor) -> torch.Tensor:
    if distributed.is_enabled():
        torch.distributed.all_reduce(tensor)
    return tensor


def choose_patch_indices(num_patches: int, patches_per_image: int, device: torch.device) -> torch.Tensor:
    if patches_per_image <= 0 or patches_per_image >= num_patches:
        return torch.arange(num_patches, device=device)
    return torch.linspace(0, num_patches - 1, steps=patches_per_image, device=device).long()


def choose_random_patch_indices(
    num_patches: int,
    patches_per_image: int,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if generator is None or patches_per_image <= 0 or patches_per_image >= num_patches:
        return choose_patch_indices(num_patches, patches_per_image, device)
    patch_idx = torch.randperm(num_patches, generator=generator)[:patches_per_image]
    return patch_idx.sort().values.to(device=device)


def select_patch_rows(
    values: torch.Tensor,
    patches_per_image: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    patch_idx = choose_random_patch_indices(values.shape[1], patches_per_image, values.device, generator)
    return values.index_select(1, patch_idx).reshape(-1, values.shape[-1])


def patchify_raw(raw: torch.Tensor, patch_size: int) -> torch.Tensor:
    patches = F.unfold(raw, kernel_size=patch_size, stride=patch_size)
    return patches.transpose(1, 2).contiguous()


def rgb_to_gray(raw: torch.Tensor) -> torch.Tensor:
    weights = raw.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    return (raw * weights).sum(dim=1, keepdim=True)


def dct_matrix(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    n = torch.arange(size, device=device, dtype=dtype)
    k = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
    mat = torch.cos(math.pi / size * (n + 0.5) * k)
    mat[0] = mat[0] * math.sqrt(1.0 / size)
    mat[1:] = mat[1:] * math.sqrt(2.0 / size)
    return mat


def dct_band_masks(size: int, bands: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    fy = torch.arange(size, device=device, dtype=dtype).view(size, 1)
    fx = torch.arange(size, device=device, dtype=dtype).view(1, size)
    radius = torch.sqrt(fx * fx + fy * fy)
    max_radius = radius.max().clamp_min(1.0)
    band_id = torch.clamp((radius / (max_radius + 1e-12) * bands).long(), max=bands - 1)
    band_id[0, 0] = 0
    masks = F.one_hot(band_id.reshape(-1), num_classes=bands).to(dtype)
    masks[0] = 0.0
    counts = masks.sum(dim=0, keepdim=True).clamp_min(1.0)
    return masks / counts


class LocalDetailTargetBuilder:
    def __init__(
        self,
        target_names: Sequence[str],
        patch_size: int,
        rgb_hist_bins: int = 8,
        hog_bins: int = 9,
        dct_bands: int = 6,
        lbp_bins: int = 16,
        raw_pca_dim: int = 64,
        raw_pca_mean: Optional[torch.Tensor] = None,
        raw_pca_components: Optional[torch.Tensor] = None,
    ):
        self.target_names = list(target_names)
        self.patch_size = patch_size
        self.rgb_hist_bins = rgb_hist_bins
        self.hog_bins = hog_bins
        self.dct_bands = dct_bands
        self.lbp_bins = lbp_bins
        self.raw_pca_dim = raw_pca_dim
        self.raw_pca_mean = raw_pca_mean
        self.raw_pca_components = raw_pca_components
        self.slices = self._build_slices()

    def _build_slices(self) -> "OrderedDict[str, Tuple[int, int]]":
        dims = {
            "rgb_mean": 3,
            "rgb_hist": 3 * self.rgb_hist_bins,
            "hog": self.hog_bins,
            "dct_energy": self.dct_bands,
            "lbp_hist": self.lbp_bins,
            "raw_pca": self.raw_pca_dim,
        }
        slices = OrderedDict()
        start = 0
        for name in self.target_names:
            dim = dims[name]
            slices[name] = (start, start + dim)
            start += dim
        return slices

    @property
    def target_dim(self) -> int:
        if not self.slices:
            return 0
        return next(reversed(self.slices.values()))[1]

    def compute(self, raw: torch.Tensor) -> torch.Tensor:
        target_parts = []
        if "rgb_mean" in self.target_names:
            target_parts.append(self._rgb_mean(raw))
        if "rgb_hist" in self.target_names:
            target_parts.append(self._rgb_hist(raw))
        if "hog" in self.target_names:
            target_parts.append(self._hog(raw))
        if "dct_energy" in self.target_names:
            target_parts.append(self._dct_energy(raw))
        if "lbp_hist" in self.target_names:
            target_parts.append(self._lbp_hist(raw))
        if "raw_pca" in self.target_names:
            target_parts.append(self._raw_pca(raw))
        return torch.cat(target_parts, dim=-1).float()

    def _rgb_mean(self, raw: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(raw, kernel_size=self.patch_size, stride=self.patch_size)
        return pooled.flatten(2).transpose(1, 2).contiguous()

    def _rgb_hist(self, raw: torch.Tensor) -> torch.Tensor:
        hist_parts = []
        for channel in range(raw.shape[1]):
            bin_idx = torch.clamp((raw[:, channel] * self.rgb_hist_bins).long(), max=self.rgb_hist_bins - 1)
            one_hot = F.one_hot(bin_idx, num_classes=self.rgb_hist_bins).permute(0, 3, 1, 2).float()
            hist = F.avg_pool2d(one_hot, kernel_size=self.patch_size, stride=self.patch_size)
            hist_parts.append(hist.flatten(2).transpose(1, 2))
        return torch.cat(hist_parts, dim=-1).contiguous()

    def _hog(self, raw: torch.Tensor) -> torch.Tensor:
        gray = rgb_to_gray(raw)
        kernel_x = raw.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
        kernel_y = raw.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0
        grad_x = F.conv2d(gray, kernel_x, padding=1)
        grad_y = F.conv2d(gray, kernel_y, padding=1)
        magnitude = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12).squeeze(1)
        orientation = torch.remainder(torch.atan2(grad_y, grad_x).squeeze(1), math.pi)

        centers = torch.arange(self.hog_bins, device=raw.device, dtype=raw.dtype) * (math.pi / self.hog_bins)
        dist = torch.abs(orientation.unsqueeze(1) - centers.view(1, self.hog_bins, 1, 1))
        dist = torch.minimum(dist, math.pi - dist)
        weights = torch.clamp(1.0 - dist / (math.pi / self.hog_bins), min=0.0)
        weighted = weights * magnitude.unsqueeze(1)
        pooled = F.avg_pool2d(weighted, kernel_size=self.patch_size, stride=self.patch_size)
        denom = F.avg_pool2d(magnitude.unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size)
        hog = pooled / denom.clamp_min(1e-12)
        return hog.flatten(2).transpose(1, 2).contiguous()

    def _dct_energy(self, raw: torch.Tensor) -> torch.Tensor:
        gray = rgb_to_gray(raw)
        patch_size = self.patch_size
        gray_patches = F.unfold(gray, kernel_size=patch_size, stride=patch_size)
        gray_patches = gray_patches.transpose(1, 2).reshape(raw.shape[0], -1, patch_size, patch_size)
        basis = dct_matrix(patch_size, raw.device, raw.dtype)
        coeff = torch.einsum("ux,bnxy,vy->bnuv", basis, gray_patches, basis)
        energy = coeff.pow(2).reshape(raw.shape[0], -1, patch_size * patch_size)
        masks = dct_band_masks(patch_size, self.dct_bands, raw.device, raw.dtype)
        band_energy = energy @ masks
        return torch.log1p(band_energy).contiguous()

    def _lbp_hist(self, raw: torch.Tensor) -> torch.Tensor:
        gray = rgb_to_gray(raw)
        padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        center = padded[:, :, 1:-1, 1:-1]
        neighbors = [
            padded[:, :, :-2, :-2],
            padded[:, :, :-2, 1:-1],
            padded[:, :, :-2, 2:],
            padded[:, :, 1:-1, 2:],
            padded[:, :, 2:, 2:],
            padded[:, :, 2:, 1:-1],
            padded[:, :, 2:, :-2],
            padded[:, :, 1:-1, :-2],
        ]
        code = torch.zeros_like(center, dtype=torch.long)
        for bit, neighbor in enumerate(neighbors):
            code = code + (neighbor >= center).long() * (1 << bit)
        coarse_code = torch.clamp(code.squeeze(1) // max(1, 256 // self.lbp_bins), max=self.lbp_bins - 1)
        one_hot = F.one_hot(coarse_code, num_classes=self.lbp_bins).permute(0, 3, 1, 2).float()
        hist = F.avg_pool2d(one_hot, kernel_size=self.patch_size, stride=self.patch_size)
        return hist.flatten(2).transpose(1, 2).contiguous()

    def _raw_pca(self, raw: torch.Tensor) -> torch.Tensor:
        if self.raw_pca_mean is None or self.raw_pca_components is None:
            raise RuntimeError("raw_pca target requested before PCA components were fitted.")
        patch_flat = patchify_raw(raw, self.patch_size)
        mean = self.raw_pca_mean.to(device=raw.device, dtype=patch_flat.dtype)
        components = self.raw_pca_components.to(device=raw.device, dtype=patch_flat.dtype)
        return (patch_flat - mean) @ components


def extract_patch_tokens(model, images: torch.Tensor, layer) -> torch.Tensor:
    if layer == "final":
        model_out = model(images, is_training=True)
        return model_out["x_norm_patchtokens"]
    return model.get_intermediate_layers(images, n=[layer], reshape=False, return_class_token=False, norm=True)[0]


def assert_token_target_alignment(tokens: torch.Tensor, targets: torch.Tensor):
    if tokens.shape[1] != targets.shape[1]:
        raise RuntimeError(
            f"Patch-token count ({tokens.shape[1]}) does not match target patch count ({targets.shape[1]}). "
            "Check crop size and model patch size."
        )


@torch.inference_mode()
def fit_raw_patch_pca(
    train_loader,
    patch_size: int,
    patches_per_image: int,
    raw_pca_dim: int,
    log_every: int,
    patch_generator: Optional[torch.Generator] = None,
):
    device = torch.device("cuda", torch.cuda.current_device())
    raw_dim = 3 * patch_size * patch_size
    raw_sum = torch.zeros(raw_dim, device=device, dtype=torch.float64)
    raw_xtx = torch.zeros(raw_dim, raw_dim, device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)

    for step, (samples, _) in enumerate(train_loader):
        raw = samples["raw"].cuda(non_blocking=True)
        patch_flat = patchify_raw(raw, patch_size)
        patch_flat = select_patch_rows(patch_flat, patches_per_image, patch_generator).double()
        raw_sum += patch_flat.sum(dim=0)
        raw_xtx += patch_flat.t().matmul(patch_flat)
        count += patch_flat.shape[0]
        if distributed.is_main_process() and log_every > 0 and (step + 1) % log_every == 0:
            logger.info(f"raw_pca pass: processed {int(count.item())} local patches")

    distributed_sum_(raw_sum)
    distributed_sum_(raw_xtx)
    distributed_sum_(count)

    mean = raw_sum / count.clamp_min(1.0)
    cov = raw_xtx / count.clamp_min(1.0) - torch.outer(mean, mean)
    cov = (cov + cov.t()) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = torch.clamp(eigvals, min=0.0)
    pca_dim = min(raw_pca_dim, raw_dim)
    components = eigvecs[:, -pca_dim:].flip(dims=(1,)).contiguous()
    explained_variance = eigvals[-pca_dim:].flip(dims=(0,)).contiguous()
    total_variance = eigvals.sum().clamp_min(1e-12)
    return {
        "mean": mean.float(),
        "components": components.float(),
        "explained_variance": explained_variance.float(),
        "explained_variance_ratio": (explained_variance / total_variance).float(),
        "fitted_patch_count": int(count.item()),
        "raw_dim": raw_dim,
        "pca_dim": pca_dim,
    }


def empty_probe_stats(x_dim: int, y_dim: int, device: torch.device):
    return {
        "count": torch.zeros((), device=device, dtype=torch.float64),
        "x_sum": torch.zeros(x_dim, device=device, dtype=torch.float64),
        "x_sumsq": torch.zeros(x_dim, device=device, dtype=torch.float64),
        "y_sum": torch.zeros(y_dim, device=device, dtype=torch.float64),
        "y_sumsq": torch.zeros(y_dim, device=device, dtype=torch.float64),
        "xtx": torch.zeros(x_dim, x_dim, device=device, dtype=torch.float64),
        "xty": torch.zeros(x_dim, y_dim, device=device, dtype=torch.float64),
    }


def update_probe_stats(stats, x: torch.Tensor, y: torch.Tensor):
    x = x.double()
    y = y.double()
    stats["count"] += x.shape[0]
    stats["x_sum"] += x.sum(dim=0)
    stats["x_sumsq"] += (x * x).sum(dim=0)
    stats["y_sum"] += y.sum(dim=0)
    stats["y_sumsq"] += (y * y).sum(dim=0)
    stats["xtx"] += x.t().matmul(x)
    stats["xty"] += x.t().matmul(y)


def all_reduce_probe_stats(stats):
    for value in stats.values():
        distributed_sum_(value)


@torch.inference_mode()
def collect_probe_stats(
    model,
    train_loader,
    target_builder: LocalDetailTargetBuilder,
    layer,
    patches_per_image: int,
    autocast_dtype: torch.dtype,
    log_every: int,
    patch_generator: Optional[torch.Generator] = None,
):
    device = torch.device("cuda", torch.cuda.current_device())
    stats = None
    patch_count_local = 0

    for step, (samples, _) in enumerate(train_loader):
        images = samples["image"].cuda(non_blocking=True)
        raw = samples["raw"].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            tokens = extract_patch_tokens(model, images, layer).float()
        targets = target_builder.compute(raw)
        assert_token_target_alignment(tokens, targets)

        patch_idx = choose_random_patch_indices(tokens.shape[1], patches_per_image, tokens.device, patch_generator)
        x = tokens.index_select(1, patch_idx).reshape(-1, tokens.shape[-1])
        y = targets.index_select(1, patch_idx).reshape(-1, targets.shape[-1])
        if stats is None:
            stats = empty_probe_stats(x.shape[-1], y.shape[-1], device)
        update_probe_stats(stats, x, y)
        patch_count_local += x.shape[0]
        if distributed.is_main_process() and log_every > 0 and (step + 1) % log_every == 0:
            logger.info(f"probe-stat pass: processed {patch_count_local} local patches")

    if stats is None:
        raise RuntimeError("No train batches were processed.")
    all_reduce_probe_stats(stats)
    return stats


def solve_ridge_probe(stats, ridge_alpha: float, eps: float):
    count = stats["count"].clamp_min(1.0)
    x_mean = stats["x_sum"] / count
    y_mean = stats["y_sum"] / count
    x_var = stats["x_sumsq"] / count - x_mean * x_mean
    x_scale = torch.sqrt(torch.clamp(x_var, min=eps))

    xtx_centered = stats["xtx"] - count * torch.outer(x_mean, x_mean)
    xty_centered = stats["xty"] - count * torch.outer(x_mean, y_mean)
    gram = xtx_centered / (x_scale[:, None] * x_scale[None, :]) / count
    rhs = xty_centered / x_scale[:, None] / count

    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    weight = torch.linalg.solve(gram + ridge_alpha * eye, rhs)
    return {
        "weight": weight,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "train_patch_count": int(stats["count"].item()),
        "feature_dim": int(gram.shape[0]),
        "target_dim": int(rhs.shape[1]),
    }


def make_val_accumulators(slices: "OrderedDict[str, Tuple[int, int]]", device: torch.device):
    accum = {}
    for name, (start, end) in slices.items():
        dim = end - start
        accum[name] = {
            "count": torch.zeros((), device=device, dtype=torch.float64),
            "sse": torch.zeros(dim, device=device, dtype=torch.float64),
            "sum": torch.zeros(dim, device=device, dtype=torch.float64),
            "sumsq": torch.zeros(dim, device=device, dtype=torch.float64),
        }
    return accum


def update_val_accumulators(accum, slices, pred: torch.Tensor, target: torch.Tensor):
    pred = pred.double()
    target = target.double()
    for name, (start, end) in slices.items():
        pred_group = pred[:, start:end]
        target_group = target[:, start:end]
        error = pred_group - target_group
        accum[name]["count"] += target_group.shape[0]
        accum[name]["sse"] += (error * error).sum(dim=0)
        accum[name]["sum"] += target_group.sum(dim=0)
        accum[name]["sumsq"] += (target_group * target_group).sum(dim=0)


def all_reduce_val_accumulators(accum):
    for group in accum.values():
        for value in group.values():
            distributed_sum_(value)


@torch.inference_mode()
def evaluate_probe(
    model,
    val_loader,
    target_builder: LocalDetailTargetBuilder,
    probe,
    layer,
    patches_per_image: int,
    autocast_dtype: torch.dtype,
    log_every: int,
    patch_generator: Optional[torch.Generator] = None,
):
    device = torch.device("cuda", torch.cuda.current_device())
    accum = make_val_accumulators(target_builder.slices, device)
    patch_count_local = 0

    weight = probe["weight"]
    x_mean = probe["x_mean"]
    x_scale = probe["x_scale"]
    y_mean = probe["y_mean"]

    for step, (samples, _) in enumerate(val_loader):
        images = samples["image"].cuda(non_blocking=True)
        raw = samples["raw"].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            tokens = extract_patch_tokens(model, images, layer).float()
        targets = target_builder.compute(raw)
        assert_token_target_alignment(tokens, targets)

        patch_idx = choose_random_patch_indices(tokens.shape[1], patches_per_image, tokens.device, patch_generator)
        x = tokens.index_select(1, patch_idx).reshape(-1, tokens.shape[-1]).double()
        y = targets.index_select(1, patch_idx).reshape(-1, targets.shape[-1]).double()
        pred = ((x - x_mean) / x_scale).matmul(weight) + y_mean
        update_val_accumulators(accum, target_builder.slices, pred, y)
        patch_count_local += x.shape[0]
        if distributed.is_main_process() and log_every > 0 and (step + 1) % log_every == 0:
            logger.info(f"val pass: processed {patch_count_local} local patches")

    all_reduce_val_accumulators(accum)
    return accum


def summarize_val_accumulators(accum, eps: float) -> Dict[str, float]:
    metrics = OrderedDict()
    total_sse = 0.0
    total_sst = 0.0
    for name, group in accum.items():
        count = group["count"].clamp_min(1.0)
        sse = group["sse"]
        sum_y = group["sum"]
        sumsq_y = group["sumsq"]
        sst = torch.clamp(sumsq_y - sum_y * sum_y / count, min=eps)
        r2_dims = 1.0 - sse / sst
        mse_dims = sse / count
        target_std_dims = torch.sqrt(sst / count)

        metrics[f"{name}/r2_mean"] = float(r2_dims.mean().item())
        metrics[f"{name}/r2_min"] = float(r2_dims.min().item())
        metrics[f"{name}/r2_max"] = float(r2_dims.max().item())
        metrics[f"{name}/r2_variance_weighted"] = float(1.0 - sse.sum().item() / sst.sum().item())
        metrics[f"{name}/mse_mean"] = float(mse_dims.mean().item())
        metrics[f"{name}/target_std_mean"] = float(target_std_dims.mean().item())
        metrics[f"{name}/num_dims"] = int(sse.numel())
        metrics[f"{name}/num_val_patches"] = int(count.item())
        total_sse += float(sse.sum().item())
        total_sst += float(sst.sum().item())

    metrics["all/r2_variance_weighted"] = float(1.0 - total_sse / max(total_sst, eps))
    metrics["all/sse"] = total_sse
    metrics["all/sst"] = total_sst
    return metrics


def parse_visualization_targets(value: str, available: Sequence[str]) -> List[str]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        return []
    available_set = set(available)
    return [name for name in requested if name in available_set]


def predict_targets_from_tokens(tokens: torch.Tensor, probe: Dict) -> torch.Tensor:
    weight = probe["weight"]
    x_mean = probe["x_mean"]
    x_scale = probe["x_scale"]
    y_mean = probe["y_mean"]
    return ((tokens.double() - x_mean) / x_scale).matmul(weight) + y_mean


def target_group_to_grid_image(values: torch.Tensor, target_name: str, grid_h: int, grid_w: int, size: Tuple[int, int]):
    values = values.detach().cpu().float()
    if target_name == "rgb_mean" and values.shape[-1] >= 3:
        rgb = values[:, :3].view(grid_h, grid_w, 3).clamp(0.0, 1.0)
        array = (rgb.numpy() * 255.0 + 0.5).astype("uint8")
        from PIL import Image

        return Image.fromarray(array, mode="RGB").resize(size, Image.NEAREST)
    scalar = values.norm(dim=-1).view(grid_h, grid_w)
    return patch_values_to_image(scalar, grid_h, grid_w, size)


@torch.inference_mode()
def save_local_detail_visualizations(
    model,
    val_loader,
    target_builder: LocalDetailTargetBuilder,
    probe: Dict,
    layer,
    autocast_dtype: torch.dtype,
    output_dir: str,
    args,
):
    locked_indices = []
    if args.visualization_indices_file:
        locked_indices = load_visualization_indices(args.visualization_indices_file)
    locked_index_set = set(int(idx) for idx in locked_indices)
    target_sample_count = len(locked_index_set) if locked_index_set else args.visualization_samples
    if not args.save_visualizations or target_sample_count <= 0:
        return
    write_outputs = distributed.is_main_process()
    vis_targets = parse_visualization_targets(args.visualization_targets, target_builder.slices.keys())
    if not vis_targets:
        if write_outputs:
            logger.info("No requested local-detail visualization targets are available in this run.")
        return
    vis_dir = None
    if write_outputs:
        vis_dir = ensure_dir(args.visualization_dir or os.path.join(output_dir, "visualizations", "local_detail"))
    manifest = []
    saved_sample_indices = []
    saved = 0
    for _step, (samples, indices) in enumerate(val_loader):
        images = samples["image"].cuda(non_blocking=True)
        raw = samples["raw"].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            tokens = extract_patch_tokens(model, images, layer).float()
        targets = target_builder.compute(raw).float()
        assert_token_target_alignment(tokens, targets)
        pred = predict_targets_from_tokens(tokens.reshape(-1, tokens.shape[-1]), probe).float()
        pred = pred.view(tokens.shape[0], tokens.shape[1], -1)

        grid_h = raw.shape[-2] // target_builder.patch_size
        grid_w = raw.shape[-1] // target_builder.patch_size
        for batch_idx in range(raw.shape[0]):
            if saved >= target_sample_count:
                if write_outputs:
                    save_visualization_manifest(vis_dir, manifest)
                    if args.write_visualization_indices_file:
                        save_visualization_indices(args.write_visualization_indices_file, saved_sample_indices)
                    logger.info(f"Wrote {len(manifest)} local-detail visualizations to {vis_dir}")
                return
            sample_id = int(indices[batch_idx].item()) if torch.is_tensor(indices) else saved
            if locked_index_set and sample_id not in locked_index_set:
                continue
            if write_outputs:
                base = tensor_to_pil_image(raw[batch_idx])
                for target_name in vis_targets:
                    start, end = target_builder.slices[target_name]
                    gt_values = targets[batch_idx, :, start:end]
                    pred_values = pred[batch_idx, :, start:end]
                    err_values = (pred_values - gt_values).pow(2).mean(dim=-1)
                    panels = [
                        base,
                        target_group_to_grid_image(gt_values, target_name, grid_h, grid_w, base.size),
                        target_group_to_grid_image(pred_values, target_name, grid_h, grid_w, base.size),
                        patch_values_to_image(err_values.view(grid_h, grid_w), grid_h, grid_w, base.size),
                    ]
                    labels = ["image", f"GT {target_name}", f"pred {target_name}", "squared error"]
                    out_name = f"{saved:04d}_sample{sample_id}_{safe_name(target_name)}.pdf"
                    out_path = vis_dir / out_name
                    save_visualization_pdf(make_panel_grid(panels, labels, columns=4), out_path)
                    manifest.append(
                        {
                            "sample_rank": saved,
                            "sample_index": sample_id,
                            "target": target_name,
                            "path": str(out_path),
                        }
                    )
                saved_sample_indices.append(sample_id)
            saved += 1
    if write_outputs:
        save_visualization_manifest(vis_dir, manifest)
        if args.write_visualization_indices_file:
            save_visualization_indices(args.write_visualization_indices_file, saved_sample_indices)
        logger.info(f"Wrote {len(manifest)} local-detail visualizations to {vis_dir}")


def prepare_dataloaders(args, subset_seed: Optional[int] = None):
    transform = LocalDetailEvalTransform(resize_size=args.resize_size, crop_size=args.crop_size)
    dataset = make_dataset(dataset_str=args.dataset_str, transform=transform)
    seed = args.subset_seed if subset_seed is None else subset_seed
    train_dataset, val_dataset, subset_indices = build_train_val_subsets(
        dataset,
        train_images=args.train_images,
        val_images=args.val_images,
        seed=seed,
    )
    train_loader = make_indexed_loader(train_dataset, args.batch_size, args.num_workers)
    val_loader = make_indexed_loader(val_dataset, args.batch_size, args.num_workers)
    return train_loader, val_loader, subset_indices


def save_results(output_dir: str, results: Dict, run_subset_indices: Sequence[Dict], probe: Dict, pca_info: Optional[Dict], args):
    if not distributed.is_main_process():
        return
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results_local_detail_decodability.json")
    sampled_indices_files = []
    for item in run_subset_indices:
        indices_path = os.path.join(
            output_dir,
            "sampled_indices_"
            f"train{args.train_images}_val{args.val_images}_"
            f"seed{item['effective_subset_seed']}_run{item['run_index']}.pt",
        )
        torch.save(item["indices"].cpu(), indices_path)
        sampled_indices_files.append(
            {
                "run_index": item["run_index"],
                "run_seed": item["run_seed"],
                "effective_subset_seed": item["effective_subset_seed"],
                "file": indices_path,
            }
        )
    results["sampled_indices_files"] = sampled_indices_files
    if sampled_indices_files:
        results["sampled_indices_file"] = sampled_indices_files[0]["file"]
        for item in results.get("run_metrics", []):
            for saved in sampled_indices_files:
                if int(item.get("run_index", -1)) == int(saved["run_index"]):
                    item["sampled_indices_file"] = saved["file"]
                    break

    if args.save_probe:
        probe_path = os.path.join(output_dir, "ridge_probe.pt")
        save_payload = {
            "probe": {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in probe.items()},
            "target_slices": dict(results["target_slices"]),
            "args": vars(args),
        }
        if pca_info is not None:
            save_payload["raw_pca"] = {
                key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in pca_info.items()
            }
        torch.save(save_payload, probe_path)
        results["probe_file"] = probe_path

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote local-detail decodability results to {results_path}")


def main(args):
    targets = parse_targets(args.targets)
    layer = parse_layer(args.layer)
    args.output_dir = append_layer_to_output_dir(args.output_dir, args.layer)
    run_seeds = parse_int_list(args.run_seeds)
    model, autocast_dtype = setup_and_build_model(args)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    patch_size = get_patch_size(model)

    run_metrics = []
    run_records = []
    run_subset_indices = []
    first_probe = None
    first_pca_info = None
    first_target_builder = None
    first_val_loader = None

    for run_index, run_seed in enumerate(run_seeds):
        effective_subset_seed = int(args.subset_seed) + int(run_seed)
        set_run_seed(effective_subset_seed)
        patch_generator = torch.Generator()
        patch_generator.manual_seed(effective_subset_seed)
        logger.info(
            f"Starting local-detail repeat {run_index + 1}/{len(run_seeds)} "
            f"with run_seed={run_seed}, effective_subset_seed={effective_subset_seed}"
        )
        train_loader, val_loader, subset_indices = prepare_dataloaders(args, subset_seed=effective_subset_seed)
        run_subset_indices.append(
            {
                "run_index": run_index,
                "run_seed": int(run_seed),
                "effective_subset_seed": effective_subset_seed,
                "indices": subset_indices,
            }
        )

        pca_info = None
        raw_pca_mean = None
        raw_pca_components = None
        raw_pca_dim = args.raw_pca_dim
        if "raw_pca" in targets:
            logger.info("Fitting raw-patch PCA on the train split.")
            pca_info = fit_raw_patch_pca(
                train_loader=train_loader,
                patch_size=patch_size,
                patches_per_image=args.patches_per_image,
                raw_pca_dim=args.raw_pca_dim,
                log_every=args.log_every,
                patch_generator=patch_generator,
            )
            raw_pca_mean = pca_info["mean"]
            raw_pca_components = pca_info["components"]
            raw_pca_dim = int(pca_info["pca_dim"])
            if distributed.is_main_process():
                logger.info(
                    "raw_pca explained variance ratio sum: "
                    f"{float(pca_info['explained_variance_ratio'].sum().item())}"
                )

        target_builder = LocalDetailTargetBuilder(
            target_names=targets,
            patch_size=patch_size,
            rgb_hist_bins=args.rgb_hist_bins,
            hog_bins=args.hog_bins,
            dct_bands=args.dct_bands,
            lbp_bins=args.lbp_bins,
            raw_pca_dim=raw_pca_dim,
            raw_pca_mean=raw_pca_mean,
            raw_pca_components=raw_pca_components,
        )

        logger.info("Collecting ridge sufficient statistics from frozen patch tokens.")
        stats = collect_probe_stats(
            model=model,
            train_loader=train_loader,
            target_builder=target_builder,
            layer=layer,
            patches_per_image=args.patches_per_image,
            autocast_dtype=autocast_dtype,
            log_every=args.log_every,
            patch_generator=patch_generator,
        )
        probe = solve_ridge_probe(stats, ridge_alpha=args.ridge_alpha, eps=args.standardize_eps)

        logger.info("Evaluating ridge probe on held-out images.")
        val_accum = evaluate_probe(
            model=model,
            val_loader=val_loader,
            target_builder=target_builder,
            probe=probe,
            layer=layer,
            patches_per_image=args.patches_per_image,
            autocast_dtype=autocast_dtype,
            log_every=args.log_every,
            patch_generator=patch_generator,
        )
        metrics_i = summarize_val_accumulators(val_accum, eps=args.standardize_eps)

        if first_probe is None:
            first_probe = probe
            first_pca_info = pca_info
            first_target_builder = target_builder
            first_val_loader = val_loader

        run_record = OrderedDict()
        run_record["run_index"] = run_index
        run_record["run_seed"] = int(run_seed)
        run_record["effective_subset_seed"] = effective_subset_seed
        run_record["feature_dim"] = probe["feature_dim"]
        run_record["target_dim"] = probe["target_dim"]
        run_record["train_patch_count"] = probe["train_patch_count"]
        if pca_info is not None:
            run_record["raw_pca_dim"] = int(pca_info["pca_dim"])
            run_record["raw_pca_fitted_patch_count"] = int(pca_info["fitted_patch_count"])
            run_record["raw_pca_explained_variance_ratio_sum"] = float(
                pca_info["explained_variance_ratio"].sum().item()
            )
        run_record["metrics"] = metrics_i
        run_records.append(run_record)
        run_metrics.append(metrics_i)

    metrics = summarize_metric_runs(run_metrics)
    probe = first_probe
    pca_info = first_pca_info
    target_builder = first_target_builder

    results = OrderedDict()
    results["dataset"] = args.dataset_str
    results["train_images"] = args.train_images
    results["val_images"] = args.val_images
    results["subset_seed"] = args.subset_seed
    results["subset_seed_mode"] = "per_run"
    results["effective_subset_seeds"] = [item["effective_subset_seed"] for item in run_subset_indices]
    results["run_seeds"] = run_seeds
    results["num_runs"] = len(run_seeds)
    results["patch_size"] = patch_size
    results["patches_per_image"] = args.patches_per_image
    results["layer"] = args.layer
    results["layer_output_dir"] = args.output_dir
    results["save_visualizations"] = bool(args.save_visualizations)
    results["visualization_samples"] = args.visualization_samples
    results["visualization_targets"] = args.visualization_targets
    results["visualization_dir"] = args.visualization_dir or os.path.join(
        args.output_dir, "visualizations", "local_detail"
    )
    results["visualization_indices_file"] = args.visualization_indices_file
    results["write_visualization_indices_file"] = args.write_visualization_indices_file
    results["targets"] = targets
    results["target_slices"] = {name: [start, end] for name, (start, end) in target_builder.slices.items()}
    results["ridge_alpha"] = args.ridge_alpha
    results["feature_dim"] = probe["feature_dim"]
    results["target_dim"] = probe["target_dim"]
    results["train_patch_count"] = probe["train_patch_count"]
    if pca_info is not None:
        results["raw_pca_dim"] = int(pca_info["pca_dim"])
        results["raw_pca_fitted_patch_count"] = int(pca_info["fitted_patch_count"])
        results["raw_pca_explained_variance_ratio_sum"] = float(
            pca_info["explained_variance_ratio"].sum().item()
        )
    results["metrics"] = metrics
    results["run_metrics"] = run_records

    if distributed.is_main_process():
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
    save_results(args.output_dir, results, run_subset_indices, probe, pca_info, args)
    save_local_detail_visualizations(
        model=model,
        val_loader=first_val_loader,
        target_builder=target_builder,
        probe=probe,
        layer=layer,
        autocast_dtype=autocast_dtype,
        output_dir=args.output_dir,
        args=args,
    )

    if distributed.is_enabled():
        torch.distributed.barrier()
    return 0


if __name__ == "__main__":
    description = "Linear decodability of local low-level patch attributes from frozen patch tokens"
    args_parser = get_args_parser(description=description)
    parsed_args = args_parser.parse_args()
    sys.exit(main(parsed_args))
