# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Subset

import jea.distributed as distributed
from jea.data import (
    DatasetWithEnumeratedTargets,
    SamplerType,
    make_data_loader,
    make_dataset,
)
from jea.data.transforms import make_classification_eval_transform
from jea.eval.setup import get_args_parser as get_setup_args_parser
from jea.eval.setup import setup_and_build_model
from jea.eval.utils import all_gather_and_flatten

logger = logging.getLogger("JEA")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_str",
        type=str,
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        help="Number of images to sample deterministically from the dataset.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        help="Random seed used to select the deterministic image subset.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the final feature matrix on CPU.",
    )
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Whether to save the final feature matrix to output_dir.",
    )
    parser.set_defaults(
        dataset_str="ImageNet:split=TRAIN",
        batch_size=256,
        num_images=10000,
        subset_seed=0,
        num_workers=5,
        gather_on_cpu=False,
        save_features=False,
    )
    return parser


@torch.no_grad()
def patch_effective_rank_values(
    patch_tok: torch.Tensor, epsilon: float = 1e-7
) -> torch.Tensor:
    """Compute per-image normalized effective rank from patch tokens."""
    patch_tok = patch_tok.float()
    sing_vals = torch.linalg.svdvals(patch_tok)
    prob = sing_vals / sing_vals.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    prob = prob.clamp_min(epsilon)
    entropy = -torch.sum(prob * torch.log(prob), dim=-1)
    num_patches = patch_tok.shape[1]
    return torch.exp(entropy) / num_patches


def build_deterministic_subset(dataset, num_images: int, seed: int):
    """Sample a fixed random subset that is reproducible across runs."""
    if num_images <= 0:
        raise ValueError(f"num_images must be > 0, got {num_images}")
    if len(dataset) < num_images:
        raise ValueError(
            f"Requested {num_images} images, but dataset only has {len(dataset)} samples."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    subset_indices = torch.randperm(len(dataset), generator=generator)[:num_images]
    subset = Subset(dataset, subset_indices.tolist())
    return subset, subset_indices.cpu()


def _compute_mean_std(values: torch.Tensor) -> Tuple[float, float]:
    """Return population mean and standard deviation for scalar values."""
    values = values.double()
    return values.mean().item(), values.std(unbiased=False).item()


@torch.inference_mode()
def extract_feature_matrix_and_metrics(
    model, dataset, batch_size, num_workers, gather_on_cpu=False
):
    """Gather one uncentered patch-rank value per sampled image, without padding duplicates."""
    indexed = DatasetWithEnumeratedTargets(dataset)
    loader = make_data_loader(
        dataset=indexed,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=num_workers > 0,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    storage = torch.device("cpu") if gather_on_cpu else device
    ranks = torch.zeros(len(indexed), device=storage)
    seen = torch.zeros(len(indexed), device=storage, dtype=torch.bool)
    features = None
    for images, (indices, _) in loader:
        output = model(images.cuda(non_blocking=True), is_training=True)
        local_features = output["x_norm_clstoken"].float()
        local_ranks = patch_effective_rank_values(output["x_norm_patchtokens"])
        if features is None:
            features = torch.zeros(
                len(indexed), local_features.shape[-1], device=storage
            )
        indices = all_gather_and_flatten(indices.cuda(non_blocking=True)).to(storage)
        features.index_copy_(
            0, indices, all_gather_and_flatten(local_features).to(storage)
        )
        ranks.index_copy_(0, indices, all_gather_and_flatten(local_ranks).to(storage))
        seen[indices] = True
    if features is None or not seen.all():
        raise RuntimeError("Not all sampled images were processed")
    mean, std = _compute_mean_std(ranks)
    return features, {
        "patch_effective_rank_mean": mean,
        "patch_effective_rank_std": std,
    }


def eval_metrics_with_model(
    model,
    output_dir: str,
    dataset_str: str = "ImageNet:split=TRAIN",
    batch_size: int = 256,
    num_images: int = 10000,
    subset_seed: int = 0,
    num_workers: int = 5,
    gather_on_cpu: bool = False,
    transform=None,
    save_features: bool = False,
):
    transform = transform or make_classification_eval_transform()
    dataset = make_dataset(
        dataset_str=dataset_str,
        transform=transform,
    )
    subset_dataset, subset_indices = build_deterministic_subset(
        dataset, num_images, subset_seed
    )

    logger.info(
        f"Using a deterministic random subset of {len(subset_dataset)} images "
        f"from {dataset_str} with subset_seed={subset_seed}."
    )

    feat_mat, metric_results = extract_feature_matrix_and_metrics(
        model=model,
        dataset=subset_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        gather_on_cpu=gather_on_cpu,
    )

    metric_results["num_images"] = len(subset_dataset)
    metric_results["feature_dim"] = feat_mat.shape[1]
    metric_results["dataset"] = dataset_str
    metric_results["subset_seed"] = subset_seed

    indices_file_path = os.path.join(
        output_dir, f"sampled_indices_{num_images}_seed{subset_seed}.pt"
    )
    metrics_file_path = os.path.join(output_dir, "results_cal_metrics.json")
    features_file_path = os.path.join(
        output_dir, f"sampled_features_{num_images}_seed{subset_seed}.pt"
    )

    if distributed.is_main_process():
        torch.save(subset_indices, indices_file_path)
        metric_results["sampled_indices_file"] = indices_file_path

        if save_features:
            torch.save(feat_mat.cpu(), features_file_path)
            metric_results["features_file"] = features_file_path

        with open(metrics_file_path, "w") as f:
            json.dump(metric_results, f, indent=2)

        for key, value in metric_results.items():
            logger.info(f"{key}: {value}")

    if distributed.is_enabled():
        torch.distributed.barrier()

    return metric_results


def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        eval_metrics_with_model(
            model=model,
            output_dir=args.output_dir,
            dataset_str=args.dataset_str,
            batch_size=args.batch_size,
            num_images=args.num_images,
            subset_seed=args.subset_seed,
            num_workers=args.num_workers,
            gather_on_cpu=args.gather_on_cpu,
            transform=None,
            save_features=args.save_features,
        )
    return 0


if __name__ == "__main__":
    description = "Calculate metrics on a deterministic sampled subset"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
