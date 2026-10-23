import logging
from functools import partial
from typing import Any

from omegaconf import OmegaConf
import torch

import jea.distributed as distributed
from jea.data import (
    DataAugmentation,
    SamplerType,
    collate_data_and_cast,
    ibotMaskingGenerator,
    make_data_loader,
    make_dataset,
)


logger = logging.getLogger("JEA")


def build_data_transform(cfg) -> DataAugmentation:
    return DataAugmentation(
        cfg.crops.aug_type,
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.global_crops_number,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        patch_size=cfg.student.patch_size,
    )


def build_collate_fn(cfg, inputs_dtype: torch.dtype):
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2

    use_patch_masks = cfg.ibot.use_loss and cfg.ibot.loss_weight > 0
    ibot_mask_generator = (
        ibotMaskingGenerator(
            input_size=(img_size // patch_size, img_size // patch_size),
            max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
        )
        if use_patch_masks
        else None
    )

    return partial(
        collate_data_and_cast,
        ibot_mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        ibot_mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        ibot_mask_generator=ibot_mask_generator,
        dtype=inputs_dtype,
    )


def build_train_dataset_and_loader(cfg, *, start_iter: int, inputs_dtype: torch.dtype):
    """Build the train dataset and its infinite distributed loader."""
    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=build_data_transform(cfg),
        target_transform=lambda _: (),
    )
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=cfg.train.seed,
        sampler_type=SamplerType.SHARDED_INFINITE,
        sampler_advance=start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        persistent_workers=cfg.train.get("persistent_workers", True),
        prefetch_factor=cfg.train.get("prefetch_factor", 2),
        collate_fn=build_collate_fn(cfg, inputs_dtype),
    )
    return dataset, data_loader


def resolve_epoch_length(cfg, dataset_size: int) -> int:
    """Infer the effective epoch length when the config leaves it unset."""
    epoch_length = cfg.train.OFFICIAL_EPOCH_LENGTH
    if epoch_length > 0:
        return epoch_length

    epoch_length = dataset_size // (cfg.train.batch_size_per_gpu * distributed.get_global_size())
    if epoch_length <= 0:
        raise ValueError("Dataset is smaller than one global batch; set train.OFFICIAL_EPOCH_LENGTH explicitly")
    logger.info(
        "OFFICIAL_EPOCH_LENGTH was unset. Using %s based on dataset size and global batch size.",
        epoch_length,
    )
    cfg.train.OFFICIAL_EPOCH_LENGTH = epoch_length
    return epoch_length


def flatten_config(cfg) -> dict[str, Any]:
    """Convert an OmegaConf config into a flat dict for experiment loggers."""
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_cfg, dict):
        return {}
    return _flatten_mapping(resolved_cfg)


def _flatten_mapping(mapping: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    flat_items: dict[str, Any] = {}
    for key, value in mapping.items():
        full_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flat_items.update(_flatten_mapping(value, full_key, sep=sep))
        else:
            flat_items[full_key] = value
    return flat_items
