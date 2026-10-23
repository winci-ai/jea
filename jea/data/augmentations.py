# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
from torchvision import transforms
from torchvision.transforms import functional as F

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)


logger = logging.getLogger("JEA")


class DataAugmentation(object):
    def __init__(
        self,
        aug_type,
        global_crops_scale,
        local_crops_scale,
        global_crops_number,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        patch_size=16,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.patch_size = patch_size

        if global_crops_number != 2:
            raise ValueError("DataAugmentation currently expects exactly 2 global crops.")
        if global_crops_size % patch_size != 0 or local_crops_size % patch_size != 0:
            raise ValueError("global_crops_size and local_crops_size must be divisible by patch_size.")
        if local_crops_size > global_crops_size:
            raise ValueError("local_crops_size must not exceed global_crops_size.")

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"global_crops_number: {global_crops_number}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"patch_size: {patch_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        if aug_type != 'asym':
            raise ValueError("The paper uses asymmetric DINO augmentation (aug_type=asym)")
        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])

        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def _sample_global_geometry(self, image):
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image,
            scale=self.global_crops_scale,
            ratio=(3.0 / 4.0, 4.0 / 3.0),
        )
        crop = F.resized_crop(
            image,
            i,
            j,
            h,
            w,
            [self.global_crops_size, self.global_crops_size],
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        if torch.rand(1) < 0.5:
            crop = F.hflip(crop)
        return crop

    def __call__(self, image):
        output = {}

        # Sample global geometry once, then apply independent photometric views.
        global_geometry_crops = [self._sample_global_geometry(image) for _ in range(self.global_crops_number)]
        global_crops = [
            self.global_transfo1(global_geometry_crops[0]),
            self.global_transfo2(global_geometry_crops[1]),
        ]

        output["global_crops"] = global_crops

        # global crops for teacher:
        output["global_crops_teacher"] = global_crops

        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]

        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
