# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


def collate_data_and_cast(
    samples_list,
    ibot_mask_ratio_tuple,
    ibot_mask_probability,
    dtype,
    n_tokens=None,
    ibot_mask_generator=None,
):
    """Collate multi-crop samples and build the iBOT masking metadata once per batch."""

    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)

    if ibot_mask_generator is not None:
        N = n_tokens
        n_samples_masked = int(B * ibot_mask_probability)
        probs = torch.linspace(*ibot_mask_ratio_tuple, n_samples_masked + 1)
        upperbound = 0
        ibot_masks_list = []
        for i in range(0, n_samples_masked):
            prob_min = probs[i]
            prob_max = probs[i + 1]
            ibot_masks_list.append(torch.BoolTensor(ibot_mask_generator(int(N * random.uniform(prob_min, prob_max)))))
            upperbound += int(N * prob_max)

        for i in range(n_samples_masked, B):
            ibot_masks_list.append(torch.BoolTensor(ibot_mask_generator(0)))

        random.shuffle(ibot_masks_list)

        ibot_collated_masks = torch.stack(ibot_masks_list).flatten(1)
        ibot_mask_indices_list = ibot_collated_masks.flatten().nonzero().flatten()

        masks_weight = (1 / ibot_collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(ibot_collated_masks)[ibot_collated_masks]
        n_masked_patches = torch.full((1,), fill_value=ibot_mask_indices_list.shape[0], dtype=torch.long)
        
    else:
        ibot_collated_masks = ibot_mask_indices_list = masks_weight = upperbound = n_masked_patches = None

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "ibot_collated_masks": ibot_collated_masks,
        "ibot_mask_indices_list": ibot_mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": n_masked_patches,
    }
