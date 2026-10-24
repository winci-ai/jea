# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import math
import copy
import torch
from torch import nn
from jea.loss import DINOLoss, KoLeoLoss, iBOTPatchCELoss, GramLoss
from jea.models import build_model_from_cfg
from jea.layers import DINOHead
from jea.utils.utils import has_batchnorms, load_pretrained_weights
from jea.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from jea.fsdp import (
    ac_compile_parallelize,
    get_param_dtype,
    foreach_ema_,
    reshard_fsdp_model,
)

logger = logging.getLogger("JEA")


class SSLMetaArch(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # iBOT uses a fixed loss weight (1.0 in the paper); use_loss=false gives plain DINO.
        self.ibot_loss_weight = float(cfg.ibot.loss_weight)
        if not math.isfinite(self.ibot_loss_weight) or self.ibot_loss_weight < 0:
            raise ValueError("ibot.loss_weight must be finite and nonnegative")
        if not cfg.ibot.use_loss:
            self.ibot_loss_weight = 0.0
        if cfg.jea.loss_weight <= 0:
            raise ValueError("jea.loss_weight must be positive")
        if cfg.crops.global_crops_number != 2:
            raise ValueError("DINO expects exactly two global crops")
        if get_param_dtype(cfg) == torch.float16:
            raise ValueError(
                "FSDP2 training supports bf16/fp32; use compute_precision.param_dtype=bf16"
            )
        self.grad_scaler = None
        student_model_dict = dict()
        teacher_model_dict = dict()
        gram_model_dict = dict()
        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")
        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(
                f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}"
            )
            if "teacher" in chkpt:
                load_pretrained_weights(
                    student_backbone, cfg.student.pretrained_weights, "teacher"
                )
            else:
                student_backbone.load_state_dict(
                    chkpt.get("model", chkpt), strict=False
                )
        self.embed_dim = embed_dim
        self.jea_loss_weight = cfg.jea.loss_weight
        self.jea_loss = DINOLoss(cfg.jea.dino_prototypes)
        self.do_koleo = cfg.jea.koleo_loss_weight > 0
        head = partial(
            DINOHead,
            in_dim=embed_dim,
            hidden_dim=cfg.jea.head_hidden_dim,
            bottleneck_dim=cfg.jea.head_out_dim,
            out_dim=cfg.jea.dino_prototypes,
            nlayers=cfg.jea.head_nlayers,
        )
        student_model_dict["jea_head"] = head()
        teacher_model_dict["jea_head"] = head()
        if self.do_koleo:
            self.koleo_loss = KoLeoLoss()
        self.do_ibot = bool(cfg.ibot.use_loss and self.ibot_loss_weight > 0)
        self.ibot_separate_head = cfg.ibot.separate_head
        if self.do_ibot:
            if (
                max(cfg.ibot.mask_ratio_min_max) <= 0
                or cfg.ibot.mask_sample_probability <= 0
            ):
                raise ValueError(
                    "iBOT requires positive masking ratios and probability"
                )
            prototypes = cfg.ibot.get("head_n_prototypes") or cfg.jea.dino_prototypes
            if not self.ibot_separate_head and prototypes != cfg.jea.dino_prototypes:
                raise ValueError(
                    "Shared DINO/iBOT heads must use the same prototype count"
                )
            self.ibot_patch_loss = iBOTPatchCELoss(prototypes)
            if self.ibot_separate_head:
                patch_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    hidden_dim=cfg.jea.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.get("head_bottleneck_dim")
                    or cfg.jea.head_out_dim,
                    out_dim=prototypes,
                    nlayers=cfg.jea.head_nlayers,
                )
                student_model_dict["ibot_head"] = patch_head()
                teacher_model_dict["ibot_head"] = patch_head()
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(
            f"Student and Teacher are built: they are both {cfg.student.arch} network."
        )
        self.gram_use_loss = cfg.gram.use_loss
        self.has_gram_teacher = False
        self.gram_teacher_initialized = False
        if self.gram_use_loss:
            self.has_gram_teacher = True
            gram_backbone = copy.deepcopy(teacher_backbone).cpu()
            if cfg.gram.ckpt:
                logger.info(
                    f"OPTIONS -- pretrained weights of gram teacher: loading from {cfg.gram.ckpt}"
                )
                load_pretrained_weights(gram_backbone, cfg.gram.ckpt, "teacher")
                self.gram_teacher_initialized = True
            gram_model_dict["backbone"] = gram_backbone
            self.gram_teacher = nn.ModuleDict(gram_model_dict)
            for p in self.gram_teacher.parameters():
                p.requires_grad = False
            self.gram_loss = GramLoss(
                apply_norm=cfg.gram.normalized,
                remove_only_teacher_neg=cfg.gram.remove_only_teacher_neg,
                remove_neg=cfg.gram.remove_neg,
            )

            self.gram_ckpt = cfg.gram.ckpt
            self.gram_img_level = cfg.gram.img_level
            self.gram_ep_first_update = cfg.gram.ep_first_update
            self.gram_rep_update = cfg.gram.rep_update
            self.gram_update_frequency = cfg.gram.update_frequency
            self.gram_ep_load_ema_teacher = cfg.gram.ep_load_ema_teacher
            self.gram_params_lists = None
            if self.gram_ckpt is None and self.gram_ep_load_ema_teacher < 0:
                raise ValueError(
                    "If no gram checkpoint is provided, `gram.ep_load_ema_teacher` must be set to a non-negative value."
                )
            logger.info("OPTIONS -- GRAM")
            logger.info(f"OPTIONS -- GRAM -- ckpt: {cfg.gram.ckpt}")
            if self.cfg.gram.rep_update:
                logger.info(
                    f"OPTIONS -- GRAM -- repeated update: {cfg.gram.rep_update}"
                )
                logger.info(
                    f"OPTIONS -- GRAM -- update freq: {cfg.gram.update_frequency}"
                )
                logger.info(
                    f"OPTIONS -- GRAM -- epoch first update: {cfg.gram.ep_first_update}"
                )
            logger.info(
                f"OPTIONS -- GRAM -- apply normalization: {cfg.gram.normalized}"
            )
            logger.info(f"OPTIONS -- GRAM -- img_level: {cfg.gram.img_level}")
            logger.info(f"OPTIONS -- GRAM -- remove_neg: {cfg.gram.remove_neg}")
            logger.info(
                f"OPTIONS -- GRAM -- remove_only_teacher_neg: {cfg.gram.remove_only_teacher_neg}"
            )

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def _move_images_to_device(self, images: torch.Tensor) -> torch.Tensor:
        images = images.cuda(non_blocking=True)
        return images

    def forward_backward(self, images, teacher_temp, gram_loss_weight):
        n_global_crops = self.cfg.crops.global_crops_number
        n_local_crops = self.cfg.crops.local_crops_number
        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
        global_crops = self._move_images_to_device(images["collated_global_crops"])
        local_crops = self._move_images_to_device(images["collated_local_crops"])
        do_ibot_loss = self.do_ibot and self.ibot_loss_weight > 0
        use_patch_masks = do_ibot_loss
        if use_patch_masks:
            ibot_masks = images["ibot_collated_masks"].cuda(non_blocking=True)
            ibot_mask_indices_list = images["ibot_mask_indices_list"].cuda(
                non_blocking=True
            )
            n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
            n_masked_patches = ibot_mask_indices_list.shape[0]
            masks_weight = images["masks_weight"].cuda(non_blocking=True)
        else:
            ibot_masks = None

        @torch.no_grad()
        def get_teacher_output():
            teacher_output = self.teacher.backbone(global_crops, is_training=True)
            teacher_cls_tokens = teacher_output["x_norm_clstoken"]
            cls_views = teacher_cls_tokens.chunk(n_global_crops)
            teacher_cls_tokens = torch.cat((cls_views[1], cls_views[0]))
            masked_teacher_patches = None
            if do_ibot_loss:
                masked_teacher_patches = torch.index_select(
                    teacher_output["x_norm_patchtokens"].flatten(0, 1),
                    dim=0,
                    index=ibot_mask_indices_list,
                )
            if do_ibot_loss and (not self.ibot_separate_head):
                n_cls_tokens = teacher_cls_tokens.shape[0]
                tokens_after_head = self.teacher.jea_head(
                    torch.cat((teacher_cls_tokens, masked_teacher_patches), dim=0)
                )
                teacher_cls_after_head = tokens_after_head[:n_cls_tokens]
                teacher_patch_after_head = tokens_after_head[n_cls_tokens:]
            else:
                teacher_cls_after_head = self.teacher.jea_head(teacher_cls_tokens)
                teacher_patch_after_head = (
                    self.teacher.ibot_head(masked_teacher_patches)
                    if do_ibot_loss
                    else None
                )
            return (teacher_cls_after_head, teacher_patch_after_head)

        teacher_cls_tokens_after_head, ibot_teacher_patch_tokens_after_head = (
            get_teacher_output()
        )
        teacher_dino_softmaxed_centered_list = self.jea_loss.sinkhorn_knopp_teacher(
            teacher_cls_tokens_after_head, teacher_temp=teacher_temp
        ).view(n_global_crops, -1, *teacher_cls_tokens_after_head.shape[1:])
        if do_ibot_loss:
            masked_teacher_ibot_softmaxed_centered = (
                self.ibot_patch_loss.sinkhorn_knopp_teacher(
                    ibot_teacher_patch_tokens_after_head,
                    teacher_temp=teacher_temp,
                    n_masked_patches_tensor=n_masked_patches_tensor,
                )
            )
        reshard_fsdp_model(self.teacher.backbone)
        reshard_fsdp_model(self.teacher.jea_head)
        if do_ibot_loss and self.ibot_separate_head:
            reshard_fsdp_model(self.teacher.ibot_head)
        gram_teacher_patch_tokens = None
        if self.gram_use_loss and gram_loss_weight > 0 and self.has_gram_teacher:
            if not self.gram_teacher_initialized:
                raise RuntimeError(
                    "Initialize Gram from gram.ckpt or EMA before enabling Gram loss"
                )
            with torch.no_grad():
                gram_input_crops = global_crops
                gram_teacher_output_dict = self.gram_teacher.backbone(
                    gram_input_crops, is_training=True
                )
                gram_teacher_patch_tokens = gram_teacher_output_dict[
                    "x_norm_patchtokens"
                ]
                del gram_input_crops
                del gram_teacher_output_dict
                reshard_fsdp_model(self.gram_teacher)
        loss_dict = {}
        loss_accumulator = 0
        student_global_backbone_output_dict, student_local_backbone_output_dict = (
            self.student.backbone(
                [global_crops, local_crops],
                ibot_masks=[ibot_masks, None],
                is_training=True,
            )
        )
        student_global_cls_tokens = student_global_backbone_output_dict[
            "x_norm_clstoken"
        ]
        ibot_student_patch_tokens = student_global_backbone_output_dict[
            "x_norm_patchtokens"
        ]
        inputs_for_student_head_list = []
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))
        if do_ibot_loss:
            _dim = ibot_student_patch_tokens.shape[-1]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(
                n_masked_patches, _dim
            )
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(
                    ibot_student_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=ibot_mask_indices_list,
                )
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(
                    buffer_tensor_patch_tokens.unsqueeze(0)
                )
            else:
                student_ibot_masked_patch_tokens_after_head = self.student.ibot_head(
                    buffer_tensor_patch_tokens
                )[:n_masked_patches]
        head_sizes = [tokens.shape[1] for tokens in inputs_for_student_head_list]
        cat_inputs = torch.cat(inputs_for_student_head_list, dim=1)
        outputs_list = list(self.student.jea_head(cat_inputs).split(head_sizes, dim=1))
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)
        dino_global_crops_loss = (
            self.jea_loss(
                student_output_list=[student_global_cls_tokens_after_head],
                teacher_out_softmaxed_centered_list=[
                    teacher_dino_softmaxed_centered_list.flatten(0, 1)
                ],
            )
            * 2
            / (n_global_crops_loss_terms + n_local_crops_loss_terms)
        )
        jea_loss = dino_global_crops_loss
        if n_local_crops > 0:
            dino_local_crops_loss = self.jea_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(
                    n_local_crops
                ),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            jea_loss += dino_local_crops_loss
        loss_dict["jea_loss"] = jea_loss
        loss_accumulator += jea_loss * self.jea_loss_weight
        if self.do_koleo:
            koleo_loss = (
                sum(
                    (
                        self.koleo_loss(p)
                        for p in student_global_cls_tokens.chunk(n_global_crops)
                    )
                )
                / n_global_crops
            )
            loss_dict["koleo_loss"] = koleo_loss / 2
            koleo_scale = n_global_crops
            loss_accumulator += (
                koleo_loss * self.cfg.jea.koleo_loss_weight * koleo_scale
            )
        if do_ibot_loss and (not self.ibot_separate_head):
            student_ibot_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(
                0
            )[:n_masked_patches]
        if do_ibot_loss:
            ibot_patch_loss = self.ibot_patch_loss.forward_masked(
                student_ibot_masked_patch_tokens_after_head,
                masked_teacher_ibot_softmaxed_centered,
                student_masks_flat=ibot_masks,
                n_masked_patches=n_masked_patches,
                masks_weight=masks_weight,
            )
            loss_dict["ibot_loss"] = ibot_patch_loss / 2
            loss_accumulator += ibot_patch_loss * self.ibot_loss_weight
        if gram_teacher_patch_tokens is not None:
            gram_loss = self.gram_loss(
                ibot_student_patch_tokens,
                gram_teacher_patch_tokens,
                img_level=self.gram_img_level,
            )
            loss_dict["gram_loss"] = gram_loss / 2
            loss_accumulator += gram_loss * gram_loss_weight
        self.backprop_loss(loss_accumulator)
        return loss_dict

    @torch.no_grad()
    def update_teacher(self, m):
        reshard_fsdp_model(self.student)
        reshard_fsdp_model(self.teacher)
        for name in self.teacher:
            foreach_ema_(self.teacher[name], self.student[name], m)

    @torch.no_grad()
    def update_gram(self, m=0):
        if self.has_gram_teacher:
            reshard_fsdp_model(self.teacher)
            reshard_fsdp_model(self.gram_teacher)
            foreach_ema_(self.gram_teacher.backbone, self.teacher.backbone, m)
            self.gram_teacher_initialized = True

    @torch.no_grad()
    def gram_load_ema_teacher(self):
        if self.has_gram_teacher and (not self.gram_teacher_initialized):
            self.update_gram(m=0)
            self.gram_teacher.eval()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        if self.has_gram_teacher:
            self.gram_teacher.eval()
        return self

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("Preparing FSDP2 student, EMA teacher and runtime Gram teacher")
        if has_batchnorms(self.student):
            raise NotImplementedError("BatchNorm is unsupported in FSDP2 training")
        for name in self.teacher:
            self.teacher[name].load_state_dict(self.student[name].state_dict())
        inference_models = [("teacher", self.teacher)]
        if self.has_gram_teacher:
            inference_models.append(("gram_teacher", self.gram_teacher))
        ac_compile_parallelize(self.student, inference_models, self.cfg)
        for name, module in self.named_children():
            if name not in {"student", "teacher", "gram_teacher"}:
                module.cuda()
        self.teacher.eval()
        if self.has_gram_teacher:
            self.gram_teacher.eval()
