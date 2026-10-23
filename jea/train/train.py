# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os

import torch
import wandb

import jea.distributed as distributed
from jea.fsdp import FSDPCheckpointer, PeriodicFSDPCheckpointer, clip_grad_norm_, full_model_state_dict, get_param_dtype
from jea.logging import MetricLogger
from jea.train.runtime import (
    build_train_dataset_and_loader,
    flatten_config,
    resolve_epoch_length,
)
from jea.train.ssl_meta_arch import SSLMetaArch
from jea.utils.config import setup
from jea.utils.utils import CosineScheduler, linear_warmup_cosine_decay, load_pretrained_weights
from jea.eval.knn import eval_knn_with_model

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger("JEA")


def build_loss_weight_schedule(loss_cfg, epoch_length, total_iterations):
    if not loss_cfg.get("use_loss", True):
        return linear_warmup_cosine_decay(
            start=0,
            peak=0,
            end=0,
            warmup_iterations=0,
            total_iterations=total_iterations,
            cosine_iterations=0,
        )

    schedule_cfg = loss_cfg.loss_weight_schedule
    return linear_warmup_cosine_decay(
        start=schedule_cfg.start,
        peak=schedule_cfg.peak,
        end=schedule_cfg.end,
        warmup_iterations=epoch_length * schedule_cfg.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=epoch_length * schedule_cfg.cosine_epochs if "cosine_epochs" in schedule_cfg else None,
    )


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("jea training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".""".strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default=None,
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    fused = cfg.optim.get("fused", False)
    return torch.optim.AdamW(
        params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
        fused=fused, foreach=not fused,
    )


def build_schedulers(cfg):
    epoch_length = cfg.train.OFFICIAL_EPOCH_LENGTH
    total_iterations = cfg.optim.epochs * epoch_length

    lr_schedule = CosineScheduler(
        base_value=cfg.optim.lr,
        final_value=cfg.optim.min_lr,
        total_iters=total_iterations,
        warmup_iters=cfg.optim.warmup_epochs * epoch_length,
        start_warmup_value=0,
    )
    wd_schedule = CosineScheduler(
        base_value=cfg.optim.weight_decay,
        final_value=cfg.optim.weight_decay_end,
        total_iters=total_iterations,
    )
    momentum_schedule = CosineScheduler(
        base_value=cfg.teacher.momentum_teacher,
        final_value=cfg.teacher.final_momentum_teacher,
        total_iters=total_iterations,
    )
    teacher_temp_schedule = CosineScheduler(
        base_value=cfg.teacher.dino_teacher_temp,
        final_value=cfg.teacher.dino_teacher_temp,
        total_iters=cfg.teacher.dino_warmup_teacher_temp_epochs * epoch_length,
        warmup_iters=cfg.teacher.dino_warmup_teacher_temp_epochs * epoch_length,
        start_warmup_value=cfg.teacher.dino_warmup_teacher_temp,
    )
    last_layer_lr_schedule = CosineScheduler(
        base_value=cfg.optim.lr,
        final_value=cfg.optim.min_lr,
        total_iters=total_iterations,
        warmup_iters=cfg.optim.warmup_epochs * epoch_length,
        start_warmup_value=0,
    )
    last_layer_lr_schedule.schedule[: cfg.optim.dino_freeze_last_layer_epochs * epoch_length] = 0

    gram_loss_schedule = build_loss_weight_schedule(cfg.gram, epoch_length, total_iterations)

    logger.info("Schedulers ready.")
    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
        gram_loss_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def reduce_loss_dict(loss_dict):
    names = list(loss_dict)
    values = torch.stack([loss_dict[name].detach().float() for name in names])
    if distributed.get_global_size() > 1:
        torch.distributed.all_reduce(values)
    values = (values / distributed.get_global_size()).cpu().tolist()
    return dict(zip(names, values))


def maybe_log_wandb(
    loss_dict_reduced,
    grad_norms,
    lr,
    wd,
    mom,
    ibot_loss_weight,
    gram_loss_weight,
    losses_reduced,
    iteration,
):
    if not distributed.is_main_process() or wandb.run is None:
        return

    if iteration == 0 or (iteration + 1) % 100 == 0:
        wandb.log(loss_dict_reduced, commit=False)
        if grad_norms:
            wandb.log({"encoder_grad": grad_norms[0], "head_grad": grad_norms[-1]}, commit=False)

        wandb.log(
            {
                "learning_rate": lr,
                "weight_decay": wd,
                "momentum": mom,
                "ibot_loss_weight": ibot_loss_weight,
                "gram_loss_weight": gram_loss_weight,
                "total_loss": losses_reduced,
                "iters": iteration,
            }
        )


def do_test(cfg, model, iteration):
    teacher_state_dict = full_model_state_dict(model.teacher)
    #student_state_dict = model.student.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)

        torch.save({"teacher": teacher_state_dict}, os.path.join(eval_dir, "teacher_checkpoint.pth"))
        #torch.save({"student": student_state_dict}, os.path.join(eval_dir, "student_checkpoint.pth"))

    torch.cuda.synchronize()
    
    if cfg.evaluation.knn:
        knn_output_dir = os.path.join(cfg.train.output_dir, f"eval/{iteration}/knn")
        os.makedirs(knn_output_dir, exist_ok=True)
        knn_results = eval_knn_with_model(model=model.teacher.backbone, 
                                          output_dir=knn_output_dir,
                                          train_dataset_str=cfg.train.dataset_path,
                                          val_dataset_str=cfg.train.dataset_path.replace('TRAIN', 'VAL'),
                                          )
        
        if torch.distributed.get_rank() == 0:
            wandb.log(knn_results, commit=False)


def do_train(cfg, model, resume: bool = False):
    model.train()
    inputs_dtype = get_param_dtype(cfg)
    grad_scaler = model.grad_scaler

    optimizer = build_optimizer(cfg, model.get_params_groups())
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    dataset, data_loader = build_train_dataset_and_loader(cfg, start_iter=start_iter, inputs_dtype=inputs_dtype)
    epoch_length = resolve_epoch_length(cfg, len(dataset))
    max_iter = cfg.optim.epochs * epoch_length
    eval_period_iterations = int(cfg.evaluation.eval_period_epochs * epoch_length)
    saveckp_epochs = {int(epoch) for epoch in cfg.train.get("saveckp_epochs", [])}

    gram_it_load_ema_teacher = None
    gram_it_first_update = None
    gram_update_frequency = None
    if cfg.gram.use_loss:
        gram_it_load_ema_teacher = int(model.gram_ep_load_ema_teacher * epoch_length)
        gram_it_first_update = int(model.gram_ep_first_update * epoch_length)
        gram_update_frequency = int(model.gram_update_frequency * epoch_length)

    periodic_checkpointer = PeriodicFSDPCheckpointer(
        checkpointer,
        period=max(1, cfg.train.saveckp_freq * epoch_length),
        max_iter=max_iter,
        max_to_keep=1,
    )

    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
        gram_loss_schedule,
    ) = build_schedulers(cfg)
    ibot_loss_weight = model.ibot_loss_weight

    logger.info("Starting training from iteration %s", start_iter)
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)

    if cfg.gram.use_loss:
        num_gram_updates = 0

    if (
        cfg.gram.use_loss
        and model.has_gram_teacher
        and cfg.gram.rep_update
        and start_iter > 0
        and start_iter >= gram_it_first_update
    ):
        num_gram_updates = math.ceil((start_iter + 1 - gram_it_first_update) / gram_update_frequency)
        logger.info("Gram was updated %s times before iteration %s", num_gram_updates, start_iter)

    iteration = start_iter
    for data in metric_logger.log_every(data_loader, 10, "Training", max_iter, start_iter):
        if iteration >= max_iter:
            break

        current_batch_size = data["collated_global_crops"].shape[0] // max(cfg.crops.global_crops_number, 1)

        if (
            cfg.gram.use_loss
            and model.has_gram_teacher
            and iteration == gram_it_load_ema_teacher
            and not cfg.gram.ckpt
        ):
            logger.info("Loading EMA teacher into Gram teacher before iteration %s", iteration)
            model.gram_load_ema_teacher()

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        gram_loss_weight = gram_loss_schedule[iteration]

        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(
            data,
            teacher_temp,
            gram_loss_weight,
        )

        grad_norms = []
        if grad_scaler is not None:
            if cfg.optim.clip_grad:
                grad_scaler.unscale_(optimizer)
                for submodule in model.student.values():
                    total_grad_norm = clip_grad_norm_(submodule, cfg.optim.clip_grad)
                    grad_norms.append(total_grad_norm.item())
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for submodule in model.student.values():
                    grad_norms.append(clip_grad_norm_(submodule, cfg.optim.clip_grad).item())
            optimizer.step()

        model.update_teacher(mom)

        if (
            cfg.gram.use_loss
            and model.has_gram_teacher
            and model.gram_rep_update
            and (iteration + 1) >= gram_it_first_update
            and (iteration + 1 - gram_it_first_update) % gram_update_frequency == 0
            and (cfg.gram.max_updates is None or num_gram_updates < cfg.gram.max_updates)
        ):
            logger.info("Updating Gram teacher from EMA teacher after iteration %s", iteration)
            model.update_gram()
            num_gram_updates += 1

        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = (
            loss_dict_reduced["jea_loss"] * model.jea_loss_weight
            + loss_dict_reduced.get("ibot_loss", 0.0) * ibot_loss_weight
            + loss_dict_reduced.get("koleo_loss", 0.0) * cfg.jea.koleo_loss_weight * cfg.crops.global_crops_number
            + loss_dict_reduced.get("gram_loss", 0.0) * gram_loss_weight
        )
        if not math.isfinite(losses_reduced):
            raise AssertionError("Non-finite reduced loss")

        metric_logger.update(
            lr=lr,
            wd=wd,
            mom=mom,
            last_layer_lr=last_layer_lr,
            ibot_loss_weight=ibot_loss_weight,
            gram_loss_weight=gram_loss_weight,
            current_batch_size=current_batch_size,
            total_loss=losses_reduced,
            **loss_dict_reduced,
        )

        if eval_period_iterations > 0 and (iteration + 1) % eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()

        periodic_checkpointer.step(iteration)
        finished_epoch = (iteration + 1) // epoch_length
        if (iteration + 1) % epoch_length == 0 and finished_epoch in saveckp_epochs:
            logger.info("Saving checkpoint for epoch %s at iteration %s", finished_epoch, iteration)
            checkpointer.save(f"model_epoch_{finished_epoch:04d}", iteration=iteration)

        maybe_log_wandb(
            loss_dict_reduced=loss_dict_reduced,
            grad_norms=grad_norms,
            lr=lr,
            wd=wd,
            mom=mom,
            ibot_loss_weight=ibot_loss_weight,
            gram_loss_weight=gram_loss_weight,
            losses_reduced=losses_reduced,
            iteration=iteration,
        )
        iteration += 1

    metric_logger.synchronize_between_processes()
    return {name: meter.global_avg for name, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)
    torch.backends.cudnn.benchmark = cfg.train.get("cudnn_benchmark", False)

    model = SSLMetaArch(cfg)
    model.prepare_for_distributed_training()
    logger.info("Model:\n%s", model)

    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    if distributed.is_main_process():
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.env_name,
                   mode=cfg.wandb.get("mode", "disabled"), config=flatten_config(cfg))

    try:
        do_train(cfg, model, resume=not args.no_resume)
    finally:
        if distributed.is_main_process() and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main(get_args_parser(add_help=True).parse_args())
