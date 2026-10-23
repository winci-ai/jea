# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math
import logging
import os

from omegaconf import OmegaConf
import jea.distributed as distributed
from jea.logging import setup_logging
from jea.utils import utils
from jea.configs import JEA_default_config


logger = logging.getLogger("JEA")


def apply_scaling_rules_to_cfg(cfg):  # to fix
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        base_lr = cfg.optim.base_lr
        cfg.optim.lr = base_lr
        cfg.optim.lr *= math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_global_size() / 1024.0)
        logger.info(f"sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
    else:
        raise NotImplementedError
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        # Export concrete paths, including the Gram anchor environment override.
        OmegaConf.save(config=cfg, f=f, resolve=True)
    return saved_cfg_path


def get_cfg_from_args(args):
    default_cfg = OmegaConf.create(JEA_default_config)
    cfg = OmegaConf.load(args.config_file)
    cli_cfg = OmegaConf.from_cli(args.opts or [])
    return OmegaConf.merge(default_cfg, cfg, cli_cfg)


def default_setup(args):
    distributed.enable(overwrite=True)
    seed = getattr(args, "seed", 0)
    rank = distributed.get_global_rank()

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("JEA")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    if args.output_dir is None:
        args.output_dir = cfg.train.output_dir = os.path.join(cfg.train.output_dir, cfg.wandb.env_name)
    else:
        cfg.train.output_dir = args.output_dir

    os.makedirs(args.output_dir, exist_ok=True)
    if not hasattr(args, "seed"):
        args.seed = cfg.train.seed
    default_setup(args)
    apply_scaling_rules_to_cfg(cfg)
    write_config(cfg, args.output_dir)
    return cfg
