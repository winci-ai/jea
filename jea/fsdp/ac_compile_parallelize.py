"""Activation checkpointing -> compilation -> FSDP2, following DINOv3.

Reference: facebookresearch/dinov3, dinov3/fsdp/ac_compile_parallelize.py.
Uses public FSDP2 APIs and preserves existing chunk indices/checkpoint names.
"""
from functools import partial
import logging
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard, register_fsdp_forward_method
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.utils.checkpoint import create_selective_checkpoint_contexts

logger = logging.getLogger("JEA")
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def get_param_dtype(cfg, role="student", name="backbone"):
    precision = cfg.compute_precision
    override = precision.get(role, {}).get(name, {}).get("mixed_precision", {})
    return DTYPES[override.get("param_dtype", precision.param_dtype)]


def _block_slots(backbone):
    if backbone.chunked_blocks:
        for chunk in backbone.blocks:
            for index, block in enumerate(chunk):
                if not isinstance(block, nn.Identity):
                    yield chunk, index
    else:
        for index in range(len(backbone.blocks)):
            yield backbone.blocks, index


def activation_checkpoint_backbone(backbone, full=False):
    kwargs = {"preserve_rng_state": True}
    if not full:
        saved_ops = [
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
        ]
        kwargs["context_fn"] = partial(create_selective_checkpoint_contexts, saved_ops)
    for parent, index in list(_block_slots(backbone)):
        parent[index] = checkpoint_wrapper(parent[index], **kwargs)


def ac_compile_parallelize(student, inference_models, cfg):
    if not dist.is_initialized():
        raise RuntimeError("Initialize the process group with torchrun before FSDP2 setup")
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP2 training requires CUDA")
    if cfg.train.get("cudagraphs", False) and not cfg.train.get("compile", False):
        raise ValueError("train.cudagraphs requires train.compile=true")
    if cfg.train.get("checkpointing", False):
        activation_checkpoint_backbone(student.backbone, cfg.train.get("checkpointing_full", False))
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
    device = torch.device("cuda", torch.cuda.current_device())
    for role, model in [("student", student), *inference_models]:
        # FSDP2 does not broadcast initial weights automatically. Construct on CPU
        # and move/shard one model at a time to limit initialization GPU memory.
        model.to(device)
        for tensor in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(tensor.detach(), src=0)
        for name, module in model.items():
            if role == "student" and get_param_dtype(cfg, role, name) == torch.float16:
                raise ValueError("Student FSDP2 precision must be bf16 or fp32 (no FP16 gradient scaler)")
            override = cfg.compute_precision.get(role, {}).get(name, {})
            mp = override.get("mixed_precision", {})
            policy = MixedPrecisionPolicy(
                param_dtype=get_param_dtype(cfg, role, name),
                reduce_dtype=DTYPES[mp.get("reduce_dtype", cfg.compute_precision.reduce_dtype)],
            )
            reshard = cfg.compute_precision.get("reshard_after_forward", True)
            options = dict(mesh=mesh, mp_policy=policy, reshard_after_forward=reshard)
            if name == "backbone":
                blocks = []
                for parent, index in list(_block_slots(module)):
                    block = parent[index]
                    if cfg.train.get("compile", False):
                        if cfg.train.get("cudagraphs", False):
                            block.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": True})
                        else:
                            block.compile()
                    fully_shard(block, **options)
                    blocks.append(block)
                if cfg.compute_precision.get("prefetch", True):
                    for previous, following in zip(blocks, blocks[1:]):
                        previous.set_modules_to_forward_prefetch([following])
                        following.set_modules_to_backward_prefetch([previous])
                fully_shard(module, **options)
                register_fsdp_forward_method(module, "get_intermediate_layers")
            else:
                if cfg.train.get("compile", False):
                    module.compile()
                fully_shard(module, **options)
        logger.info("FSDP2 ready: %s (block sharding, reshard=%s)", role, reshard)
