"""FSDP2 training utilities (PyTorch >= 2.7)."""
from collections import defaultdict
import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from .ac_compile_parallelize import ac_compile_parallelize, get_param_dtype
from .checkpoint import FSDPCheckpointer, PeriodicFSDPCheckpointer, full_model_state_dict


def reshard_fsdp_model(model):
    # Release inference-only root parameters using the public API.
    for module in reversed(list(model.modules())):
        if isinstance(module, FSDPModule):
            module.reshard()


def clip_grad_norm_(module, max_norm):
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm, foreach=True)
    return norm.full_tensor() if isinstance(norm, DTensor) else norm


def _canonical_name(name):
    return name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")


@torch.no_grad()
def foreach_ema_(target, source, momentum):
    """Update corresponding local shards without gathering full model weights."""
    source_params = {_canonical_name(n): p for n, p in source.named_parameters()}
    target_params = {_canonical_name(n): p for n, p in target.named_parameters()}
    if source_params.keys() != target_params.keys():
        raise ValueError("EMA models must have identical parameter names")
    groups = defaultdict(lambda: ([], []))
    for name, dst in target_params.items():
        src = source_params[name]
        if isinstance(dst, DTensor) != isinstance(src, DTensor):
            raise ValueError("EMA models must use the same sharding layout")
        if isinstance(dst, DTensor):
            if dst.placements != src.placements or dst.device_mesh != src.device_mesh:
                raise ValueError(f"EMA sharding mismatch for {name}")
            dst, src = dst.to_local(), src.to_local()
        if dst.shape != src.shape:
            raise ValueError(f"EMA shape mismatch for {name}")
        dsts, srcs = groups[(dst.device, dst.dtype)]
        dsts.append(dst)
        srcs.append(src.to(device=dst.device, dtype=dst.dtype))
    for dsts, srcs in groups.values():
        torch._foreach_mul_(dsts, momentum)
        torch._foreach_add_(dsts, srcs, alpha=1 - momentum)
    source_buffers = {_canonical_name(n): b for n, b in source.named_buffers()}
    for name, buffer in target.named_buffers():
        buffer.copy_(source_buffers[_canonical_name(name)])
