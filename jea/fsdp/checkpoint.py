"""Distributed checkpoints and portable teacher exports for FSDP2."""
import json
import logging
import os
from pathlib import Path
import random
import shutil
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict,
)
from torch.distributed.tensor import DTensor

logger = logging.getLogger("JEA")


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def full_model_state_dict(model, rank0_only=True):
    """Collectively gather one tensor at a time; retain only CPU tensors."""
    result = {}
    for key, value in get_model_state_dict(model).items():
        if isinstance(value, DTensor):
            value = value.full_tensor()
        if not rank0_only or _rank() == 0:
            result[key] = value.detach().cpu()
    return result


class FSDPCheckpointer:
    def __init__(self, model, save_dir="", *, optimizer=None, save_to_disk=True, **checkpointables):
        self.model = model
        self.save_dir = str(save_dir)
        self.optimizer = optimizer
        self.save_to_disk = save_to_disk
        self.checkpointables = checkpointables

    def _state(self):
        model_state = get_model_state_dict(self.model)
        # Gram is always persisted as a normal full .pth file, never DCP shards.
        model_state = {k: v for k, v in model_state.items() if not k.startswith("gram_teacher.")}
        state = {"model": model_state, "iteration": -1,
                 "gram_teacher_initialized": getattr(self.model, "gram_teacher_initialized", False)}
        if self.optimizer is not None:
            state["optimizer"] = get_optimizer_state_dict(self.model, self.optimizer)
        state.update({name: obj.state_dict() for name, obj in self.checkpointables.items()})
        return state

    def save(self, name, **metadata):
        if not self.save_dir or not self.save_to_disk:
            return
        if Path(name).name != name:
            raise ValueError("Checkpoint name must not contain a directory")
        destination = Path(self.save_dir) / name
        pending = destination.with_name(destination.name + ".incomplete")
        # Only rank 0 checks before creating the directory; otherwise a slower
        # rank can mistake the new pending directory for a stale checkpoint.
        exists = [destination.exists() or pending.exists() if _rank() == 0 else None]
        if dist.is_initialized():
            dist.broadcast_object_list(exists, src=0)
        if exists[0]:
            raise FileExistsError(f"Checkpoint already exists: {destination} or {pending}")
        if _rank() == 0:
            pending.mkdir(parents=True)
        _barrier()
        state = self._state()
        state["iteration"] = metadata.get("iteration", -1)
        dcp.save(state, checkpoint_id=pending)
        if getattr(self.model, "has_gram_teacher", False):
            gram_state = full_model_state_dict(self.model.gram_teacher)
            if _rank() == 0:
                torch.save({"teacher": gram_state}, pending / "gram_teacher.pth")
        rng = {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state()
        torch.save(rng, pending / f"rng_rank_{_rank()}.pth")
        _barrier()
        if _rank() == 0:
            metadata["world_size"] = dist.get_world_size() if dist.is_initialized() else 1
            (pending / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            pending.rename(destination)
        _barrier()
        self.tag_last_checkpoint(name)
        logger.info("Saved FSDP2 checkpoint: %s", destination)

    def load(self, path):
        if not path:
            return {}
        checkpoint = Path(path)
        if not checkpoint.is_dir():
            raise ValueError(
                "Training resume requires an FSDP2 distributed checkpoint directory. "
                "FSDP1 rank_*.pth checkpoints cannot be resumed directly; export full "
                "teacher weights in the old environment and use student.pretrained_weights."
            )
        state = self._state()
        dcp.load(state, checkpoint_id=checkpoint)
        missing = set_model_state_dict(
            self.model, state.pop("model"), options=StateDictOptions(strict=False)
        )
        if missing.unexpected_keys or any(not k.startswith("gram_teacher.") for k in missing.missing_keys):
            raise RuntimeError(f"Incompatible training checkpoint: {missing}")
        if getattr(self.model, "has_gram_teacher", False):
            gram_path = checkpoint / "gram_teacher.pth"
            if gram_path.is_file():
                gram_state = torch.load(gram_path, map_location="cpu", weights_only=True)
                set_model_state_dict(
                    self.model.gram_teacher, gram_state["teacher"],
                    options=StateDictOptions(full_state_dict=True, strict=True),
                )
            elif not state["gram_teacher_initialized"] and getattr(self.model, "gram_teacher_initialized", False):
                # Branch a non-Gram run using an explicitly supplied anchor.
                # Preserve the loaded gram.ckpt; restore student, EMA and optimizer normally.
                state["gram_teacher_initialized"] = True
                logger.info("Starting Gram branch with the explicitly loaded gram.ckpt")
            else:
                raise FileNotFoundError(
                    "Missing gram_teacher.pth: a new Gram branch requires gram.ckpt; "
                    "an existing Gram run requires its saved Gram sidecar"
                )
        if self.optimizer is not None:
            set_optimizer_state_dict(self.model, self.optimizer, state.pop("optimizer"))
        for name, obj in self.checkpointables.items():
            obj.load_state_dict(state.pop(name))
        if hasattr(self.model, "gram_teacher_initialized"):
            self.model.gram_teacher_initialized = state["gram_teacher_initialized"]
        metadata_file = checkpoint / "metadata.json"
        metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rng_path = checkpoint / f"rng_rank_{_rank()}.pth"
        if metadata.get("world_size") == world_size and rng_path.exists():
            # Local checkpoint sidecar contains Python/NumPy RNG tuples.
            rng = torch.load(rng_path, map_location="cpu", weights_only=False)
            torch.set_rng_state(rng["torch"])
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            if torch.cuda.is_available() and "cuda" in rng:
                torch.cuda.set_rng_state(rng["cuda"])
        logger.info("Loaded FSDP2 checkpoint: %s", path)
        return {**metadata, **state}

    def resume_or_load(self, path, *, resume=True):
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
        return self.load(path)

    def has_checkpoint(self):
        return bool(self.save_dir) and (Path(self.save_dir) / "last_checkpoint").is_file()

    def get_checkpoint_file(self):
        name = (Path(self.save_dir) / "last_checkpoint").read_text(encoding="utf-8").strip()
        return str(Path(self.save_dir) / name)

    def tag_last_checkpoint(self, name):
        if _rank() == 0:
            marker = Path(self.save_dir) / "last_checkpoint.tmp"
            marker.write_text(name, encoding="utf-8")
            os.replace(marker, Path(self.save_dir) / "last_checkpoint")
        _barrier()


class PeriodicFSDPCheckpointer:
    """Directory-aware retention; milestone checkpoints are kept separately."""
    def __init__(self, checkpointer, period, max_iter, max_to_keep=2):
        self.checkpointer = checkpointer
        self.period = period
        self.max_iter = max_iter
        self.max_to_keep = max_to_keep

    def step(self, iteration):
        if (iteration + 1) % self.period == 0:
            self.checkpointer.save(f"model_{iteration:07d}", iteration=iteration)
            if _rank() == 0 and self.max_to_keep:
                root = Path(self.checkpointer.save_dir).resolve()
                checkpoints = sorted(p for p in root.glob("model_*")
                                     if p.is_dir() and p.name.removeprefix("model_").isdigit())
                for old in checkpoints[:-self.max_to_keep]:
                    if old.resolve().parent != root or old.is_symlink():
                        raise ValueError(f"Refusing to remove checkpoint outside {root}")
                    shutil.rmtree(old)
            _barrier()
        if iteration >= self.max_iter - 1:
            self.checkpointer.save("model_final", iteration=iteration)
