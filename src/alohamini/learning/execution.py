"""Torch DDP/AMP execution; checkpoints are taken only between complete updates."""

import os
from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from alohamini.learning.training_state import EpisodeAwareSampler


def validate_local_workers(processes, device):
    """Validate local launch capacity before creating run files or a process group."""
    if type(processes) is not int or processes < 1:
        raise ValueError("num_processes must be a positive integer")
    device = torch.device(device)
    if device.type == "cuda":
        available = torch.cuda.device_count()
        if not torch.cuda.is_available() or processes > available:
            raise ValueError(
                f"Requested {processes} CUDA worker(s), but only {available} GPU(s) are visible; "
                "DDP requires one visible GPU per local worker"
            )
        if processes == 1 and device.index is not None and device.index >= available:
            raise ValueError(f"Requested {device}, but only {available} GPU(s) are visible")


class Execution:
    def __init__(self, device, precision="none"):
        if precision not in ("none", "bfloat16", "float16"):
            raise ValueError("mixed_precision must be none, bfloat16 or float16")
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(self.world_size)))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not 0 <= self.rank < self.world_size or not 0 <= local_rank < self.local_world_size:
            raise ValueError("Invalid distributed rank configuration")
        validate_local_workers(self.local_world_size, device)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable in this environment")
            if self.world_size > 1:
                self.device = torch.device("cuda", local_rank)
            elif self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(self.device)
        if precision == "float16" and self.device.type != "cuda":
            raise ValueError("float16 training requires CUDA GradScaler")
        self.precision = precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "float16")
        self.owns_group = self.world_size > 1 and not dist.is_initialized()
        if self.owns_group:
            dist.init_process_group(
                "nccl" if self.device.type == "cuda" else "gloo", timeout=timedelta(minutes=5)
            )

    @property
    def main(self):
        return self.rank == 0

    def close(self):
        if self.owns_group:
            dist.destroy_process_group()

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def wrap(self, model):
        if self.world_size == 1:
            return model
        return DistributedDataParallel(
            model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    def autocast(self):
        return (
            nullcontext()
            if self.precision == "none"
            else torch.autocast(self.device.type, dtype=getattr(torch, self.precision))
        )

    def update(self, model, batches, optimizer, clip, processor):
        """Global valid-target reconstruction mean plus global sample-mean KL.

        CPU microbatches are retained for counting; activations are never retained
        across microbatches. DDP averages gradients, hence the world-size factor.
        """
        model.train()
        optimizer.zero_grad(set_to_none=True)
        counts = torch.tensor(
            [
                sum(int((~b["action_is_pad"]).sum()) for b in batches),
                sum(len(b["action"]) for b in batches),
            ],
            device=self.device,
            dtype=torch.float64,
        )
        if self.world_size > 1:
            dist.all_reduce(counts)
        if (counts <= 0).any():
            raise ValueError("An update needs valid action targets and samples")
        metrics = torch.zeros((), device=self.device, dtype=torch.float64)
        for index, raw in enumerate(batches):
            batch = processor(raw)
            batch["_reconstruction_weight"] = (
                self.world_size * int((~raw["action_is_pad"]).sum()) / counts[0].item()
            )
            batch["_kl_weight"] = self.world_size * len(raw["action"]) / counts[1].item()
            synchronize = (
                nullcontext()
                if index == len(batches) - 1 or self.world_size == 1
                else model.no_sync()
            )
            with synchronize, self.autocast():
                loss, _ = model(batch)
                finite = torch.isfinite(loss).int()
                if self.world_size > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite:
                    raise RuntimeError("Non-finite loss; no optimizer step performed")
                metrics += loss.detach().double()
                self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            clip if clip > 0 else float("inf"),
            error_if_nonfinite=not self.scaler.is_enabled(),
        )
        old_scale = self.scaler.get_scale()
        self.scaler.step(optimizer)
        self.scaler.update()
        applied = self.scaler.get_scale() >= old_scale
        optimizer.zero_grad(set_to_none=True)
        if self.world_size > 1:
            dist.all_reduce(metrics)
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        if applied and callable(getattr(raw_model, "update", None)):
            raw_model.update()
        return dict(
            loss=metrics.item() / self.world_size,
            grad_norm=norm.item() if torch.isfinite(norm) else None,
            lr=optimizer.param_groups[0]["lr"],
            optimizer_step=applied,
        )


class RankBatchSampler:
    """Deterministic epoch shuffling; resume by consumed batches, not worker prefetch.

    For DDP only, pad the final global batch by repeating its epoch's first
    samples (or drop it explicitly). Every rank has the same batch count.
    Single-process partial batches retain the previous trainer's behavior.
    """

    def __init__(self, size, batch_size, seed, rank=0, world_size=1, drop_last=False, consumed=0):
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size, self.drop_last = rank, world_size, drop_last
        self.sampler = EpisodeAwareSampler([0], [size], shuffle=True, seed=seed)
        global_batch = batch_size * world_size
        self.batches = (
            size // global_batch if drop_last else (size + global_batch - 1) // global_batch
        )
        if not self.batches:
            raise ValueError("drop_last would discard every global batch")
        self.epoch, self.offset = divmod(consumed, self.batches)

    def __len__(self):
        return self.batches

    def __iter__(self):
        self.sampler.set_epoch(self.epoch)
        order = list(self.sampler)
        width = self.batch_size * self.world_size
        if self.world_size > 1 and not self.drop_last:
            order = (order * ((self.batches * width + self.size - 1) // self.size))[
                : self.batches * width
            ]
        offset = self.offset
        self.epoch += 1
        self.offset = 0
        return iter(
            [
                order[
                    i * width + self.rank * self.batch_size : min(
                        i * width + (self.rank + 1) * self.batch_size, len(order)
                    )
                ]
                for i in range(offset, self.batches)
            ]
        )
