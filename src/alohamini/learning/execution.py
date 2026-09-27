"""Accelerate execution; checkpoints are taken only between complete updates."""

import os
from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    DistributedType,
    FullyShardedDataParallelPlugin,
    GradientAccumulationPlugin,
    InitProcessGroupKwargs,
    gather_object,
    patch_environment,
)

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
                "Distributed training requires one visible GPU per local worker"
            )
        if processes == 1 and device.index is not None and device.index >= available:
            raise ValueError(f"Requested {device}, but only {available} GPU(s) are visible")


class Execution:
    def __init__(self, device, precision="none", backend="ddp"):
        if backend not in ("ddp", "fsdp2"):
            raise ValueError("distributed_backend must be ddp or fsdp2")
        self.sharded = backend == "fsdp2"
        if precision not in ("none", "bfloat16", "float16"):
            raise ValueError("mixed_precision must be none, bfloat16 or float16")
        for flag in (
            "ACCELERATE_USE_DEEPSPEED",
            "ACCELERATE_USE_MEGATRON_LM",
            "ACCELERATE_USE_PARALLELISM_CONFIG",
        ):
            if os.environ.get(flag, "false").lower() == "true":
                raise ValueError(
                    f"{flag} is not supported by the native trainer yet; use single-device or DDP"
                )
        if os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true" and not self.sharded:
            raise ValueError("Use distributed_backend=fsdp2 to enable native FSDP training")
        if self.sharded and (torch.device(device).type != "cuda" or "LOCAL_RANK" not in os.environ):
            raise ValueError("FSDP2 requires CUDA and a torchrun launch; use --background")
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
        self.owns_group = (self.world_size > 1 or self.sharded) and not dist.is_initialized()
        plugin = None
        if self.sharded:
            from torch.distributed.fsdp import MixedPrecisionPolicy

            plugin = FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=["ACTEncoderLayer", "ACTDecoderLayer"],
                reshard_after_forward=True,
                cpu_offload=False,
                cpu_ram_efficient_loading=False,
                activation_checkpointing=False,
                # Keep labels, master parameters and reductions in FP32. Existing
                # autocast controls compute precision without recasting batch targets.
                mixed_precision_policy=MixedPrecisionPolicy(
                    param_dtype=torch.float32, reduce_dtype=torch.float32
                ),
            )
        # Respect the explicit device, including cuda:N, instead of auto-selecting GPU 0.
        with patch_environment(
            ACCELERATE_TORCH_DEVICE=str(self.device),
            ACCELERATE_USE_FSDP=str(self.sharded).lower(),
        ):
            self.accelerator = Accelerator(
                fsdp_plugin=plugin,
                cpu=self.device.type == "cpu",
                mixed_precision={"none": "no", "bfloat16": "bf16", "float16": "fp16"}[precision],
                # Losses already carry global valid-target/sample weights. Dividing
                # again by the number of microbatches would shrink the gradients.
                gradient_accumulation_plugin=GradientAccumulationPlugin(
                    num_steps=1, adjust_scheduler=False, sync_with_dataloader=False
                ),
                step_scheduler_with_optimizer=False,
                dynamo_backend="no",
                kwargs_handlers=[
                    DistributedDataParallelKwargs(find_unused_parameters=True),
                    InitProcessGroupKwargs(
                        backend=(
                            ("nccl" if self.device.type == "cuda" else "gloo")
                            if self.world_size > 1 or self.sharded
                            else None
                        ),
                        timeout=timedelta(minutes=5),
                    ),
                ],
            )
        if (
            self.accelerator.distributed_type
            not in (
                (DistributedType.FSDP,)
                if self.sharded
                else (DistributedType.NO, DistributedType.MULTI_CPU, DistributedType.MULTI_GPU)
            )
            or self.accelerator.device != self.device
            or self.accelerator.num_processes != self.world_size
            or self.accelerator.process_index != self.rank
        ):
            raise ValueError(
                "Accelerate runtime differs from this run; start a fresh training process"
            )
        self.scaler = self.accelerator.scaler

    @property
    def main(self):
        return self.accelerator.is_main_process

    def close(self):
        if self.owns_group:
            self.accelerator.end_training()
        self.accelerator.free_memory()

    def barrier(self):
        if self.accelerator.distributed_type == DistributedType.MULTI_CPU:
            # Accelerate 1.15 passes local rank as a GPU device ID even for Gloo.
            # On a CUDA-capable CPU-training host that can select a nonexistent GPU.
            dist.barrier()
        else:
            self.accelerator.wait_for_everyone()

    def gather(self, value):
        return gather_object([value])

    def prepare(self, model, optimizer):
        if self.sharded:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                set_model_state_dict,
            )
            from torch.distributed.fsdp import register_fsdp_forward_method

            if model.name not in ("act", "am_act"):
                raise ValueError("FSDP2 currently supports ACT and AM-ACT only")
            if any(p.device.type != "cpu" for p in model.parameters()):
                raise ValueError("Construct the FSDP2 model on CPU before preparing it")
            classes = {type(module).__name__ for module in model.modules()}
            self.accelerator.state.fsdp_plugin.transformer_cls_names_to_wrap = sorted(
                classes & {"ACTEncoderLayer", "ACTDecoderLayer", "BasicBlock", "Bottleneck"}
            )
            # All ranks construct on CPU; synchronize rank-zero initialization only
            # after sharding, without materializing the full model on a single GPU.
            initial = model.state_dict() if self.main else {}
            model, optimizer = self.accelerator.prepare(model, optimizer)
            set_model_state_dict(
                model,
                initial,
                options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
            )
            register_fsdp_forward_method(model, "predict_action_chunk")
            return model, optimizer
        # RankBatchSampler already shards/resumes data. Scheduler steps are counted
        # in successful updates, not batches or ranks. Neither is prepared twice.
        # Optimizer.load_state_dict already places moments correctly and keeps
        # non-capturable Adam step counters on CPU, just like a fresh optimizer.
        return self.accelerator.prepare(model, optimizer, device_placement=[True, False])

    def distributed_checkpoint(self, path, model, optimizer, *, load=False):
        """Collective sharded model/optimizer I/O; platform RNG is saved separately."""
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        raw_optimizer = optimizer.optimizer
        model_state, optimizer_state = get_state_dict(model, raw_optimizer)
        state = {"model": model_state, "optimizer": optimizer_state}
        if load:
            dcp.load(state, checkpoint_id=path)
            set_state_dict(
                model,
                raw_optimizer,
                model_state_dict=state["model"],
                optim_state_dict=state["optimizer"],
            )
        else:
            dcp.save(state, checkpoint_id=path)

    def autocast(self):
        return self.accelerator.autocast()

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
        counts = self.accelerator.reduce(counts, reduction="sum")
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
                if index == len(batches) - 1 or self.world_size == 1 or self.sharded
                else self.accelerator.no_sync(model)
            )
            with synchronize, self.autocast():
                loss, _ = model(batch)
                finite = torch.isfinite(loss).int()
                finite = self.accelerator.reduce(finite, reduction="sum")
                if finite.item() != self.world_size:
                    raise RuntimeError("Non-finite loss; no optimizer step performed")
                metrics += loss.detach().double()
                self.accelerator.backward(loss)
        norm = self.accelerator.clip_grad_norm_(
            model.parameters(),
            clip if clip > 0 else float("inf"),
        )
        if self.scaler is None and not torch.isfinite(norm):
            raise RuntimeError("Non-finite gradients; no optimizer step performed")
        optimizer.step()
        applied = not optimizer.step_was_skipped
        optimizer.zero_grad(set_to_none=True)
        metrics = self.accelerator.reduce(metrics, reduction="sum")
        raw_model = self.accelerator.unwrap_model(model)
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
