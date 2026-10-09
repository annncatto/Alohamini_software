#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import math
import os
import random
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from alohamini.learning.policy import save_checkpoint

logger = logging.getLogger(__name__)


class EpisodeAwareSampler:
    """Sampler over episode frames that stores only per-episode boundaries.

    Logical positions map to frame indices on the fly (O(num_episodes) construction memory)
    instead of materializing a Python list of every frame index.

    Each epoch is shuffled with a `torch.randperm` seeded from `(seed, epoch)`, so the data order
    is a pure function of `(seed, epoch)`: it reproduces on every rank without synchronizing the
    global RNG (no `generator` to sync across distributed ranks), and `state_dict` /
    `load_state_dict` resume a run sample-exactly by regenerating the epoch's permutation and
    continuing from the saved offset. Each call to `__iter__` advances the epoch. During a
    resumed epoch, `__len__` still reports the full length.

    Epoch advancement: `__iter__` eagerly advances the epoch, and `set_epoch` / `load_state_dict`
    set it explicitly. Within a single run callers should rely on exactly one of these mechanisms,
    not both: advancing the epoch by hand *and* letting `__iter__` auto-advance over the same
    iterations would skip or repeat epochs. The training loop drives it purely through `__iter__`
    (via `cycle`); `set_epoch` / `load_state_dict` are used only to (re)position before iteration
    starts (e.g. on resume or in tests).
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
    ):
        """
        Args:
            dataset_from_indices: Start index of each episode in the dataset.
            dataset_to_indices: End index of each episode in the dataset.
            episode_indices_to_use: Episode indices to use; None means all.
            drop_n_first_frames: Frames to drop from the start of each episode.
            drop_n_last_frames: Frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            seed: Seed the permutation is derived from (together with the epoch).
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError(
                f"dataset_from_indices and dataset_to_indices must have the same length, "
                f"got {len(from_indices)} and {len(to_indices)}"
            )

        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(from_indices), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        starts = from_indices + drop_n_first_frames
        lengths = to_indices - drop_n_last_frames - starts
        for episode_idx in np.flatnonzero(used & (lengths <= 0)):
            logger.warning(
                "Episode %d has %d frames but drop_n_first_frames=%d and "
                "drop_n_last_frames=%d removes all frames. Skipping.",
                episode_idx,
                to_indices[episode_idx] - from_indices[episode_idx],
                drop_n_first_frames,
                drop_n_last_frames,
            )
        used &= lengths > 0
        if not used.any():
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self._starts = starts[used]
        self._cum_lengths = np.cumsum(lengths[used])
        self._num_frames = int(self._cum_lengths[-1])
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._start_index = 0
        self._absolute_to_relative = absolute_to_relative_idx

    @property
    def indices(self) -> list[int]:
        """Materialized frame indices in unshuffled order; O(num_frames), introspection only."""
        return [self._frame_index(k) for k in range(self._num_frames)]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        # Derive a per-epoch seed from (seed, epoch) so the permutation is a pure function of both
        # and reproduces identically on every rank without touching the global RNG.
        epoch_seed = int(
            np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0]
        )
        return torch.Generator().manual_seed(epoch_seed)

    def _frame_index(self, position: int) -> int:
        episode = int(np.searchsorted(self._cum_lengths, position, side="right"))
        position_in_episode = position - (int(self._cum_lengths[episode - 1]) if episode > 0 else 0)
        absolute_idx = int(self._starts[episode]) + position_in_episode
        if self._absolute_to_relative is not None:
            return self._absolute_to_relative[absolute_idx]
        return absolute_idx

    def __iter__(self) -> Iterator[int]:
        # Advance epoch state eagerly, not on first consumption of the generator.
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        if self.shuffle:
            order = torch.randperm(self._num_frames, generator=self._epoch_generator(epoch))
            for k in range(start, self._num_frames):
                yield self._frame_index(int(order[k]))
        else:
            for k in range(start, self._num_frames):
                yield self._frame_index(k)

    def __len__(self) -> int:
        return self._num_frames


def compute_sampler_state(step: int, num_frames: int, batch_size: int, num_processes: int) -> dict:
    """Map an optimization step to an `EpisodeAwareSampler` state for sample-exact resume.

    Under accelerate's batch sharding, one step consumes `batch_size * num_processes` sampler
    positions and each rank sees `ceil(ceil(num_frames / batch_size) / num_processes)` batches
    per epoch (`even_batches` padding included). The start index provably stays below
    `num_frames`; the `min` is defensive.

    Assumptions (resume is only sample-exact when they hold):
        - `num_processes` and `batch_size` match the run that wrote the checkpoint. Both scale how
          many positions a step consumes, so the epoch/offset are wrong if either changed. The
          caller passes the checkpoint's `num_processes` and `batch_size` and warns on a mismatch.
        - accelerate uses `even_batches=True` (its default). The `ceil(... / num_processes)` term
          mirrors that padding; with `even_batches=False` the per-epoch batch count differs and
          the boundary is off.
    """
    batches_per_epoch = math.ceil(math.ceil(num_frames / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    start_index = min(batches_into_epoch * batch_size * num_processes, num_frames)
    return {"epoch": epoch, "start_index": start_index}


def rng_state():
    """Tensor/primitives only, so training state can be loaded with weights_only=True."""
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy[0], torch.tensor(numpy[1].astype(np.int64)), *numpy[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], numpy[1].numpy().astype(np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _link(target, destination):
    """Replace only our pointer, never a real checkpoint directory."""
    if destination.exists() and not destination.is_symlink():
        raise FileExistsError(destination)
    pending = destination.with_name(f".{destination.name}.pending-{time.time_ns()}")
    pending.symlink_to(os.path.relpath(target, destination.parent))
    pending.replace(destination)


def save_training_checkpoint(
    output,
    step,
    cfg,
    model,
    optimizer,
    stats,
    samples,
    training,
    *,
    scheduler=None,
    execution_state=None,
    execution=None,
):
    """Adapt LeRobot's step/pretrained_model layout to AlohaMini policy manifests.

    A complete model, optimizer and RNG state are published together; latest
    pointers advance only after the complete checkpoint has been written.
    """
    name = f"{step:0{max(6, len(str(cfg['steps'])))}d}"
    checkpoint = output / "checkpoints" / name
    sharded = execution is not None and execution.sharded
    stage = None
    if not sharded or execution.main:
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        stage = str(tempfile.mkdtemp(prefix=name + ".pending-", dir=checkpoint.parent))
    model_state = None
    if sharded:
        stage = execution.gather(stage)[0]
        execution.distributed_checkpoint(Path(stage) / "distributed", model, optimizer)
        model_state = execution.accelerator.get_state_dict(model)
        if not execution.main:
            return checkpoint
    stage = Path(stage)
    state = {"step": step, "rng": rng_state()}
    if not sharded:
        state["optimizer"] = optimizer.state_dict()
    else:
        state["distributed_backend"] = "fsdp2"
    state.update(execution_state or {})
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    save_checkpoint(
        stage / "pretrained_model",
        model,
        stats,
        samples,
        training=training,
        model_state=model_state,
    )
    (stage / "pretrained_model/train_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (stage / "training_state").mkdir()
    torch.save(state, stage / "training_state/state.pt")
    stage.rename(checkpoint)
    _link(checkpoint, output / "checkpoints/last")
    _link(checkpoint / "pretrained_model", output / "checkpoint")
    return checkpoint


def load_training_checkpoint(path, cfg, samples, validation):
    """Reject changed data/optimization semantics before continuing a run."""
    path = Path(path).expanduser().resolve()
    if not (path / "training_state/state.pt").is_file():
        raise ValueError("Checkpoint has no resumable training state")
    saved = json.loads((path / "pretrained_model/train_config.json").read_text())
    manifest = json.loads((path / "pretrained_model/policy.json").read_text())
    if any(boundary["frame_index"] != 0 for boundary in manifest.get("sample_boundaries", [])):
        raise ValueError(
            "Checkpoint used within-episode sample boundaries; start a new run "
            "to train with episode-only windows"
        )
    if manifest.get("sample_filter") != samples.sample_filter:
        raise ValueError(
            "Checkpoint used a different sample-selection policy; resume would change the "
            "training population. Use the original trainer to resume, or --policy.path "
            "to start a new run with the selected episodes."
        )
    if "optimizer" not in saved:
        from types import SimpleNamespace

        from alohamini.learning.optim import resolve_optimization

        # Interpret old single-device checkpoints with the optimizer actually
        # used by that trainer, not a newly introduced policy default.
        saved = {**saved, "grad_clip_norm": saved.get("grad_clip_norm", 10.0)}
        saved = resolve_optimization(saved, SimpleNamespace(**manifest["config"]))
        if "grad_clip_norm" not in cfg:
            saved.pop("grad_clip_norm")
    saved.setdefault("world_size", 1)
    saved.setdefault("gradient_accumulation_steps", 1)
    saved.setdefault("distributed_backend", "ddp")
    cfg = {"distributed_backend": "ddp", **cfg}
    mutable = {
        "steps",
        "save_freq",
        "log_freq",
        "log_every",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "cpu_threads",
        "resume",
        "_checkpoint",
        "eval_steps",
        "eval_batch_size",
        "eval_num_workers",
        "eval_prefetch_factor",
        "eval_persistent_workers",
        "eval_log_freq",
        "video_cache_size",
        "video_backend",
        "camera_workers",
        "return_uint8",
    }

    def comparable(config, key):
        value = config.get(key)
        if key == "paper_preset" and isinstance(value, dict):
            # Preset metadata may repeat permitted overrides (notably SmolVLA
            # total steps). Compare their source fields under the rules above.
            value = {
                **value,
                "overrides": {
                    k: v for k, v in value.get("overrides", {}).items() if k not in mutable
                },
            }
        return value

    differences = [
        k
        for k in saved.keys() | cfg.keys()
        if k not in mutable and comparable(saved, k) != comparable(cfg, k)
    ]
    if differences:
        raise ValueError(f"Resume configuration changed: {sorted(differences)}")
    output = Path(cfg["output_dir"])
    complete = [
        entry
        for entry in (output / "checkpoints").iterdir()
        if entry.name.isdigit() and (entry / "training_state/state.pt").is_file()
    ]
    if not complete or max(complete, key=lambda entry: int(entry.name)).resolve() != path:
        raise ValueError(
            "Resume must use the latest complete checkpoint; older history is preserved"
        )
    if saved.get("scheduler", {}).get("type", "none") != "none" and saved["steps"] != cfg["steps"]:
        raise ValueError("Scheduled resume must retain total steps to preserve its LR schedule")
    if manifest["table_sha256"] != samples.table_sha256 or (
        validation is not None
        and manifest["training"].get("validation_table_sha256") != validation.table_sha256
    ):
        raise ValueError("Dataset changed since checkpoint; resume rejected")
    state = torch.load(path / "training_state/state.pt", map_location="cpu", weights_only=True)
    if not 0 < state["step"] < cfg["steps"]:
        raise ValueError(
            "--steps must exceed the checkpoint step (total updates, not additional updates)"
        )
    return path, manifest, state
