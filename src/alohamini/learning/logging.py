# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training console formatting adapted from LeRobot's logging utilities."""

import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime

import torch
from tqdm import tqdm

from alohamini.learning.metrics import format_metrics


class _LeRobotFormatter(logging.Formatter):
    """Console prefix from LeRobot utils.init_logging, using real caller locations."""

    def format(self, record):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        location = f"{record.pathname}:{record.lineno}"
        message = f"{record.levelname} {timestamp} {location[-15:]:>15} {record.getMessage()}"
        # Keep actionable tracebacks; LeRobot's custom formatter drops exc_info.
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        return message


class _TrainingConsoleFilter(logging.Filter):
    def filter(self, record):
        # These platform diagnostics have no corresponding LeRobot training output.
        return record.levelno >= logging.WARNING or (
            not getattr(record, "training_diagnostic", False)
            and record.name
            not in {
                "alohamini.learning.data",
                "alohamini.learning.validation",
            }
        )


def format_big_number(num, precision=0):
    """LeRobot's decimal suffixes and rounding, including rounded episode counts."""
    for suffix in ("", "K", "M", "B", "T", "Q"):
        if abs(num) < 1000.0:
            return f"{num:.{precision}f}{suffix}"
        num /= 1000.0
    return num


@contextmanager
def training_logging(main, *, training=False):
    """Scope console configuration to this package and restore embedded callers."""
    logger = logging.getLogger("alohamini")
    previous = logger.handlers[:], logger.level, logger.propagate
    handler = logging.StreamHandler() if main else logging.NullHandler()
    handler.setFormatter(
        _LeRobotFormatter()
        if training
        else logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    if training:
        handler.addFilter(_TrainingConsoleFilter())
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield
    finally:
        logger.handlers, logger.level, logger.propagate = previous
        handler.close()


@contextmanager
def log_stage(message):
    logger = logging.getLogger(__name__)
    logger.info("%s", message, extra={"training_diagnostic": True})
    started = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("%s failed after %.1fs", message, time.perf_counter() - started)
        raise
    else:
        logger.info(
            "%s completed in %.1fs",
            message,
            time.perf_counter() - started,
            extra={"training_diagnostic": True},
        )


class TrainingProgress:
    """LeRobot-style interval means, with actual consumed sample counts.

    Loss is already globally reduced by Execution.update. Timing and memory
    use the maximum rank's interval mean, matching the reference trainer.
    """

    def __init__(
        self, execution, *, frames, episodes, steps, samples=0, initial_step=0, metric_specs=()
    ):
        self.execution = execution
        self.frames, self.episodes, self.steps = frames, episodes, steps
        self.samples = samples
        self.initial_step = self.step = initial_step
        self.bar = None
        self.metric_specs = metric_specs
        self.reset()

    @contextmanager
    def track(self):
        """Use the reference's tqdm defaults, also for redirected/background output."""
        if self.execution.main:
            self.bar = tqdm(
                total=self.steps - self.initial_step,
                desc="Training",
                unit="step",
                disable="SLURM_JOB_ID" in os.environ,
                position=0,
                leave=True,
            )
        try:
            yield self
        finally:
            if self.bar is not None:
                self.bar.close()
                self.bar = None

    def reset(self):
        self.count = 0
        self.window_samples = 0
        self.sums = {}
        self.metric_sums = {}

    def update(self, record, samples):
        previous_step = self.step
        self.step = record.get("step", self.step + 1)
        self.count += 1
        self.samples += samples
        self.window_samples += samples
        for key in ("loss", "grad_norm", "lr", "dataloading_s", "update_s", "gpu_mem_gb"):
            if key in record:
                self.sums[key] = self.sums.get(key, 0.0) + record[key]
        for name, value in record.get("metrics", {}).items():
            self.metric_sums[name] = self.metric_sums.get(name, 0.0) + value
        if self.bar is not None:
            self.bar.update(self.step - previous_step)

    def summary(self, step):
        means = {key: value / self.count for key, value in self.sums.items()}
        means["metrics"] = {key: value / self.count for key, value in self.metric_sums.items()}
        keys = [key for key in ("dataloading_s", "update_s", "gpu_mem_gb") if key in means]
        if self.execution.world_size > 1:
            values = torch.tensor(
                [means[key] for key in keys], device=self.execution.device, dtype=torch.float64
            )
            values = self.execution.accelerator.reduce(values, reduction="max").tolist()
            means.update(zip(keys, values, strict=True))
        step_s = means["dataloading_s"] + means["update_s"]
        means.update(
            step=step,
            samples=self.samples,
            episodes=self.samples * self.episodes / self.frames,
            epochs=self.samples / self.frames,
            samples_per_s=self.window_samples / (self.count * step_s) if step_s > 0 else 0.0,
        )
        return means

    def log(self, step):
        # All ranks must participate in summary collectives before the main gate.
        m = self.summary(step)
        if self.execution.main:
            memory = f" mem_gb:{m['gpu_mem_gb']:.2f}" if "gpu_mem_gb" in m else ""
            logging.getLogger(__name__).info(
                "step:%s smpl:%s ep:%s epch:%.2f loss:%.3f grdn:%.3f "
                "lr:%.1e updt_s:%.3f data_s:%.3f smp/s:%.0f%s%s",
                format_big_number(step),
                format_big_number(m["samples"]),
                format_big_number(m["episodes"]),
                m["epochs"],
                m["loss"],
                m["grad_norm"],
                m["lr"],
                m["update_s"],
                m["dataloading_s"],
                m["samples_per_s"],
                memory,
                format_metrics(m["metrics"], self.metric_specs),
                stacklevel=2,
            )
        self.reset()


def consumed_samples(sampler, consumed):
    """Count batches including partial single-rank tails and distributed padding."""
    epochs, batches = divmod(consumed, sampler.batches)
    width = sampler.batch_size * sampler.world_size
    epoch_size = (
        sampler.batches * width if sampler.drop_last or sampler.world_size > 1 else sampler.size
    )
    return epochs * epoch_size + batches * width
