"""Training console summaries; per-update JSON metrics stay in the run directory."""

import logging
import time
from contextlib import contextmanager

import torch


@contextmanager
def training_logging(main):
    """Scope console configuration to this package and restore embedded callers."""
    logger = logging.getLogger("alohamini")
    previous = logger.handlers[:], logger.level, logger.propagate
    handler = logging.StreamHandler() if main else logging.NullHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
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
    logger.info("%s", message)
    started = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("%s failed after %.1fs", message, time.perf_counter() - started)
        raise
    else:
        logger.info("%s completed in %.1fs", message, time.perf_counter() - started)


class TrainingProgress:
    """LeRobot-style interval means, with actual consumed sample counts.

    Loss is already globally reduced by Execution.update. Timing and memory
    use the maximum rank's interval mean, matching the reference trainer.
    """

    def __init__(self, execution, *, frames, episodes, steps, samples=0):
        self.execution = execution
        self.frames, self.episodes, self.steps = frames, episodes, steps
        self.samples = samples
        self.reset()

    def reset(self):
        self.count = 0
        self.window_samples = 0
        self.sums = {}

    def update(self, record, samples):
        self.count += 1
        self.samples += samples
        self.window_samples += samples
        for key in ("loss", "grad_norm", "lr", "dataloading_s", "update_s", "gpu_mem_gb"):
            if key in record:
                self.sums[key] = self.sums.get(key, 0.0) + record[key]

    def summary(self, step):
        means = {key: value / self.count for key, value in self.sums.items()}
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
            eta_s=max(0, self.steps - step) * step_s,
        )
        return means

    def log(self, step):
        # All ranks must participate in summary collectives before the main gate.
        m = self.summary(step)
        if self.execution.main:
            memory = f" mem_gb:{m['gpu_mem_gb']:.2f}" if "gpu_mem_gb" in m else ""
            logging.getLogger(__name__).info(
                "step:%d/%d smpl:%d ep:%.2f epch:%.3f loss:%.3f grdn:%.3f "
                "lr:%.2e updt_s:%.3f data_s:%.3f smp/s:%.1f%s eta_s:%.0f",
                step,
                self.steps,
                m["samples"],
                m["episodes"],
                m["epochs"],
                m["loss"],
                m["grad_norm"],
                m["lr"],
                m["update_s"],
                m["dataloading_s"],
                m["samples_per_s"],
                memory,
                m["eta_s"],
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
