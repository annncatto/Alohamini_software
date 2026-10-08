"""Ordered evaluation with explicit timing and algorithm-declared loss denominators."""

import logging
import math
import time

import torch
from torch.utils.data import default_collate

logger = logging.getLogger(__name__)


def timed_batches(loader, *, device="cpu", label="Evaluation", log_freq=50, action_offsets=None):
    """Time exposed loader waits and computation, including transfers/processing.

    Synchronize CUDA at compute boundaries so kernel time is not mislabeled as
    loader waiting. Prefetch can run concurrently; data_s is the exposed wait.
    """
    if type(log_freq) is not int or log_freq < 1:
        raise ValueError("eval_log_freq must be a positive integer")
    device = torch.device(device)
    logger.info("%s: starting %d batches, %d samples", label, len(loader), len(loader.dataset))
    started = last_log = time.perf_counter()
    iterator = iter(loader)
    data_s = compute_s = 0.0
    samples = valid = 0
    for number in range(1, len(loader) + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        before = time.perf_counter()
        batch = next(iterator)
        data_s += time.perf_counter() - before
        before = time.perf_counter()
        yield batch
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_s += time.perf_counter() - before
        samples += len(batch["action"])
        if "action_is_pad" in batch:
            mask = ~batch["action_is_pad"]
            if action_offsets is not None:
                mask = mask & (torch.tensor(action_offsets) >= 0)[None]
            valid += int(mask.sum())
        now = time.perf_counter()
        if number == 1 or number % log_freq == 0 or number == len(loader) or now - last_log >= 10:
            elapsed = now - started
            logger.info(
                "%s batches:%d/%d samples:%d valid_action_steps:%d "
                "data_s:%.3f compute_s:%.3f smp/s:%.1f eta_s:%.1f",
                label,
                number,
                len(loader),
                samples,
                valid,
                data_s / number,
                compute_s / number,
                samples / max(elapsed, 1e-9),
                elapsed / number * (len(loader) - number),
            )
            last_log = now


def loss_denominators(samples, reduction):
    """Count selected numeric targets without loading images or fitting statistics."""
    totals = {}
    for start in range(0, len(samples), 128):
        batch = default_collate(
            [samples.action_metadata(i) for i in range(start, min(start + 128, len(samples)))]
        )
        counts = reduction(batch)
        if totals and totals.keys() != counts.keys():
            raise ValueError("Evaluation loss terms changed between batches")
        for key, value in counts.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError("Evaluation loss denominators must be finite and nonnegative")
            totals[key] = totals.get(key, 0) + value
    if not totals or any(value <= 0 for value in totals.values()):
        raise ValueError("Evaluation loss terms require positive denominators")
    return totals


def evaluate_loss(model, processor, loader, reduction, totals, *, device, log_freq=50):
    """Sum weighted batch contributions, separately for each model loss term."""
    loss_sum = 0.0
    for raw in timed_batches(loader, device=device, label="Validation", log_freq=log_freq):
        counts = reduction(raw)
        batch = processor(raw)
        batch.update({key: counts[key] / total for key, total in totals.items()})
        loss, _ = model(batch)
        value = loss.item()
        if not math.isfinite(value):
            raise RuntimeError("Non-finite validation loss")
        loss_sum += value
    return loss_sum
