"""Shared loader settings for training and ordered offline evaluation."""

import torch
from torch.utils.data import DataLoader


def loader_options(cfg, *, evaluation=False):
    result = {
        "num_workers": cfg.get("num_workers", 0),
        "prefetch_factor": cfg.get("prefetch_factor", 4),
        "persistent_workers": cfg.get("persistent_workers", True),
    }
    if evaluation:
        for key in result:
            if cfg.get(f"eval_{key}") is not None:
                result[key] = cfg[f"eval_{key}"]
    return result


def make_loader(
    samples,
    *,
    device="cpu",
    batch_size=8,
    batch_sampler=None,
    num_workers=0,
    prefetch_factor=4,
    persistent_workers=True,
    generator=None,
):
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(prefetch_factor) is not int or prefetch_factor < 1:
        raise ValueError("prefetch_factor must be a positive integer")
    if type(persistent_workers) is not bool:
        raise ValueError("persistent_workers must be a boolean")
    if batch_sampler is None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    return DataLoader(
        samples,
        **(
            {"batch_sampler": batch_sampler}
            if batch_sampler is not None
            else {"batch_size": batch_size, "shuffle": False, "drop_last": False}
        ),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers else None,
        persistent_workers=persistent_workers and num_workers > 0,
        pin_memory=torch.device(device).type == "cuda",
        generator=generator,
    )
