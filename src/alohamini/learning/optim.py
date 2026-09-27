"""Resolved optimizer/scheduler recipes, independent of the training backend."""

import math

import torch


def resolve_optimization(settings, policy_config):
    """Policy defaults < explicit training overrides; preserve backbone parameter groups."""
    cfg = dict(settings)
    supplied = dict(cfg.get("optimizer", {}))
    kind = supplied.get("type", "adamw")
    if kind not in ("adam", "adamw", "sgd"):
        raise ValueError("optimizer.type must be adam, adamw or sgd")
    optimizer = dict(
        type=kind,
        lr=policy_config.optimizer_lr,
        weight_decay=policy_config.optimizer_weight_decay,
        grad_clip_norm=cfg.get(
            "grad_clip_norm", getattr(policy_config, "optimizer_grad_clip_norm", 10.0)
        ),
    )
    if kind == "sgd":
        optimizer.update(momentum=0.0, dampening=0.0, nesterov=False)
    else:
        optimizer.update(
            betas=list(getattr(policy_config, "optimizer_betas", (0.9, 0.999))),
            eps=getattr(policy_config, "optimizer_eps", 1e-8),
        )
    if supplied.keys() - optimizer.keys():
        raise ValueError(
            f"Unsupported {kind} options: {sorted(supplied.keys() - optimizer.keys())}"
        )
    optimizer.update(supplied)
    for key in ("lr", "weight_decay", "grad_clip_norm"):
        if not math.isfinite(optimizer[key]) or optimizer[key] < 0:
            raise ValueError(f"optimizer.{key} must be finite and nonnegative")
    cfg["optimizer"] = optimizer
    preset = (
        dict(
            type=getattr(policy_config, "scheduler_type", "cosine"),
            warmup_steps=policy_config.scheduler_warmup_steps,
            decay_steps=policy_config.scheduler_decay_steps,
            decay_lr=policy_config.scheduler_decay_lr,
        )
        if hasattr(policy_config, "scheduler_decay_steps")
        else dict(type="none")
    )
    supplied_scheduler = cfg.get("scheduler", {})
    if supplied_scheduler is None or supplied_scheduler.get("type") == "none":
        scheduler = {"type": "none"}
    else:
        scheduler = {**preset, **supplied_scheduler}
    kind = scheduler.get("type", "none")
    allowed = {"type"} if kind == "none" else {"type", "warmup_steps", "decay_steps", "decay_lr"}
    if kind not in ("none", "cosine", "warmup_cosine") or scheduler.keys() - allowed:
        raise ValueError("Unsupported scheduler configuration")
    if kind in ("cosine", "warmup_cosine"):
        scheduler = {"warmup_steps": 0, "decay_steps": cfg["steps"], "decay_lr": 0.0, **scheduler}
        if (
            type(scheduler["warmup_steps"]) is not int
            or type(scheduler["decay_steps"]) is not int
            or not 0 <= scheduler["warmup_steps"] < scheduler["decay_steps"]
            or not math.isfinite(scheduler["decay_lr"])
            or not 0 <= scheduler["decay_lr"] <= optimizer["lr"]
            or optimizer["lr"] <= 0
        ):
            raise ValueError("Invalid cosine warmup/decay configuration")
    cfg["scheduler"] = scheduler
    return cfg


def make_optimizer_and_scheduler(cfg, model):
    options = dict(cfg["optimizer"])
    kind = options.pop("type")
    options.pop("grad_clip_norm")
    optimizer = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[
        kind
    ](model.get_optim_params(), **options)
    return optimizer, make_scheduler(cfg, optimizer)


def make_scheduler(cfg, optimizer):
    schedule = cfg["scheduler"]
    if schedule["type"] == "none":
        return None
    # Retain the official SmolVLA cosine convention, including short-run scaling.
    warmup, decay = schedule["warmup_steps"], schedule["decay_steps"]
    if schedule["type"] == "cosine" and cfg["steps"] < decay:
        warmup = int(warmup * cfg["steps"] / decay)
        decay = cfg["steps"]
    alpha = schedule["decay_lr"] / cfg["optimizer"]["lr"]

    def multiplier(step):
        if schedule["type"] == "warmup_cosine":
            if step < warmup:
                initial = 1 / (warmup + 1)
                return initial + (1 - initial) * step / warmup
            progress = min(1.0, (step - warmup) / max(1, decay - warmup))
            return alpha + (1 - alpha) * (1 + math.cos(math.pi * progress)) / 2
        if step < warmup:
            return (
                1 / (warmup + 1) if step <= 0 else (1 / (warmup + 1) - 1) * (1 - step / warmup) + 1
            )
        return (1 - alpha) * 0.5 * (1 + math.cos(math.pi * min(step, decay) / decay)) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
