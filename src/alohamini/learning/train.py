# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local ACT/AM-ACT training, adapted from lerobot.scripts.lerobot_train.

Retains the single-device FP32 update order, policy AdamW presets, epoch
sampling, periodic checkpoints and resumable training state. Dataset/policy
factories use the native platform; no remote jobs, Hub or robot connection.
"""

import fcntl
import json
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from alohamini.learning.data import DEFAULT_IMAGE_SIZE, NativeSamples
from alohamini.learning.policy import NativePolicy, Processor, make_policy
from alohamini.learning.training_state import (
    EpisodeAwareSampler,
    compute_sampler_state,
    load_training_checkpoint,
    restore_rng,
    save_training_checkpoint,
)
from alohamini.paths import WorkspacePaths


def launch_training(settings):
    """Launch a detached local trainer with exclusive config/log/PID files."""
    paths = WorkspacePaths()
    run = (
        Path(settings["output_dir"]).expanduser().resolve()
        if settings.get("output_dir")
        else paths.run(settings["run_name"])
    )
    if run.exists() and not settings.get("resume"):
        raise FileExistsError(run)
    directory = paths.logs / "training"
    directory.mkdir(parents=True, exist_ok=True)
    label = run.name + (f"_resume_{time.time_ns()}" if settings.get("resume") else "")
    config_path = directory / f"{label}.json"
    log_path = directory / f"{label}.log"
    pid_path = directory / f"{label}.pid"
    if any(p.exists() for p in (config_path, log_path, pid_path)):
        raise FileExistsError("Training launch files already exist; choose a new run_name")
    with config_path.open("x") as stream:
        json.dump(settings, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with log_path.open("xb") as stream:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "alohamini.learning.train", "--config", str(config_path)],
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    with pid_path.open("x") as stream:
        stream.write(f"{process.pid}\n")
    return {
        "pid": process.pid,
        "log": str(log_path),
        "pid_file": str(pid_path),
        "config": str(config_path),
        "checkpoint": str(run / "checkpoint"),
    }


def offline_evaluate(policy, samples, *, batch_size=8):
    """Physical-unit MAE per dimension, excluding padding; no robot connection.

    Predicts independent chunks, not a closed-loop task success rate. Execution
    scaling is included so reported outputs match the checkpoint's action meaning.
    """
    if (
        samples.info["robot_metadata"] != policy.robot_metadata
        or samples.state != policy.manifest["state"]
        or samples.cameras != policy.manifest["cameras"]
        or samples.image_size != tuple(policy.manifest["image_size"])
        or samples.info["fps"] != policy.fps
    ):
        raise ValueError("Offline dataset does not match checkpoint observation/action contract")
    total = torch.zeros(len(policy.names), dtype=torch.float64)
    count = 0
    policy.reset()
    for batch in DataLoader(samples, batch_size=batch_size):
        mask = ~batch["action_is_pad"]
        with torch.inference_mode():
            predicted = policy.processor.action(
                policy.model.predict_action_chunk(policy.processor(batch))
            ).cpu()
        if predicted.shape != batch["action"].shape:
            raise ValueError("Evaluation chunk_size must match checkpoint")
        total += ((predicted - batch["action"]).abs() * mask[..., None]).sum((0, 1))
        count += int(mask.sum())
    return {
        "valid_action_steps": count,
        "mae_by_action": dict(zip(policy.names, (total / count).tolist(), strict=True)),
    }


def update_policy(model, batch, optimizer, grad_clip_norm):
    """Single-device branch of the original update_policy, without Accelerator."""
    start_time = time.perf_counter()
    model.train()
    loss, output_dict = model.forward(batch)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss; no optimizer step performed")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        grad_clip_norm if grad_clip_norm > 0 else float("inf"),
        error_if_nonfinite=True,
    )
    optimizer.step()
    optimizer.zero_grad()
    if callable(getattr(model, "update", None)):
        model.update()
    return {
        **(output_dict or {}),
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "lr": optimizer.param_groups[0]["lr"],
        "update_s": time.perf_counter() - start_time,
    }


def train(settings):
    """Local offline training with periodic, fully resumable checkpoints."""
    cfg = deepcopy(settings)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "This local trainer supports one process; distributed training is not enabled"
        )
    steps, batch_size = cfg.get("steps", 1000), cfg.get("batch_size", 8)
    if type(steps) is not int or steps < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("steps and batch_size must be positive integers")
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this environment; select cpu explicitly if needed")
    deterministic = cfg.get("cudnn_deterministic", False)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    save_freq = cfg.get("save_freq", 20_000)
    log_freq = cfg.get("log_freq", cfg.get("log_every", 200))
    eval_steps = cfg.get("eval_steps", 0)
    for key, number, minimum in (
        ("save_freq", save_freq, 1),
        ("log_freq", log_freq, 0),
        ("eval_steps", eval_steps, 0),
        ("num_workers", cfg.get("num_workers", 0), 0),
    ):
        if type(number) is not int or number < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    options = dict(cfg.get("model", {}))
    chunk = options.get("chunk_size", 100)
    # An explicit shorter chunk also needs an explicit execution horizon.
    options.setdefault("n_action_steps", chunk)
    root = Path(cfg["dataset"]).expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    state = cfg.get("state", "none")
    if state == "auto":
        state = (
            "joint_position,base_velocity,lift_height"
            if "observation.state" in info["features"]
            else "none"
        )
    args = dict(
        root=cfg["dataset"],
        chunk_size=chunk,
        state=state,
        cameras=cfg.get("cameras"),
        image_size=tuple(cfg.get("image_size", DEFAULT_IMAGE_SIZE)),
        review_note=cfg.get("review_note", ""),
    )
    val_episodes = cfg.get("val_episodes", [])
    train_episodes = cfg.get("train_episodes")
    if train_episodes is None:
        count = info.get("total_episodes")
        if count is None:
            count = len(list((root / "episodes").glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]")))
        train_episodes = [i for i in range(count) if i not in val_episodes]
    if set(train_episodes) & set(val_episodes):
        raise ValueError("Train and validation episodes must be disjoint")
    if eval_steps and not val_episodes:
        raise ValueError("--eval_steps requires held-out --dataset.eval_episodes")
    samples = NativeSamples(**args, episodes=train_episodes)
    validation = NativeSamples(**args, episodes=val_episodes) if val_episodes else None
    options["input_features"] = samples.input_features
    options["output_features"] = samples.output_features
    output = (
        Path(cfg["output_dir"]).expanduser().resolve()
        if cfg.get("output_dir")
        else WorkspacePaths().run(cfg["run_name"])
    )
    cfg.update(
        output_dir=str(output),
        dataset=str(root),
        state=state,
        steps=steps,
        batch_size=batch_size,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        seed=seed,
        device=device,
    )
    if not cfg.get("resume"):
        output.mkdir(parents=True, exist_ok=False)
    with (output / "training.lock").open("a+b") as run_lock:
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another trainer is using this output directory") from exc
        resume_state = None
        if cfg.get("resume"):
            checkpoint_path, manifest, resume_state = load_training_checkpoint(
                cfg["_checkpoint"], cfg, samples, validation
            )
            policy = NativePolicy(checkpoint_path / "pretrained_model", device=device)
            model, processor, stats = policy.model, policy.processor, manifest["stats"]
            del policy
        else:
            stats = samples.statistics()
            model = make_policy(cfg.get("policy", "act"), options, stats).to(device)
            processor = Processor(stats, device)
        attempt = time.time_ns()
        config_name = (
            "train.json"
            if resume_state is None
            else f"resume-{resume_state['step']}-{attempt}.json"
        )
        with (output / config_name).open("x") as stream:
            json.dump(cfg, stream, ensure_ascii=False, indent=2)
        print(
            f"Training samples={len(samples)} excluded={samples.excluded}; run={output}", flush=True
        )
        optimizer = torch.optim.AdamW(
            model.get_optim_params(),
            lr=model.config.optimizer_lr,
            weight_decay=model.config.optimizer_weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        # ACT and AM-ACT's original get_scheduler_preset() both return None.
        start_step = 0
        if resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
            start_step = resume_state["step"]
        sampler = EpisodeAwareSampler([0], [len(samples)], shuffle=True, seed=seed)
        sampler.load_state_dict(compute_sampler_state(start_step, len(samples), batch_size, 1))
        workers = cfg.get("num_workers", 0)
        loader = DataLoader(
            samples,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=device.startswith("cuda"),
            prefetch_factor=cfg.get("prefetch_factor", 4) if workers else None,
            persistent_workers=cfg.get("persistent_workers", True) and workers > 0,
            drop_last=False,
            generator=torch.Generator().manual_seed(seed),
        )
        iterator = iter(loader)
        if resume_state:
            restore_rng(resume_state["rng"])
        optimizer.zero_grad()
        training = {
            "seed": seed,
            "train_episodes": train_episodes,
            "val_episodes": val_episodes,
            "overfit_smoke": cfg.get("overfit_smoke", False),
            "dataset": str(samples.root),
            "samples": len(samples),
            "excluded_samples": samples.excluded,
            "torch": str(torch.__version__),
        }
        if validation:
            training["validation_table_sha256"] = validation.table_sha256
        if not validation:
            print(
                "No held-out episodes: training loss does not measure generalization.", flush=True
            )
        metrics_name = (
            "metrics.jsonl" if not resume_state else f"metrics-from-{start_step}-{attempt}.jsonl"
        )
        with (output / metrics_name).open("x") as stream:
            for step in range(start_step + 1, steps + 1):
                started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = processor(batch)
                data_s = time.perf_counter() - started
                record = {
                    "step": step,
                    "dataloading_s": data_s,
                    **update_policy(model, batch, optimizer, cfg.get("grad_clip_norm", 10.0)),
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                if step == start_step + 1 or (log_freq and step % log_freq == 0) or step == steps:
                    print(json.dumps(record), flush=True)
                if validation and eval_steps and step % eval_steps == 0:
                    # Evaluation must not perturb dropout/CVAE RNG for subsequent updates.
                    from alohamini.learning.training_state import rng_state

                    rng = rng_state()
                    try:
                        model.eval()
                        losses = []
                        with torch.no_grad():
                            for val_batch in DataLoader(validation, batch_size=batch_size):
                                loss, _ = model(processor(val_batch))
                                losses.append(loss.item())
                        print(
                            json.dumps({"step": step, "eval_loss": sum(losses) / len(losses)}),
                            flush=True,
                        )
                    finally:
                        restore_rng(rng)
                if step % save_freq == 0 or step == steps:
                    saved = save_training_checkpoint(
                        output,
                        step,
                        cfg,
                        model,
                        optimizer,
                        stats,
                        samples,
                        {**training, "steps": step},
                    )
                    print(f"Checkpoint policy after step {step}: {saved}", flush=True)
        checkpoint = output / "checkpoint"
        del model, optimizer
        if validation:
            policy = NativePolicy(checkpoint, device=device)
            metrics = offline_evaluate(policy, validation, batch_size=batch_size)
            (output / "offline-evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics), flush=True)
        else:
            print("No held-out evaluation or generalization claim.", flush=True)
        print(f"Checkpoint: {checkpoint}", flush=True)
        return checkpoint


def main():
    from alohamini.learning.train_config import parse_training_args

    cfg, background = parse_training_args()
    if background:
        job = launch_training(cfg)
        print(json.dumps(job, ensure_ascii=False, indent=2))
        print("tail -f", job["log"])
    else:
        train(cfg)


if __name__ == "__main__":
    main()
