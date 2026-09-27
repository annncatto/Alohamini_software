# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local policy training, adapted from lerobot.scripts.lerobot_train.

Retains policy presets and epoch sampling, with shared optimizer creation,
Accelerate DDP/AMP, accumulation and resumable per-rank state. Dataset/policy factories
use AlohaMini interfaces; no Hub or robot connection.
"""

import fcntl
import json
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from accelerate import __version__ as accelerate_version
from torch.utils.data import DataLoader

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.execution import Execution, RankBatchSampler, validate_local_workers
from alohamini.learning.optim import make_optimizer_and_scheduler, resolve_optimization
from alohamini.learning.policy import NativePolicy, make_policy, make_processor
from alohamini.learning.processor import DEFAULT_IMAGE_SIZE
from alohamini.learning.training_state import (
    load_training_checkpoint,
    restore_rng,
    rng_state,
    save_training_checkpoint,
)
from alohamini.paths import WorkspacePaths
from alohamini.policies.registry import algorithm


def launch_training(settings):
    """Launch a detached local trainer with exclusive config/log/PID files."""
    paths = WorkspacePaths()
    processes = settings.get("num_processes", 1)
    validate_local_workers(processes, settings.get("device", "cuda"))
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
        command = [sys.executable, "-u", "-m", "alohamini.learning.train"]
        if processes > 1 or settings.get("distributed_backend") == "fsdp2":
            command = [
                sys.executable,
                "-u",
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={processes}",
                "-m",
                "alohamini.learning.train",
            ]
        process = subprocess.Popen(
            [*command, "--config", str(config_path)],
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
    policy.reset()

    def predict(batch):
        with torch.inference_mode(), policy.inference_context():
            prepared = policy.processor(batch)
            return policy.execution_action(
                policy.model.predict_action_chunk(prepared), context=prepared.get("_action_context")
            ).cpu()

    return _offline_evaluate_chunks(samples, predict, policy.names, batch_size=batch_size)


def _offline_evaluate_chunks(samples, predict, names, *, batch_size):
    """Accumulate physical-unit MAE for normal or collectively sharded predictions."""
    total = torch.zeros(len(names), dtype=torch.float64)
    count = 0
    for batch in DataLoader(samples, batch_size=batch_size):
        mask = ~batch["action_is_pad"]
        offsets = samples.delta_indices.get("action")
        if offsets is not None:
            mask = mask & (torch.tensor(offsets) >= 0)[None]
        predicted = predict(batch)
        if predicted.shape != batch["action"].shape:
            raise ValueError("Evaluation chunk_size must match checkpoint")
        total += ((predicted - batch["action"]).abs() * mask[..., None]).sum((0, 1))
        count += int(mask.sum())
    return {
        "valid_action_steps": count,
        "mae_by_action": dict(zip(names, (total / count).tolist(), strict=True)),
    }


def update_policy(model, batch, optimizer, grad_clip_norm, *, mixed_precision="none"):
    """Single-device branch of the original update_policy, without Accelerator."""
    start_time = time.perf_counter()
    model.train()
    autocast = (
        torch.autocast(next(model.parameters()).device.type, dtype=torch.bfloat16)
        if mixed_precision == "bfloat16"
        else nullcontext()
    )
    with autocast:
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
    components = algorithm(settings.get("policy", "act"))
    settings = components.apply_preset(settings)
    if settings.get("deterministic_algorithms"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    backend = settings.get("distributed_backend", "ddp")
    if backend == "fsdp2" and not components.fsdp_classes:
        raise ValueError(f"{settings.get('policy')}: FSDP2 boundaries are not adapted yet")
    execution = Execution(
        settings.get("device", "cuda"), settings.get("mixed_precision", "none"), backend
    )
    try:
        if settings.get("num_processes", execution.local_world_size) != execution.local_world_size:
            raise ValueError("Use --background or torchrun to launch the requested num_processes")
        return _train(settings, execution)
    finally:
        execution.close()


def _train(settings, execution):
    """Local offline training with periodic, fully resumable checkpoints."""
    components = algorithm(settings.get("policy", "act"))
    cfg = components.apply_preset(settings)
    cfg.setdefault("distributed_backend", "ddp")
    precision = cfg.get("mixed_precision", "none")
    accumulation = cfg.setdefault("gradient_accumulation_steps", 1)
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    cfg["world_size"] = execution.world_size
    steps, batch_size = cfg.get("steps", 1000), cfg.get("batch_size", 8)
    if type(steps) is not int or steps < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("steps and batch_size must be positive integers")
    seed = cfg.get("seed", 42)
    random.seed(seed + execution.rank)
    np.random.seed(seed + execution.rank)
    torch.manual_seed(seed + execution.rank)
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    device = str(execution.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this environment; select cpu explicitly if needed")
    deterministic = cfg.get("cudnn_deterministic", False)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.use_deterministic_algorithms(cfg.get("deterministic_algorithms", False))
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
    options = components.options(cfg, device)
    root = Path(cfg["dataset"]).expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    state = cfg.get("state", "none")
    if state == "auto":
        state = (
            "joint_position,base_velocity,lift_height"
            if "observation.state" in info["features"]
            else "none"
        )
    cameras = cfg.get("cameras")
    if cameras is None:
        cameras = info.get(
            "cameras",
            info.get("robot_metadata", {}).get(
                "cameras",
                [
                    key.removeprefix("observation.images.")
                    for key in info["features"]
                    if key.startswith("observation.images.")
                ],
            ),
        )
    args = dict(
        root=cfg["dataset"],
        **components.sample_spec(
            options,
            cameras=cameras,
        ),
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
    samples = AlohaMiniDataset(**args, episodes=train_episodes)
    validation = AlohaMiniDataset(**args, episodes=val_episodes) if val_episodes else None
    if cfg.get("drop_last", False) and len(samples) < batch_size:
        raise ValueError("drop_last would discard every sample; reduce batch_size")
    options["input_features"] = samples.input_features
    options["output_features"] = samples.output_features
    output = (
        Path(cfg["output_dir"]).expanduser().resolve()
        if cfg.get("output_dir")
        else WorkspacePaths().run(cfg["run_name"])
    )
    policy_config = components.config_class(**options)
    cfg = resolve_optimization(cfg, policy_config)
    cfg.update(
        output_dir=str(output),
        dataset=str(root),
        state=state,
        steps=steps,
        batch_size=batch_size,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        seed=seed,
        device=cfg.get("device", "cuda"),
    )
    if execution.main and not cfg.get("resume"):
        output.mkdir(parents=True, exist_ok=False)
    execution.barrier()
    with (output / "training.lock").open("a+b") if execution.main else nullcontext() as run_lock:
        try:
            if execution.main:
                fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another trainer is using this output directory") from exc
        resume_state = None
        if cfg.get("resume"):
            checkpoint_path, manifest, resume_state = load_training_checkpoint(
                cfg["_checkpoint"], cfg, samples, validation
            )
            policy = NativePolicy(
                checkpoint_path / "pretrained_model", device="cpu" if execution.sharded else device
            )
            model, processor, stats = policy.model, policy.processor, manifest["stats"]
            del policy
            if execution.sharded:
                processor = make_processor(model, stats, device)
        else:
            stats = components.statistics(samples, options)
            model = make_policy(cfg.get("policy", "act"), options, stats)
            if not execution.sharded:
                model = model.to(device)
            components.initialize(model, cfg)
            processor = make_processor(model, stats, device)
        attempt = time.time_ns()
        config_name = (
            "train.json"
            if resume_state is None
            else f"resume-{resume_state['step']}-{attempt}.json"
        )
        if execution.main:
            with (output / config_name).open("x") as stream:
                json.dump(cfg, stream, ensure_ascii=False, indent=2)
        if execution.main:
            print(
                f"Training samples={len(samples)} excluded={samples.excluded}; run={output}",
                flush=True,
            )
        optimizer, scheduler = make_optimizer_and_scheduler(cfg, model)
        start_step = 0
        consumed = 0
        if resume_state:
            if not execution.sharded:
                optimizer.load_state_dict(resume_state["optimizer"])
            start_step = resume_state["step"]
            if scheduler:
                scheduler.load_state_dict(resume_state["scheduler"])
            consumed = resume_state.get("consumed_batches", start_step)
            if execution.scaler is not None and resume_state.get("scaler"):
                execution.scaler.load_state_dict(resume_state["scaler"])
        sampler = RankBatchSampler(
            len(samples),
            batch_size,
            seed,
            execution.rank,
            execution.world_size,
            cfg.get("drop_last", False),
            consumed,
        )
        workers = cfg.get("num_workers", 0)
        loader_generator = torch.Generator().manual_seed(seed + execution.rank)
        local_resume = None
        if resume_state and "ranks" in resume_state:
            local_resume = resume_state["ranks"][execution.rank]
            loader_generator.set_state(local_resume["loader_epoch_rng"])
        loader_epoch_rng = loader_generator.get_state()
        loader = DataLoader(
            samples,
            batch_sampler=sampler,
            num_workers=workers,
            pin_memory=device.startswith("cuda"),
            prefetch_factor=cfg.get("prefetch_factor", 4) if workers else None,
            persistent_workers=cfg.get("persistent_workers", True) and workers > 0,
            generator=loader_generator,
        )
        wrapped, optimizer = execution.prepare(model, optimizer)
        if resume_state and execution.sharded:
            execution.distributed_checkpoint(
                checkpoint_path / "distributed", wrapped, optimizer, load=True
            )
        iterator = iter(loader)
        if (
            local_resume
            and "loader_next_rng" in local_resume
            and (consumed % sampler.batches or (workers and cfg.get("persistent_workers", True)))
        ):
            loader_generator.set_state(local_resume["loader_next_rng"])
        if resume_state:
            restore_rng(local_resume["rng"] if local_resume else resume_state["rng"])
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
            "execution_backend": "accelerate",
            "distributed_backend": cfg["distributed_backend"],
            "accelerate": accelerate_version,
            "mixed_precision": precision,
            "world_size": execution.world_size,
            "gradient_accumulation_steps": accumulation,
            "optimizer": cfg["optimizer"],
            "scheduler": cfg["scheduler"],
            "paper_preset": cfg.get("paper_preset"),
        }
        if validation:
            training["validation_table_sha256"] = validation.table_sha256
        if execution.main and not validation:
            print(
                "No held-out episodes: training loss does not measure generalization.", flush=True
            )
        metrics_name = (
            "metrics.jsonl" if not resume_state else f"metrics-from-{start_step}-{attempt}.jsonl"
        )
        with (output / metrics_name).open("x") if execution.main else nullcontext() as stream:
            step = start_step
            skipped = 0
            while step < steps:
                started = time.perf_counter()
                batches = []
                for _ in range(accumulation):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        loader_epoch_rng = loader_generator.get_state()
                        iterator = iter(loader)
                        batch = next(iterator)
                    batches.append(batch)
                    consumed += 1
                data_s = time.perf_counter() - started
                update = execution.update(
                    wrapped,
                    batches,
                    optimizer,
                    cfg["optimizer"]["grad_clip_norm"],
                    processor,
                    reduction=components.loss_counts,
                )
                if not update["optimizer_step"]:
                    skipped += 1
                    if execution.main:
                        print(
                            f"AMP overflow: skipped update; scale={execution.scaler.get_scale()}",
                            flush=True,
                        )
                    if skipped >= 20:
                        raise RuntimeError(
                            "20 consecutive AMP overflows; check data/model precision"
                        )
                    continue
                skipped = 0
                step += 1
                record = {
                    "step": step,
                    "dataloading_s": data_s,
                    **update,
                    "update_s": time.perf_counter() - started - data_s,
                }
                if scheduler:
                    scheduler.step()
                if stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                if execution.main and (
                    step == start_step + 1 or (log_freq and step % log_freq == 0) or step == steps
                ):
                    print(json.dumps(record), flush=True)
                if (
                    (execution.main or execution.sharded)
                    and validation
                    and eval_steps
                    and step % eval_steps == 0
                ):
                    # Evaluation must not perturb dropout/CVAE RNG for subsequent updates.
                    rng = rng_state()
                    try:
                        model.eval()
                        losses = []
                        with torch.no_grad(), execution.autocast():
                            for val_batch in DataLoader(validation, batch_size=batch_size):
                                loss, _ = model(processor(val_batch))
                                losses.append(loss.item())
                        if execution.main:
                            print(
                                json.dumps({"step": step, "eval_loss": sum(losses) / len(losses)}),
                                flush=True,
                            )
                    finally:
                        restore_rng(rng)
                if step % save_freq == 0 or step == steps:
                    # Snapshot consumed work, not sampler prefetch; keep one RNG per rank.
                    ranks = execution.gather(
                        dict(
                            rng=rng_state(),
                            loader_epoch_rng=(
                                loader_epoch_rng
                                if consumed % sampler.batches
                                else loader_generator.get_state()
                            ),
                            loader_next_rng=loader_generator.get_state(),
                        )
                    )
                    if execution.main or execution.sharded:
                        saved = save_training_checkpoint(
                            output,
                            step,
                            cfg,
                            model,
                            optimizer,
                            stats,
                            samples,
                            {**training, "steps": step},
                            scheduler=scheduler,
                            execution=execution,
                            execution_state=dict(
                                ranks=ranks,
                                consumed_batches=consumed,
                                scaler=execution.scaler.state_dict() if execution.scaler else {},
                            ),
                        )
                        if execution.main:
                            print(f"Checkpoint policy after step {step}: {saved}", flush=True)
                    execution.barrier()
        checkpoint = output / "checkpoint"
        if execution.sharded and validation:
            from alohamini.learning.processor import scale_action

            # Every rank traverses identical batches: custom prediction methods
            # also need FSDP's collective pre/post-forward hooks.
            model.eval()
            model.reset()

            def predict(batch):
                with torch.no_grad(), execution.autocast():
                    prepared = processor(batch)
                    return scale_action(
                        processor.action(
                            model.predict_action_chunk(prepared),
                            context=prepared.get("_action_context"),
                        ),
                        getattr(model.config, "inference_action_scale_dims", ()),
                        getattr(model.config, "inference_action_scale", 1.0),
                    ).cpu()

            metrics = _offline_evaluate_chunks(
                validation,
                predict,
                validation.info["features"]["action"]["names"],
                batch_size=batch_size,
            )
            if execution.main:
                (output / "offline-evaluation.json").write_text(
                    json.dumps(metrics, indent=2) + "\n"
                )
                print(json.dumps(metrics), flush=True)
        elif execution.main and validation:
            del wrapped, model, optimizer
            policy = NativePolicy(checkpoint, device=device)
            metrics = offline_evaluate(policy, validation, batch_size=batch_size)
            (output / "offline-evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics), flush=True)
        elif execution.main:
            print("No held-out evaluation or generalization claim.", flush=True)
        if execution.main:
            print(f"Checkpoint: {checkpoint}", flush=True)
        execution.barrier()
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
