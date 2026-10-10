# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local policy training, adapted from lerobot.scripts.lerobot_train.

Retains policy presets and epoch sampling, with shared optimizer creation,
Accelerate DDP/AMP, accumulation and resumable per-rank state. Dataset/policy factories
use AlohaMini interfaces; no Hub or robot connection.
"""

import fcntl
import json
import logging
import os
import random
import shlex
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from accelerate import __version__ as accelerate_version

from alohamini.learning.checkpoint import (
    initialize_policy,
    resolve_pretrained,
    validate_pretrained,
)
from alohamini.learning.data import AlohaMiniDataset, DatasetInspection
from alohamini.learning.execution import Execution, RankBatchSampler, validate_local_workers
from alohamini.learning.loading import loader_options, make_loader, resolve_data_pipeline
from alohamini.learning.logging import (
    TrainingProgress,
    consumed_samples,
    format_big_number,
    log_stage,
    training_logging,
)
from alohamini.learning.metrics import MetricAccumulator, format_metrics
from alohamini.learning.optim import make_optimizer_and_scheduler, resolve_optimization
from alohamini.learning.policy import NativePolicy, make_policy, make_processor
from alohamini.learning.statistics import sample_arguments, training_statistics, write_statistics
from alohamini.learning.training_state import (
    load_training_checkpoint,
    restore_rng,
    rng_state,
    save_training_checkpoint,
)
from alohamini.learning.validation import evaluate_loss, loss_denominators, timed_batches
from alohamini.paths import WorkspacePaths
from alohamini.policies.registry import algorithm

# Keep the package logger when launched with ``python -m`` (__name__ == "__main__").
logger = logging.getLogger("alohamini.learning.train")


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
            [*command, "--config", str(config_path), "--foreground"],
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


def offline_evaluate(
    policy,
    samples,
    *,
    batch_size=8,
    num_workers=0,
    prefetch_factor=4,
    persistent_workers=True,
    log_freq=50,
    loader=None,
):
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
        or (samples.selection.feature if samples.selection else None)
        != policy.manifest["state_feature"]
    ):
        raise ValueError("Offline dataset does not match checkpoint observation/action contract")
    policy.reset()

    def predict(batch):
        # Offline chunks are independent observations, not a shared task phase.
        policy.reset()
        with torch.inference_mode(), policy.inference_context():
            prepared = policy.processor(policy.fixed_input(batch))
            return policy.execution_action(
                policy.model.predict_action_chunk(prepared), context=prepared.get("_action_context")
            ).cpu()

    device = next(policy.model.parameters()).device
    if loader is None:
        loader = make_loader(
            samples,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            generator=torch.Generator().manual_seed(0),
        )
    return _offline_evaluate_chunks(
        samples,
        predict,
        policy.names,
        batch_size=batch_size,
        loader=loader,
        device=device,
        log_freq=log_freq,
    )


def _offline_evaluate_chunks(
    samples,
    predict,
    names,
    *,
    batch_size,
    loader=None,
    device="cpu",
    log_freq=50,
):
    """Accumulate physical-unit MAE for normal or collectively sharded predictions."""
    total = torch.zeros(len(names), dtype=torch.float64)
    count = 0
    if loader is None:
        loader = make_loader(samples, device=device, batch_size=batch_size)
    for batch in timed_batches(
        loader,
        device=device,
        label="Offline MAE",
        log_freq=log_freq,
        action_offsets=samples.delta_indices.get("action"),
    ):
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
    settings = resolve_pretrained(settings)
    components = algorithm(settings.get("policy", "act"))
    settings = components.apply_preset(settings)
    settings = resolve_data_pipeline(settings)
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
        with training_logging(execution.main, training=True):
            logger.info("%s", pformat(settings))
            logger.info("Logs will be saved locally.")
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
    eval_batch_size = cfg.get("eval_batch_size", batch_size)
    eval_log_freq = cfg.get("eval_log_freq", 50)
    for key, number, minimum in (
        ("save_freq", save_freq, 1),
        ("log_freq", log_freq, 0),
        ("eval_steps", eval_steps, 0),
        ("num_workers", cfg.get("num_workers", 0), 0),
        ("eval_batch_size", eval_batch_size, 1),
        ("eval_log_freq", eval_log_freq, 1),
    ):
        if type(number) is not int or number < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    options = components.options(cfg, device)
    args, train_episodes, val_episodes = sample_arguments(cfg, components, options)
    root, state = Path(args["root"]), args["state"]
    if eval_steps and not val_episodes:
        raise ValueError("--eval_steps requires held-out --dataset.eval_episodes")
    logger.info("Creating dataset")
    inspection = DatasetInspection()
    with log_stage(f"Creating training dataset ({len(train_episodes)} episodes)"):
        samples = AlohaMiniDataset(**args, episodes=train_episodes, inspection=inspection)
    validation = None
    if val_episodes:
        with log_stage(f"Creating validation dataset ({len(val_episodes)} episodes)"):
            validation = AlohaMiniDataset(**args, episodes=val_episodes, inspection=inspection)
    if cfg.get("drop_last", False) and len(samples) < batch_size:
        raise ValueError("drop_last would discard every sample; reduce batch_size")
    pretrained_manifest = validate_pretrained(cfg, samples)
    options["input_features"] = samples.input_features
    options["output_features"] = samples.output_features
    if not cfg.get("resume") and hasattr(components, "prepare_options"):
        components.prepare_options(options, samples)
    output = (
        Path(cfg["output_dir"]).expanduser().resolve()
        if cfg.get("output_dir")
        else WorkspacePaths().run(cfg["run_name"])
    )
    policy_config = components.config_class(**options)
    if not cfg.get("resume") and hasattr(components, "prepare_options"):
        # Resume must validate the same resolved discrete coordinates/weights,
        # including groups that intentionally omit these non-regression axes.
        cfg["model"] = asdict(policy_config)
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
    logger.debug("Resolved training configuration:\n%s", pformat(cfg))
    logger.debug("Policy configuration:\n%s", pformat(policy_config))
    logger.debug(
        "Dataset: %s; training frames=%d samples=%d window_excluded=%d episodes=%d; "
        "validation samples=%d episodes=%d",
        root,
        len(samples.rows),
        len(samples),
        samples.excluded,
        len(train_episodes),
        len(validation) if validation is not None else 0,
        len(val_episodes),
    )
    logger.debug(
        "Device=%s precision=%s backend=%s steps=%d; effective batch size: %d x %d x %d = %d "
        "(per rank x ranks x accumulation)",
        device,
        precision,
        cfg["distributed_backend"],
        steps,
        batch_size,
        execution.world_size,
        accumulation,
        batch_size * execution.world_size * accumulation,
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
        logger.info("Creating policy")
        resume_state = None
        if cfg.get("resume"):
            logger.debug("Loading checkpoint: %s", cfg["_checkpoint"])
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
            with log_stage(
                "Loading normalization statistics"
                if cfg.get("stats") or cfg.get("normalization") == "checkpoint"
                else "Computing normalization statistics"
            ):
                artifact = training_statistics(
                    components,
                    samples,
                    options,
                    cfg.get("policy", "act"),
                    cfg.get("stats"),
                    checkpoint=pretrained_manifest
                    if cfg.get("normalization") == "checkpoint"
                    else None,
                )
            stats = artifact["stats"]
            if execution.main:
                write_statistics(output / "statistics.json", artifact)
            with log_stage("Creating policy and processors"):
                model = make_policy(cfg.get("policy", "act"), options, stats)
                if not execution.sharded:
                    model = model.to(device)
                if cfg.get("pretrained_path"):
                    logger.debug("Loading initial policy weights: %s", cfg["pretrained_path"])
                initialize_policy(components, model, cfg)
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
        logger.info("Creating optimizer and scheduler")
        optimizer, scheduler = make_optimizer_and_scheduler(cfg, model)
        logger.info("Output dir: %s", output)
        logger.info("cfg.steps=%d (%s)", steps, format_big_number(steps))
        logger.info("dataset.num_frames=%d (%s)", len(samples), format_big_number(len(samples)))
        logger.info("dataset.num_episodes=%d", len(train_episodes))
        effective_batch_size = batch_size * execution.world_size * accumulation
        if accumulation == 1:
            logger.info(
                "Effective batch size: %d x %d = %d",
                batch_size,
                execution.world_size,
                effective_batch_size,
            )
        else:
            logger.info(
                "Effective batch size: %d x %d x %d = %d",
                batch_size,
                execution.world_size,
                accumulation,
                effective_batch_size,
            )
        learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("num_learnable_params=%d (%s)", learnable, format_big_number(learnable))
        logger.info("num_total_params=%d (%s)", total, format_big_number(total))
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
        if resume_state:
            logger.debug("Resuming at step %d; consumed microbatches=%d", start_step, consumed)
        metric_specs = components.metric_specs(model.config)
        progress = TrainingProgress(
            execution,
            frames=len(samples),
            episodes=len(train_episodes),
            steps=steps,
            samples=consumed_samples(sampler, consumed),
            initial_step=start_step,
            metric_specs=metric_specs,
        )
        workers = cfg.get("num_workers", 0)
        logger.debug("Creating dataloader: workers=%d batch_size=%d", workers, batch_size)
        loader_generator = torch.Generator().manual_seed(seed + execution.rank)
        local_resume = None
        if resume_state and "ranks" in resume_state:
            local_resume = resume_state["ranks"][execution.rank]
            loader_generator.set_state(local_resume["loader_epoch_rng"])
        loader_epoch_rng = loader_generator.get_state()
        loader = make_loader(
            samples,
            device=device,
            batch_sampler=sampler,
            **loader_options(cfg),
            generator=loader_generator,
        )
        eval_loader = eval_totals = None
        if validation is not None and (execution.main or execution.sharded):
            logger.debug(
                "Creating reusable evaluation loader: batch_size=%d settings=%s",
                eval_batch_size,
                loader_options(cfg, evaluation=True),
            )
            eval_loader = make_loader(
                validation,
                device=device,
                batch_size=eval_batch_size,
                **loader_options(cfg, evaluation=True),
                generator=torch.Generator().manual_seed(seed),
            )
            if eval_steps:
                eval_totals = loss_denominators(validation, components.loss_counts)
                logger.debug("Validation loss reduction: algorithm denominators %s", eval_totals)
        with log_stage("Preparing execution backend"):
            wrapped, optimizer = execution.prepare(model, optimizer)
        if resume_state and execution.sharded:
            execution.distributed_checkpoint(
                checkpoint_path / "distributed", wrapped, optimizer, load=True
            )
        with log_stage("Starting dataloader workers"):
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
            logger.warning("No held-out episodes: training loss does not measure generalization.")
        metrics_name = (
            "metrics.jsonl" if not resume_state else f"metrics-from-{start_step}-{attempt}.jsonl"
        )
        if execution.main:
            schema = {
                "version": 1,
                "training_console": "mean of successful global optimizer steps in log window",
                "training_jsonl": "one successful global optimizer step per record",
                "validation": "dataset means using each declared denominator",
                "loss": "original differentiable objective; separately weighted terms",
                "components": [spec.schema() for spec in metric_specs],
            }
            (output / metrics_name.replace(".jsonl", "-schema.json")).write_text(
                json.dumps(schema, indent=2) + "\n"
            )
        with (
            progress.track(),
            (output / metrics_name).open("x") if execution.main else nullcontext() as stream,
            (output / ("validation-" + metrics_name)).open("x")
            if execution.main and validation
            else nullcontext() as validation_stream,
        ):
            logger.debug("Per-update metrics: %s", output / metrics_name)
            logger.info(
                "Start offline training on a fixed dataset, with effective batch size: %d",
                effective_batch_size,
            )
            step = start_step
            skipped = 0
            while step < steps:
                started = time.perf_counter()
                previous_consumed = consumed
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
                if execution.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(execution.device)
                update = execution.update(
                    wrapped,
                    batches,
                    optimizer,
                    cfg["optimizer"]["grad_clip_norm"],
                    processor,
                    reduction=components.loss_counts,
                    metric_specs=metric_specs,
                )
                if not update["optimizer_step"]:
                    skipped += 1
                    if execution.main:
                        logger.warning(
                            "AMP overflow: skipped update; scale=%s", execution.scaler.get_scale()
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
                if execution.device.type == "cuda":
                    record["gpu_mem_gb"] = (
                        torch.cuda.max_memory_allocated(execution.device) / 1024**3
                    )
                count = consumed_samples(sampler, consumed)
                window_count = count - consumed_samples(sampler, previous_consumed)
                progress.samples = count - window_count
                progress.update(record, window_count)
                if scheduler:
                    scheduler.step()
                if stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                if log_freq > 0 and step % log_freq == 0:
                    progress.log(step)
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
                        eval_metrics = MetricAccumulator(metric_specs, device=device, phase="eval")
                        with torch.no_grad(), execution.autocast():
                            eval_loss = evaluate_loss(
                                model,
                                processor,
                                eval_loader,
                                components.loss_counts,
                                eval_totals,
                                device=device,
                                log_freq=eval_log_freq,
                                metrics=eval_metrics,
                            )
                        if execution.main:
                            evaluation = {"step": step, "loss": eval_loss, **eval_metrics.result()}
                            validation_stream.write(json.dumps(evaluation, allow_nan=False) + "\n")
                            validation_stream.flush()
                            logger.info(
                                "step %d: eval_loss=%.4f%s",
                                step,
                                eval_loss,
                                format_metrics(evaluation["metrics"], metric_specs),
                            )
                    finally:
                        restore_rng(rng)
                if step % save_freq == 0 or step == steps:
                    logger.info("Checkpoint policy after step %d", step)
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
                            logger.debug("Saved checkpoint: %s", saved)
                    execution.barrier()
        checkpoint = output / "checkpoint"
        if validation:
            logger.debug("Running final offline evaluation")
        if execution.sharded and validation:
            from alohamini.learning.processor import scale_action

            # Every rank traverses identical batches: custom prediction methods
            # also need FSDP's collective pre/post-forward hooks.
            model.eval()
            model.reset()

            def predict(batch):
                model.reset()
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
                loader=eval_loader,
                device=device,
                log_freq=eval_log_freq,
            )
            if execution.main:
                (output / "offline-evaluation.json").write_text(
                    json.dumps(metrics, indent=2) + "\n"
                )
                logger.debug("Final offline evaluation: %s", json.dumps(metrics))
        elif execution.main and validation:
            del wrapped, model, optimizer
            policy = NativePolicy(checkpoint, device=device)
            metrics = offline_evaluate(
                policy,
                validation,
                batch_size=eval_batch_size,
                loader=eval_loader,
                log_freq=eval_log_freq,
            )
            (output / "offline-evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
            logger.debug("Final offline evaluation: %s", json.dumps(metrics))
        elif execution.main:
            logger.debug("No held-out evaluation or generalization claim.")
        if execution.main:
            logger.info("End of training")
        execution.barrier()
        return checkpoint


def main():
    from alohamini.learning.train_config import parse_training_args

    cfg, background = parse_training_args()
    if background:
        job = launch_training(cfg)
        print(json.dumps(job, ensure_ascii=False, indent=2))
        print("tail -f", shlex.quote(job["log"]))
    else:
        train(cfg)


if __name__ == "__main__":
    main()
