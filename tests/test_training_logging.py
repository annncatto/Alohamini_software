import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_native_learning import model_options
from test_native_learning import recording as recording

from alohamini.learning.execution import RankBatchSampler
from alohamini.learning.logging import (
    TrainingProgress,
    consumed_samples,
    log_stage,
    training_logging,
)
from alohamini.learning.train import launch_training


def test_interval_means_and_slowest_rank():
    reductions = []

    def reduce(values, reduction):
        reductions.append(reduction)
        return torch.maximum(values, torch.tensor([4.0, 6.0, 3.0]))

    execution = SimpleNamespace(
        main=True, world_size=2, device="cpu", accelerator=SimpleNamespace(reduce=reduce)
    )
    progress = TrainingProgress(execution, frames=10, episodes=2, steps=10, samples=20)
    for loss in (2.0, 4.0):
        progress.update(
            dict(loss=loss, grad_norm=4, lr=0.01, dataloading_s=1, update_s=2, gpu_mem_gb=2),
            samples=8,
        )
    result = progress.summary(4)
    assert reductions == ["max"]
    assert result["loss"] == 3
    assert result["samples"] == 36
    assert result["epochs"] == 3.6
    assert result["episodes"] == 7.2
    assert result["samples_per_s"] == 0.8
    assert result["eta_s"] == 60
    assert result["gpu_mem_gb"] == 3
    progress.reset()
    assert progress.count == 0 and progress.samples == 36 and progress.sums == {}


@pytest.mark.parametrize("ranks,drop,expected", [(1, False, 10), (1, True, 9), (2, False, 12)])
def test_sample_counts_handle_epoch_tail_and_resume(ranks, drop, expected):
    sampler = RankBatchSampler(10, 3, seed=1000, world_size=ranks, drop_last=drop)
    assert consumed_samples(sampler, sampler.batches) == expected
    assert consumed_samples(sampler, sampler.batches + 1) == expected + 3 * ranks


def test_console_scope_and_failure_restore(capsys):
    logger = logging.getLogger("alohamini")
    previous = logger.handlers[:], logger.level, logger.propagate
    root_handlers = logging.getLogger().handlers[:]
    with pytest.raises(ValueError, match="broken"):
        with training_logging(True), log_stage("Reading dataset"):
            raise ValueError("broken")
    error = capsys.readouterr().err
    assert "INFO Reading dataset" in error
    assert "ERROR Reading dataset failed" in error and "Traceback" in error
    assert (logger.handlers, logger.level, logger.propagate) == previous
    assert logging.getLogger().handlers == root_handlers
    with training_logging(False):
        logger.warning("worker warning")
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("eval_workers", [0, 2])
def test_detached_training_logs_and_raw_metrics(recording, tmp_path, monkeypatch, eval_workers):
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    processes = []
    popen = subprocess.Popen

    def capture(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", capture)
    job = launch_training(
        dict(
            dataset=str(recording),
            output_dir=str(tmp_path / "run"),
            device="cpu",
            policy="act",
            state="none",
            train_episodes=[0],
            val_episodes=[1],
            steps=4,
            batch_size=3,
            num_workers=0,
            log_freq=3,
            eval_steps=2,
            eval_num_workers=eval_workers,
            eval_prefetch_factor=2,
            eval_batch_size=2,
            save_freq=4,
            image_size=[32, 32],
            model=model_options(state=False),
        )
    )
    process = processes[0]
    try:
        assert process.wait(timeout=120) == 0, Path(job["log"]).read_text()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
    log = Path(job["log"]).read_text()
    for phrase in (
        "Training configuration:",
        "Checking dataset integrity",
        "Loaded episodes",
        "Computing normalization statistics",
        "Creating policy",
        "Model parameters:",
        "Creating optimizer",
        "Creating dataloader",
        "effective batch size:",
        "step:1/4",
        "step:3/4",
        "step:4/4",
        "eval_loss=",
        "Validation batches:",
        "Offline MAE batches:",
        "algorithm denominators",
        "Saving checkpoint",
        "Final offline evaluation",
        "End of training",
    ):
        assert phrase in log
    assert "step:2/4" not in log
    records = [
        json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in records] == [1, 2, 3, 4]
    mean = (records[1]["loss"] + records[2]["loss"]) / 2
    summary = next(line for line in log.splitlines() if "step:3/4" in line)
    assert f"loss:{mean:.3f}" in summary
    assert "smp/s:" in summary and "epch:" in summary and "mem_gb:" not in summary
    assert Path(job["pid_file"]).read_text().strip() == str(process.pid)
