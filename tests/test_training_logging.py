import io
import json
import logging
import re
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
    progress = TrainingProgress(
        execution, frames=10, episodes=2, steps=10, samples=20, initial_step=2
    )
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
    assert result["gpu_mem_gb"] == 3
    progress.reset()
    assert progress.count == 0 and progress.samples == 36 and progress.sums == {}


def test_metric_line_matches_lerobot_format(capsys):
    progress = TrainingProgress(
        SimpleNamespace(main=True, world_size=1),
        frames=10000,
        episodes=20,
        steps=100000,
        samples=1996,
        initial_step=998,
    )
    for loss in (2.0, 4.0):
        progress.update(
            dict(
                loss=loss,
                grad_norm=4,
                lr=1e-5,
                dataloading_s=0.5,
                update_s=0.125,
                gpu_mem_gb=1.8928,
            ),
            samples=2,
        )
    with training_logging(True, training=True):
        progress.log(1000)
    text = capsys.readouterr().err
    assert text.endswith(
        "step:1K smpl:2K ep:4 epch:0.20 loss:3.000 grdn:4.000 "
        "lr:1.0e-05 updt_s:0.125 data_s:0.500 smp/s:3 mem_gb:1.89\n"
    )
    assert re.match(r"INFO \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} .{15} step:", text)
    assert "elapsed:" not in text and "ETA:" not in text


@pytest.mark.parametrize("tty", [True, False])
def test_reference_progress_bar_and_cleanup_on_interrupt(monkeypatch, tty):
    class Terminal(io.StringIO):
        def isatty(self):
            return tty

    terminal = Terminal()
    monkeypatch.setattr("sys.stderr", terminal)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    progress = TrainingProgress(
        SimpleNamespace(main=True, world_size=1),
        frames=100,
        episodes=2,
        steps=100,
        initial_step=50,
    )
    with pytest.raises(RuntimeError, match="stop"):
        with training_logging(True, training=True), progress.track():
            assert progress.bar.n == 0 and progress.bar.total == 50
            assert progress.bar.unit == "step" and progress.bar.mininterval == 0.1
            assert progress.bar.smoothing == 0.3
            progress.update(
                dict(step=51, loss=1, grad_norm=2, lr=0.01, dataloading_s=0.1, update_s=0.2),
                samples=2,
            )
            logging.getLogger("alohamini").info("step 51: eval_loss=1.0000")
            raise RuntimeError("stop")
    text = terminal.getvalue()
    assert "Training:" in text and "1/50" in text
    assert "[00:00<" in text and "step/s]" in text
    assert "step 51: eval_loss=1.0000" in text
    assert "\r" in text and "Training loop interrupted" not in text
    assert progress.bar is None


@pytest.mark.parametrize("main,slurm", [(False, False), (True, True)])
def test_reference_progress_suppression(monkeypatch, capsys, main, slurm):
    if slurm:
        monkeypatch.setenv("SLURM_JOB_ID", "123")
    else:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    progress = TrainingProgress(
        SimpleNamespace(main=main, world_size=1),
        frames=100,
        episodes=2,
        steps=10,
    )
    with progress.track():
        progress.update(dict(step=1), samples=2)
    assert capsys.readouterr().err == ""


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
        with training_logging(True, training=True), log_stage("Reading dataset"):
            raise ValueError("broken")
    error = capsys.readouterr().err
    assert "INFO" not in error
    assert error.startswith("ERROR ")
    assert "Reading dataset failed" in error and "Traceback" in error
    assert (logger.handlers, logger.level, logger.propagate) == previous
    assert logging.getLogger().handlers == root_handlers
    with training_logging(False, training=True):
        logger.warning("worker warning")
    assert capsys.readouterr().err == ""
    diagnostic = logging.getLogger("alohamini.learning.validation")
    with training_logging(True, training=True):
        diagnostic.info("Validation batches:1/2")
        diagnostic.warning("Data warning")
    text = capsys.readouterr().err
    assert "Validation batches:" not in text and "Data warning" in text
    with training_logging(True):
        diagnostic.info("Validation batches:1/2")
    assert "Validation batches:" in capsys.readouterr().err


@pytest.mark.parametrize("eval_workers,log_freq", [(0, 3), (2, 0)])
def test_detached_training_logs_and_raw_metrics(
    recording, tmp_path, monkeypatch, eval_workers, log_freq
):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
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
            log_freq=log_freq,
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
        "Logs will be saved locally.",
        "Creating dataset",
        "Creating policy",
        "Creating optimizer and scheduler",
        "Output dir:",
        "cfg.steps=4 (4)",
        "dataset.num_frames=",
        "dataset.num_episodes=1",
        "num_learnable_params=",
        "num_total_params=",
        "Effective batch size: 3 x 1 = 3",
        "Start offline training on a fixed dataset, with effective batch size: 3",
        "step 2: eval_loss=",
        "step 4: eval_loss=",
        "Checkpoint policy after step 4",
        "End of training",
        "Training:",
        "4/4 [",
    ):
        assert phrase in log
    for phrase in (
        "step:1",
        "step:2",
        "step:4",
        "Validation batches:",
        "Offline MAE batches:",
        "algorithm denominators",
        "Saving checkpoint",
        "elapsed:",
        "ETA:",
        "steps/s:",
        "Training configuration:",
        "completed in",
        "Final offline evaluation:",
    ):
        assert phrase not in log
    assert "\r" in Path(job["log"]).read_bytes().decode()
    assert re.search(
        r"Training:.*\d/4 \[\d+:\d+<[^\]]+,\s*[^\]]*(?:step/s|s/step)\]INFO "
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} .{15} ",
        log,
    )
    assert re.search(r"eval_loss=\d+\.\d{4} l1:\d+\.\d{4}(?:\n|\r)", log)
    records = [
        json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in records] == [1, 2, 3, 4]
    for record in records:
        assert record["loss"] == pytest.approx(
            record["metrics"]["loss_l1"] + record["metrics"]["loss_kl_weighted"], rel=1e-6
        )
    evaluations = [
        json.loads(line)
        for line in (tmp_path / "run/validation-metrics.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in evaluations] == [2, 4]
    assert all(set(r["metrics"]) == {"loss_l1"} for r in evaluations)
    schema = json.loads((tmp_path / "run/metrics-schema.json").read_text())
    assert schema["components"][0]["denominator"] == "valid_action_elements"
    summaries = [line for line in log.splitlines() if " step:" in line]
    if log_freq:
        assert len(summaries) == 1 and "step:3 " in summaries[0]
        mean = sum(record["loss"] for record in records[:3]) / 3
        assert f"loss:{mean:.3f}" in summaries[0]
        for key, label in (("loss_l1", "l1"), ("loss_kl", "kl"), ("loss_kl_weighted", "kl_w")):
            mean = sum(r["metrics"][key] for r in records[:3]) / 3
            assert f"{label}:{mean:.4f}" in summaries[0]
        assert "smp/s:" in summaries[0] and "epch:" in summaries[0]
        assert "mem_gb:" not in summaries[0]
    else:
        assert not summaries
    assert (tmp_path / "run/offline-evaluation.json").is_file()
    assert Path(job["pid_file"]).read_text().strip() == str(process.pid)
