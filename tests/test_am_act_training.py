import json
import subprocess
from pathlib import Path

import torch
from safetensors.torch import load_file
from test_am_act import conditional_options
from test_native_learning import recording as recording

from alohamini.learning.train import launch_training
from alohamini.learning.train_config import parse_training_args


def test_conditional_training_statistics_validation_and_exact_resume(
    recording, tmp_path, monkeypatch
):
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    processes = []
    original = subprocess.Popen

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"]
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    config = dict(
        dataset=str(recording),
        policy="am_act",
        state="none",
        device="cpu",
        train_episodes=[0],
        val_episodes=[1],
        batch_size=3,
        num_workers=0,
        save_freq=1,
        eval_steps=1,
        log_freq=1,
        image_size=[32, 32],
        seed=1000,
        cudnn_deterministic=True,
        deterministic_algorithms=True,
        model=conditional_options(
            kl_weight=1,
            latent_kl_warmup_steps=5000,
            action_loss_groups={"arms": list(range(14)), "lift": [17]},
        ),
    )

    def run(settings):
        job = launch_training(settings)
        process = processes[-1]
        try:
            assert process.wait(timeout=120) == 0, Path(job["log"]).read_text()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        assert Path(job["pid_file"]).read_text().strip() == str(process.pid)
        return Path(job["checkpoint"])

    baseline = run({**config, "output_dir": str(tmp_path / "baseline"), "steps": 2})
    partial = run({**config, "output_dir": str(tmp_path / "resumed"), "steps": 1})
    cfg, _ = parse_training_args(
        [
            "--config_path",
            str(partial.resolve() / "train_config.json"),
            "--resume=true",
            "--steps=2",
        ]
    )
    resumed = run(cfg)
    expected, actual = (
        load_file(baseline / "model.safetensors"),
        load_file(resumed / "model.safetensors"),
    )
    assert actual["kl_updates"] == 2
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    manifest = json.loads((resumed / "policy.json").read_text())
    saved = manifest["config"]
    assert saved["discrete_action_dims"] == [14, 15, 16]
    assert saved["discrete_action_class_counts"] == [[0, 1, 3], [0, 1, 3], [0, 4, 0]]
    assert saved["discrete_action_class_weights"][2] == [5, 1, 5]
    assert saved["prior_type"] == "conditional_gaussian"
    rows = [
        json.loads(line)
        for line in (tmp_path / "baseline/validation-metrics.jsonl").read_text().splitlines()
    ]
    assert "classification_dim_14_macro_recall" in rows[-1]["metrics"]
