"""Training --policy.path loads ACT weights instead of silently starting afresh."""

import json
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from test_native_learning import model_options
from test_native_learning import recording as recording

from alohamini.learning.checkpoint import write_checkpoint_metadata
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import make_policy, save_checkpoint
from alohamini.learning.processor import act_statistics
from alohamini.learning.statistics import sample_arguments, training_statistics
from alohamini.learning.train import launch_training
from alohamini.learning.train_config import parse_training_args
from alohamini.policies.registry import algorithm


@pytest.fixture(params=["act", "am_act"])
def pretrained(request, recording, tmp_path):
    kind = request.param
    samples = AlohaMiniDataset(
        recording, episodes=[0], state="none", chunk_size=3, image_size=(32, 32)
    )
    component = algorithm(kind)
    model = make_policy(kind, model_options(state=False))
    with torch.no_grad():
        model.model.action_head.bias.fill_(0.125)
    stats = act_statistics(samples)
    # A distinguishable saved statistic: fine-tuning must fit the new data.
    stats["action"]["mean"] = [123.0] * 18
    path = tmp_path / "pretrained"
    save_checkpoint(path, model, stats, samples, training={"step": 987})
    return kind, path, model, component


def config(pretrained, recording, tmp_path, *extra):
    kind, path, _, _ = pretrained
    return parse_training_args(
        [
            f"--policy.type={kind}",
            f"--policy.path={path}",
            f"--dataset.root={recording}",
            f"--output_dir={tmp_path / 'run'}",
            "--policy.device=cpu",
            "--num_workers=0",
            "--steps=1",
            "--batch_size=2",
            "--dataset.episodes=[0]",
            "--save_freq=1",
            "--log_freq=1",
            *extra,
        ]
    )[0]


def test_loads_configuration_weights_and_refits_statistics(pretrained, recording, tmp_path):
    kind, _, source, component = pretrained
    cfg = config(pretrained, recording, tmp_path, "--policy.kl_weight=2.5")
    assert cfg["state"] == "none"
    assert cfg["image_size"] == [32, 32]
    assert cfg["cameras"] == ["forward"]
    assert cfg["model"]["dim_model"] == 32
    assert cfg["model"]["kl_weight"] == 2.5
    assert cfg["model"]["pretrained_backbone_weights"] is None
    options = component.options(cfg, "cpu")
    args, episodes, _ = sample_arguments(cfg, component, options)
    samples = AlohaMiniDataset(**args, episodes=episodes)
    component.validate_pretrained(cfg, samples)
    artifact = training_statistics(component, samples, options, kind)
    assert artifact["stats"]["action"]["mean"] != [123.0] * 18
    target = make_policy(kind, options, artifact["stats"])
    component.initialize(target, cfg)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)


def test_bad_path_kind_and_incompatible_weights_are_explicit(pretrained, recording, tmp_path):
    kind, path, _, component = pretrained
    with pytest.raises(ValueError, match="policy.path"):
        config(pretrained, recording, tmp_path, f"--policy.path={path / 'missing'}")
    other = "am_act" if kind == "act" else "act"
    with pytest.raises(ValueError, match="checkpoint"):
        config(pretrained, recording, tmp_path, f"--policy.type={other}")
    cfg = config(pretrained, recording, tmp_path, "--policy.dim_model=64")
    model = make_policy(kind, component.options(cfg, "cpu"))
    with pytest.raises(RuntimeError, match="size mismatch"):
        component.initialize(model, cfg)
    (path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        config(pretrained, recording, tmp_path)


def test_equal_shape_does_not_hide_changed_action_names_or_units(pretrained, recording, tmp_path):
    _, path, _, component = pretrained
    cfg = config(pretrained, recording, tmp_path)
    samples = AlohaMiniDataset(
        recording, episodes=[0], state="none", chunk_size=3, image_size=(32, 32)
    )
    manifest_path = path / "policy.json"
    original = json.loads(manifest_path.read_text())
    changed = json.loads(manifest_path.read_text())
    names = changed["source_info"]["features"]["action"]["names"]
    names[0], names[1] = names[1], names[0]
    write_checkpoint_metadata(path, changed)
    with pytest.raises(ValueError, match="action coordinates"):
        component.validate_pretrained(cfg, samples)
    motor = original["source_info"]["features"]["action"]["names"][0].removesuffix(".pos")
    original["source_info"]["robot_metadata"]["motors"][motor]["normalization"] = "degrees"
    write_checkpoint_metadata(path, original)
    with pytest.raises(ValueError, match="action units"):
        component.validate_pretrained(cfg, samples)


def test_detached_finetune_starts_from_weights_and_resume_needs_no_source(
    pretrained, recording, tmp_path, monkeypatch
):
    _, path, source, _ = pretrained
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    processes = []
    popen = subprocess.Popen

    def capture(*args, **kwargs):
        assert kwargs["start_new_session"]
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", capture)

    def run(cfg):
        job = launch_training(cfg)
        process = processes[-1]
        try:
            assert process.wait(timeout=120) == 0, Path(job["log"]).read_text()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        assert Path(job["pid_file"]).read_text().strip() == str(process.pid)
        return Path(job["checkpoint"]).resolve()

    # Zero LR makes every trainable tensor an exact witness of the loaded weights.
    cfg = config(
        pretrained, recording, tmp_path, "--optimizer.lr=0", "--policy.optimizer_lr_backbone=0"
    )
    checkpoint = run(cfg)
    saved = load_file(checkpoint / "model.safetensors")
    for key, value in source.named_parameters():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    state = torch.load(checkpoint.parent / "training_state/state.pt", weights_only=True)
    assert state["step"] == 1
    manifest = json.loads((checkpoint / "policy.json").read_text())
    assert manifest["stats"]["action"]["mean"] != [123.0] * 18
    # Resuming the new run must not reopen the original fine-tuning checkpoint.
    path.rename(path.with_name("source-moved"))
    resumed, _ = parse_training_args(
        [
            f"--config_path={checkpoint / 'train_config.json'}",
            "--resume=true",
            "--steps=2",
        ]
    )
    final = run(resumed)
    final_state = torch.load(final.parent / "training_state/state.pt", weights_only=True)
    assert final_state["step"] == 2


def test_am_act_discrete_classes_keep_physical_values_after_refitting(recording, tmp_path):
    samples = AlohaMiniDataset(
        recording, episodes=[0], state="none", chunk_size=3, image_size=(32, 32)
    )
    component = algorithm("am_act")
    options = model_options(
        state=False, discrete_action_dims=[0], discrete_action_values=[[-1.0, 0.0, 1.0]]
    )
    old_stats = act_statistics(samples)
    old_stats["action"]["mean"] = [10.0] * 18
    old_stats["action"]["std"] = [2.0] * 18
    source = make_policy("am_act", options, old_stats)
    path = tmp_path / "classes"
    save_checkpoint(path, source, old_stats, samples, training={})
    cfg = component.apply_preset({"policy": "am_act", "pretrained_path": str(path)})
    new_stats = act_statistics(samples)
    target = make_policy("am_act", component.options(cfg, "cpu"), new_stats)
    component.initialize(target, cfg)
    centers = torch.tensor(target.config.discrete_action_normalized_values[0])
    physical = centers * new_stats["action"]["std"][0] + new_stats["action"]["mean"][0]
    torch.testing.assert_close(physical, torch.tensor([-1.0, 0.0, 1.0]))
    assert (
        target.config.discrete_action_normalized_values
        != source.config.discrete_action_normalized_values
    )
