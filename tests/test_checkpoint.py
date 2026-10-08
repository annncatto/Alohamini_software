"""Common native checkpoint routing, legacy compatibility and asset handling."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_diffusion_fastwam import diffusion_options, fake_frozen_assets, fastwam_options
from test_native_learning import model_options
from test_native_learning import recording as recording
from test_smolvla import options as smol_options
from test_smolvla import tiny_assets as tiny_assets

from alohamini.learning.checkpoint import (
    checkpoint_sidecars,
    initialize_policy,
    read_checkpoint,
    resolve_pretrained,
    validate_pretrained,
)
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.learning.statistics import sample_arguments, training_statistics
from alohamini.learning.train import launch_training
from alohamini.learning.train_config import parse_training_args
from alohamini.policies.registry import algorithm


@pytest.fixture(params=["act", "am_act", "diffusion", "smolvla", "pi05", "fastwam"])
def native(request, recording, tmp_path, monkeypatch, tiny_assets):
    kind = request.param
    component = algorithm(kind)
    options = model_options() if kind in ("act", "am_act") else {}
    if kind == "diffusion":
        options = diffusion_options()
    if kind == "smolvla":
        options = smol_options(tiny_assets)
        options.update(max_state_dim=32, max_action_dim=32, compile_model=False)
    if kind == "fastwam":
        fake_frozen_assets(monkeypatch)
        options = fastwam_options()
        for key in ("vae_model_id", "text_encoder_model_id", "tokenizer_model_id"):
            options[key] = str(tmp_path)
    if kind == "pi05":
        # Keep the actual adapter's save/load and resource paths; replace only
        # the multi-billion-parameter network and tokenizer for this CPU test.
        from alohamini.policies.pi05 import adapter, modeling_pi05

        monkeypatch.setattr(modeling_pi05, "PI0Pytorch", lambda config: torch.nn.Linear(18, 18))
        monkeypatch.setattr(adapter, "PI05BatchProcessor", lambda *args: SimpleNamespace())
        tokenizer = tmp_path / "original.model"
        tokenizer.write_bytes(b"test tokenizer resource")
        options = dict(
            network={"action_horizon": 3}, n_action_steps=2, tokenizer_path=str(tokenizer)
        )
    samples = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **component.sample_spec(options, cameras=["forward"]),
    )
    options.update(input_features=samples.input_features, output_features=samples.output_features)
    stats = component.statistics(samples, options)
    model = component.build(options, stats)
    with torch.no_grad():
        next(model.parameters()).fill_(0.125)
    path = tmp_path / "checkpoint"
    save_checkpoint(path, model, stats, samples, training={})
    return kind, path, model, samples


def test_all_policies_native_finetune_and_evaluation_without_retyping_config(
    native, recording, tmp_path
):
    kind, path, source, samples = native
    cfg, _ = parse_training_args(
        [
            f"--policy.path={path}",
            f"--dataset.root={recording}",
            f"--output_dir={tmp_path / 'new-run'}",
            "--policy.device=cpu",
            "--dataset.episodes=[0]",
            "--normalization=checkpoint",
        ]
    )
    assert cfg["policy"] == kind  # Infer the algorithm from a native checkpoint.
    assert cfg["image_size"] == [32, 32]
    component = algorithm(kind)
    opts = component.options(cfg, "cpu")
    args, episodes, _ = sample_arguments(cfg, component, opts)
    new_samples = AlohaMiniDataset(**args, episodes=episodes)
    manifest = validate_pretrained(cfg, new_samples)
    artifact = training_statistics(component, new_samples, opts, kind, checkpoint=manifest)
    assert artifact["stats"] == manifest["stats"]
    target = component.build(opts, artifact["stats"])
    initialize_policy(component, target, cfg)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    evaluated = NativePolicy(path, device="cpu")
    for key, value in source.state_dict().items():
        torch.testing.assert_close(evaluated.model.state_dict()[key], value, rtol=0, atol=0)
    # A fine-tuned model can be saved and loaded again, including its resources.
    second = tmp_path / "next-checkpoint"
    save_checkpoint(second, target, artifact["stats"], new_samples, training={})
    assert read_checkpoint(second)[1]["kind"] == kind
    moved = second.rename(tmp_path / "relocated")
    path.rename(tmp_path / "old-source")
    assert NativePolicy(moved, device="cpu").model.name == kind
    relocated = resolve_pretrained({"pretrained_path": str(moved)})
    for key in ("vlm_model_name", "tokenizer_path"):
        if key in relocated["model"]:
            assert Path(relocated["model"][key]).is_relative_to(moved)
    step = tmp_path / "000001"
    step.mkdir()
    moved.rename(step / "pretrained_model")
    assert resolve_pretrained({"pretrained_path": str(step)})["policy"] == kind
    assert NativePolicy(step).model.name == kind


def test_legacy_native_metadata_remains_readable(native, recording, tmp_path):
    kind, path, source, _ = native
    manifest = json.loads((path / "policy.json").read_text())
    manifest.pop("layout_version")
    manifest.pop("assets")
    (path / "policy.json").write_text(json.dumps(manifest))
    (path / "config.json").unlink()
    (path / "preprocessing.json").unlink()
    cfg = resolve_pretrained({"pretrained_path": str(path)})
    assert cfg["policy"] == kind
    loaded = NativePolicy(path)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[key], rtol=0, atol=0)


def test_new_metadata_is_consistent_and_corruption_is_explicit(native):
    kind, path, _, _ = native
    _, manifest = read_checkpoint(path)
    assert json.loads((path / "config.json").read_text())["type"] == kind
    for filename, data in checkpoint_sidecars(manifest).items():
        assert json.loads((path / filename).read_text()) == data
    config = json.loads((path / "config.json").read_text())
    config["type"] = "different"
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="config.json disagrees"):
        read_checkpoint(path)


def test_checkpoint_statistics_conflicts_and_base_import_routing(tmp_path):
    with pytest.raises(ValueError, match="native"):
        resolve_pretrained({"normalization": "checkpoint"})
    with pytest.raises(ValueError, match="not both"):
        resolve_pretrained({"normalization": "checkpoint", "stats": "statistics.json"})
    with pytest.raises(ValueError, match="normalization must"):
        resolve_pretrained({"normalization": "invalid"})
    base = tmp_path / "external-base"
    base.mkdir()
    events = []
    component = SimpleNamespace(initialize=lambda model, cfg: events.append(cfg["pretrained_path"]))
    initialize_policy(component, object(), {"pretrained_path": str(base)})
    assert events == [str(base)]


def test_resource_resolution_and_explicit_override(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    # Use the same field resolution for all assets without building large models.
    path = tmp_path / "native"
    path.mkdir()
    manifest = {
        "format": "alohamini-policy",
        "version": 1,
        "kind": "fastwam",
        "config": dict.fromkeys(
            ("vae_model_id", "text_encoder_model_id", "tokenizer_model_id"), str(assets)
        ),
        "state": "none",
        "cameras": ["forward"],
        "image_size": [32, 32],
    }
    (path / "policy.json").write_text(json.dumps(manifest))
    (path / "model.safetensors").touch()
    new = tmp_path / "replacement"
    new.mkdir()
    cfg = resolve_pretrained({"pretrained_path": str(path), "model": {"vae_model_id": str(new)}})
    assert cfg["model"]["vae_model_id"] == str(new)
    assets.rmdir()
    with pytest.raises(FileNotFoundError, match="resource"):
        resolve_pretrained({"pretrained_path": str(path)})
    # Source assets are not consulted when resuming a run that already saved them.
    assert resolve_pretrained({"resume": True, "pretrained_path": str(path)})["resume"]


@pytest.mark.parametrize("native", ["diffusion", "smolvla"], indirect=True)
def test_detached_native_finetune_then_resume(native, recording, tmp_path, monkeypatch):
    _, path, source, _ = native
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

    cfg, _ = parse_training_args(
        [
            f"--policy.path={path}",
            f"--dataset.root={recording}",
            f"--output_dir={tmp_path / 'finetuned'}",
            "--policy.device=cpu",
            "--dataset.episodes=[0]",
            "--normalization=checkpoint",
            "--steps=1",
            "--batch_size=1",
            "--num_workers=0",
            "--optimizer.lr=0",
            "--scheduler.type=none",
            "--mixed_precision=none",
        ]
    )
    checkpoint = run(cfg)
    policy = NativePolicy(checkpoint)
    name, value = next(source.named_parameters())
    torch.testing.assert_close(dict(policy.model.named_parameters())[name], value, rtol=0, atol=0)
    assert policy.manifest["stats"] == read_checkpoint(path)[1]["stats"]
    path.rename(tmp_path / "source-unavailable")
    resumed, _ = parse_training_args(
        [f"--config_path={checkpoint / 'train_config.json'}", "--resume=true", "--steps=2"]
    )
    final = run(resumed)
    state = torch.load(final.parent / "training_state/state.pt", weights_only=True)
    assert state["step"] == 2


@pytest.mark.parametrize("native", ["diffusion"], indirect=True)
def test_statistics_command_inherits_native_config(native, recording, tmp_path):
    from alohamini.learning.statistics import prepare_statistics

    _, path, _, _ = native
    cfg = tmp_path / "stats-config.json"
    cfg.write_text(
        json.dumps(
            {"pretrained_path": str(path), "normalization": "checkpoint", "train_episodes": [0]}
        )
    )
    output = prepare_statistics(recording, tmp_path / "stats.json", config=cfg)
    artifact = json.loads(output.read_text())
    assert artifact["stats"] == read_checkpoint(path)[1]["stats"]
    assert artifact["contract"]["policy"] == "diffusion"


@pytest.mark.parametrize("native", ["pi05"], indirect=True)
def test_checkpoint_statistics_reject_changed_delta_coordinates(native):
    _, path, _, samples = native
    cfg = resolve_pretrained(
        {"pretrained_path": str(path), "normalization": "checkpoint", "model": {"delta_dims": []}}
    )
    with pytest.raises(ValueError, match="unchanged delta_dims"):
        validate_pretrained(cfg, samples)
