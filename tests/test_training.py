import ast
import json
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from test_native_learning import model_options
from test_native_learning import recording as recording

from alohamini.learning.train import launch_training, update_policy
from alohamini.learning.train_config import parse_training_args
from alohamini.learning.training_state import EpisodeAwareSampler, compute_sampler_state


def test_legacy_style_cli_and_override(tmp_path):
    cfg, background = parse_training_args(
        [
            "--dataset.repo_id=local/pickup",
            "--dataset.root=/tmp/dataset",
            "--policy.type=am_act",
            "--policy.device=cuda",
            "--policy.push_to_hub=false",
            "--output_dir=/tmp/run",
            "--steps=100000",
            "--batch_size=2",
            "--save_freq=10000",
            "--wandb.enable=false",
            "--policy.chunk_size=30",
            "--policy.pretrained_backbone_weights=null",
            "--policy.discrete_action_dims=[14]",
            "--policy.discrete_action_values=[[-0.15,0,0.15]]",
            "--background",
        ]
    )
    assert background and cfg["policy"] == "am_act" and cfg["state"] == "auto"
    assert cfg["model"]["chunk_size"] == 30
    assert cfg["model"]["pretrained_backbone_weights"] is None
    assert cfg["model"]["discrete_action_values"] == [[-0.15, 0, 0.15]]
    assert cfg["save_freq"] == 10000 and cfg["dataset_repo_id"] == "local/pickup"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    changed, _ = parse_training_args(
        ["--config", str(path), "--steps=50", "--policy.chunk_size=10"]
    )
    assert changed["steps"] == 50 and changed["model"]["chunk_size"] == 10


@pytest.mark.parametrize(
    "option",
    [
        "--policy.push_to_hub=true",
        "--wandb.enable=true",
        "--job.target=cloud",
        "--policy.unknown=1",
    ],
)
def test_cli_rejects_remote_or_unknown_options(option):
    with pytest.raises(SystemExit):
        parse_training_args(["--dataset.root=/tmp/data", "--output_dir=/tmp/run", option])


@pytest.mark.parametrize("kind", ["act", "am_act"])
def test_local_training_cli_needs_no_service_switches(kind):
    cfg, background = parse_training_args(
        [
            "--dataset.root=/tmp/data",
            "--output_dir=/tmp/run",
            f"--policy.type={kind}",
            "--dataset.eval_episodes=[1]",
            "--background",
        ]
    )
    assert background and cfg["policy"] == kind
    assert cfg["state"] == "auto" and cfg["val_episodes"] == [1]
    assert not any("hub" in name or "wandb" in name for name in cfg)


@pytest.mark.parametrize(
    "kind,storage", [("act", "native"), ("act", "v3"), ("am_act", "vision"), ("am_act", "edited")]
)
def test_offline_evaluation_entry(recording, tmp_path, monkeypatch, capsys, kind, storage):
    from alohamini.datasets.edit import edit_dataset, parse_args
    from alohamini.datasets.lerobotv3 import export_lerobot
    from alohamini.learning.data import AlohaMiniDataset
    from alohamini.learning.evaluate import main
    from alohamini.learning.policy import make_policy, save_checkpoint

    def no_robot(*args, **kwargs):
        raise AssertionError("Offline evaluation must not connect to a robot")

    monkeypatch.setattr("alohamini.client.HostClient._connect", no_robot)
    dataset = recording
    if storage in ("v3", "vision"):
        dataset = tmp_path / "v3"
        export_lerobot(recording, dataset)
        if storage == "vision":
            visual = tmp_path / "vision"
            export_lerobot(dataset, visual, vision_only=True)
            dataset = visual
    elif storage == "edited":
        dataset = tmp_path / "edited"
        edit_dataset(
            parse_args(
                [
                    "--root",
                    str(recording),
                    "--output",
                    str(dataset),
                    "--operation.type",
                    "modify_tasks",
                    "--operation.new_task",
                    "pick",
                ]
            )
        )
    state = "none" if storage == "vision" else "joint_position,base_velocity,lift_height"
    samples = AlohaMiniDataset(
        dataset, episodes=[0], state=state, chunk_size=3, image_size=(32, 32)
    )
    model = make_policy(kind, model_options(state=state != "none"))
    checkpoint, output = tmp_path / "checkpoint", tmp_path / "evaluation.json"
    save_checkpoint(checkpoint, model, samples.statistics(), samples, training={})
    args = [
        "--policy.path",
        str(checkpoint),
        "--dataset.root",
        str(dataset),
        "--dataset.episodes",
        "[1]",
        "--output",
        str(output),
    ]
    capsys.readouterr()
    assert main(args) == 0
    report = json.loads(output.read_text())
    assert report["episodes"] == [1] and report["valid_action_steps"] > 0
    assert len(report["mae_by_action"]) == 18
    assert json.loads(capsys.readouterr().out)["samples"] == report["samples"]
    with pytest.raises(FileExistsError):
        main(args)


def test_resume_rejects_changed_within_episode_window_semantics(tmp_path):
    from alohamini.learning.training_state import load_training_checkpoint

    (tmp_path / "training_state").mkdir()
    (tmp_path / "training_state/state.pt").touch()
    (tmp_path / "pretrained_model").mkdir()
    (tmp_path / "pretrained_model/train_config.json").write_text("{}")
    (tmp_path / "pretrained_model/policy.json").write_text(
        json.dumps({"sample_boundaries": [{"episode_index": 0, "frame_index": 2}]})
    )
    with pytest.raises(ValueError, match="within-episode sample boundaries"):
        load_training_checkpoint(tmp_path, {}, None, None)


def test_sampler_is_copied_from_original_and_resume_order_matches():
    source = Path("/home/anncatto/lerobot_alohamini/src/lerobot/datasets/sampler.py")
    if not source.exists():
        pytest.skip("Original migration source unavailable")
    import alohamini.learning.training_state as migrated

    old = ast.parse(source.read_text())
    new = ast.parse(Path(migrated.__file__).read_text())
    for name in ("EpisodeAwareSampler", "compute_sampler_state"):
        old_node = next(n for n in old.body if getattr(n, "name", None) == name)
        new_node = next(n for n in new.body if getattr(n, "name", None) == name)
        assert ast.dump(old_node) == ast.dump(new_node)
    sampler = EpisodeAwareSampler([0], [7], seed=42, shuffle=True)
    epoch0, epoch1 = list(sampler), list(sampler)
    resumed = EpisodeAwareSampler([0], [7], seed=42, shuffle=True)
    resumed.load_state_dict(compute_sampler_state(2, 7, 3, 1))
    assert list(resumed) == epoch0[6:]
    assert list(resumed) == epoch1


def test_update_policy_matches_original_single_device_math():
    source = Path("/home/anncatto/lerobot_alohamini/src/lerobot/scripts/lerobot_train.py")
    if not source.exists():
        pytest.skip("Original migration source unavailable")
    import time
    from contextlib import nullcontext
    from copy import deepcopy
    from types import SimpleNamespace

    tree = ast.parse(source.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "update_policy"
    )
    node.decorator_list = []
    # Execute only the unchanged source function, with its single-device dependencies.
    namespace = dict(
        torch=torch,
        time=time,
        nullcontext=nullcontext,
        Any=object,
        MetricsTracker=object,
        PreTrainedPolicy=object,
        Optimizer=object,
        has_method=lambda obj, name: callable(getattr(obj, name, None)),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)

    class Model(torch.nn.Linear):
        def forward(self, batch):
            return super().forward(batch).square().mean(), {}

    model = Model(3, 2)
    other = deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    original_optimizer = torch.optim.AdamW(other.parameters(), lr=1e-5)
    accelerator = SimpleNamespace(
        autocast=nullcontext,
        backward=lambda loss: loss.backward(),
        clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
        unwrap_model=lambda obj, **_: obj,
    )
    batch = torch.ones(2, 3)
    result = update_policy(model, batch, optimizer, 10.0)
    original, _ = namespace["update_policy"](
        SimpleNamespace(), other, batch, original_optimizer, 10.0, accelerator
    )
    assert result["loss"] == original.loss
    for left, right in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_failed_checkpoint_does_not_publish_and_retry_preserves_partial(tmp_path, monkeypatch):
    import alohamini.learning.training_state as state

    def save_model(path, *args, **kwargs):
        path.mkdir()
        (path / "policy.json").write_text("{}")

    monkeypatch.setattr(state, "save_checkpoint", save_model)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    arguments = (tmp_path, 1, {"steps": 2}, model, optimizer, {}, None, {})
    original = torch.save

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        state.save_training_checkpoint(*arguments)
    assert not (tmp_path / "checkpoints/last").exists()
    assert not (tmp_path / "checkpoint").exists()
    partial = list((tmp_path / "checkpoints").glob("000001.pending-*"))
    assert len(partial) == 1
    monkeypatch.setattr(torch, "save", original)
    checkpoint = state.save_training_checkpoint(*arguments)
    assert (tmp_path / "checkpoints/last").resolve() == checkpoint
    assert partial[0].is_dir()


def test_nonfinite_update_does_not_change_parameters():
    class Model(torch.nn.Linear):
        def forward(self, batch):
            return super().forward(batch).mean(), {}

    model = Model(1, 1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(RuntimeError, match="Non-finite"):
        update_policy(model, torch.tensor([[float("nan")]]), optimizer, 10)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_eval_frequency_requires_held_out_episodes(recording, tmp_path):
    from alohamini.learning.train import train

    with pytest.raises(ValueError, match="held-out"):
        train(
            dict(
                dataset=str(recording),
                output_dir=str(tmp_path / "run"),
                device="cpu",
                eval_steps=10,
                train_episodes=[0],
                val_episodes=[],
            )
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "device,precision,process_count,accumulation,workers",
    [
        ("cpu", "none", 1, 1, 0),
        ("cpu", "none", 2, 2, 0),
        ("cuda", "none", 1, 2, 0),
        ("cuda", "bfloat16", 1, 2, 0),
        ("cuda", "float16", 1, 2, 0),
        ("cpu", "none", 1, 2, 2),
        ("cuda", "none", 2, 2, 0),
        ("cuda", "bfloat16", 2, 2, 0),
        ("cuda", "float16", 2, 2, 0),
    ],
)
def test_periodic_checkpoint_resume_exact_and_no_data_mutation(
    recording,
    tmp_path,
    monkeypatch,
    device,
    precision,
    process_count,
    accumulation,
    workers,
    backend="ddp",
    kind="am_act",
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "cuda" and torch.cuda.device_count() < process_count:
        pytest.skip("Multi-GPU resume verification requires two visible GPUs")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    processes = []
    original = subprocess.Popen

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"]
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    cfg = dict(
        dataset=str(recording),
        state="none",
        policy=kind,
        distributed_backend=backend,
        device=device,
        mixed_precision=precision,
        num_processes=process_count,
        gradient_accumulation_steps=accumulation,
        cudnn_deterministic=True,
        deterministic_algorithms=True,
        seed=1000,
        train_episodes=[0],
        val_episodes=[1],
        batch_size=3,
        num_workers=workers,
        save_freq=1,
        log_freq=1,
        eval_steps=1,
        image_size=[32, 32],
        model={**model_options(state=False), "dropout": 0.2},
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

    baseline = run({**cfg, "output_dir": str(tmp_path / "baseline"), "steps": 3})
    interrupted = run({**cfg, "output_dir": str(tmp_path / "resumed"), "steps": 1})
    first = interrupted.resolve()
    assert (first / "train_config.json").is_file()
    resumed_cfg, _ = parse_training_args(
        [
            "--config_path",
            str(first / "train_config.json"),
            "--resume=true",
            "--steps=3",
        ]
    )
    resumed = run(resumed_cfg)
    assert first.is_dir()  # Earlier checkpoint not overwritten.
    for step in (1, 2, 3):
        assert (tmp_path / f"resumed/checkpoints/{step:06d}/training_state/state.pt").is_file()
    baseline_weights = load_file(baseline / "model.safetensors")
    resumed_weights = load_file(resumed / "model.safetensors")
    for key in baseline_weights:
        torch.testing.assert_close(baseline_weights[key], resumed_weights[key], rtol=0, atol=0)

    def identical(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                identical(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                identical(x, y)
        else:
            assert a == b

    states = [
        torch.load(p.parent / "training_state/state.pt", weights_only=True)
        for p in (baseline.resolve(), resumed.resolve())
    ]
    identical(states[0], states[1])
    if backend == "fsdp2":
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

        distributed = []
        for label, checkpoint in (("baseline", baseline), ("resumed", resumed)):
            destination = tmp_path / f"{label}-state.pt"
            dcp_to_torch_save(checkpoint.resolve().parent / "distributed", destination)
            distributed.append(torch.load(destination, weights_only=True))
        identical(distributed[0], distributed[1])
        from alohamini.learning.policy import NativePolicy

        # A deployment checkpoint must load independently of Accelerate/FSDP.
        portable = NativePolicy(resumed, device="cpu")
        assert not any(
            hasattr(parameter, "placements") for parameter in portable.model.parameters()
        )
    metrics = [
        json.loads(line)
        for line in next((tmp_path / "resumed").glob("metrics-from-1-*.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert [m["step"] for m in metrics] == [2, 3]
    assert (
        json.loads((tmp_path / "resumed/offline-evaluation.json").read_text())["valid_action_steps"]
        > 0
    )

    # Reusing an existing run without resume and resuming from an old step both fail safely.
    with pytest.raises(FileExistsError):
        launch_training({**cfg, "output_dir": str(tmp_path / "resumed"), "steps": 4})
    from alohamini.learning.data import AlohaMiniDataset
    from alohamini.learning.training_state import load_training_checkpoint

    data = AlohaMiniDataset(recording, episodes=[0], state="none", image_size=(32, 32))
    saved_cfg = json.loads((resumed / "train_config.json").read_text())
    saved_cfg["steps"] = 4
    with pytest.raises(ValueError, match="latest"):
        load_training_checkpoint(first.parent, saved_cfg, data, None)


@pytest.mark.parametrize(
    "kind,precision,processes",
    [
        ("act", "none", 1),
        ("am_act", "bfloat16", 1),
        ("am_act", "float16", 1),
        ("act", "none", 2),
        ("am_act", "bfloat16", 2),
    ],
)
def test_fsdp_checkpoint_resume(recording, tmp_path, monkeypatch, kind, precision, processes):
    test_periodic_checkpoint_resume_exact_and_no_data_mutation(
        recording,
        tmp_path,
        monkeypatch,
        "cuda",
        precision,
        processes,
        2,
        0,
        backend="fsdp2",
        kind=kind,
    )
