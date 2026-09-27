"""Execution math and numerical statistics; no hardware or remote models."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alohamini.datasets.statistics import ExactQuantileStats, diagnose_statistics
from alohamini.learning.execution import Execution, RankBatchSampler, validate_local_workers
from alohamini.learning.optim import make_optimizer_and_scheduler, resolve_optimization


@pytest.fixture(autouse=True)
def isolated_accelerate_state():
    # Production jobs run in separate processes. Tests exercise CPU and different
    # CUDA precision modes in one interpreter, which Accelerate does not support.
    from accelerate.state import AcceleratorState, GradientState

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    yield
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7))

    def forward(self, batch):
        mask = ~batch["action_is_pad"]
        error = (self.weight * batch["action"] - 1).square().squeeze(-1)
        reconstruction = (error * mask).sum() / mask.sum().clamp_min(1)
        kl = (self.weight * batch["state"]).square().mean()
        return (
            reconstruction * batch.get("_reconstruction_weight", 1)
            + kl * batch.get("_kl_weight", 1)
        ), {}


def test_accumulation_matches_full_batch_with_unequal_padding():
    torch.manual_seed(0)
    full = {
        "action": torch.randn(3, 4, 1),
        "state": torch.randn(3, 2),
        "action_is_pad": torch.tensor([[False] * 4, [False, True, True, True], [False] * 4]),
    }
    batches = [{k: v[:2] for k, v in full.items()}, {k: v[2:] for k, v in full.items()}]
    model = TinyPolicy()
    reference = deepcopy(model)
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.01) for m in (model, reference)]
    runtime = Execution("cpu")
    wrapped, optimizer = runtime.prepare(model, optimizers[0])
    runtime.update(wrapped, batches, optimizer, 0, lambda b: dict(b))
    reference(full)[0].backward()
    optimizers[1].step()
    torch.testing.assert_close(model.weight, reference.weight, rtol=1e-6, atol=1e-7)
    runtime.close()


def test_optimizer_presets_overrides_and_groups():
    config = SimpleNamespace(
        optimizer_lr=0.01, optimizer_weight_decay=0.2, optimizer_grad_clip_norm=3.0
    )
    cfg = resolve_optimization({"steps": 10, "optimizer": {"lr": 0.02}}, config)
    assert cfg["optimizer"]["grad_clip_norm"] == 3.0
    assert cfg["optimizer"]["lr"] == 0.02
    model = TinyPolicy()
    model.get_optim_params = lambda: [{"params": [model.weight], "lr": 0.001}]
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, model)
    assert optimizer.param_groups[0]["lr"] == 0.001 and scheduler is None
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_optimization(
            {"steps": 10, "optimizer": {"type": "sgd", "betas": [0.9, 0.99]}}, config
        )


def test_cuda_launch_checks_capacity_before_writing_logs(tmp_path, monkeypatch):
    from alohamini.learning.train import launch_training

    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="only 1 GPU"):
        launch_training({"device": "cuda", "num_processes": 2, "run_name": "too_many"})
    assert not (tmp_path / "workspace/logs/training").exists()
    validate_local_workers(2, "cpu")


def test_multi_node_uses_local_worker_count(monkeypatch):
    import alohamini.learning.train as training

    runtime = SimpleNamespace(world_size=4, local_world_size=2, close=lambda: None)
    monkeypatch.setattr(training, "Execution", lambda *args: runtime)
    monkeypatch.setattr(training, "_train", lambda cfg, execution: execution.world_size)
    assert training.train({"device": "cpu", "num_processes": 2}) == 4
    with pytest.raises(ValueError, match="torchrun"):
        training.train({"device": "cpu", "num_processes": 3})


def test_nccl_binds_local_not_global_rank(monkeypatch):
    import torch.distributed as dist
    from accelerate.utils import DistributedType

    import alohamini.learning.execution as execution

    for key, value in dict(WORLD_SIZE="4", LOCAL_WORLD_SIZE="2", RANK="3", LOCAL_RANK="1").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    devices, options = [], []
    monkeypatch.setattr(torch.cuda, "set_device", devices.append)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    def accelerator(**kwargs):
        import os

        assert os.environ["ACCELERATE_TORCH_DEVICE"] == "cuda:1"
        options.append(kwargs)
        return SimpleNamespace(
            device=torch.device("cuda:1"),
            process_index=3,
            num_processes=4,
            distributed_type=DistributedType.MULTI_GPU,
            scaler=None,
            end_training=lambda: None,
            free_memory=lambda: None,
        )

    monkeypatch.setattr(execution, "Accelerator", accelerator)
    runtime = Execution("cuda")
    assert runtime.rank == 3 and runtime.device == torch.device("cuda:1")
    assert devices == [torch.device("cuda:1")]
    assert options[0]["kwargs_handlers"][1].backend == "nccl"
    runtime.close()


@pytest.mark.parametrize("size,drop", [(11, False), (11, True), (2, False)])
def test_rank_sampler_resume_ignores_prefetch(size, drop):
    for rank in (0, 1):
        sampler = RankBatchSampler(size, 2, 1000, rank, 2, drop)
        first = list(sampler)
        second = list(sampler)
        resumed = RankBatchSampler(size, 2, 1000, rank, 2, drop, consumed=1)
        if len(first) > 1:
            assert list(resumed) == first[1:]
        assert list(resumed) == second


def test_exact_stats_constant_sparse_and_small_variance():
    values = np.column_stack(
        [np.full(1000, 1234.1), 1e8 + np.linspace(0, 0.001, 1000), np.r_[np.zeros(999), 1]]
    )
    forward, reverse = ExactQuantileStats(), ExactQuantileStats()
    for block in np.array_split(values, 7):
        forward.update(block)
    reverse.update(values[::-1])
    a, b = forward.get_statistics(), reverse.get_statistics()
    assert a["std"][0] == 0
    assert a["std"][1] > 0
    np.testing.assert_allclose(a["std"], values.std(0), rtol=1e-4, atol=1e-10)
    for q in ("q01", "q99"):
        np.testing.assert_array_equal(a[q], b[q])
    report = diagnose_statistics(a)
    assert report[0]["constant"]
    assert not report[2]["constant"] and report[2]["narrow_quantiles"]
    assert report[2]["normalized_max"] > 1000000


def test_episode_quantiles_are_not_averaged():
    from alohamini.datasets.lerobot_tools import aggregate_feature_stats

    stats = []
    for value in (0, 100):
        tracker = ExactQuantileStats()
        tracker.update(np.full((10, 1), value))
        stats.append(tracker.get_statistics())
    assert "q99" not in aggregate_feature_stats(stats)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fp16_overflow_skips_update_and_retains_scaler():
    runtime = Execution("cuda", "float16")
    model = TinyPolicy().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    wrapped, optimizer = runtime.prepare(model, optimizer)
    before = model.weight.detach().clone()
    # Infinite gradients with a finite forward loss: GradScaler must skip.
    hook = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    batch = {
        "action": torch.ones(1, 2, 1),
        "state": torch.ones(1, 2),
        "action_is_pad": torch.zeros(1, 2, dtype=torch.bool),
    }
    old_scale = runtime.scaler.get_scale()
    result = runtime.update(
        wrapped, [batch], optimizer, 1, lambda b: {k: v.cuda() for k, v in b.items()}
    )
    assert not result["optimizer_step"] and runtime.scaler.get_scale() < old_scale
    torch.testing.assert_close(model.weight, before, atol=0, rtol=0)
    hook.remove()
    runtime.close()


@pytest.mark.parametrize("backend", ["DEEPSPEED", "MEGATRON_LM", "PARALLELISM_CONFIG"])
def test_unimplemented_backends_fail_before_initialization(monkeypatch, backend):
    monkeypatch.setenv(f"ACCELERATE_USE_{backend}", "true")
    with pytest.raises(ValueError, match="not supported"):
        Execution("cpu")


def test_fsdp_requires_explicit_backend_and_cuda_launch(monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    with pytest.raises(ValueError, match="distributed_backend=fsdp2"):
        Execution("cpu")
    with pytest.raises(ValueError, match="CUDA.*torchrun"):
        Execution("cpu", backend="fsdp2")
    with pytest.raises(ValueError, match="ddp or fsdp2"):
        Execution("cpu", backend="zero")


def test_explicit_loss_weights_ignore_accelerate_accumulation_env(monkeypatch):
    monkeypatch.setenv("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", "8")
    runtime = Execution("cpu")
    assert runtime.accelerator.gradient_accumulation_steps == 1
    assert not runtime.accelerator.gradient_state.sync_with_dataloader
    runtime.close()


def test_nonfinite_gradients_do_not_update_without_scaler():
    runtime = Execution("cpu")
    model = TinyPolicy()
    wrapped, optimizer = runtime.prepare(model, torch.optim.SGD(model.parameters(), lr=0.01))
    before = model.weight.detach().clone()
    hook = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    batch = {
        "action": torch.ones(1, 2, 1),
        "state": torch.ones(1, 2),
        "action_is_pad": torch.zeros(1, 2, dtype=torch.bool),
    }
    with pytest.raises(RuntimeError, match="Non-finite gradients"):
        runtime.update(wrapped, [batch], optimizer, 1, lambda b: dict(b))
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    hook.remove()
    runtime.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ddp_matches_global_masked_loss(tmp_path, device):
    import subprocess
    import sys
    from pathlib import Path

    if device == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("NCCL verification requires two visible GPUs")
    log, pid = tmp_path / "ddp.log", tmp_path / "ddp.pid"
    with log.open("wb") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(Path(__file__).resolve()),
                "--distributed-check",
                device,
            ],
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid.write_text(str(process.pid))
    try:
        assert process.wait(timeout=60) == 0, log.read_text()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


if __name__ == "__main__":
    import sys

    runtime = Execution(sys.argv[-1])
    try:
        runtime.barrier()
        torch.manual_seed(10)
        full = {
            "action": torch.randn(4, 4, 1),
            "state": torch.randn(4, 2),
            "action_is_pad": torch.tensor(
                [[False] * 4, [False, True, True, True], [False, False, True, True], [False] * 4]
            ),
        }
        local = [
            {k: v[i : i + 1] for k, v in full.items()}
            for i in (runtime.rank * 2, runtime.rank * 2 + 1)
        ]
        full = {k: v.to(runtime.device) for k, v in full.items()}
        model, reference = TinyPolicy().to(runtime.device), TinyPolicy().to(runtime.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        wrapped, optimizer = runtime.prepare(model, optimizer)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        result = runtime.update(
            wrapped, local, optimizer, 0, lambda b: {k: v.to(runtime.device) for k, v in b.items()}
        )
        expected_loss, _ = reference(full)
        expected_loss.backward()
        expected_optimizer.step()
        torch.testing.assert_close(model.weight, reference.weight, rtol=1e-6, atol=1e-7)
        assert result["loss"] == pytest.approx(expected_loss.item(), rel=1e-6)
        runtime.barrier()
    finally:
        runtime.close()
