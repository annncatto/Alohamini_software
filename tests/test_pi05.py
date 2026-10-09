"""Focused PI0.5 migration checks; no downloads, full weights or robot access."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alohamini.datasets.statistics import RunningQuantileStats


def test_centered_statistics_preserve_small_variation():
    values = np.array([[93.06253, 1e8 + i * 0.001] for i in range(200)], np.float64)
    stats = RunningQuantileStats()
    for chunk in np.array_split(values, 5):
        stats.update(chunk)
    result = stats.get_statistics()
    assert result["std"][0] < 1e-12
    np.testing.assert_allclose(result["std"], values.std(0), atol=1e-7)


def test_float32_constant_has_no_artificial_variance():
    values = np.full((10013, 3), [93.06253, 0.063246, -72.011], np.float32)
    stats = RunningQuantileStats()
    for chunk in np.array_split(values, 6):
        stats.update(chunk)
    np.testing.assert_array_equal(stats.get_statistics()["std"], np.zeros(3))


def processor():
    from alohamini.policies.pi05.processor_pi05 import PI05Processor

    class Tokenizer:
        def tokenize(self, prompt, state):
            assert state.shape == (18,)  # Tokenization precedes padding to 32.
            return np.array([2, 3]), np.array([True, True])

    stats = {key: {"q01": [-2.0] * 18, "q99": [3.0] * 18} for key in ("state", "actions")}
    return PI05Processor(stats, Tokenizer())


def test_quantile_formula_and_differentiable_inverse():
    p = processor()
    x = np.arange(18, dtype=np.float32)
    norm = p.normalize("state", x)
    np.testing.assert_allclose(norm, (x.astype(np.float64) + 2) / (5 + 1e-6) * 2 - 1, rtol=1e-7)
    # Values outside q01/q99 remain outside [-1,1].
    assert norm[-1] > 1
    y = torch.zeros((1, 2, 32), requires_grad=True)
    out = p.restore_actions(y, x)
    out.sum().backward()
    assert torch.isfinite(y.grad).all()
    assert not y.grad[..., 18:].any()


def test_explicit_physical_span_floor_is_reversible():
    from alohamini.policies.pi05.processor_pi05 import PI05Processor

    original = processor()
    stats = {key: {"q01": [0.0] * 18, "q99": [0.0] * 18} for key in ("state", "actions")}
    adjusted = PI05Processor(
        stats, original.tokenizer, span_floors={"state": [0.01] * 18, "actions": [0.01] * 18}
    )
    values = np.full((2, 18), 0.001)
    normalized = adjusted.normalize("actions", values)
    assert np.max(np.abs(normalized)) < 1
    padded = torch.tensor(np.pad(normalized, ((0, 0), (0, 14))))
    np.testing.assert_allclose(adjusted.restore_actions(padded, np.zeros(18)), values, atol=1e-9)
    with pytest.raises(ValueError, match="18 finite"):
        PI05Processor(stats, original.tokenizer, span_floors={"state": [0.1]})


def test_delta_targets_and_black_masked_camera():
    from alohamini.policies.pi05.processor_pi05 import ARM_DELTA_DIMS, relative_actions

    p = processor()
    state = np.arange(18, dtype=np.float32)
    actions = np.stack([state + 1, state + 2])
    obs, targets = p.prepare(
        state, {"forward": np.zeros((480, 640, 3), np.uint8)}, "pick up", actions
    )
    delta = relative_actions(state, actions)
    np.testing.assert_allclose(delta[0, list(ARM_DELTA_DIMS)], 1)
    np.testing.assert_array_equal(
        delta[:, [6, 13, 14, 15, 16, 17]], actions[:, [6, 13, 14, 15, 16, 17]]
    )
    assert obs.images["base_0_rgb"].shape == (1, 3, 224, 224)
    assert not obs.image_masks["left_wrist_0_rgb"].any()
    assert targets.shape == (1, 2, 32)
    assert not targets[..., 18:].any()
    np.testing.assert_allclose(p.restore_actions(targets, state)[0], actions, atol=1e-5)


def test_official_model_small_forward_and_sampling(monkeypatch):
    pytest.importorskip("transformers")
    from transformers import PaliGemmaConfig

    from alohamini.policies.pi05 import gemma_pytorch
    from alohamini.policies.pi05.configuration_pi05 import PI05Config
    from alohamini.policies.pi05.modeling_pi05 import PI0Pytorch

    # Shrink only the vision backbone for the smoke test, never production defaults.
    def tiny_config():
        config = PaliGemmaConfig()
        config.vision_config.hidden_size = 32
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 4
        config.vision_config.patch_size = 112
        return config

    monkeypatch.setattr(gemma_pytorch, "CONFIG_MAPPING", {"paligemma": tiny_config})
    config = PI05Config(
        dtype="float32", paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=2
    )
    model = PI0Pytorch(config)
    # OpenPI fixes the real model's multimodal projection to 2048. Match the
    # test-only dummy text width without changing production architecture.
    model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear = torch.nn.Linear(
        32, 64
    )
    state = np.zeros(18, np.float32)
    obs, actions = processor().prepare(
        state, {"forward": np.zeros((224, 224, 3), np.uint8)}, "pick", np.zeros((2, 18), np.float32)
    )
    loss = model(obs, actions, noise=torch.zeros_like(actions), time=torch.full((1,), 0.5))
    assert loss.shape == (1, 2, 32)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert torch.isfinite(model.action_out_proj.weight.grad).all()
    from alohamini.learning.metrics import MetricAccumulator
    from alohamini.policies.pi05.adapter import PI05Algorithm, PI05TrainModel

    # Exercise the production training adapter around the same small network.
    wrapper = PI05TrainModel.__new__(PI05TrainModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.network = model
    torch.manual_seed(71)
    expected = model(obs, actions).mean()
    torch.manual_seed(71)
    raw = {"observation": obs, "action": actions}
    actual, outputs = wrapper(raw)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    accumulator = MetricAccumulator(PI05Algorithm.metric_specs(None))
    accumulator.add(outputs, raw, PI05Algorithm.loss_counts(raw))
    assert accumulator.result()["metrics"]["loss_flow"] == pytest.approx(actual.item())
    assert accumulator.result()["metric_totals"]["loss_flow"]["count"] == 1
    model.eval()
    predicted = model.sample_actions(torch.device("cpu"), obs, num_steps=2)
    assert predicted.shape == (1, 2, 32)
    assert torch.isfinite(predicted).all()
    # Local transformer definitions must not replace the official classes globally.
    from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

    assert GemmaRMSNorm.__module__.startswith("transformers.")


def test_policy_queue_and_reset():
    pytest.importorskip("transformers")
    from alohamini.policies.pi05.policy import PI05Policy

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(action_horizon=3, action_dim=32)
            self.calls = 0

        def sample_actions(self, device, observation, num_steps):
            self.calls += 1
            return torch.zeros((1, 3, 32))

    model = Model()
    policy = PI05Policy(model, processor(), n_action_steps=2)
    args = (np.zeros(18), {"forward": np.zeros((224, 224, 3), np.uint8)}, "pick")
    for _ in range(3):
        assert policy.select_action(*args).shape == (18,)
    assert model.calls == 2
    policy.reset()
    policy.select_action(*args)
    assert model.calls == 3
