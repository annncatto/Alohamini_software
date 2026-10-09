from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from test_native_learning import batch, model_options

from alohamini.learning.execution import Execution
from alohamini.learning.logging import TrainingProgress
from alohamini.learning.metrics import MetricAccumulator, MetricSpec, loss_count
from alohamini.learning.policy import make_policy
from alohamini.policies.registry import algorithm


def test_element_means_zero_counts_detachment_and_one_packed_reduction():
    spec = MetricSpec("l1", "l1", "elements", loss_count("n"))
    accumulator = MetricAccumulator([spec])
    for value, count in ((0.1, 200), (1.0, 20), (float("nan"), 0)):
        accumulator.add({"l1": torch.tensor(value, requires_grad=True)}, {}, {"n": count})
    assert not accumulator.totals.requires_grad and accumulator.totals.grad_fn is None
    calls = []

    def reduce(tensor, reduction):
        calls.append(reduction)
        # A second rank with different valid counts: numerator=10, count=100.
        return tensor + torch.tensor([[10, 100]])

    result = accumulator.result(reduce)
    assert calls == ["sum"]
    assert result["metrics"]["l1"] == pytest.approx(50 / 320)
    assert result["metric_totals"]["l1"]["count"] == 320


def test_window_is_mean_of_steps_not_last_batch_or_dataset_mean():
    progress = TrainingProgress(
        SimpleNamespace(main=True, world_size=1), frames=100, episodes=1, steps=2
    )
    for l1, kl in ((0.1, 2), (1, 4)):
        progress.update(
            dict(
                loss=l1 + 10 * kl,
                grad_norm=1,
                lr=1e-5,
                dataloading_s=0.1,
                update_s=0.1,
                metrics={"loss_l1": l1, "loss_kl_weighted": 10 * kl},
            ),
            1,
        )
    summary = progress.summary(2)
    assert summary["metrics"]["loss_l1"] == pytest.approx(0.55)
    assert summary["loss"] == pytest.approx(sum(summary["metrics"].values()))
    progress.reset()
    assert progress.metric_sums == {}


@pytest.mark.parametrize("kind", ["act", "am_act"])
def test_real_policy_metrics_do_not_change_loss_gradients_or_update(kind):
    from accelerate.state import AcceleratorState, GradientState

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    options = model_options(kl_weight=3)
    if kind == "am_act":
        options.update(
            discrete_action_dims=[14],
            discrete_action_values=[[-1.0, 0.0, 1.0]],
            discrete_action_class_weights=[[2, 1, 3]],
            fixed_action_dims=[17],
            action_loss_groups={"arm": list(range(14)), "base": [14, 15, 16]},
            action_loss_weights={"arm": 1, "base": 2},
            discrete_action_loss_weight=2,
        )
    model = make_policy(kind, options, {"action": {"mean": [0.0] * 18, "std": [1.0] * 18}})
    reference = deepcopy(model)
    raw = batch()
    recipe = algorithm(kind)
    specs = recipe.metric_specs(model.config)
    runtime = Execution("cpu")
    gradients = [[], []]
    handles = [
        m.model.action_head.weight.register_hook(
            lambda grad, i=i: gradients[i].append(grad.detach().clone())
        )
        for i, m in enumerate((model, reference))
    ]
    try:
        results = []
        for i, policy in enumerate((model, reference)):
            wrapped, optimizer = runtime.prepare(
                policy, torch.optim.SGD(policy.parameters(), lr=0.01)
            )
            torch.manual_seed(73)
            results.append(
                runtime.update(
                    wrapped,
                    [raw],
                    optimizer,
                    0,
                    lambda b: dict(b),
                    reduction=recipe.loss_counts,
                    metric_specs=specs if i == 0 else (),
                )
            )
        assert results[0]["loss"] == results[1]["loss"]
        torch.testing.assert_close(gradients[0][0], gradients[1][0], rtol=0, atol=0)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        values = results[0]["metrics"]
        terms = ["loss_l1", "loss_kl_weighted"]
        assert values["loss_kl_weighted"] == pytest.approx(3 * values["loss_kl"])
        if kind == "am_act":
            terms.append("loss_classification_weighted")
            assert values["loss_l1"] == pytest.approx(
                (values["loss_l1_arm"] + 2 * values["loss_l1_base"]) / 3
            )
            assert values["loss_classification_weighted"] == pytest.approx(
                2 * values["loss_classification"]
            )
            assert results[0]["metric_totals"]["loss_l1_base"]["count"] == 10
        assert results[0]["loss"] == pytest.approx(sum(values[t] for t in terms), rel=1e-6)
        assert results[0]["metric_totals"]["loss_kl"]["count"] == 2
    finally:
        for handle in handles:
            handle.remove()
        runtime.close()
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()


@pytest.mark.parametrize("masked", [False, True])
def test_diffusion_declares_actual_mask_convention(masked):
    recipe = algorithm("diffusion")
    recipe.options({"model": {"do_mask_loss_for_padding": masked}}, "cpu")
    config = SimpleNamespace(
        do_mask_loss_for_padding=masked, horizon=3, action_feature=SimpleNamespace(shape=(18,))
    )
    raw = batch()
    accumulator = MetricAccumulator(recipe.metric_specs(config))
    accumulator.add({"diffusion_loss": torch.tensor(2.0)}, raw, recipe.loss_counts(raw))
    assert accumulator.result()["metric_totals"]["loss_diffusion"] == {
        "sum": 2 * (90 if masked else 108),
        "count": 90 if masked else 108,
    }


def test_flow_and_fastwam_do_not_inherit_act_padding_or_double_weight():
    raw = batch()
    for kind, outputs, expected in (
        ("pi05", {"flow_loss": 7.0}, {"loss_flow": 7.0}),
        (
            "fastwam",
            {"loss_video": 3.0, "loss_action": 5.0},
            {"loss_video_weighted": 3.0, "loss_action_weighted": 5.0},
        ),
    ):
        recipe = algorithm(kind)
        accumulator = MetricAccumulator(recipe.metric_specs(None))
        accumulator.add(outputs, raw, recipe.loss_counts(raw))
        result = accumulator.result()
        assert result["metrics"] == expected
        assert all(t["count"] == 2 for t in result["metric_totals"].values())


def test_smolvla_intermediate_means_keep_zero_padding_denominator():
    recipe = algorithm("smolvla")
    config = SimpleNamespace(action_feature=SimpleNamespace(shape=(18,)), max_action_dim=32)
    accumulator = MetricAccumulator(recipe.metric_specs(config))
    raw = batch()
    outputs = {
        "loss": 2.0,
        "losses_after_forward": 3.0,
        "losses_after_in_ep_bound": 1.0,
        "losses_after_rm_padding": 1.0,
    }
    accumulator.add(outputs, raw, recipe.loss_counts(raw))
    counts = accumulator.result()["metric_totals"]
    assert counts["loss_flow"]["count"] == 90
    assert counts["losses_after_rm_padding"]["count"] == 108


def test_missing_or_invalid_metrics_fail_explicitly():
    accumulator = MetricAccumulator([MetricSpec("l1", "l1", "elements", loss_count("n"))])
    with pytest.raises(KeyError):
        accumulator.add({}, {}, {"n": 1})
    with pytest.raises(ValueError, match="scalar"):
        accumulator.add({"l1": torch.ones(2)}, {}, {"n": 1})
    with pytest.raises(ValueError, match="denominator"):
        accumulator.add({"l1": 1.0}, {}, {"n": -1})
    accumulator.add({"l1": float("nan")}, {}, {"n": 1})
    with pytest.raises(RuntimeError, match="non-finite"):
        accumulator.result()


def test_disabled_vae_has_no_kl_specs():
    for kind in ("act", "am_act"):
        recipe = algorithm(kind)
        config = recipe.config_class(**model_options(use_vae=False))
        assert [s.name for s in recipe.metric_specs(config)] == ["loss_l1"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_diagnostics_detach_and_pack_python_and_tensor_values():
    specs = [MetricSpec(name, name, "samples", loss_count("n")) for name in ("tensor", "float")]
    accumulator = MetricAccumulator(specs, device="cuda")
    accumulator.add(
        {"tensor": torch.tensor(2.0, device="cuda", requires_grad=True), "float": 3.0}, {}, {"n": 5}
    )
    assert accumulator.totals.device.type == "cuda" and accumulator.totals.grad_fn is None
    assert accumulator.result()["metrics"] == {"tensor": 2.0, "float": 3.0}
