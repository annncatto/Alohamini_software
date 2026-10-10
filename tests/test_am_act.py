from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from test_native_learning import batch, model_options

from alohamini.learning.metrics import MetricAccumulator
from alohamini.learning.policy import make_policy
from alohamini.policies.am_act.classification import (
    BASE_CLASSES,
    class_weights,
    classification_metrics,
    prepare_options,
)
from alohamini.policies.am_act.latent_prior import gaussian_kl
from alohamini.policies.registry import algorithm


def conditional_options(**overrides):
    options = model_options(
        state=False,
        posterior_condition="image_action",
        prior_type="conditional_gaussian",
        latent_inference_mode="sample",
    )
    options.update(overrides)
    return options


def test_weights_keep_stop_one_and_clip_each_direction():
    assert class_weights([4, 100, 25], reference_index=1) == [5, 1, 2]
    assert class_weights([0, 100, 400], reference_index=1) == [5, 1, 1]
    assert class_weights([0, 0, 1], reference_index=1) == [5, 1, 1]
    assert class_weights([4, 100, 25], "inverse_frequency", reference_index=1) == [5, 1, 4]
    assert class_weights([0, 100, 1], "none", reference_index=1) == [1, 1, 1]


def test_auto_classes_use_named_unique_numeric_rows_only():
    # Deliberately permuted coordinates: no inference from vector width or index.
    names = ["theta.vel", "arm", "y.vel", "x.vel"]
    data = SimpleNamespace(
        info={"features": {"action": {"names": names}}},
        rows=[{"action": [0, 2, 0, x]} for x in [0, 0, -0.15, 0.15, 0.15]],
        _used_rows={"action": {0, 1, 2, 3}},
    )
    options = {}
    prepare_options(options, data)
    assert options["discrete_action_dims"] == [3, 2, 0]
    assert options["discrete_action_values"] == list(BASE_CLASSES.values())
    assert options["discrete_action_class_counts"] == [[1, 2, 1], [0, 4, 0], [0, 4, 0]]
    assert options["discrete_action_class_weights"][1] == [5, 1, 5]
    options["discrete_action_class_weights"] = [[2, 1, 3]] * 3
    options["discrete_action_weight_source"] = "manual"
    prepare_options(options, data)
    assert options["discrete_action_class_weights"] == [[2, 1, 3]] * 3


def test_loss_groups_cannot_silently_omit_continuous_axes():
    with pytest.raises(ValueError, match="omit continuous"):
        make_policy("am_act", model_options(action_loss_groups={"left": list(range(7))}))


def test_conditional_kl_matches_torch_distribution():
    from torch.distributions import Normal, kl_divergence

    torch.manual_seed(1)
    mq, lq, mp, lp = [torch.randn(3, 8) for _ in range(4)]
    expected = kl_divergence(Normal(mq, (lq / 2).exp()), Normal(mp, (lp / 2).exp()))
    torch.testing.assert_close(gaussian_kl(mq, lq, mp, lp), expected)
    torch.testing.assert_close(gaussian_kl(mq, lq, mq, lq), torch.zeros_like(mq))


@pytest.mark.parametrize("pooling,queries", [("gap", 1), ("multi_query_attention", 4)])
def test_shared_vision_posterior_padding_and_prior_gradients(pooling, queries):
    options = conditional_options(latent_visual_pooling=pooling)
    options["input_features"]["observation.images.wrist"] = {"type": "VISUAL", "shape": (3, 32, 32)}
    model = make_policy("am_act", options)
    data = batch(False)
    data["observation.images.wrist"] = torch.rand(2, 3, 32, 32)
    calls = []
    handle = model.model.backbone.register_forward_hook(lambda *args: calls.append(1))
    torch.manual_seed(12)
    loss, metrics = model(data)
    loss.backward()
    assert len(calls) == 2
    assert torch.isfinite(loss) and metrics["kld_loss"] >= 0
    assert model.model.latent_prior.net[-1].weight.grad.abs().sum() > 0
    assert model.model.vae_encoder_latent_output_proj.weight.grad.abs().sum() > 0
    assert model.model.visual_pooling.camera.weight.grad.abs().sum() > 0
    assert model.model.vae_encoder_pos_enc.shape[1] == 1 + queries + options["chunk_size"]
    changed = deepcopy(data)
    changed["action"][data["action_is_pad"]] = 10000
    torch.manual_seed(12)
    changed_loss, _ = model(changed)
    torch.testing.assert_close(changed_loss, loss)
    handle.remove()


def test_phase_retains_z_but_updates_vision_and_reset_discards_queue():
    model = make_policy("am_act", conditional_options())
    data = batch(False)
    calls = []
    handle = model.model.backbone.register_forward_hook(lambda *args: calls.append(1))
    prior_calls = []
    prior_handle = model.model.latent_prior.register_forward_hook(
        lambda *args: prior_calls.append(1)
    )
    first = model.select_action(data)
    z = model._cached_z.clone()
    model.select_action(data)
    assert len(calls) == 1  # existing chunk execution does not rerun vision
    changed = {
        **data,
        "observation.images.forward": torch.zeros_like(data["observation.images.forward"]),
    }
    updated = model.select_action(changed)
    assert len(calls) == 2
    assert len(prior_calls) == 1
    torch.testing.assert_close(model._cached_z, z, rtol=0, atol=0)
    assert not torch.equal(first, updated)
    model.refresh_latent()
    assert model._cached_z is None and not model._action_queue
    model.select_action(changed)
    assert len(prior_calls) == 2
    assert not torch.equal(model._cached_z, z)
    handle.remove()
    prior_handle.remove()


def test_latent_modes_and_explicit_z_change_predictions():
    data = batch(False)
    model = make_policy("am_act", conditional_options(latent_inference_mode="mean"))
    a = model.predict_action_chunk(data)
    torch.testing.assert_close(model.predict_action_chunk(data), a, rtol=0, atol=0)
    prepared = {**data, "observation.images": [data["observation.images.forward"]]}
    zeros = torch.zeros(2, model.config.latent_dim)
    with torch.no_grad():
        a = model.model(prepared, latent_sample=zeros)[0]
        b = model.model(prepared, latent_sample=torch.ones_like(zeros))[0]
    assert (a - b).abs().max() > 1e-5


def test_warmup_advances_on_update_only_and_survives_strict_loading():
    options = conditional_options(latent_kl_warmup_steps=4, kl_weight=1)
    model = make_policy("am_act", options)
    for _ in range(2):
        _, metrics = model(batch(False))
        assert float(metrics["kld_loss_weighted"]) == pytest.approx(metrics["kld_loss"] / 4)
    assert model.kl_updates == 0
    model.update()
    restored = make_policy("am_act", options)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.kl_updates == 1
    _, metrics = restored(batch(False))
    assert float(metrics["kld_loss_weighted"]) == pytest.approx(metrics["kld_loss"] / 2)


def test_recall_macro_and_confusion_aggregate_counts_not_batch_means():
    recipe = algorithm("am_act")
    config = recipe.config_class(
        **model_options(discrete_action_dims=[14], discrete_action_values=[[-1, 0, 1]])
    )
    specs = [s for s in recipe.metric_specs(config) if s.name.startswith("classification_")]
    acc = MetricAccumulator(specs)
    # Actual negative: 1/3 correct; actual stop: 2/2 correct; positive absent.
    for truth, prediction, valid in [
        ([0, 1, 2], [0, 1, 0], [True, True, False]),
        ([0, 0, 1], [1, 2, 1], [True, True, True]),
    ]:
        logits = torch.nn.functional.one_hot(torch.tensor([prediction]), 3).float()
        outputs = classification_metrics(logits, torch.tensor([truth]), torch.tensor([valid]), 14)
        acc.add(outputs, {}, {})
    values = acc.result()["metrics"]
    assert values["classification_dim_14_class_0_recall"] == pytest.approx(1 / 3)
    assert values["classification_dim_14_macro_recall"] == pytest.approx(2 / 3)
    assert "classification_dim_14_class_2_recall" not in values
    assert values["classification_confusion_dim_14_0_2"] == 1
    assert values["classification_confusion_dim_14_1_1"] == 2


def test_physical_labels_match_frequency_rule_even_at_midpoints():
    from alohamini.policies.am_act.classification import nearest_class
    from alohamini.policies.am_act.processor import AMACTProcessor

    recipe = algorithm("am_act")
    config = recipe.config_class(
        **model_options(discrete_action_dims=[14], discrete_action_values=[[-0.15, 0, 0.15]])
    )
    stats = {"action": {"mean": [0.01] * 18, "std": [0.02] * 18}}
    stats["observation.state"] = {"mean": [0.0] * 18, "std": [1.0] * 18}
    stats["observation.images.forward"] = {"mean": [[[0.0]]] * 3, "std": [[[1.0]]] * 3}
    processor = AMACTProcessor.from_config(config, stats)
    raw = batch()
    raw["action"][..., 14] = torch.tensor([[-0.075, 0, 0.075], [-0.15, 0.15, 0]])
    original = raw["action"].clone()
    processed = processor(raw)
    expected = nearest_class(original[..., 14], [-0.15, 0, 0.15])
    torch.testing.assert_close(processed["discrete_action_labels"][..., 0], expected)
    torch.testing.assert_close(raw["action"], original)


def test_fixed_speed_classes_survive_temporal_ensemble():
    options = model_options()
    options.update(
        n_action_steps=1,
        temporal_ensemble_coeff=0.01,
        discrete_action_dims=[14],
        discrete_action_values=[[-0.15, 0, 0.15]],
    )
    stats = {"action": {"mean": [0.0] * 18, "std": [1.0] * 18}}
    model = make_policy("am_act", options, stats)
    old = torch.zeros(1, 3, 18)
    old[..., 14] = -0.15
    new = torch.ones(1, 3, 18)
    new[..., 14] = 0.15
    chunks = iter([old, new])
    model.predict_action_chunk = lambda _: next(chunks)
    model.select_action({})
    actual = model.select_action({})
    assert actual[0, 14] == pytest.approx(0.15)
    assert 0 < actual[0, 0] < 1  # continuous joints still use both chunks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_tensor_and_conditional_bfloat16_backward():
    device = torch.device("cuda")
    tensor = torch.arange(9.0, device=device).reshape(3, 3)
    torch.testing.assert_close((tensor @ tensor).cpu(), tensor.cpu() @ tensor.cpu())
    model = make_policy("am_act", conditional_options(latent_kl_warmup_steps=5000)).to(device)
    data = {key: value.to(device) for key, value in batch(False).items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = model(data)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
