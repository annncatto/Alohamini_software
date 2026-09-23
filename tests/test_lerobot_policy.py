"""Optional adapter tests; checkpoints are small local ACT models, never robot commands."""

import json
import socket
from unittest.mock import Mock

import numpy as np
import pytest
from test_dataset import jpeg
from test_replay import replay_snapshot

pytest.importorskip("alohamini_lerobot")
torch = pytest.importorskip("torch")

from alohamini_lerobot.policy import LeRobotPolicy, _execution_config  # noqa: E402
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402

from alohamini.datasets.lerobot import export_lerobot  # noqa: E402
from alohamini.datasets.native import (  # noqa: E402
    LocalDataset,
    StateSelection,
    motor_feedback_frame,
)


@pytest.fixture
def training(tmp_path):
    snap = replay_snapshot()
    snap.images = {"forward": jpeg(shape=(32, 32, 3))}
    snap.payload["_motor_feedback"] = {
        "version": 1,
        "motors": {
            name: {
                "current_ma": 650,
                "velocity_raw": 20,
                "sample_started_s": 1.0,
                "sample_finished_s": 1.01,
            }
            for name in snap.payload["_robot_metadata"]["motors"]
        },
    }
    snap.payload["_host_timing"] = {"state_sample_finished_monotonic_s": 1.02}
    source = tmp_path / "native"
    ds = LocalDataset(source, fps=30, task="pick", robot_metadata=snap.payload["_robot_metadata"])
    ds.begin_episode()
    for _ in range(2):
        values = [snap.payload[name] for name in ds.names]
        ds.add_frame(
            {
                "observation.state": values,
                "action": values,
                **motor_feedback_frame(ds.features, snap.payload["_motor_feedback"]),
            },
            snap.images,
            {},
        )
    ds.close()
    export = tmp_path / "training"
    export_lerobot(source, export, state="joint_current,joint_velocity")
    return snap, export


def config(*, state=True):
    inputs = {"observation.images.forward": PolicyFeature(FeatureType.VISUAL, (3, 32, 32))}
    if state:
        inputs["observation.state"] = PolicyFeature(FeatureType.STATE, (28,))
    return ACTConfig(
        input_features=inputs,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (18,))},
        device="cpu",
        chunk_size=2,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        pretrained_backbone_weights=None,
        normalization_mapping={
            name: NormalizationMode.IDENTITY for name in ("STATE", "VISUAL", "ACTION")
        },
    )


def stub(export, *, state=True):
    policy = Mock(config=config(state=state))
    policy.to.return_value = policy
    policy.select_action.return_value = torch.arange(18, dtype=torch.float32).unsqueeze(0)
    pre, post = Mock(side_effect=lambda value: value), Mock(side_effect=lambda value: value)
    return LeRobotPolicy(policy, pre, post, export)


def test_live_state_matches_export_selection_and_rgb(training):
    snap, export = training
    adapter = stub(export)
    result = adapter.select_action(snap)
    obs = adapter.preprocessor.call_args.args[0]
    selection = StateSelection(adapter.source, "joint_current,joint_velocity")
    row = {
        "observation.state": [snap.payload[name] for name in selection.source_names],
        **motor_feedback_frame(adapter.source["features"], snap.payload["_motor_feedback"]),
    }
    np.testing.assert_allclose(obs["observation.state"].numpy()[0], selection.frame(row))
    image = obs["observation.images.forward"]
    assert image.shape == (1, 3, 32, 32)
    assert image[0, 0, 0, 0] > 0.9 and image[0, 2, 0, 0] < 0.1
    assert list(result) == list(adapter.action_names)
    assert list(result.values()) == list(range(18))
    adapter.reset()
    for component in (adapter.policy, adapter.preprocessor, adapter.postprocessor):
        component.reset.assert_called_once()


def test_visual_only_does_not_require_unused_current_or_velocity(training):
    snap, export = training
    adapter = stub(export, state=False)
    snap.payload.pop("_motor_feedback")
    adapter.select_action(snap)
    assert adapter.policy.select_action.call_args.args[0]["observation.state"].shape == (1, 0)


def test_missing_feedback_rejected_not_filled_with_zero(training):
    snap, export = training
    adapter = stub(export)
    snap.payload["_motor_feedback"]["motors"].pop("arm_left_elbow_flex")
    with pytest.raises(ValueError, match="Unavailable selected"):
        adapter.select_action(snap)
    adapter.policy.select_action.assert_not_called()


@pytest.mark.parametrize("change", ["units", "state_order", "action", "shape"])
def test_training_contract_mismatch_rejected(training, change):
    _, export = training
    path = export / "meta" / ("alohamini.json" if change == "units" else "info.json")
    value = json.loads(path.read_text())
    if change == "units":
        value["state_units"][0] = "mA"
    elif change == "state_order":
        value["features"]["observation.state"]["names"].reverse()
    elif change == "action":
        value["features"]["action"]["names"][0] = "tcp.dx"
    else:
        value["features"]["observation.images.forward"]["shape"] = [64, 64, 3]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        stub(export)


@pytest.mark.parametrize(
    "bad", [torch.zeros(18), torch.zeros(2, 18), torch.full((1, 18), float("nan"))]
)
def test_malformed_action_rejected(training, bad):
    snap, export = training
    adapter = stub(export)
    adapter.policy.select_action.return_value = bad
    with pytest.raises(ValueError):
        adapter.select_action(snap)


def test_missing_camera_and_wrong_shape_rejected(training):
    snap, export = training
    adapter = stub(export)
    snap.images = {}
    with pytest.raises(ValueError, match="unavailable"):
        adapter.select_action(snap)
    snap.images = {"forward": jpeg(shape=(64, 64, 3))}
    with pytest.raises(ValueError, match="shape"):
        adapter.select_action(snap)


@pytest.mark.parametrize("state", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ensemble", [False, True])
def test_real_act_checkpoint_processors_and_chunk_reset_are_offline(
    training, tmp_path, monkeypatch, state, device, ensemble
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable in the adapter test environment")
    snap, export = training
    torch.set_num_threads(1)
    torch.manual_seed(3)
    cfg = config(state=state)
    cfg.device = device
    cfg.normalization_mapping["STATE"] = NormalizationMode.MEAN_STD
    cfg.normalization_mapping["ACTION"] = NormalizationMode.MEAN_STD
    model = ACTPolicy(cfg).to(device).eval()
    stats = {"action": {"mean": torch.ones(18) * 5, "std": torch.ones(18) * 2}}
    if state:
        stats["observation.state"] = {"mean": torch.ones(28) * 0.3, "std": torch.ones(28) * 0.5}
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint, config_filename="policy_preprocessor.json")
    post.save_pretrained(checkpoint, config_filename="policy_postprocessor.json")

    def no_network(*args, **kwargs):
        raise AssertionError("Model loading/inference must remain local")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    # A full checkpoint must not fetch the ImageNet backbone even if its old config requests it.
    config_path = checkpoint / "config.json"
    saved = json.loads(config_path.read_text())
    saved["pretrained_backbone_weights"] = "ResNet18_Weights.IMAGENET1K_V1"
    config_path.write_text(json.dumps(saved))
    original_config = config_path.read_bytes()
    overrides = {"n_action_steps": 1, "temporal_ensemble_coeff": 0.01} if ensemble else {}
    adapter = LeRobotPolicy.from_pretrained(checkpoint, export, device=device, **overrides)
    if ensemble:
        reference_model = ACTPolicy(_execution_config(cfg, **overrides)).to(device).eval()
        reference_model.load_state_dict(model.state_dict())
    else:
        reference_model = model
    reference = LeRobotPolicy(reference_model, pre, post, export)
    for _ in range(3):  # Includes a cached action and a new chunk.
        np.testing.assert_allclose(
            list(adapter.select_action(snap).values()),
            list(reference.select_action(snap).values()),
            atol=1e-6,
        )
    adapter.reset()
    reference.reset()
    np.testing.assert_allclose(
        list(adapter.select_action(snap).values()),
        list(reference.select_action(snap).values()),
        atol=1e-6,
    )
    assert config_path.read_bytes() == original_config


def test_loader_rejects_custom_model_before_loading_weights(training, tmp_path):
    _, export = training
    checkpoint = tmp_path / "custom"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").touch()
    (checkpoint / "config.json").write_text(json.dumps({"type": "am_act"}))
    with pytest.raises(ValueError, match="not am_act"):
        LeRobotPolicy.from_pretrained(checkpoint, export, device="cpu")


def test_stale_motor_feedback_is_not_policy_state(training):
    snap, export = training
    adapter = stub(export)
    snap.payload["_host_timing"]["state_sample_finished_monotonic_s"] = 2
    with pytest.raises(ValueError, match="stale"):
        adapter.select_action(snap)
    adapter.policy.select_action.assert_not_called()


def test_relative_processors_and_missing_statistics_are_rejected(tmp_path):
    from alohamini_lerobot.policy import _processors

    for step, message in (
        ({"registry_name": "relative_actions_processor", "config": {}}, "Unsupported"),
        ({"registry_name": "normalizer_processor", "state_file": "missing.safetensors"}, "Missing"),
    ):
        (tmp_path / "policy_preprocessor.json").write_text(json.dumps({"steps": [step]}))
        with pytest.raises(ValueError, match=message):
            _processors(tmp_path)


def test_cli_checks_training_fps_before_opening_robot(monkeypatch, capsys):
    from alohamini_lerobot.policy import main

    import alohamini.apps.evaluation as app

    model = Mock(fps=20)
    monkeypatch.setattr(LeRobotPolicy, "from_pretrained", Mock(return_value=model))
    client = Mock()
    monkeypatch.setattr(app, "HostClient", client)
    assert (
        main(
            [
                "--policy.path",
                "/local/model",
                "--training-dataset",
                "/local/data",
                "--host",
                "127.0.0.1",
                "--robot_model",
                "alohamini2pro",
                "--task",
                "pick",
            ]
        )
        == 1
    )
    assert "FPS" in capsys.readouterr().err
    client.assert_not_called()


@pytest.mark.parametrize("steps", [0, -1, 3, True, 1.5])
def test_invalid_execution_steps_rejected(steps):
    with pytest.raises(ValueError, match="n_action_steps"):
        _execution_config(config(), n_action_steps=steps)


@pytest.mark.parametrize("coefficient", [float("nan"), float("inf"), True])
def test_invalid_ensemble_coefficient_rejected(coefficient):
    with pytest.raises(ValueError, match="finite"):
        _execution_config(config(), n_action_steps=1, temporal_ensemble_coeff=coefficient)


def test_execution_options_preserve_defaults_and_can_disable_saved_ensemble():
    cfg = config()
    assert _execution_config(cfg) == cfg
    with pytest.raises(ValueError, match="requires"):
        _execution_config(cfg, temporal_ensemble_coeff=0.01)
    enabled = _execution_config(cfg, n_action_steps=1, temporal_ensemble_coeff=0)
    assert enabled.temporal_ensemble_coeff == 0
    disabled = _execution_config(enabled, n_action_steps=2, temporal_ensemble_coeff=None)
    assert disabled.temporal_ensemble_coeff is None and disabled.n_action_steps == 2
    assert cfg.n_action_steps == 2 and cfg.temporal_ensemble_coeff is None


@pytest.mark.parametrize("coefficient", [None, 0.0, 0.01])
def test_act_queue_and_same_timestep_ensemble_through_adapter(training, coefficient):
    snap, export = training
    cfg = config()
    cfg.chunk_size = 3
    cfg = _execution_config(
        cfg,
        n_action_steps=2 if coefficient is None else 1,
        temporal_ensemble_coeff=coefficient,
    )
    model = ACTPolicy(cfg)
    # Real ACT queue/ensembler, deterministic predicted chunks; no robot I/O.
    model.predict_action_chunk = Mock(
        side_effect=[
            torch.tensor([base, base + 1, base + 2], dtype=torch.float32)
            .reshape(1, 3, 1)
            .expand(1, 3, 18)
            .clone()
            for base in (10, 20, 30, 40)
        ]
    )
    adapter = LeRobotPolicy(
        model, Mock(side_effect=lambda x: x), Mock(side_effect=lambda x: x), export
    )
    actual = [next(iter(adapter.select_action(snap).values())) for _ in range(3)]
    if coefficient is None:
        np.testing.assert_allclose(actual, [10, 11, 20])
        assert model.predict_action_chunk.call_count == 2
        expected_after_reset = 30
    else:
        weights = np.exp(-coefficient * np.arange(3))
        expected = [
            10,
            np.average([11, 20], weights=weights[:2]),
            np.average([12, 21, 30], weights=weights),
        ]
        np.testing.assert_allclose(actual, expected, atol=1e-5)
        assert model.predict_action_chunk.call_count == 3
        expected_after_reset = 40
    adapter.reset()
    assert next(iter(adapter.select_action(snap).values())) == expected_after_reset
