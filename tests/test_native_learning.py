import importlib
import json
import os
import subprocess
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_dataset import frame, jpeg, metadata

from alohamini.datasets.record import StateSelection, motor_feedback_frame
from alohamini.datasets.record import _EpisodeWriter as LocalDataset
from alohamini.learning.data import AlohaMiniDataset, capture_timeline
from alohamini.learning.policy import (
    NativePolicy,
    evaluate_robot,
    make_policy,
    save_checkpoint,
)
from alohamini.learning.processor import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    Processor,
    act_statistics,
    image_tensor,
    scale_action,
)
from alohamini.learning.train import launch_training, offline_evaluate, train

torch.set_num_threads(2)


def model_options(*, state=True, **kwargs):
    inputs = {"observation.images.forward": {"type": "VISUAL", "shape": (3, 32, 32)}}
    if state:
        inputs["observation.state"] = {"type": "STATE", "shape": (18,)}
    return dict(
        input_features=inputs,
        output_features={"action": {"type": "ACTION", "shape": (18,)}},
        chunk_size=3,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_vae_encoder_layers=1,
        dropout=0.0,
        pretrained_backbone_weights=None,
        **kwargs,
    )


def batch(state=True):
    result = {
        "observation.images.forward": torch.rand(2, 3, 32, 32),
        "action": torch.randn(2, 3, 18),
        "action_is_pad": torch.tensor([[False, False, True], [False, False, False]]),
    }
    if state:
        result["observation.state"] = torch.randn(2, 18)
    return result


@pytest.mark.parametrize("kind", ["act", "am_act"])
@pytest.mark.parametrize("use_state", [True, False])
def test_native_model_forward_backward_and_reset(kind, use_state):
    model = make_policy(kind, model_options(state=use_state))
    data = batch(use_state)
    loss, _ = model(data)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.model.action_head.weight.grad is not None
    prediction = model.predict_action_chunk(data)
    torch.testing.assert_close(model.select_action(data), prediction[:, 0])
    torch.testing.assert_close(model.select_action(data), prediction[:, 1])
    model.reset()
    torch.testing.assert_close(model.select_action(data), prediction[:, 0])


@pytest.mark.parametrize("kind", ["act", "am_act"])
def test_temporal_ensemble_and_reset(kind):
    options = model_options()
    options.update(n_action_steps=1, temporal_ensemble_coeff=0.01)
    model = make_policy(kind, options)
    calls = iter([torch.arange(3.0).reshape(1, 3, 1), torch.full((1, 3, 1), 10.0)])
    model.predict_action_chunk = lambda _: next(calls)
    assert model.select_action({}).item() == 0
    expected = (1 + 10 * np.exp(-0.01)) / (1 + np.exp(-0.01))
    assert model.select_action({}).item() == pytest.approx(expected)
    model.reset()
    assert model.temporal_ensembler.ensembled_actions is None


@pytest.mark.parametrize("kind", ["act", "am_act"])
@pytest.mark.parametrize("use_state", [True, False])
def test_against_actual_old_fork(kind, use_state):
    """Run separately in the old environment with the source fork on PYTHONPATH."""
    if not os.environ.get("ALOHAMINI_COMPARE_OLD_FORK"):
        pytest.skip("Explicit old-fork comparison; not the optional released LeRobot adapter")
    pytest.importorskip("lerobot")
    from lerobot.configs.types import FeatureType, PolicyFeature

    prefix = "ACT" if kind == "act" else "AMACT"
    config_type = getattr(
        importlib.import_module(f"lerobot.policies.{kind}.configuration_{kind}"), f"{prefix}Config"
    )
    policy_type = getattr(
        importlib.import_module(f"lerobot.policies.{kind}.modeling_{kind}"), f"{prefix}Policy"
    )
    options = model_options(state=use_state)
    stats = {"action": {"mean": [0.0] * 18, "std": [1.0] * 18}}
    if kind == "am_act":
        options.update(
            discrete_action_dims=[14],
            discrete_action_values=[[-1.0, 0.0, 1.0]],
            discrete_action_class_weights=[[2.0, 1.0, 3.0]],
            fixed_action_dims=[17],
            action_loss_groups={"arm": list(range(14)), "base": [14, 15, 16]},
            action_loss_weights={"arm": 1.0, "base": 2.0},
            observation_state_dims=[0, 2, 4] if use_state else [],
        )
    native = make_policy(kind, options, stats)
    legacy_options = asdict(native.config)
    # New opt-in AM-ACT controls have no equivalent in the frozen old fork.
    legacy_fields = {f.name for f in fields(config_type)}
    legacy_options = {key: value for key, value in legacy_options.items() if key in legacy_fields}
    for key in ("input_features", "output_features"):
        legacy_options[key] = {
            k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
            for k, v in legacy_options[key].items()
        }
    legacy = policy_type(config_type(**legacy_options), dataset_stats=stats)
    native.load_state_dict(legacy.state_dict(), strict=True)
    data = batch(use_state)
    legacy_data = dict(data)
    if not use_state:
        legacy_data["observation.state"] = torch.empty(2, 0)
    torch.manual_seed(42)
    old_loss, old_metrics = legacy(legacy_data)
    old_loss.backward()
    torch.manual_seed(42)
    new_loss, new_metrics = native(data)
    new_loss.backward()
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
    assert {key: new_metrics[key] for key in old_metrics} == old_metrics
    for old, new in zip(legacy.parameters(), native.parameters(), strict=True):
        if old.grad is not None:
            torch.testing.assert_close(new.grad, old.grad, rtol=0, atol=0)
    for _ in range(5):
        torch.testing.assert_close(
            native.select_action(data), legacy.select_action(legacy_data), rtol=0, atol=0
        )


@pytest.fixture
def recording(tmp_path):
    root = tmp_path / "dataset"
    robot_metadata = metadata()
    for motor in robot_metadata["motors"].values():
        motor["drive_mode"] = 0
        motor["range_min"], motor["range_max"] = 0, 4095
    dataset = LocalDataset(root, fps=30, task="test", robot_metadata=robot_metadata)
    for episode in range(2):
        dataset.begin_episode()
        for i in range(4):
            value = frame(dataset)
            value["action"][:] = i + 10 * episode
            stamp = 100 + episode + i / 30
            feedback = {
                name: {
                    "sample_started_s": stamp - 0.001,
                    "sample_finished_s": stamp,
                    "velocity_raw": i,
                    "current_ma": 1000.0,
                }
                for name in dataset.features["observation.motor_current_ma"]["names"]
            }
            value.update(motor_feedback_frame(dataset.features, {"version": 1, "motors": feedback}))
            record = {
                "safety": {"feedback_valid": True},
                "client_timing": {
                    "observation_received_monotonic_s": stamp + 1000,
                    "action_sample_started_monotonic_s": stamp + 1000.001,
                    "action_sample_finished_monotonic_s": stamp + 1000.002,
                    "command_sent_monotonic_s": stamp + 1000.003,
                },
                "host_timing": {
                    "state_sample_monotonic_s": stamp,
                    "camera_capture_monotonic_s": {"forward": stamp},
                },
                "requested_action": dict.fromkeys(dataset.names, float(i + 10 * episode)),
            }
            assert dataset.add_frame(value, {"forward": jpeg()}, record)
        dataset.save_episode()
    dataset.close()
    return root


def samples(root, **kwargs):
    return AlohaMiniDataset(
        root,
        chunk_size=3,
        image_size=(32, 32),
        state="none",
        review_note="Synthetic fixture",
        **kwargs,
    )


def test_capture_timeline_preserves_jitter_and_ignores_event_rows(recording):
    path = recording / "episodes/episode_000000/safety.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for row in records:
        if row.get("frame_index") is not None:
            i = row["frame_index"]
            row["host_timing"]["camera_capture_monotonic_s"]["forward"] = 100 + i / 29.5
            row["alignment_error_s"] = 0.005
    records.append({"event": {"type": "test"}})
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    before = path.read_bytes()
    timing = capture_timeline(recording, 0)
    assert timing["measured_fps"] == pytest.approx(29.5)
    np.testing.assert_allclose(timing["nominal_time_s"], np.arange(4) / 30)
    np.testing.assert_allclose(timing["capture_time_s"], np.arange(4) / 29.5)
    np.testing.assert_allclose(timing["camera_intervals_s"]["forward"], 1 / 29.5)
    np.testing.assert_allclose(timing["alignment_error_s"], 0.005)
    assert path.read_bytes() == before
    np.testing.assert_allclose(capture_timeline(recording, 1)["drift_s"], 0, atol=1e-12)


def test_capture_timeline_missing_host_time_is_not_filled_from_pc(recording):
    path = recording / "episodes/episode_000000/safety.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for row in records:
        if row.get("frame_index") == 2:
            row["host_timing"]["camera_capture_monotonic_s"] = {}
            row["client_timing"] = {"observation_received_monotonic_s": 999}
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert np.isnan(capture_timeline(recording, 0)["capture_time_s"][2])


def test_notebook_runs_offline_without_confirmation(recording, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    notebook = json.loads(
        (Path(__file__).parents[1] / "examples/learning/local_act.ipynb").read_text()
    )
    cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    ids = list(cells)
    assert ids.index("capture-timing") < ids.index("local-act-4")
    assert "timing-review" not in cells
    assert "RUN_TRAINING = False" in cells["local-act-1"]
    assert "ENABLE_ROBOT = False" in cells["local-act-1"]

    def unexpected(*args, **kwargs):
        pytest.fail("Notebook must not prompt, launch training or connect to a robot")

    monkeypatch.setattr("builtins.input", unexpected)
    monkeypatch.setattr("alohamini.learning.train.launch_training", unexpected)
    monkeypatch.setattr("alohamini.learning.policy.evaluate_robot", unexpected)
    scope = {
        "settings": {
            "dataset": str(recording),
            "train_episodes": [0],
            "val_episodes": [1],
            "policy": "act",
            "state": "none",
            "image_size": [32, 32],
            "device": "cpu",
            "model": model_options(state=False),
        },
        "CHECKPOINT": recording / "no_checkpoint",
        "RUN_TRAINING": False,
        "ENABLE_ROBOT": False,
    }
    # The teaching training config no longer sets an inference queue length.
    scope["settings"]["model"].pop("n_action_steps")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = compile(cells[cell["id"]], cell["id"], "exec")
            if cell["id"] != "local-act-1":
                exec(source, scope)
    assert len(scope["samples"]) == 4
    assert scope["samples"].review_note == ""
    scope["plt"].close("all")


def test_dataset_warnings_do_not_require_review_note(recording, caplog):
    path = recording / "episodes/episode_000000/safety.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for row in records:
        if row.get("frame_index") in (2, 3):
            row["host_timing"]["camera_capture_monotonic_s"]["forward"] += 0.04
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    data = AlohaMiniDataset(
        recording, episodes=[0], chunk_size=3, state="none", image_size=(32, 32)
    )
    assert data.report["warnings"] > 0
    assert data.review_note == ""
    assert data[1]["action_is_pad"].tolist() == [False, False, False]
    assert "No samples are removed by quality warnings" in caplog.text


def test_dataset_errors_still_prevent_training(recording):
    path = recording / "meta/info.json"
    info = json.loads(path.read_text())
    info["format"] = "invalid"
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="Dataset integrity check failed"):
        AlohaMiniDataset(recording, episodes=[0], state="none")


def test_chunks_do_not_cross_episodes_and_stats_are_train_only(recording):
    data = samples(recording, episodes=[0, 1])
    assert data[3]["action_is_pad"].tolist() == [False, True, True]
    assert data[3]["action"][:, 0].tolist() == [3.0, 3.0, 3.0]
    training = samples(recording, episodes=[0])
    assert training.statistics()["action"]["mean"] == [1.5] * 18


def test_default_image_size_preserves_original_resolution(recording):
    data = AlohaMiniDataset(recording, episodes=[0], state="none")
    assert DEFAULT_IMAGE_SIZE == (480, 640)
    assert data.input_features["observation.images.forward"].shape == (3, 480, 640)
    assert data.observation(0)["observation.images.forward"].shape == (3, 480, 640)
    settings = json.loads((Path(__file__).parents[1] / "examples/learning/act.json").read_text())
    assert settings["image_size"] == list(DEFAULT_IMAGE_SIZE)


def test_act_statistics_use_imagenet_without_decoding_images(recording, monkeypatch):
    from torchvision.transforms import Normalize

    data = samples(recording, episodes=[0])
    empirical = data.statistics()
    original = data._value

    def numeric_only(index, key):
        assert not key.startswith("observation.images."), "Fixed RGB stats need no image scan"
        return original(index, key)

    monkeypatch.setattr(data, "_value", numeric_only)
    stats = act_statistics(data)
    assert stats["action"] == empirical["action"]
    for camera in data.cameras:
        key = f"observation.images.{camera}"
        assert stats[key]["mean"] == [[[v]] for v in IMAGENET_MEAN]
        assert stats[key]["std"] == [[[v]] for v in IMAGENET_STD]
        pixels = torch.rand(2, 3, 480, 640)
        actual = Processor(stats)({key: pixels})[key]
        expected = Normalize(IMAGENET_MEAN, IMAGENET_STD)(pixels)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert data.statistics(keys=[]) == {}
    with pytest.raises(ValueError, match="Unknown statistics fields"):
        data.statistics(keys=["unknown"])


def test_checkpoint_preserves_existing_image_statistics(recording, tmp_path):
    data = samples(recording, episodes=[0])
    old_stats = data.statistics()
    key = f"observation.images.{data.cameras[0]}"
    old_stats[key] = {"mean": [[[0.1]], [[0.2]], [[0.3]]], "std": [[[0.5]]] * 3}
    model = make_policy("act", model_options(state=False))
    checkpoint = tmp_path / "old_stats"
    save_checkpoint(checkpoint, model, old_stats, data, training={})
    loaded = NativePolicy(checkpoint)
    assert loaded.manifest["stats"][key] == old_stats[key]
    pixels = torch.ones(1, 3, 32, 32)
    actual = loaded.processor({key: pixels})[key]
    torch.testing.assert_close(actual, Processor(old_stats)({key: pixels})[key])
    assert not torch.allclose(actual, Processor(act_statistics(data))({key: pixels})[key])


@pytest.mark.parametrize("image_size", [None, [240, 320]])
def test_trainer_image_size_default_and_explicit_override(recording, monkeypatch, image_size):
    module = importlib.import_module("alohamini.learning.train")
    expected = DEFAULT_IMAGE_SIZE if image_size is None else tuple(image_size)

    class StopBeforeTraining(Exception):
        pass

    def load_samples(**kwargs):
        assert kwargs["image_size"] == expected
        raise StopBeforeTraining

    monkeypatch.setattr(module, "AlohaMiniDataset", load_samples)
    settings = {
        "dataset": str(recording),
        "train_episodes": [0],
        "val_episodes": [1],
        "device": "cpu",
    }
    if image_size is not None:
        settings["image_size"] = image_size
    with pytest.raises(StopBeforeTraining):
        module.train(settings)


def test_feedback_selection_masks_and_units(recording):
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_velocity,joint_current",
        image_size=(32, 32),
        review_note="Synthetic fixture",
    )
    assert data[0]["observation.state"].shape == (28,)
    assert data[0]["observation.state"][14:].tolist() == [1.0] * 14
    data.rows[0]["motor_feedback.current_ma_valid"][0] = 0
    with pytest.raises(ValueError, match="Unavailable"):
        data[0]


@pytest.mark.parametrize("offset", [0.04, -1 / 30])
def test_camera_gap_or_repeat_does_not_split_windows(recording, offset):
    path = recording / "episodes/episode_000000/safety.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for row in records:
        if row.get("frame_index") in (2, 3):
            row["host_timing"]["camera_capture_monotonic_s"]["forward"] += offset
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    data = samples(recording, episodes=[0])
    assert data[1]["action_is_pad"].tolist() == [False, False, False]
    assert data[1]["action"][:, 0].tolist() == [1, 2, 3]
    assert data.report["warnings"] > 0


def test_gripper_open_close_transitions_remain_in_one_window(recording):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = recording / "episodes/episode_000000/frames.parquet"
    table = pq.read_table(path)
    actions = table["action"].to_pylist()
    info = json.loads((recording / "meta/info.json").read_text())
    gripper = info["features"]["action"]["names"].index("arm_left_gripper.pos")
    for action, value in zip(actions, [100.0, 0.0, 0.0, 100.0], strict=True):
        action[gripper] = value
    table = table.set_column(
        table.column_names.index("action"),
        table.schema.field("action"),
        pa.array(actions, type=table.schema.field("action").type),
    )
    pq.write_table(table, path)
    safety_path = path.with_name("safety.jsonl")
    records = [json.loads(line) for line in safety_path.read_text().splitlines()]
    for record in records:
        if record.get("frame_index") is not None:
            record["requested_action"]["arm_left_gripper.pos"] = actions[record["frame_index"]][
                gripper
            ]
    safety_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    data = AlohaMiniDataset(recording, episodes=[0], cameras=[], chunk_size=4)
    assert data[0]["action"][:, gripper].tolist() == [100.0, 0.0, 0.0, 100.0]
    assert data[0]["action_is_pad"].tolist() == [False] * 4


def test_default_sample_is_a_single_row(recording):
    data = AlohaMiniDataset(recording, episodes=[0], state="none", image_size=(32, 32))
    assert data[0]["action"].shape == (18,)
    assert data[0]["observation.images.forward"].shape == (3, 32, 32)
    assert not any(k.endswith("_is_pad") for k in data[0])


def test_independent_history_future_and_sparse_windows(recording):
    from torch.utils.data import DataLoader

    data = AlohaMiniDataset(
        recording,
        episodes=[0, 1],
        image_size=(32, 32),
        delta_indices={
            "observation.images.forward": [-1, 0],
            "observation.state": [-2, 0],
            "action": [-1, 0, 1, 3],
            "motor_feedback.sample_finished_s": [0, 2],
        },
    )
    first = data[0]
    assert first["action"][:, 0].tolist() == [0, 0, 1, 3]
    assert first["action_is_pad"].tolist() == [True, False, False, False]
    assert first["observation.state_is_pad"].tolist() == [True, False]
    assert first["observation.images.forward_is_pad"].tolist() == [True, False]
    assert first["motor_feedback.sample_finished_s"].shape[0] == 2
    assert data[3]["action"][:, 0].tolist() == [2, 3, 3, 3]
    assert data[3]["action_is_pad"].tolist() == [False, False, True, True]
    assert data[4]["action"][:, 0].tolist() == [10, 10, 11, 13]
    batch = next(iter(DataLoader(data, batch_size=2)))
    assert batch["observation.images.forward"].shape == (2, 2, 3, 32, 32)
    assert batch["observation.state"].shape == (2, 2, 18)
    assert batch["action"].shape == (2, 4, 18)
    assert data.statistics()["action"]["mean"] == [6.5] * 18


def test_state_only_windows_do_not_decode_images(recording, monkeypatch):
    module = importlib.import_module("alohamini.learning.data")
    data = AlohaMiniDataset(
        recording, episodes=[0], cameras=[], delta_indices={"observation.state": [-1, 0]}
    )

    def unexpected_decode(*args):
        raise AssertionError("Unselected camera decoded")

    monkeypatch.setattr(module, "image_rgb", unexpected_decode)
    assert data[0]["observation.state"].shape == (2, 18)
    assert "observation.images.forward" not in data[0]


def test_invalid_selected_state_raises_instead_of_filtering_windows(recording):
    path = recording / "episodes/episode_000000/safety.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for row in records:
        if row.get("frame_index") == 1:
            row["safety"]["feedback_valid"] = False
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="Episode 0, frame 1: observation.state.*feedback_valid"):
        AlohaMiniDataset(recording, episodes=[0], cameras=[], delta_indices={"action": [-1, 0, 1]})
    visual = AlohaMiniDataset(recording, episodes=[0], state="none", chunk_size=3)
    assert visual.sample_indices == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="Episode 0, frame 1"):
        AlohaMiniDataset(
            recording, episodes=[0], cameras=[], delta_indices={"observation.state": [-1, 0]}
        )
    assert len(AlohaMiniDataset(recording, episodes=[1], chunk_size=3)) == 4


def test_invalid_current_requires_explicit_data_choice_and_remains_inspectable(
    recording, monkeypatch
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = recording / "episodes/episode_000000/frames.parquet"
    table = pq.read_table(path)
    key = "motor_feedback.current_ma_valid"
    masks = table[key].to_pylist()
    masks[1][0] = 0.0
    table = table.set_column(
        table.column_names.index(key),
        table.schema.field(key),
        pa.array(masks, type=table.schema.field(key).type),
    )
    pq.write_table(table, path)
    kwargs = dict(root=recording, episodes=[0], cameras=[], chunk_size=3)
    positions = AlohaMiniDataset(**kwargs)
    assert positions.sample_indices == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="Episode 0, frame 1: observation.state.*current"):
        AlohaMiniDataset(**kwargs, state="joint_current")
    with pytest.raises(ValueError, match="Episode 0, frame 1: observation.motor_current_ma"):
        AlohaMiniDataset(
            recording,
            episodes=[0],
            cameras=[],
            state="none",
            delta_indices={"observation.motor_current_ma": [0, 1]},
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    notebook = json.loads(
        (Path(__file__).parents[1] / "examples/learning/local_act.ipynb").read_text()
    )
    cell = next(c for c in notebook["cells"] if c["id"] == "local-act-8")
    scope = {
        "AlohaMiniDataset": AlohaMiniDataset,
        "np": np,
        "plt": plt,
        "settings": {"dataset": str(recording), "train_episodes": [0], "image_size": [32, 32]},
    }
    exec("".join(cell["source"]), scope)
    assert scope["values"].shape == (4, 28)
    assert np.isnan(scope["values"][1]).all()
    assert np.isfinite(scope["values"][[0, 2, 3]]).all()
    plt.close("all")


@pytest.mark.parametrize(
    "event",
    [
        {"type": "sequence_boundary", "reason": "manual_reset"},
        {"type": "sequence_boundary"},
        {"type": "watchdog_recovered"},
        {"type": "response_timeout"},
        {"type": "capture_wait", "reason": "camera delay"},
    ],
)
def test_events_do_not_split_episode_windows(recording, event):
    path = recording / "episodes/episode_000000/safety.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    position = next(i for i, r in enumerate(records) if r.get("frame_index") == 2)
    records.insert(
        position,
        {
            "episode_index": 0,
            "frame_index": None,
            "event": event,
            "client_monotonic_s": 1100.05,
        },
    )
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    data = AlohaMiniDataset(recording, episodes=[0], chunk_size=3, state="none")
    assert data.segment_ends == [4] * 4
    assert data[1]["action_is_pad"].tolist() == [False, False, False]
    assert data[1]["action"][:, 0].tolist() == [1, 2, 3]


@pytest.mark.parametrize("flag", ["fault", "watchdog_active", "joint_holds"])
def test_protection_flags_preserve_all_targets_and_statistics(recording, flag):
    path = recording / "episodes/episode_000000/safety.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for r in records:
        if r.get("frame_index") == 1:
            r["safety"][flag] = {"arm_left_elbow_flex": {}} if flag == "joint_holds" else True
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    data = AlohaMiniDataset(recording, episodes=[0], cameras=[], chunk_size=3)
    assert data.sample_indices == [0, 1, 2, 3]
    assert data.excluded == 0
    assert data.segment_ends == [4] * 4
    assert data[0]["action_is_pad"].tolist() == [False, False, False]
    assert data[0]["action"][:, 0].tolist() == [0, 1, 2]
    assert data[3]["action_is_pad"].tolist() == [False, True, True]
    assert data.statistics()["action"]["mean"] == [1.5] * 18
    assert data.records[1]["safety"][flag]


def test_controller_identity_alone_is_not_a_stop_event(recording):
    path = recording / "episodes/episode_000000/safety.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for r in records:
        if r.get("frame_index") is not None:
            r["safety"]["control_owner"] = "leader_a" if r["frame_index"] < 2 else "leader_b"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    data = AlohaMiniDataset(recording, episodes=[0], cameras=[], chunk_size=3)
    assert data.segment_ends == [4] * 4
    assert data[1]["action_is_pad"].tolist() == [False, False, False]


def test_protection_event_between_frames_does_not_split_windows(recording):
    path = recording / "episodes/episode_000000/safety.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    position = next(i for i, r in enumerate(records) if r.get("frame_index") == 2)
    records.insert(
        position,
        {
            "episode_index": 0,
            "frame_index": None,
            "event": {"type": "response_recovered"},
            "safety": {"joint_holds": {"arm_left_elbow_flex": {}}},
            "client_monotonic_s": 1100.05,
        },
    )
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    data = AlohaMiniDataset(recording, episodes=[0], cameras=[], chunk_size=3)
    assert data.segment_ends == [4] * 4
    assert data[1]["action_is_pad"].tolist() == [False, False, False]
    assert path.read_text() == "".join(json.dumps(r) + "\n" for r in records)


@pytest.mark.parametrize(
    "windows",
    [
        {"action": []},
        {"action": [0.5]},
        {"action": [True]},
        {"missing": [0]},
        {"observation.state": [0]},
        {"action": 3},
    ],
)
def test_invalid_window_specifications(recording, windows):
    with pytest.raises(ValueError):
        AlohaMiniDataset(recording, episodes=[0], state="none", delta_indices=windows)


def test_chunk_shorthand_matches_explicit_offsets(recording):
    old_call = samples(recording, episodes=[0])
    explicit = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="none",
        image_size=(32, 32),
        delta_indices={"action": range(3)},
    )
    for i in range(4):
        for key in old_call[i]:
            torch.testing.assert_close(old_call[i][key], explicit[i][key])
    with pytest.raises(ValueError, match="either"):
        AlohaMiniDataset(recording, episodes=[0], chunk_size=3, delta_indices={"action": [0]})


def test_recorded_reward_and_terminal_windows(recording, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from alohamini.datasets.lerobotv3 import export_lerobot

    # Additional learning labels belong to a processed dataset, not the fixed
    # native hardware recording schema. V3 permits these declared numeric fields.
    root = tmp_path / "reward_v3"
    export_lerobot(recording, root)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    for key, dtype in (("next.reward", "float32"), ("next.done", "bool")):
        info["features"][key] = {"dtype": dtype, "shape": [1], "names": [key]}
    info_path.write_text(json.dumps(info))
    for path in (root / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        indices = table["frame_index"].to_pylist()
        table = table.append_column("next.reward", pa.array([[float(i == 3)] for i in indices]))
        table = table.append_column("next.done", pa.array([[i == 3] for i in indices]))
        pq.write_table(table, path)
    data = AlohaMiniDataset(
        root,
        episodes=[0],
        cameras=[],
        delta_indices={"next.reward": [0, 1], "next.done": [0, 1]},
    )
    assert data[2]["next.reward"].tolist() == [[0.0], [1.0]]
    assert data[2]["next.done"].dtype == torch.bool
    assert data[3]["next.reward_is_pad"].tolist() == [False, True]
    assert data.statistics()["next.reward"]["mean"] == [0.25]


def test_processor_preserves_all_window_masks():
    processor = Processor({"observation.state": {"mean": [1.0], "std": [2.0]}})
    batch = {
        "observation.state": torch.tensor([[[3.0], [5.0]]]),
        "observation.state_is_pad": torch.tensor([[True, False]]),
    }
    result = processor(batch)
    torch.testing.assert_close(result["observation.state"], torch.tensor([[[1.0], [2.0]]]))
    assert result["observation.state_is_pad"].dtype == torch.bool
    assert result["observation.state_is_pad"].tolist() == [[True, False]]


@pytest.mark.parametrize(
    "key", ["control_epoch", "host_session_id", "watchdog_events", "joint_hold_events"]
)
def test_control_changes_and_gripper_holds_do_not_split_windows(recording, key):
    path = recording / "episodes/episode_000000/safety.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for row in records:
        if row.get("frame_index") is not None:
            row["safety"][key] = (
                str(int(row["frame_index"] >= 2))
                if key == "host_session_id"
                else int(row["frame_index"] >= 2)
            )
            row["safety"]["gripper_holds"] = {"arm_left_gripper": {"current_ma": 500}}
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    data = samples(recording, episodes=[0])
    assert len(data) == 4
    assert data[1]["action_is_pad"].tolist() == [False, False, False]


@pytest.mark.parametrize("kind", ["act", "am_act"])
def test_checkpoint_round_trip_and_offline_eval(recording, tmp_path, kind):
    data = samples(recording, episodes=[0])
    stats = data.statistics()
    options = model_options(state=False)
    if kind == "am_act":
        options.update(
            discrete_action_dims=[14],
            discrete_action_values=[[0.0, 1.0, 2.0, 3.0]],
            discrete_action_class_weights=[[1.0, 2.0, 2.0, 1.0]],
        )
    model = make_policy(kind, options, stats)
    checkpoint = tmp_path / "policy"
    save_checkpoint(checkpoint, model, stats, data, training={"train_episodes": [0]})
    manifest = json.loads((checkpoint / "policy.json").read_text())
    assert manifest["sample_boundary"] == "episode"
    assert "sample_boundaries" not in manifest
    loaded = NativePolicy(checkpoint)
    raw = data[0]
    original = model.predict_action_chunk(Processor(stats)({k: v[None] for k, v in raw.items()}))
    torch.testing.assert_close(loaded.predict(raw), Processor(stats).action(original)[0])
    metrics = offline_evaluate(loaded, data)
    assert len(metrics["mae_by_action"]) == 18
    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, model, stats, data, training={})


@pytest.mark.parametrize("kind", ["act", "am_act", "smolvla", "diffusion", "fastwam", "pi05"])
def test_policy_statistics_reuse_without_building_model(recording, tmp_path, monkeypatch, kind):
    from alohamini.learning.statistics import training_statistics, write_statistics
    from alohamini.policies.registry import algorithm

    data = AlohaMiniDataset(
        recording, episodes=[0], chunk_size=3, image_size=(32, 32), state=StateSelection.DEFAULT
    )
    components = algorithm(kind)
    options = dict(input_features=data.input_features, output_features=data.output_features)
    artifact = training_statistics(components, data, options, kind)
    path = write_statistics(tmp_path / "stats.json", artifact)
    monkeypatch.setattr(components, "statistics", lambda *a: pytest.fail("Unexpected recompute"))
    assert training_statistics(components, data, options, kind, path) == artifact
    original_hashes = data.table_sha256
    data.table_sha256 = {**original_hashes, "changed": "changed"}
    with pytest.raises(ValueError, match="do not match"):
        training_statistics(components, data, options, kind, path)
    data.table_sha256 = original_hashes
    data.episodes = [1]
    with pytest.raises(ValueError, match="do not match"):
        training_statistics(components, data, options, kind, path)
    with pytest.raises(FileExistsError):
        write_statistics(path, artifact)


def test_statistics_cli_and_training_selection_match(recording, tmp_path, monkeypatch):
    from alohamini.cli import main
    from alohamini.learning.statistics import sample_arguments, training_statistics
    from alohamini.policies.registry import algorithm

    cfg = dict(
        policy="act",
        state="none",
        dataset=str(recording),
        val_episodes=[1],
        image_size=[32, 32],
        model={"chunk_size": 3, "pretrained_backbone_weights": None},
    )
    config = tmp_path / "train.json"
    config.write_text(json.dumps(cfg))
    path = tmp_path / "policy-stats.json"
    assert (
        main(["dataset", "stats", str(recording), "--config", str(config), "--output", str(path)])
        == 0
    )
    components = algorithm("act")
    options = components.options(cfg, "cpu")
    args, episodes, validation = sample_arguments(cfg, components, options)
    assert episodes == [0] and validation == [1]
    data = AlohaMiniDataset(**args, episodes=episodes)
    options.update(input_features=data.input_features, output_features=data.output_features)
    artifact = training_statistics(components, data, options, "act", path)
    assert artifact["stats"]["action"]["mean"] == [1.5] * 18
    options["chunk_size"] = 4
    with pytest.raises(ValueError, match="do not match"):
        training_statistics(components, data, options, "act", path)
    path.write_text(json.dumps({"action": {"mean": [1.0]}}))
    with pytest.raises(ValueError, match="policy statistics"):
        training_statistics(components, data, options, "act", path)


def test_basic_statistics_report_does_not_modify_dataset(recording, tmp_path):
    import hashlib

    from alohamini.learning.statistics import prepare_statistics

    def hashes():
        return {
            str(p.relative_to(recording)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in recording.rglob("*")
            if p.is_file()
        }

    before = hashes()
    path = prepare_statistics(recording, tmp_path / "basic.json")
    artifact = json.loads(path.read_text())
    assert artifact["format"] == "alohamini-dataset-statistics"
    assert artifact["stats"]["action"]["mean"] == [6.5] * 18
    assert artifact["stats"]["action"]["count"] == [8] * 18
    assert artifact["provenance"]["source_sha256"]
    assert hashes() == before


def test_real_hardware_requires_explicit_enable():
    with pytest.raises(ValueError, match="enable_robot"):
        evaluate_robot("missing-checkpoint")


@pytest.mark.parametrize("kind", ["act", "am_act"])
@pytest.mark.parametrize("ensemble", [None, 0.01])
def test_evaluation_advances_chunks_and_ensemble_with_new_images(
    recording, tmp_path, monkeypatch, kind, ensemble
):
    """Exercise checkpoint, RGB processor, real policy queues and evaluator without motors."""
    from copy import deepcopy
    from unittest.mock import Mock

    from test_evaluation import EvaluationClient
    from test_replay import Clock, replay_snapshot

    from alohamini.apps.evaluation import run_evaluation

    data = samples(recording, episodes=[0])
    # The lightweight training fixture omits hardware identities; deployment needs them.
    data.info["robot_metadata"] = deepcopy(replay_snapshot().payload["_robot_metadata"])
    stats = act_statistics(data)
    options = model_options(state=False)
    options.update(
        n_action_steps=1 if ensemble is not None else 2, temporal_ensemble_coeff=ensemble
    )
    checkpoint = tmp_path / "policy"
    save_checkpoint(checkpoint, make_policy(kind, options, stats), stats, data, training={})
    policy = NativePolicy(checkpoint)
    reset = Mock(wraps=policy.reset)
    monkeypatch.setattr(policy, "reset", reset)
    seen = []

    def predict(batch):
        # Stub only the network: keep ACT/AM-ACT's actual queue and temporal ensemble.
        pixels = batch["observation.images.forward"]
        seen.append(pixels.clone())
        result = torch.zeros(1, 3, 18)
        result[0, :, 0] = pixels.mean() + torch.arange(3.0)
        return result

    monkeypatch.setattr(policy.model, "predict_action_chunk", predict)
    clock = Clock()
    client = EvaluationClient(clock)
    client.state.payload["_robot_metadata"] = deepcopy(policy.robot_metadata)

    def update_images(current):
        current.state.images["forward"] = jpeg((int(clock.now * 200), 20, 30))
        current.state.payload["_host_timing"] = {
            "state_sample_finished_monotonic_s": 10 + clock.now,
            "camera_capture_monotonic_s": {"forward": 9.6 + clock.now},
        }

    client.on_read = update_images
    send = client.send_command
    attempts = 0

    def intermittent(action, *, based_on):
        nonlocal attempts
        attempts += 1
        return None if attempts == 2 else send(action, based_on=based_on)

    client.send_command = intermittent
    monkeypatch.setattr("alohamini.apps.evaluation.time.monotonic", lambda: clock.now)
    monkeypatch.setattr("alohamini.apps.evaluation.time.sleep", clock.sleep)
    monkeypatch.setattr("alohamini.apps.evaluation.stop_owned_robot", lambda *args: None)
    count = run_evaluation(client, policy, "alohamini2pro", duration_s=0.3)
    assert count >= 6
    reset.assert_called_once()
    assert len(seen) == (attempts if ensemble is not None else (attempts + 1) // 2)
    assert not torch.equal(seen[0], seen[-1])
    assert len({entry[1]["arm_left_shoulder_pan.pos"] for entry in client.sent}) > 2


@pytest.mark.parametrize("kind", ["act", "am_act"])
def test_training_defaults_restore_imagenet_initialization(kind):
    prefix = "ACT" if kind == "act" else "AMACT"
    cls = getattr(
        importlib.import_module(f"alohamini.policies.{kind}.configuration_{kind}"),
        f"{prefix}Config",
    )
    options = model_options()
    del options["pretrained_backbone_weights"]
    config = cls(**options)
    assert config.pretrained_backbone_weights == "ResNet18_Weights.IMAGENET1K_V1"
    assert cls(**options, pretrained_backbone_weights=None).pretrained_backbone_weights is None
    settings = json.loads((Path(__file__).parents[1] / "examples/learning/act.json").read_text())
    assert "n_action_steps" not in settings["model"]
    assert settings["model"]["pretrained_backbone_weights"] == config.pretrained_backbone_weights


def test_backbone_uses_workspace_cache_and_never_silently_falls_back(tmp_path, monkeypatch):
    from alohamini.policies.backbone import make_resnet

    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path))
    config = SimpleNamespace(
        vision_backbone="resnet18",
        replace_final_stride_with_dilation=False,
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
    )
    backbone = torch.nn.Linear(2, 1)
    saved = {k: torch.full_like(v, 0.5) for k, v in backbone.state_dict().items()}
    monkeypatch.setattr("torchvision.models.resnet18", lambda **kwargs: backbone)
    calls = []

    def load(url, **options):
        assert url == "https://download.pytorch.org/models/resnet18-f37072fd.pth"
        assert options["model_dir"] == str(tmp_path / "pretrained")
        assert options["check_hash"] and options["weights_only"]
        calls.append(url)
        return saved

    monkeypatch.setattr("torch.hub.load_state_dict_from_url", load)
    make_resnet(config)
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(value, saved[name])
    config.pretrained_backbone_weights = None
    make_resnet(config)
    assert len(calls) == 1

    def failed(*args, **kwargs):
        raise OSError("download failed")

    monkeypatch.setattr("torch.hub.load_state_dict_from_url", failed)
    config.pretrained_backbone_weights = "ResNet18_Weights.IMAGENET1K_V1"
    with pytest.raises(OSError, match="download failed"):
        make_resnet(config)


@pytest.mark.parametrize("kind", ["act", "am_act"])
@pytest.mark.parametrize(
    "overrides,steps,coefficient",
    [
        ({}, 2, None),
        ({"n_action_steps": 1, "temporal_ensemble_coeff": 0.01}, 1, 0.01),
        ({"n_action_steps": 3, "temporal_ensemble_coeff": None}, 3, None),
    ],
)
def test_evaluate_robot_overrides_execution_without_changing_weights(
    recording,
    tmp_path,
    monkeypatch,
    kind,
    overrides,
    steps,
    coefficient,
):
    data = samples(recording, episodes=[0])
    model = make_policy(kind, model_options(state=False), data.statistics())
    checkpoint = tmp_path / "native"
    save_checkpoint(checkpoint, model, data.statistics(), data, training={})
    # Loading a trained checkpoint must not reinitialize from ImageNet or download.
    manifest_path = checkpoint / "policy.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["pretrained_backbone_weights"] = "ResNet18_Weights.IMAGENET1K_V1"
    from alohamini.learning.checkpoint import write_checkpoint_metadata

    write_checkpoint_metadata(checkpoint, manifest)
    before = manifest_path.read_bytes()

    def no_download(*args, **kwargs):
        pytest.fail("Checkpoint inference must not download backbone weights")

    monkeypatch.setattr("torchvision.models.ResNet18_Weights.get_state_dict", no_download)
    monkeypatch.setattr("torch.hub.load_state_dict_from_url", no_download)

    def evaluate(host, robot_model, **options):
        policy = options["policy_factory"]()
        assert host == "127.0.0.1" and robot_model == "alohamini2pro"
        assert options["fps"] == data.info["fps"]
        assert options["episode_time_s"] == 2
        assert "n_action_steps" not in options and "temporal_ensemble_coeff" not in options
        assert policy.config.n_action_steps == steps
        assert policy.config.temporal_ensemble_coeff == coefficient
        assert policy.config.pretrained_backbone_weights is None
        for name, value in model.state_dict().items():
            torch.testing.assert_close(policy.model.state_dict()[name], value, rtol=0, atol=0)
        return "evaluated"

    monkeypatch.setattr("alohamini.apps.evaluation.evaluate", evaluate)
    assert (
        evaluate_robot(
            checkpoint,
            enable_robot=True,
            host="127.0.0.1",
            device="cpu",
            episode_time_s=2,
            **overrides,
        )
        == "evaluated"
    )
    assert manifest_path.read_bytes() == before
    for invalid in ({"n_action_steps": 4}, {"temporal_ensemble_coeff": 0.01}):
        with pytest.raises((ValueError, NotImplementedError)):
            evaluate_robot(checkpoint, enable_robot=True, host="127.0.0.1", device="cpu", **invalid)


def test_training_rejects_split_leak_before_creating_run(recording):
    with pytest.raises(ValueError, match="disjoint"):
        train(
            {
                "dataset": str(recording),
                "train_episodes": [0],
                "val_episodes": [0],
                "device": "cpu",
                "run_name": "unused",
            }
        )


def test_normalization_and_physical_scaling():
    proc = Processor({"action": {"mean": [2.0, 10.0], "std": [4.0, 0.0]}})
    raw = torch.tensor([[6.0, 10.0]])
    normalized = proc({"action": raw})["action"]
    torch.testing.assert_close(normalized, torch.tensor([[1.0, 0.0]]))
    physical = proc.action(normalized)
    torch.testing.assert_close(physical, raw)
    torch.testing.assert_close(scale_action(physical, [0], 0.5), torch.tensor([[3.0, 10.0]]))
    torch.testing.assert_close(physical, raw)


def test_shared_rgb_transform_preserves_resolution_and_channels(monkeypatch):
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 255, 128, 0

    def unexpected_resize(*args, **kwargs):
        pytest.fail("Matching-resolution images must not be resized")

    monkeypatch.setattr("alohamini.learning.processor.F.interpolate", unexpected_resize)
    actual = image_tensor(rgb)
    assert actual.shape == (3, 480, 640)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual[:, 0, 0], torch.tensor([1.0, 128 / 255, 0.0]))


def test_processor_modes_preserve_labels_masks_and_inverse_gradients():
    config = SimpleNamespace(
        input_features={"observation.state": SimpleNamespace(type="STATE")},
        output_features={"action": SimpleNamespace(type="ACTION")},
        normalization_mapping={"STATE": "IDENTITY", "ACTION": "MEAN_STD"},
    )
    proc = Processor.from_config(config, {"action": {"mean": [1.0, 2.0], "std": [2.0, 3.0]}})
    raw = {
        "observation.state": torch.tensor([[4.0, 5.0]]),
        "action": torch.tensor([[[3.0, 5.0]]]),
        "action_is_pad": torch.tensor([[False]]),
        "next.done": torch.tensor([True]),
        "class_label": torch.tensor([2]),
    }
    actual = proc(raw)
    for key in raw.keys() - {"action"}:
        torch.testing.assert_close(actual[key], raw[key])
    torch.testing.assert_close(actual["action"], torch.ones(1, 1, 2))
    prediction = torch.zeros(1, 3, 2, requires_grad=True)
    proc.action(prediction).sum().backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([2.0, 3.0]).expand_as(prediction))
    state = raw["observation.state"]
    assert proc.unnormalize("observation.state", state) is state


@pytest.mark.parametrize(
    "stats",
    [{}, {"action": {"mean": [0.0]}}, {"action": {"mean": [0.0], "std": [-1.0]}}],
)
def test_processor_rejects_missing_or_invalid_selected_statistics(stats):
    with pytest.raises(ValueError, match="action:"):
        Processor(stats, modes={"action": "MEAN_STD"})


@pytest.mark.parametrize("kind", ["act", "am_act"])
def test_checkpoint_restores_processor_modes_and_execution_scaling(recording, tmp_path, kind):
    data = samples(recording, episodes=[0])
    stats = data.statistics()
    options = model_options(
        state=False, normalization_mapping={"VISUAL": "IDENTITY", "ACTION": "IDENTITY"}
    )
    if kind == "am_act":
        options.update(inference_action_scale_dims=[0], inference_action_scale=0.5)
    model = make_policy(kind, options, stats).eval()
    checkpoint = tmp_path / kind
    save_checkpoint(checkpoint, model, stats, data, training={})
    loaded = NativePolicy(checkpoint)
    raw = data.observation(0)
    predicted = model.predict_action_chunk({k: v[None] for k, v in raw.items()})[0]
    torch.testing.assert_close(loaded.processor.action(predicted), predicted)
    expected = scale_action(predicted, [0], 0.5) if kind == "am_act" else predicted
    torch.testing.assert_close(loaded.predict(raw), expected)
    total, count = torch.zeros(18, dtype=torch.float64), 0
    for item in data:
        mask = ~item["action_is_pad"]
        total += ((loaded.predict(item) - item["action"]).abs() * mask[:, None]).sum(0)
        count += int(mask.sum())
    result = offline_evaluate(loaded, data)
    np.testing.assert_allclose(list(result["mae_by_action"].values()), total / count, rtol=1e-5)


def test_am_act_discrete_centers_reject_incompatible_normalization():
    with pytest.raises(ValueError, match="centers currently require"):
        make_policy(
            "am_act",
            model_options(
                normalization_mapping={
                    "VISUAL": "MEAN_STD",
                    "STATE": "MEAN_STD",
                    "ACTION": "IDENTITY",
                },
                discrete_action_dims=[14],
                discrete_action_values=[[-1.0, 0.0, 1.0]],
            ),
        )


def test_native_cli_checkpoint_uses_existing_evaluator(recording, tmp_path, monkeypatch):
    from alohamini.cli import main

    data = samples(recording, episodes=[0])
    stats = data.statistics()
    model = make_policy("act", model_options(state=False))
    checkpoint = tmp_path / "native"
    save_checkpoint(checkpoint, model, stats, data, training={})
    called = []

    def evaluate(**options):
        policy = options["policy_factory"]()
        assert isinstance(policy, NativePolicy)
        assert policy.config.n_action_steps == 1
        assert policy.config.temporal_ensemble_coeff == 0.01
        called.append(options["host"])

    monkeypatch.setattr("alohamini.apps.evaluation.evaluate", evaluate)
    assert (
        main(
            [
                "evaluate",
                "--host",
                "127.0.0.1",
                "--robot_model",
                "alohamini2pro",
                "--policy.path",
                str(checkpoint),
                "--device",
                "cpu",
                "--policy.n_action_steps",
                "1",
                "--policy.temporal_ensemble_coeff",
                "0.01",
            ]
        )
        == 0
    )
    assert called == ["127.0.0.1"]


def test_checkpoint_contract_mismatch_fails_before_robot(recording, tmp_path):
    data = samples(recording, episodes=[0])
    model = make_policy("act", model_options(state=False))
    checkpoint = tmp_path / "native"
    save_checkpoint(checkpoint, model, data.statistics(), data, training={})
    path = checkpoint / "policy.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["input_features"]["observation.images.forward"]["shape"] = [3, 64, 64]
    from alohamini.learning.checkpoint import write_checkpoint_metadata

    write_checkpoint_metadata(checkpoint, manifest)
    with pytest.raises(ValueError, match="contract"):
        NativePolicy(checkpoint)


@pytest.mark.parametrize("kind,storage", [("act", "native"), ("am_act", "visual_v3")])
def test_detached_trainer_with_held_out_episode(recording, tmp_path, monkeypatch, kind, storage):
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    processes = []
    original_popen = subprocess.Popen

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    if storage == "visual_v3":
        from alohamini.datasets.lerobotv3 import export_lerobot

        full = tmp_path / "v3"
        visual = tmp_path / "visual"
        export_lerobot(recording, full)
        export_lerobot(full, visual, vision_only=True)
        recording = visual
    settings = {
        "dataset": str(recording),
        "run_name": "integration",
        "policy": kind,
        "device": "cpu",
        "num_workers": 0,
        "train_episodes": [0],
        "val_episodes": [1],
        "state": "none",
        "image_size": [32, 32],
        "steps": 1,
        "batch_size": 2,
        "review_note": "Synthetic integration fixture",
        "model": model_options(state=False),
    }
    if storage == "visual_v3":
        from alohamini.learning.statistics import prepare_statistics

        config_path = tmp_path / "prepare.json"
        config_path.write_text(json.dumps(settings))
        settings["stats"] = str(
            prepare_statistics(recording, tmp_path / "prepared.json", config=config_path)
        )
    job = launch_training(settings)
    try:
        result = processes[0].wait(timeout=45)
    finally:
        if processes[0].poll() is None:
            processes[0].terminate()
            processes[0].wait(timeout=5)
    from pathlib import Path

    assert result == 0, Path(job["log"]).read_text()
    run = Path(job["checkpoint"]).parent
    assert json.loads((run / "offline-evaluation.json").read_text())["valid_action_steps"] > 0
    manifest = json.loads((run / "checkpoint/policy.json").read_text())
    artifact = json.loads((run / "statistics.json").read_text())
    assert manifest["stats"] == artifact["stats"]
    if settings.get("stats"):
        assert artifact == json.loads(Path(settings["stats"]).read_text())
    assert manifest["training"]["train_episodes"] == [0]
    assert manifest["training"]["val_episodes"] == [1]
    assert manifest["stats"]["action"]["mean"] == [1.5] * 18
    for camera in manifest["cameras"]:
        image_stats = manifest["stats"][f"observation.images.{camera}"]
        assert image_stats["mean"] == [[[v]] for v in IMAGENET_MEAN]
        assert image_stats["std"] == [[[v]] for v in IMAGENET_STD]
    assert manifest["kind"] == kind
    assert "observation.state" not in manifest["config"]["input_features"]
    assert Path(job["pid_file"]).read_text().strip() == str(job["pid"])


def test_v3_samples_preserve_native_chunks_images_and_statistics(recording, tmp_path):
    from alohamini.datasets.lerobotv3 import export_lerobot

    full, visual = tmp_path / "v3", tmp_path / "visual"
    export_lerobot(recording, full)
    export_lerobot(full, visual, vision_only=True)
    native = samples(recording, episodes=[0, 1])
    converted = samples(visual, episodes=[0, 1])
    assert "SAFETY_CAPTURE_CLOCK_MISSING" not in {
        issue["code"] for issue in converted.report["issues"]
    }
    assert native.locations == converted.locations
    assert native.segment_ends == converted.segment_ends
    assert native.input_features == converted.input_features
    assert native.output_features == converted.output_features
    for i in range(len(native)):
        for key in native[i]:
            torch.testing.assert_close(native[i][key], converted[i][key], rtol=0, atol=0)
    assert native.statistics() == converted.statistics()
    # Images remain disk references, not a dataset-sized bytes list in RAM.
    assert isinstance(converted.rows[0]["observation.images.forward"], tuple)
    with pytest.raises(ValueError, match="no original state"):
        AlohaMiniDataset(visual, episodes=[0])
    original_state = AlohaMiniDataset(full, episodes=[0], image_size=(32, 32))
    assert original_state.input_features["observation.state"].shape == (18,)


def test_v3_training_rejects_missing_sidecar_and_changed_action_contract(recording, tmp_path):
    from alohamini.datasets.lerobotv3 import export_lerobot

    full = tmp_path / "v3"
    export_lerobot(recording, full)
    safety = full / "meta/safety/episode_000000.jsonl"
    safety.rename(safety.with_suffix(".saved"))
    with pytest.raises(ValueError, match="sidecar"):
        samples(full, episodes=[0])
    safety.with_suffix(".saved").rename(safety)
    path = full / "meta/alohamini.json"
    info = json.loads(path.read_text())
    info["source_info"]["features"]["action"]["names"][0] = "wrong_axis"
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="coordinates"):
        samples(full, episodes=[0])
