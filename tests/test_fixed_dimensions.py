"""Task-specific fixed axes: metadata, physical means and simulated preparation."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from test_native_learning import model_options
from test_native_learning import recording as recording
from test_recording_storage import add_episode, metadata, rows
from test_replay import Client, Clock

from alohamini.datasets.record import LocalDataset, StateSelection, state_names
from alohamini.fixed import (
    FixedGuard,
    capture,
    dataset_info,
    dimensions,
    recording_config,
    restore,
    validate,
)
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.fixed import configure, dataset_summary, summarize
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.registry import algorithm

MODEL = "alohamini2pro"
JOINT = "arm_right_shoulder_pan.pos"


def config(**targets):
    return dict(version=1, targets=targets, source="test")


def test_named_dimensions_and_strict_saved_targets():
    selected = dimensions(["arm_right", "lift_axis", JOINT], MODEL)
    assert len(selected) == 8 and JOINT in selected
    assert dimensions("base", MODEL) == ["x.vel", "y.vel", "theta.vel"]
    for invalid in (config(arm_right=0), config(**{"x.vel": 1}), config(**{JOINT: float("nan")})):
        with pytest.raises(ValueError):
            validate(invalid, MODEL)
    with pytest.raises(ValueError):
        dimensions(["camera"], MODEL)


def test_record_capture_resume_and_v3_preserve_raw_feedback(tmp_path):
    client = Client(Clock())
    client.state.payload[JOINT] = 17.0
    client.state.payload["x.vel"] = 0.1
    fixed = capture(client.state, dimensions([JOINT, "base"], MODEL))
    assert fixed["targets"]["x.vel"] == 0
    assert fixed["targets"][JOINT] == 17
    root = tmp_path / "fixed"
    with_dataset = LocalDataset(
        root, fps=30, task="pick", robot_metadata=metadata(), fixed_dimensions=fixed
    )
    add_episode(with_dataset)
    with_dataset.save_episode()
    with_dataset.close()
    info = dataset_info(root)
    assert info["fixed_dimensions"] == fixed
    assert len(rows(root)[0]["observation.state"]) == 18
    assert recording_config(root, None, MODEL, resume=True) == (fixed, list(fixed["targets"]))
    with pytest.raises(ValueError, match="retain"):
        recording_config(root, ["arm_left"], MODEL, resume=True)
    resumed = LocalDataset(
        root, fps=30, task="pick", robot_metadata=metadata(), resume=True, fixed_dimensions=fixed
    )
    assert resumed.num_episodes == 1
    resumed.close()
    assert len(StateSelection(info).feature["names"]) == 18
    selected = StateSelection(info, exclude_fixed=True)
    assert len(selected.feature["names"]) == 14
    assert JOINT not in selected.feature["names"]
    learning = AlohaMiniDataset(root, episodes=[0], chunk_size=1, image_size=(32, 32))
    assert learning.info["fixed_dimensions"] == fixed
    assert JOINT not in learning.selection.feature["names"]


@pytest.fixture
def hardware(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("time.monotonic", lambda: clock.now)
    monkeypatch.setattr("time.sleep", clock.sleep)
    client = Client(clock)
    stop = Mock()
    monkeypatch.setattr("alohamini.apps.teleoperation.stop_owned_robot", stop)
    return clock, client, stop


def test_restore_ramps_holds_other_axes_and_waits_for_measured_arrival(hardware):
    clock, client, stop = hardware
    baseline = copy.deepcopy(client.state.payload)

    def follow(c):
        if c.sent:
            c.state.payload.update(c.sent[-1][1])

    client.on_read = follow
    lease = restore(client, MODEL, config(**{JOINT: 10, "lift_axis.height_mm": 110}))
    assert lease == client.sent[-1][2]
    assert clock.now >= 1.5  # 1 s ramp plus 0.5 s settled feedback.
    for (_, previous, _), (_, action, _) in zip(client.sent, client.sent[1:], strict=False):
        assert action[JOINT] >= previous[JOINT]
        assert action[JOINT] - previous[JOINT] <= 10 * 0.022 + 1e-6
        assert action["arm_left_shoulder_pan.pos"] == baseline["arm_left_shoulder_pan.pos"]
        assert action["x.vel"] == 0
    assert client.sent[-1][1][JOINT] == 10
    stop.assert_not_called()


@pytest.mark.parametrize("failure", ["stale", "context", "metadata", "timeout", "cancel", "range"])
def test_restore_failures_stop_without_continuing(hardware, failure):
    clock, client, stop = hardware
    opts = {"timeout_s": 0.2}
    fixed = config(**{JOINT: 10})
    if failure == "stale":
        read = client.read

        def stale():
            result = read()
            result.request_started_s -= 2
            return result

        client.read = stale
    elif failure == "context":

        def change(c):
            if c.sent:
                c.state.payload["_safety"]["control_epoch"] += 1

        client.on_read = change
    elif failure == "metadata":
        opts["expected_metadata"] = {}
    elif failure == "cancel":
        opts["cancelled"] = lambda: True
    elif failure == "range":
        fixed = config(**{JOINT: 10000})
    with pytest.raises((RuntimeError, ValueError, TimeoutError, InterruptedError)):
        restore(client, MODEL, fixed, **opts)
    stop.assert_called_once()
    if failure in ("stale", "metadata", "cancel", "range"):
        assert not client.sent


def test_guard_uses_actual_feedback_without_rewriting_it(hardware):
    clock, client, _ = hardware
    guard = FixedGuard(config(**{JOINT: 10}), client.state.payload["_robot_metadata"])
    guard.check(client.state)
    clock.sleep(0.51)
    with pytest.raises(RuntimeError, match="no longer held"):
        guard.check(client.state)
    assert client.state.payload[JOINT] == 0


def test_new_checkpoint_inherits_fixed_and_contains_physical_summary(recording, tmp_path):
    info_path = recording / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["fixed_dimensions"] = config(**{JOINT: 5})
    info_path.write_text(json.dumps(info))
    samples = AlohaMiniDataset(
        recording, episodes=[0], chunk_size=3, image_size=(32, 32), state=StateSelection.DEFAULT
    )
    assert JOINT not in samples.selection.feature["names"]
    opts = model_options()
    opts.update(input_features=samples.input_features, output_features=samples.output_features)
    component = algorithm("act")
    stats = component.statistics(samples, opts)
    model = component.build(opts, stats)
    path = tmp_path / "checkpoint"
    save_checkpoint(path, model, stats, samples, training={"train_episodes": [0]})
    policy = NativePolicy(path, device="cpu", fixed_dimensions=["lift_axis"])
    assert policy.fixed_dimensions["targets"][JOINT] == 5
    summary = policy.manifest["physical_action_summary"]
    j = policy.names.index("lift_axis.height_mm")
    expected = np.mean([samples.rows[i]["action"][j] for i in samples._used_rows["action"]])
    assert summary["mean"][j] == pytest.approx(expected)
    assert policy.fixed_dimensions["targets"]["lift_axis.height_mm"] == expected
    assert JOINT not in policy.selection.feature["names"]
    actual = policy.execution_action(torch.zeros(1, 18))
    assert actual[0, policy.names.index(JOINT)] == 5
    assert actual[0, j] == expected


def test_legacy_override_separates_action_and_state_means_and_preserves_input(recording):
    samples = AlohaMiniDataset(
        recording, episodes=[0], chunk_size=3, image_size=(32, 32), state=StateSelection.DEFAULT
    )
    policy = NativePolicy.__new__(NativePolicy)
    policy.robot_metadata = samples.info["robot_metadata"]
    policy.names = state_names(MODEL)
    policy.selection = samples.selection
    policy.manifest = dict(
        source_info=samples.info,
        table_sha256=samples.table_sha256,
        training=dict(dataset=str(recording), train_episodes=[0]),
        stats={"observation.state": {"mean": list(range(18))}},
    )
    configure(policy, [JOINT, "base"])
    index = policy.names.index(JOINT)
    assert policy.fixed_dimensions["targets"][JOINT] == 1.5  # Episode 0 only.
    assert policy.fixed_state_references[index] == index
    assert policy.fixed_dimensions["targets"]["x.vel"] == 0
    raw = {"observation.state": torch.full((18,), 777.0)}
    adjusted = policy.fixed_input(raw)
    assert raw["observation.state"][index] == 777
    assert adjusted["observation.state"][index] == index
    assert adjusted["observation.state"][0] == 777
    with (recording / next(iter(samples.table_sha256))).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        dataset_summary(policy.manifest)


def test_embedded_summary_works_without_original_dataset():
    names = state_names(MODEL)
    summary = summarize(np.full((2, 18), 3.0), names)
    policy = SimpleNamespace(
        robot_metadata={"robot_model": MODEL},
        names=names,
        selection=None,
        manifest={"source_info": {}, "physical_action_summary": summary},
    )
    configure(policy, ["arm_left", "lift_axis", "base"])
    assert policy.fixed_dimensions["targets"]["lift_axis.height_mm"] == 3
    assert policy.fixed_dimensions["targets"]["x.vel"] == 0


def test_record_loop_overrides_leader_and_keyboard_after_preparation(hardware, monkeypatch):
    from alohamini.apps.recording import record_loop

    clock, client, _ = hardware
    monkeypatch.setattr("time.perf_counter", lambda: clock.now)
    monkeypatch.setattr("alohamini.apps.recording.stop_owned_robot", Mock())
    client.state.payload["_robot_metadata"]["cameras"] = []
    client.state.payload["_host_timing"] = {}
    client.set_recording_cameras = Mock()
    client.read_recording = lambda **kwargs: client.read()
    keyboard = Mock(events=dict(exit_early=False, rerecord_episode=False, stop_recording=False))
    keyboard.read.return_value = {"w"}
    leader = Mock()
    leader.read.return_value = {n: 20.0 for n in state_names(MODEL) if n.endswith(".pos")}
    record_loop(
        client,
        MODEL,
        leader,
        keyboard,
        fps=30,
        duration_s=0.1,
        metadata=client.state.payload["_robot_metadata"],
        fixed_dimensions=config(**{JOINT: 0, "x.vel": 0}),
    )
    assert leader.read.call_count > 0
    for _, action, _ in client.sent:
        assert action[JOINT] == 0
        assert action["x.vel"] == 0
    assert client.sent[-1][1]["arm_left_shoulder_pan.pos"] == 20
    assert client.state.payload["arm_left_shoulder_pan.pos"] == 0


def test_evaluator_prepares_before_policy_and_enforces_fixed_targets(hardware, monkeypatch):
    from test_evaluation import EvaluationClient

    from alohamini.apps.evaluation import run_evaluation

    clock, _, _ = hardware
    client = EvaluationClient(clock)
    monkeypatch.setattr("alohamini.apps.evaluation.stop_owned_robot", Mock())
    policy = Mock()
    policy.robot_metadata = copy.deepcopy(client.state.payload["_robot_metadata"])
    policy.fixed_dimensions = config(**{JOINT: 0})
    seen = []

    def select(snapshot):
        seen.append(clock.now)
        return {**{n: snapshot.payload[n] for n in state_names(MODEL)}, JOINT: 40}

    policy.select_action.side_effect = select
    assert run_evaluation(client, policy, MODEL, duration_s=0.1) >= 2
    assert seen[0] >= 0.5
    assert all(action[JOINT] == 0 for _, action, _ in client.sent)


def test_no_leader_connection_when_both_arms_fixed():
    from alohamini.hardware.leader import BimanualLeader

    with BimanualLeader(MODEL, active_sides=()) as leader:
        assert leader.read({}) == {}


def test_cli_fixed_options_reach_checkpoint(tmp_path, monkeypatch):
    from alohamini.cli import main

    path = tmp_path / "checkpoint"
    path.mkdir()
    (path / "policy.json").write_text("{}")
    constructor = Mock(return_value=SimpleNamespace(fps=30))
    monkeypatch.setattr("alohamini.learning.policy.NativePolicy", constructor)
    monkeypatch.setattr(
        "alohamini.apps.evaluation.evaluate", lambda **opts: opts["policy_factory"]()
    )
    assert (
        main(
            [
                "evaluate",
                "--host",
                "test-only",
                "--robot_model",
                MODEL,
                "--policy.path",
                str(path),
                "--fixed-dimensions",
                "arm_left",
                "lift_axis",
                "--fixed-dataset",
                str(tmp_path / "dataset"),
            ]
        )
        == 0
    )
    assert constructor.call_args.kwargs["fixed_dimensions"] == ["arm_left", "lift_axis"]
    assert constructor.call_args.kwargs["fixed_dataset"] == str(tmp_path / "dataset")
