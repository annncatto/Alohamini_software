"""Capture task-specific holds from live feedback without reading training data."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from test_evaluation import EvaluationClient
from test_replay import Clock

from alohamini.apps.evaluation import evaluate, run_evaluation
from alohamini.datasets.record import StateSelection, dataset_features, state_names
from alohamini.learning.fixed import configure, summarize
from alohamini.learning.policy import NativePolicy

MODEL = "alohamini2pro"
JOINT = "arm_right_shoulder_pan.pos"
LIFT = "lift_axis.height_mm"


@pytest.fixture
def rig(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("time.monotonic", lambda: clock.now)
    monkeypatch.setattr("time.sleep", clock.sleep)
    client = EvaluationClient(clock)
    client.state.payload[JOINT] = 27.0
    client.state.payload[LIFT] = 287.0
    client.state.payload["x.vel"] = 0.1
    policy = NativePolicy.__new__(NativePolicy)
    policy.robot_metadata = deepcopy(client.state.payload["_robot_metadata"])
    policy.names = state_names(MODEL)
    policy.source = dict(robot_metadata=policy.robot_metadata, features=dataset_features(MODEL))
    policy.selection = StateSelection(policy.source)
    policy.manifest = dict(
        source_info=policy.source,
        cameras=[],
        stats={"observation.state": {"mean": [4.0] * len(policy.names)}},
    )
    policy.config = SimpleNamespace()
    policy.processor = SimpleNamespace(action=lambda value, **kwargs: value)
    policy.reset = Mock()
    monkeypatch.setattr("alohamini.apps.evaluation.stop_owned_robot", Mock())
    monkeypatch.setattr(
        "alohamini.learning.fixed.dataset_summary",
        Mock(side_effect=AssertionError("Unexpected dataset read")),
    )
    return clock, client, policy


def test_current_requires_no_data_overrides_inherited_once_and_keeps_raw_state(rig):
    _, client, policy = rig
    policy.source["fixed_dimensions"] = dict(version=1, targets={JOINT: -30.0})
    configure(policy, current=["arm_right", "lift_axis", "base"])
    assert policy.fixed_dimensions is None  # No old target can leak before capture.
    with pytest.raises(RuntimeError, match="live Host preparation"):
        policy.fixed_input({"observation.state": torch.zeros(18)})
    with pytest.raises(RuntimeError, match="live Host preparation"):
        policy.execution_action(torch.zeros(1, 18))
    snap = client.read()
    before = deepcopy(snap.payload)
    policy.bind_fixed_current(snap, client.client_id)
    targets = policy.fixed_dimensions["targets"]
    assert targets[JOINT] == 27 and targets[LIFT] == 287
    assert targets["x.vel"] == 0
    assert snap.payload == before
    state = {"observation.state": torch.full((18,), 99.0)}
    assert policy.fixed_input(state)["observation.state"][policy.names.index(JOINT)] == 4
    assert state["observation.state"][policy.names.index(JOINT)] == 99
    assert policy.execution_action(torch.zeros(1, 18))[0, policy.names.index(LIFT)] == 287
    client.state.payload[LIFT] = 300
    policy.bind_fixed_current(client.read(), client.client_id)
    assert policy.fixed_dimensions["targets"][LIFT] == 287
    assert policy.fixed_deployment["current_capture"]["targets"][LIFT] == 287


def test_current_can_mix_with_mean_targets(rig):
    _, client, policy = rig
    policy.manifest["physical_action_summary"] = summarize(np.full((2, 18), 12.0), policy.names)
    configure(policy, selection=["arm_right"], current=["lift_axis"])
    policy.bind_fixed_current(client.read(), client.client_id)
    assert policy.fixed_dimensions["targets"][JOINT] == 12
    assert policy.fixed_dimensions["targets"][LIFT] == 287


@pytest.mark.parametrize("change", ["session", "reference"])
def test_bound_targets_cannot_be_reused_after_reference_change(rig, change):
    _, client, policy = rig
    configure(policy, current=[LIFT])
    policy.bind_fixed_current(client.read(), client.client_id)
    if change == "session":
        client.state.payload["_safety"]["host_session_id"] = "restarted"
    else:
        client.state.payload["lift_axis.reference_sequence"] += 1
    with pytest.raises(RuntimeError, match="changed after"):
        policy.bind_fixed_current(client.read(), client.client_id)


@pytest.mark.parametrize(
    "options",
    [
        dict(selection=["arm_right"], current=[JOINT]),
        dict(dataset="/must-not-be-read", current=["lift_axis"]),
    ],
)
def test_ambiguous_or_irrelevant_options_rejected(rig, options):
    _, _, policy = rig
    with pytest.raises(ValueError):
        configure(policy, **options)


@pytest.mark.parametrize("bad", ["stale", "owner", "calibration", "nonfinite"])
def test_current_rejects_bad_feedback_without_binding_or_commands(rig, bad):
    _, client, policy = rig
    configure(policy, current=["lift_axis"])
    snap = client.read()
    if bad == "stale":
        snap.request_started_s -= 2
    elif bad == "owner":
        snap.payload["_safety"]["control_owner"] = "other"
    elif bad == "calibration":
        snap.payload["_robot_metadata"]["lift_axis"]["soft_max_mm"] = 500
    else:
        snap.payload[LIFT] = float("nan")
    with pytest.raises(ValueError):
        policy.bind_fixed_current(snap, client.client_id)
    assert not policy._fixed_current_bound and not client.sent


def test_run_episode_captures_before_preparation_and_overrides_predictions(rig):
    clock, client, policy = rig
    configure(policy, current=[JOINT, LIFT])
    seen = []

    def predict(snap):
        seen.append(clock.now)
        assert policy._fixed_current_bound
        return {n: 0.0 for n in policy.names}

    policy.select_action = predict
    assert run_evaluation(client, policy, MODEL, duration_s=0.1) >= 2
    assert seen[0] >= 0.5
    assert all(action[JOINT] == 27 and action[LIFT] == 287 for _, action, _ in client.sent)


def test_application_logs_captured_targets_before_dataset_creation(rig, monkeypatch, tmp_path):
    _, client, policy = rig
    configure(policy, current=[LIFT])
    client.__enter__ = lambda: client
    client.__exit__ = lambda *args: None
    manager = Mock()
    manager.__enter__ = Mock(return_value=client)
    manager.__exit__ = Mock(return_value=False)
    policy.select_action = Mock()
    monkeypatch.setattr("alohamini.apps.evaluation.HostClient", lambda *args, **kwargs: manager)
    monkeypatch.setattr(
        "alohamini.apps.evaluation.WorkspacePaths",
        lambda: SimpleNamespace(logs=tmp_path, dataset=lambda name: tmp_path / name),
    )
    dataset = Mock()

    def create(*args, **kwargs):
        assert kwargs["fixed_dimensions"]["targets"][LIFT] == 287
        return dataset

    monkeypatch.setattr("alohamini.apps.evaluation.LocalDataset", create)
    monkeypatch.setattr("alohamini.apps.evaluation.run_evaluation", lambda *args, **kwargs: 0)
    evaluate("test-only", MODEL, policy_factory=lambda: policy, dataset_name="eval")
    report = json.loads(next(tmp_path.rglob("fixed-*.json")).read_text())
    assert report["fixed_dimensions"]["targets"][LIFT] == 287
    assert report["current_capture"]["targets"][LIFT] == 287


def test_cli_current_selection_does_not_require_dataset(tmp_path, monkeypatch):
    from alohamini.cli import main

    (tmp_path / "policy.json").write_text("{}")
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
                str(tmp_path),
                "--fixed-current",
                "arm_right",
                "lift_axis",
            ]
        )
        == 0
    )
    assert constructor.call_args.kwargs["fixed_current"] == ["arm_right", "lift_axis"]
    assert "fixed_dataset" not in constructor.call_args.kwargs
