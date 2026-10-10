"""Cross-robot normalized deployment leaves both calibration sources intact."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from test_evaluation import EvaluationClient
from test_replay import Clock, replay_snapshot

from alohamini.apps.evaluation import run_evaluation
from alohamini.apps.replay import check_calibration
from alohamini.datasets.record import StateSelection, dataset_features, state_names
from alohamini.learning.calibration import normalized_transfer
from alohamini.learning.policy import NativePolicy

JOINT = "arm_left_shoulder_pan"


def changed_robot():
    snapshot = replay_snapshot()
    old = deepcopy(snapshot.payload["_robot_metadata"])
    snapshot.payload["_robot_metadata"]["motors"][JOINT].update(
        range_min=500, range_max=3500, homing_offset=300
    )
    return old, snapshot


def policy_for(old, mode="normalized"):
    policy = NativePolicy.__new__(NativePolicy)
    policy.robot_metadata = deepcopy(old)
    policy.source = dict(robot_metadata=deepcopy(old), features=dataset_features("alohamini2pro"))
    policy.manifest = {"state": "joint_position,joint_velocity"}
    policy.selection = StateSelection(policy.source, policy.manifest["state"])
    policy.calibration_mode = mode
    policy._robot_bound = False
    policy.fixed_dimensions = None
    policy.checkpoint_path = Path("/test-only")
    policy.checkpoint_manifest_sha256 = "test"
    return policy


def test_transfer_preserves_sources_and_strict_rejects_same_change():
    old, snapshot = changed_robot()
    saved = deepcopy(old)
    host = deepcopy(snapshot.payload["_robot_metadata"])
    with pytest.raises(ValueError, match="range_min: recorded=1000, Host=500"):
        check_calibration(old, snapshot)
    metadata, report = normalized_transfer(old, snapshot)
    check_calibration(metadata, snapshot)
    assert report["differences"][JOINT]["homing_offset"] == {"trained": 0, "Host": 300}
    assert old == saved and snapshot.payload["_robot_metadata"] == host


@pytest.mark.parametrize(
    "field,value",
    [
        ("normalization", "degrees"),
        ("normalization", "range_0_100"),
        ("drive_mode", 1),
        ("id", 99),
        ("model", "other"),
        ("range_min", 4095),
        ("homing_offset", 3000),
    ],
)
def test_transfer_rejects_non_range_contract_changes(field, value):
    old, snapshot = changed_robot()
    snapshot.payload["_robot_metadata"]["motors"][JOINT][field] = value
    with pytest.raises(ValueError):
        normalized_transfer(old, snapshot)


def test_transfer_does_not_relax_lift_calibration():
    old, snapshot = changed_robot()
    snapshot.payload["_robot_metadata"]["motors"]["lift_axis"]["homing_offset"] = 1
    with pytest.raises(ValueError, match="lift_axis"):
        normalized_transfer(old, snapshot)


def test_bind_updates_velocity_units_once_and_rejects_later_change():
    old, snapshot = changed_robot()
    policy = policy_for(old)
    index = policy.selection.feature["names"].index(f"{JOINT}.vel")
    old_scale = policy.selection.columns[index][2]
    policy.bind_robot(snapshot)
    assert policy.selection.columns[index][2] == pytest.approx(old_scale * 2000 / 3000)
    assert policy.source["robot_metadata"] == old
    assert policy.bind_robot(snapshot) is None
    snapshot.payload["_robot_metadata"]["motors"][JOINT]["range_max"] += 1
    with pytest.raises(ValueError, match="range_max"):
        policy.bind_robot(snapshot)


def test_static_check_reused_but_nested_changes_and_context_revalidate():
    snapshot = replay_snapshot()
    policy = policy_for(snapshot.payload["_robot_metadata"], "strict")
    with patch("alohamini.learning.policy.check_calibration", wraps=check_calibration) as check:
        policy._check_calibration(snapshot)
        policy._check_calibration(deepcopy(snapshot))
        assert check.call_count == 1
        snapshot.payload["_safety"]["host_session_id"] = "new-session"
        policy._check_calibration(snapshot)
        snapshot.payload["lift_axis.reference_sequence"] = 99
        policy._check_calibration(snapshot)
        assert check.call_count == 3
        snapshot.payload["_robot_metadata"]["motors"][JOINT]["range_max"] += 1
        with pytest.raises(ValueError, match="range_max"):
            policy._check_calibration(snapshot)
        # Failed checks never replace the last valid context.
        with pytest.raises(ValueError, match="range_max"):
            policy._check_calibration(snapshot)
        snapshot.payload["_robot_metadata"]["motors"][JOINT]["range_max"] -= 1
        policy.robot_metadata["motors"][JOINT]["homing_offset"] += 1
        with pytest.raises(ValueError, match="homing_offset"):
            policy._check_calibration(snapshot)


@pytest.mark.parametrize("mode", ["strict", "normalized"])
def test_evaluator_binds_before_inference_and_keeps_actual_feedback(mode, monkeypatch, tmp_path):
    clock = Clock()
    client = EvaluationClient(clock)
    old = deepcopy(client.state.payload["_robot_metadata"])
    client.state.payload["_robot_metadata"]["motors"][JOINT]["range_min"] = 500
    policy = policy_for(old, mode)
    policy.reset = Mock()
    seen = []

    def select(snapshot):
        check_calibration(policy.robot_metadata, snapshot)
        seen.append(snapshot.payload[f"{JOINT}.pos"])
        return {n: snapshot.payload[n] for n in state_names("alohamini2pro")}

    policy.select_action = select
    # A runtime mock uses no model images; the manifest still declares state selection.
    policy.manifest["cameras"] = []
    monkeypatch.setattr("time.monotonic", lambda: clock.now)
    monkeypatch.setattr("time.sleep", clock.sleep)
    monkeypatch.setattr("alohamini.apps.evaluation.stop_owned_robot", Mock())
    monkeypatch.setattr(
        "alohamini.apps.evaluation.WorkspacePaths", lambda: SimpleNamespace(logs=tmp_path)
    )
    if mode == "strict":
        with pytest.raises(ValueError, match="range_min"):
            run_evaluation(client, policy, "alohamini2pro", duration_s=0.1)
        assert not client.sent and not seen
    else:
        assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.1) >= 2
        assert seen == [0.0] * len(seen)
        assert len(list(tmp_path.rglob("calibration-transfer-*.json"))) == 1


def test_cli_routes_explicit_transfer(tmp_path, monkeypatch):
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
                "alohamini2pro",
                "--policy.path",
                str(tmp_path),
                "--calibration-mode",
                "normalized",
            ]
        )
        == 0
    )
    assert constructor.call_args.kwargs["calibration_mode"] == "normalized"
