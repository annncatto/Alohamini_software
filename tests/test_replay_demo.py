"""Standalone offline demo checks; no robot or network connection."""

import ast
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from test_replay import Client, Clock, episode_fixture

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("replay_demo", ROOT / "examples/replay_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_guard_and_calibration_checks_are_copied_without_semantic_changes():
    def definitions(path):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }

    source = definitions(ROOT / "src/alohamini/apps/replay.py")
    copied = definitions(ROOT / "examples/replay_demo.py")
    for name in ("check_calibration", "check_target_ranges", "TargetRanges", "ReplayGuard"):
        assert copied[name] == source[name]


def test_lift_smoothing_limits_step_without_exceeding_soft_limits():
    values = np.r_[np.zeros(50), np.ones(100) * 650, np.zeros(50)]
    result = demo.smooth_lift(values, 30, 0, 600)
    assert result.min() >= 0 and result.max() <= 600
    assert np.max(np.abs(np.diff(result))) <= 40 / 30 + 1e-10
    assert result[0] == result[-1] == 0


def test_loop_transition_has_no_base_motion_and_bounded_pose_rate():
    episode = episode_fixture()
    first, last = episode.actions[0].copy(), episode.actions[-1].copy()
    last[0] += 70
    last[-1] += 400
    result = demo.transition(first, last, episode.names, 30)
    assert np.all(result[:, 14:17] == 0)
    np.testing.assert_allclose(result[0, :14], first[:14])
    np.testing.assert_allclose(result[-1, :14], last[:14])
    assert np.max(np.abs(np.diff(result[:, :14], axis=0))) * 30 <= 10 + 1e-8
    assert np.max(np.abs(np.diff(result[:, -1]))) * 30 <= 40 + 1e-8


@pytest.mark.parametrize("delay", [0.002, 0.045])
def test_timeline_does_not_stretch_to_execute_every_row(delay):
    clock, episode = Clock(), episode_fixture(count=60)
    client = Client(clock)
    client.read_delay = delay
    guard = demo.ReplayGuard(client, "alohamini2pro")
    with (
        patch.object(demo.time, "monotonic", side_effect=lambda: clock.now),
        patch.object(demo.time, "sleep", side_effect=clock.sleep),
        patch.object(demo, "stop_owned_robot") as stop,
    ):
        result = demo.play_timeline(client, episode.actions, episode.names, 30, guard)
    assert 2 <= result["elapsed_s"] <= 2.05
    stop.assert_called_once()
    if delay > 1 / 30:
        assert result["skipped_rows"] > 0
        assert len(client.sent) < 60


@pytest.mark.parametrize("failure", ["stall", "ack", "interrupt", "watchdog"])
def test_faults_stop_instead_of_restarting_demo(failure):
    clock, episode = Clock(), episode_fixture(count=60)
    client = Client(clock)
    if failure == "ack":
        client.acknowledge = False

    def on_read(current):
        if not current.sent:
            return
        if failure == "stall":
            clock.sleep(0.17)
        elif failure == "interrupt":
            raise KeyboardInterrupt
        elif failure == "watchdog":
            current.state.payload["_safety"]["watchdog_events"] += 1

    client.on_read = on_read
    guard = demo.ReplayGuard(client, "alohamini2pro")
    error = KeyboardInterrupt if failure == "interrupt" else RuntimeError
    with (
        patch.object(demo.time, "monotonic", side_effect=lambda: clock.now),
        patch.object(demo.time, "sleep", side_effect=clock.sleep),
        patch.object(demo, "stop_owned_robot") as stop,
        pytest.raises(error),
    ):
        demo.play_timeline(client, episode.actions, episode.names, 30, guard)
    stop.assert_called_once()
    assert len(client.sent) < 60


def test_bundle_rejects_nonclosing_or_fast_lift_targets():
    episode = episode_fixture()
    actions = episode.actions.copy()
    actions[:, 0] = 0
    actions[:, 14:17] = 0
    bundle = dict(
        format="alohamini-demo-trajectory",
        version=1,
        fps=30,
        names=list(episode.names),
        robot_metadata=episode.metadata,
        actions=actions.tolist(),
    )
    demo.validate_bundle(bundle)
    for column, value in ((14, 0.15), (17, 200)):
        changed = actions.copy()
        changed[1, column] = value
        with pytest.raises(ValueError):
            demo.validate_bundle({**bundle, "actions": changed.tolist()})
