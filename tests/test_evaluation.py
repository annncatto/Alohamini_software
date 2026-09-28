import contextlib
import copy
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyarrow.parquet as pq
import pytest
from test_replay import Client, Clock

from alohamini.apps.evaluation import evaluate, run_evaluation
from alohamini.cli import main
from alohamini.datasets.native import LocalDataset, state_names


class EvaluationClient(Client):
    def __init__(self, clock):
        super().__init__(clock)
        self.state.payload["_robot_metadata"]["cameras"] = []
        self.state.payload["lift_axis.reference_sequence"] = 1

    def read(self, *, include_images=False):
        return super().read()

    def connect_control(self):
        return self.read()


@pytest.fixture
def setup():
    clock = Clock()
    client = EvaluationClient(clock)
    names = state_names("alohamini2pro")
    policy = Mock()
    policy.robot_metadata = copy.deepcopy(client.state.payload["_robot_metadata"])
    policy.select_action.side_effect = lambda snap: {name: snap.payload[name] for name in names}
    with (
        patch("alohamini.apps.evaluation.time.monotonic", side_effect=lambda: clock.now),
        patch("alohamini.apps.evaluation.time.sleep", side_effect=clock.sleep),
        patch("alohamini.apps.evaluation.stop_owned_robot") as stop,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        yield clock, client, policy, stop


def test_sync_episode_resets_policy_checks_ack_and_stops(setup):
    _, client, policy, stop = setup
    count = run_evaluation(client, policy, "alohamini2pro", duration_s=0.15)
    assert count >= 4
    assert len(client.sent) == count
    policy.reset.assert_called_once()
    stop.assert_called_once_with(client, "alohamini2pro", client.sent[-1][2])


def test_skipped_command_discards_policy_queue_without_waiting_for_ack(setup):
    _, client, policy, stop = setup
    send = client.send_command
    attempts = 0

    def intermittent(action, *, based_on):
        nonlocal attempts
        attempts += 1
        if attempts % 2:
            return None
        return send(action, based_on=based_on)

    client.send_command = intermittent
    count = run_evaluation(client, policy, "alohamini2pro", duration_s=0.2)
    assert count > 0
    assert count == len(client.sent)
    assert policy.reset.call_count == 1 + (attempts + 1) // 2
    stop.assert_called_once_with(client, "alohamini2pro", client.sent[-1][2])


def test_ack_observation_is_reused_without_skipping_post_inference_refresh(setup):
    _, client, policy, _ = setup
    requests = []
    original = client.read

    def read(*, include_images=False):
        requests.append((len(client.sent), include_images))
        return original(include_images=include_images)

    client.read = read
    count = run_evaluation(client, policy, "alohamini2pro", duration_s=0.15)
    assert count >= 4
    # Initial calibration + first policy input; thereafter safety + ACK per action.
    assert len(requests) == 2 + 2 * count


def test_slow_policy_over_250ms_is_not_rejected_by_snapshot_age(setup):
    clock, client, policy, _ = setup

    def select(snapshot):
        clock.sleep(0.35)
        return {name: snapshot.payload[name] for name in state_names("alohamini2pro")}

    policy.select_action.side_effect = select
    count = run_evaluation(client, policy, "alohamini2pro", duration_s=0.8)
    assert count == 2
    assert len(client.sent) == 2
    assert client.sent[1][0] - client.sent[0][0] >= 0.35


@pytest.mark.parametrize(
    "change",
    [
        {"joint_holds": {"arm_left_elbow_flex": {}}},
        {"joint_hold_events": 1},
        {"watchdog_events": 1},
        {"watchdog_active": True, "control_owner": "replay-test"},
        {"host_session_id": "restarted"},
        {"control_epoch": 1},
        {"control_owner": "another-client"},
        {"feedback_valid": False},
    ],
)
def test_safety_change_during_policy_discards_result(setup, change):
    _, client, policy, stop = setup

    def select(snapshot):
        client.state.payload["_safety"].update(change)
        return {name: snapshot.payload[name] for name in state_names("alohamini2pro")}

    policy.select_action.side_effect = select
    with pytest.raises(RuntimeError):
        run_evaluation(client, policy, "alohamini2pro", duration_s=0.1)
    assert not client.sent
    stop.assert_called_once()


@pytest.mark.parametrize("field", ["metadata", "reference"])
def test_calibration_or_lift_reference_change_rejects_old_result(setup, field):
    _, client, policy, _ = setup

    def select(snapshot):
        if field == "metadata":
            client.state.payload["_robot_metadata"]["motors"]["arm_left_elbow_flex"][
                "homing_offset"
            ] += 1
        else:
            client.state.payload["lift_axis.reference_sequence"] += 1
        return {name: snapshot.payload[name] for name in state_names("alohamini2pro")}

    policy.select_action.side_effect = select
    with pytest.raises(RuntimeError):
        run_evaluation(client, policy, "alohamini2pro", duration_s=0.1)
    assert not client.sent


@pytest.mark.parametrize("bad", [None, {}, np.zeros(18), {"extra": 1}])
def test_wrong_policy_schema_never_sends(setup, bad):
    _, client, policy, _ = setup
    policy.select_action.side_effect = None
    policy.select_action.return_value = bad
    with pytest.raises(ValueError):
        run_evaluation(client, policy, "alohamini2pro", duration_s=0.1)
    assert not client.sent


@pytest.mark.parametrize(
    "key,value",
    [
        ("arm_left_elbow_flex.pos", float("nan")),
        ("arm_left_elbow_flex.pos", 1000),
        ("lift_axis.height_mm", 1000),
        ("x.vel", True),
    ],
)
def test_nonfinite_or_out_of_range_targets_never_send(setup, key, value):
    _, client, policy, _ = setup
    original = policy.select_action.side_effect
    policy.select_action.side_effect = lambda snap: {**original(snap), key: value}
    with pytest.raises((ValueError, TypeError)):
        run_evaluation(client, policy, "alohamini2pro", duration_s=0.1)
    assert not client.sent


def test_calibration_mismatch_fails_before_policy_or_motion(setup):
    _, client, policy, _ = setup
    policy.robot_metadata["motors"]["arm_left_elbow_flex"]["homing_offset"] += 1
    with pytest.raises(ValueError, match="calibration"):
        run_evaluation(client, policy, "alohamini2pro", duration_s=0.1)
    policy.select_action.assert_not_called()
    assert not client.sent


def test_entry_calibration_mismatch_does_not_prompt_write_or_create_dataset(setup):
    _, client, policy, _ = setup
    policy.robot_metadata["motors"]["arm_left_elbow_flex"]["homing_offset"] += 1
    connection = Mock()
    connection.__enter__ = Mock(return_value=client)
    connection.__exit__ = Mock(return_value=False)
    with tempfile.TemporaryDirectory() as directory:
        with (
            patch("alohamini.apps.evaluation.HostClient", return_value=connection),
            patch("alohamini.apps.evaluation.LocalDataset") as dataset,
            patch("alohamini.apps.evaluation.WorkspacePaths") as paths,
            patch("builtins.input") as prompt,
        ):
            paths.return_value.dataset.return_value = Path(directory) / "eval"
            with pytest.raises(ValueError, match="calibration"):
                evaluate("pi", "alohamini2pro", policy_factory=lambda: policy, dataset_name="eval")
    dataset.assert_not_called()
    prompt.assert_not_called()
    policy.select_action.assert_not_called()
    assert not client.sent


def test_missing_ack_stops_without_advancing_policy(setup):
    _, client, policy, stop = setup
    client.acknowledge = False
    with pytest.raises(RuntimeError, match="未确认"):
        run_evaluation(client, policy, "alohamini2pro", duration_s=2)
    assert len(client.sent) == 1
    policy.select_action.assert_called_once()
    stop.assert_called_once()


def test_inference_finishing_after_deadline_drops_action(setup):
    clock, client, policy, _ = setup
    original = policy.select_action.side_effect

    def select(snapshot):
        clock.sleep(1)
        return original(snapshot)

    policy.select_action.side_effect = select
    assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.1) == 0
    assert not client.sent


def test_records_policy_input_not_new_post_inference_state_and_keeps_partial_episode(setup):
    _, client, policy, _ = setup
    original = policy.select_action.side_effect

    def select(snapshot):
        action = original(snapshot)
        snapshot.payload["arm_left_elbow_flex.pos"] = -20  # Policy mutations stay private.
        client.state.payload["arm_left_elbow_flex.pos"] = 20
        if policy.select_action.call_count == 2:
            raise RuntimeError("policy failed")
        return action

    policy.select_action.side_effect = select
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "evaluation"
        dataset = LocalDataset(root, fps=30, task="pick", robot_metadata=policy.robot_metadata)
        dataset.begin_episode()
        try:
            with pytest.raises(RuntimeError, match="policy failed"):
                run_evaluation(client, policy, "alohamini2pro", duration_s=0.1, dataset=dataset)
        finally:
            dataset.close()
        rows = pq.read_table(root / "episodes/episode_000000/frames.parquet").to_pylist()
        index = state_names("alohamini2pro").index("arm_left_elbow_flex.pos")
        assert len(rows) == 1
        assert rows[0]["observation.state"][index] == 0
        assert rows[0]["action"][index] == 0


def test_camera_missing_or_stale_does_not_reach_policy(setup):
    clock, client, policy, _ = setup
    client.state.payload["_robot_metadata"]["cameras"] = ["forward"]
    client.state.payload["_host_timing"] = {"camera_capture_monotonic_s": {}}
    assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.1) == 0
    policy.select_action.assert_not_called()
    client.state.images["forward"] = b"jpeg"
    client.state.payload["_host_timing"] = {
        "camera_capture_monotonic_s": {"forward": 1.0},
        "state_sample_finished_monotonic_s": 2.0,
    }
    with pytest.raises(RuntimeError, match="stale"):
        run_evaluation(client, policy, "alohamini2pro", duration_s=0.1)
    assert not client.sent


def test_cli_dispatch_and_factory_validation_do_not_connect():
    with patch("alohamini.apps.evaluation.evaluate") as call:
        assert (
            main(
                [
                    "evaluate",
                    "--host",
                    "pi",
                    "--robot_model",
                    "alohamini2pro",
                    "--policy",
                    "my_policy:load",
                    "--episode_time",
                    "8",
                ]
            )
            == 0
        )
        assert call.call_args.kwargs["policy_factory"] == "my_policy:load"
        assert call.call_args.kwargs["dataset_name"] is None
    with patch("alohamini.apps.evaluation.HostClient") as client:
        with pytest.raises(ValueError, match="module:factory"):
            evaluate("pi", "alohamini2pro", policy_factory="checkpoint_dir")
        client.assert_not_called()


@pytest.mark.parametrize("coefficient", ["none", "0", "0.01"])
def test_native_cli_loads_checkpoint_into_existing_evaluator(monkeypatch, coefficient, tmp_path):
    (tmp_path / "policy.json").write_text("{}")
    model = Mock(fps=30)
    loader = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "alohamini.learning.policy",
        SimpleNamespace(NativePolicy=loader),
    )
    with patch("alohamini.apps.evaluation.evaluate") as run:
        assert (
            main(
                [
                    "evaluate",
                    "--host",
                    "pi",
                    "--robot_model",
                    "alohamini2pro",
                    "--policy.path",
                    str(tmp_path),
                    "--policy.n_action_steps",
                    "1",
                    "--policy.temporal_ensemble_coeff",
                    coefficient,
                ]
            )
            == 0
        )
        assert run.call_args.kwargs["policy_factory"]() is model
    loader.assert_called_once_with(
        str(tmp_path),
        device="cuda",
        task="robot task",
        n_action_steps=1,
        temporal_ensemble_coeff=None if coefficient == "none" else float(coefficient),
    )


@pytest.mark.parametrize(
    "policy_args",
    [
        ["--policy.path", "/local/model"],
        ["--policy", "local:load", "--policy.n_action_steps", "2"],
    ],
)
def test_checkpoint_option_errors_do_not_open_robot(policy_args):
    with patch("alohamini.apps.evaluation.HostClient") as client:
        assert (
            main(["evaluate", "--host", "pi", "--robot_model", "alohamini2pro", *policy_args]) == 1
        )
    client.assert_not_called()


@pytest.mark.parametrize("interrupt", [False, True])
def test_application_closes_dataset_and_client_after_policy_failure(setup, interrupt):
    _, client, policy, stop = setup
    failure = KeyboardInterrupt() if interrupt else RuntimeError("policy failed")
    original = policy.select_action.side_effect

    def select(snapshot):
        if policy.select_action.call_count == 2:
            raise failure
        return original(snapshot)

    policy.select_action.side_effect = select
    module = Mock()
    module.load.return_value = policy
    connection = Mock()
    connection.__enter__ = Mock(return_value=client)
    connection.__exit__ = Mock(return_value=False)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "evaluation"
        with (
            patch(
                "alohamini.apps.evaluation.importlib", Mock(import_module=Mock(return_value=module))
            ),
            patch("alohamini.apps.evaluation.HostClient", return_value=connection),
            patch("alohamini.apps.evaluation.WorkspacePaths") as paths,
        ):
            paths.return_value.dataset.return_value = root
            with pytest.raises(type(failure)):
                evaluate(
                    "pi",
                    "alohamini2pro",
                    policy_factory="local_policy:load",
                    dataset_name="evaluation",
                    episode_time_s=0.2,
                )
        assert len(pq.read_table(root / "episodes/episode_000000/frames.parquet")) == 1
    connection.__exit__.assert_called_once()
    stop.assert_called_once()


def test_save_failure_during_policy_failure_does_not_hide_original_error(setup):
    _, client, policy, _ = setup
    policy.select_action.side_effect = RuntimeError("policy failed")
    connection = Mock()
    connection.__enter__ = Mock(return_value=client)
    connection.__exit__ = Mock(return_value=False)
    dataset = Mock(
        robot_metadata=client.state.payload["_robot_metadata"],
        fps=30,
        close=Mock(side_effect=OSError("disk failed")),
    )
    with tempfile.TemporaryDirectory() as directory:
        with (
            patch("alohamini.apps.evaluation.HostClient", return_value=connection),
            patch("alohamini.apps.evaluation.LocalDataset", return_value=dataset),
            patch("alohamini.apps.evaluation.WorkspacePaths") as paths,
        ):
            paths.return_value.dataset.return_value = Path(directory) / "evaluation"
            with pytest.raises(RuntimeError, match="policy failed"):
                evaluate(
                    "pi", "alohamini2pro", policy_factory=lambda: policy, dataset_name="evaluation"
                )
    dataset.close.assert_called_once()
    connection.__exit__.assert_called_once()
