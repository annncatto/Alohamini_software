import contextlib
import copy
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyarrow.parquet as pq
import pytest
from test_replay import Client, Clock

from alohamini.apps.evaluation import _action, evaluate, run_evaluation
from alohamini.cli import main
from alohamini.datasets.record import _EpisodeWriter as LocalDataset, state_names
from alohamini.errors import ResponseTimeoutError


class EvaluationClient(Client):
    def __init__(self, clock):
        super().__init__(clock)
        self.state.payload["_robot_metadata"]["cameras"] = []
        self.state.payload["lift_axis.reference_sequence"] = 1

    def read(self, *, include_images=False):
        return super().read()

    def connect_control(self):
        return self.read()

    def refresh(self, *, include_images=False):
        return self.read(include_images=include_images)


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


def test_sync_episode_resets_policy_submits_and_stops(setup):
    _, client, policy, stop = setup
    count = run_evaluation(client, policy, "alohamini2pro", duration_s=0.15)
    assert count >= 4
    assert len(client.sent) == count
    policy.reset.assert_called_once()
    stop.assert_called_once_with(client, "alohamini2pro", client.sent[-1][2])


@pytest.mark.parametrize("delay", [0.25, 0.4])
def test_initial_poll_timeout_waits_for_late_feedback(setup, delay):
    clock, client, policy, _ = setup
    original = client.read
    calls = 0

    def read(*, include_images=False):
        nonlocal calls
        calls += 1
        if calls == 1:
            clock.sleep(0.2)
            raise ResponseTimeoutError("initial poll timeout")
        if calls == 2:
            clock.sleep(max(0, delay - clock.now))
        return original(include_images=include_images)

    client.read = read
    count = run_evaluation(client, policy, "alohamini2pro", duration_s=0.15)
    assert count >= 4
    assert client.sent[0][0] >= delay
    policy.reset.assert_called_once()


def test_initial_feedback_loss_has_bounded_wait_and_never_sends(setup):
    clock, client, policy, stop = setup

    def read(*, include_images=False):
        clock.sleep(0.2)
        raise ResponseTimeoutError("no reply")

    client.read = read
    with pytest.raises(ResponseTimeoutError, match="No initial Host feedback"):
        run_evaluation(client, policy, "alohamini2pro", duration_s=0.15)
    assert 5 <= clock.now < 5.25
    assert not client.sent
    policy.select_action.assert_not_called()
    stop.assert_called_once_with(client, "alohamini2pro", None)


def test_skipped_command_preserves_policy_queue_without_waiting_for_ack(setup):
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
    policy.reset.assert_called_once()
    stop.assert_called_once_with(client, "alohamini2pro", client.sent[-1][2])


def test_fast_policy_uses_one_observation_per_tick_without_ack_or_extra_refresh(setup):
    _, client, policy, _ = setup
    requests = []
    original = client.read

    def read(*, include_images=False):
        requests.append((len(client.sent), include_images))
        return original(include_images=include_images)

    client.read = read
    count = run_evaluation(client, policy, "alohamini2pro", duration_s=0.15)
    assert count >= 4
    # Initial calibration, then one input per tick, with no per-command ACK read.
    assert len(requests) == 1 + count


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


def test_slow_policy_reads_once_per_tick_without_post_inference_refresh(setup):
    clock, client, policy, _ = setup
    original = policy.select_action.side_effect
    refresh = Mock(wraps=client.refresh)
    client.refresh = refresh
    client.read = Mock(wraps=client.read)

    def select(snapshot):
        clock.sleep(0.04)
        return original(snapshot)

    policy.select_action.side_effect = select
    count = run_evaluation(client, policy, "alohamini2pro", duration_s=0.2)
    assert count > 0
    refresh.assert_not_called()
    assert client.read.call_count == 1 + policy.select_action.call_count


@pytest.mark.parametrize(
    "change",
    [
        {"host_session_id": "restarted"},
        {"control_owner": "another-client"},
    ],
)
def test_safety_change_during_policy_is_detected_at_next_observation(setup, change):
    clock, client, policy, stop = setup
    client.acknowledge = False

    def select(snapshot):
        clock.sleep(0.04)
        client.state.payload["_safety"].update(change)
        return {name: snapshot.payload[name] for name in state_names("alohamini2pro")}

    policy.select_action.side_effect = select
    with pytest.raises(RuntimeError):
        run_evaluation(client, policy, "alohamini2pro", duration_s=0.1)
    # No second read: one command can be submitted using the previous context.
    # Host session/ownership validation, not this in-memory sender, rejects it.
    assert len(client.sent) == 1
    assert client.sent[0][2].host_session_id == "session"
    stop.assert_called_once()


def test_invalid_feedback_pauses_without_sending_or_prompting(setup):
    clock, client, policy, _ = setup
    original = policy.select_action.side_effect

    def select(snapshot):
        clock.sleep(0.04)
        client.state.payload["_safety"]["feedback_valid"] = False
        return original(snapshot)

    policy.select_action.side_effect = select
    with patch("builtins.input") as prompt:
        assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.1) == 1
    assert len(client.sent) == 1  # Invalid feedback is observed at the next tick.
    prompt.assert_not_called()


def test_joint_protection_event_during_inference_does_not_discard_prediction(setup):
    clock, client, policy, _ = setup
    original = policy.select_action.side_effect

    def select(snapshot):
        if not client.state.payload["_safety"]["joint_hold_events"]:
            clock.sleep(0.04)
        client.state.payload["_safety"]["joint_hold_events"] = 1
        return original(snapshot)

    policy.select_action.side_effect = select
    assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.2) > 0
    policy.reset.assert_called_once()
    assert policy.select_action.call_count == len(client.sent)


@pytest.mark.parametrize("gripper", [False, True])
def test_host_joint_warning_and_gripper_hold_do_not_reset_policy(setup, gripper):
    from test_arm_control import joint_spec

    from alohamini.runtime.arm_contact import ArmContactGuard, GripperContactCalibration

    clock, client, policy, _ = setup
    name = "arm_left_gripper" if gripper else "arm_left_elbow_flex"
    spec = joint_spec(name=name, contact=GripperContactCalibration(0, 1) if gripper else None)
    contact = ArmContactGuard({name: spec})
    present = 0.5 if gripper else 0.0
    push = 0.0 if gripper else 0.5
    retreat = 0.8 if gripper else -0.1
    current = 0.6 if gripper else 2.2
    # Pre-existing contact comes from the real Host limiter, not a hand-cleared flag.
    for now in (0.0, 0.16):
        contact.limit({name: push}, {name: present}, {name: current}, now=now)
    held = contact.holds[name] if gripper else push
    clock.now = 0.2
    original_select = policy.select_action.side_effect
    original_send = client.send_command
    applied = []

    def update_feedback(_):
        status = client.state.payload["_safety"]
        status["gripper_holds" if gripper else "joint_holds"] = contact.holds
        status["joint_hold_events"] = contact.joint_hold_events
        status["joint_stall_currents_a"] = contact.joint_stall_currents_a

    def select(snapshot):
        action = original_select(snapshot)
        # Explicit test coordinates: percent of a 1-rad gripper stroke, or degrees.
        target = push if len(client.sent) < 3 else retreat
        action[name + ".pos"] = target * 100 if gripper else np.degrees(target)
        return action

    def send(action, *, based_on):
        target = action[name + ".pos"]
        target = target / 100 if gripper else np.radians(target)
        applied.append(
            contact.limit(
                {name: target}, {name: present}, {name: 0.1 if gripper else current}, now=clock.now
            )[name]
        )
        return original_send(action, based_on=based_on)

    client.on_read = update_feedback
    client.send_command = send
    policy.select_action.side_effect = select
    run_evaluation(client, policy, "alohamini2pro", duration_s=0.25)
    assert len(applied) > 3
    assert applied[:3] == pytest.approx([held] * 3)
    assert applied[3:] == pytest.approx([retreat] * (len(applied) - 3))
    assert not contact.holds
    policy.reset.assert_called_once()


def test_expired_inference_is_not_sent_and_next_tick_resumes_after_watchdog_release(setup):
    clock, client, policy, _ = setup
    original = policy.select_action.side_effect

    def select(snapshot):
        if policy.select_action.call_count == 1:
            clock.sleep(1.1)  # This fixture advertises a 1s feedback lifetime.
            status = client.state.payload["_safety"]
            status.update(
                control_epoch=1, control_owner=None, watchdog_active=True, watchdog_events=1
            )
        return original(snapshot)

    policy.select_action.side_effect = select
    with patch("builtins.input") as prompt:
        assert run_evaluation(client, policy, "alohamini2pro", duration_s=1.5) > 1
    assert all(entry[2].control_epoch == 1 for entry in client.sent)
    assert policy.reset.call_count == 2
    prompt.assert_not_called()


def test_prolonged_feedback_loss_pauses_then_automatically_resumes(setup):
    clock, client, policy, _ = setup
    read = client.read

    def intermittent(**kwargs):
        if client.sent and clock.now < 1.5:
            clock.sleep(0.2)
            raise ResponseTimeoutError("offline")
        return read(**kwargs)

    client.read = intermittent
    with patch("builtins.input") as prompt:
        assert run_evaluation(client, policy, "alohamini2pro", duration_s=1.8) > 1
    assert client.sent[0][0] < 0.1
    # Brief loss can reuse bounded feedback; prolonged loss pauses sending.
    assert any(sent_at >= 1.5 for sent_at, _, _ in client.sent)
    assert not any(1.0 <= sent_at < 1.5 for sent_at, _, _ in client.sent)
    assert policy.reset.call_count >= 2
    prompt.assert_not_called()


def test_joint_hold_and_release_keep_policy_running_without_confirmation(setup):
    clock, client, policy, _ = setup
    read = client.read

    def protected(**kwargs):
        status = client.state.payload["_safety"]
        status["joint_hold_events"] = 1
        status["joint_holds"] = {"arm_left_elbow_flex": {}} if clock.now < 0.1 else {}
        return read(**kwargs)

    client.read = protected
    with patch("builtins.input") as prompt:
        assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.25) > 0
    assert client.sent[0][0] < 0.1
    policy.reset.assert_called_once()
    prompt.assert_not_called()


@pytest.mark.parametrize("field", ["metadata", "reference"])
def test_calibration_or_lift_reference_change_stops_on_next_observation(setup, field):
    clock, client, policy, _ = setup

    def select(snapshot):
        clock.sleep(0.04)
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
    assert len(client.sent) == 1


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
        ("arm_left_gripper.pos", float("inf")),
        ("lift_axis.height_mm", float("nan")),
        ("x.vel", True),
    ],
)
def test_nonfinite_or_invalid_targets_never_send(setup, key, value):
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


def test_missing_ack_does_not_block_action_queue_or_recording(setup):
    _, client, policy, stop = setup
    client.acknowledge = False
    count = run_evaluation(client, policy, "alohamini2pro", duration_s=0.5)
    assert count >= 14
    assert len(client.sent) == policy.select_action.call_count == count
    policy.reset.assert_called_once()
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


def test_missing_camera_reports_error_instead_of_silently_waiting(setup):
    _, client, policy, _ = setup
    client.state.payload["_robot_metadata"]["cameras"] = ["forward"]
    client.state.payload["_host_timing"] = {"camera_capture_monotonic_s": {}}
    with pytest.raises(ValueError, match="Missing policy camera: forward"):
        run_evaluation(client, policy, "alohamini2pro", duration_s=0.1)
    policy.select_action.assert_not_called()
    assert not client.sent


def test_camera_state_skew_warns_but_keeps_advancing_actions(setup, caplog):
    _, client, policy, _ = setup
    client.state.payload["_robot_metadata"]["cameras"] = ["forward"]
    client.state.images["forward"] = b"jpeg"
    client.state.payload["_host_timing"] = {
        "camera_capture_monotonic_s": {"forward": 1.0},
        "state_sample_finished_monotonic_s": 2.0,
    }
    assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.15) >= 4
    policy.reset.assert_called_once()
    assert len([r for r in caplog.records if "camera/state time difference" in r.message]) == 1


def test_slow_inference_keeps_images_without_refreshing_camera_pipeline(setup):
    clock, client, policy, _ = setup
    client.state.payload["_robot_metadata"]["cameras"] = ["forward"]
    client.state.images["forward"] = b"jpeg"
    original = policy.select_action.side_effect

    def select(snapshot):
        assert snapshot.images["forward"] == b"jpeg"
        clock.sleep(0.04)
        return original(snapshot)

    policy.select_action.side_effect = select
    client.refresh = Mock(wraps=client.refresh)
    assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.2) > 1
    client.refresh.assert_not_called()


def test_brief_read_timeout_continues_using_bounded_feedback(setup):
    clock, client, policy, _ = setup
    original = policy.select_action.side_effect
    original_read = client.read

    def read(**kwargs):
        if client.sent:
            clock.sleep(0.02)
            raise ResponseTimeoutError("read timeout")
        return original_read(**kwargs)

    def select(snapshot):
        clock.sleep(0.04)
        return original(snapshot)

    policy.select_action.side_effect = select
    client.read = read
    assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.2) > 1
    policy.reset.assert_called_once()


def test_policy_reuses_complete_cached_image_and_state_then_sees_new_pixels(setup):
    import cv2

    from alohamini.datasets.images import decode_host_image

    clock, client, policy, _ = setup
    original_read = client.read
    original_select = policy.select_action.side_effect
    seen = []
    delayed = False
    client.state.payload["_robot_metadata"]["cameras"] = ["forward"]

    def read(**kwargs):
        nonlocal delayed
        if client.sent and not delayed:
            delayed = True
            clock.sleep(0.2)
            raise ResponseTimeoutError("late camera reply")
        level = 180 if delayed else 30
        ok, jpeg = cv2.imencode(".jpg", np.full((8, 8, 3), level, np.uint8))
        assert ok
        client.state.images["forward"] = jpeg.tobytes()
        client.state.payload["arm_left_elbow_flex.pos"] = 10 if delayed else 0
        return original_read(**kwargs)

    def select(snapshot):
        seen.append(
            (
                int(decode_host_image(snapshot.images["forward"])[0, 0, 0]),
                snapshot.payload["arm_left_elbow_flex.pos"],
                snapshot.request_started_s,
                snapshot.received_s,
            )
        )
        return original_select(snapshot)

    client.read = read
    client.refresh = Mock(side_effect=AssertionError("No post-inference refresh"))
    policy.select_action.side_effect = select
    assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.4) >= 3
    assert seen[0] == seen[1]  # Image, state and both timestamps are unchanged.
    assert seen[0][:2] == (30, 0)
    assert seen[2][:2] == (180, 10)
    assert seen[2][2] > seen[1][2]
    policy.reset.assert_called_once()
    client.refresh.assert_not_called()


@pytest.mark.parametrize("delay", [0.25, 0.4])
def test_delayed_transport_recovers_images_through_evaluator_and_command_encoder(setup, delay):
    import cv2
    from test_client import DelayedResponseSocket

    from alohamini.client import HostClient
    from alohamini.datasets.images import decode_host_image

    clock, fixture_client, policy, _ = setup
    payload = copy.deepcopy(fixture_client.state.payload)
    payload["_image_encoding"] = "jpeg"
    payload["_robot_metadata"]["cameras"] = ["forward"]
    policy.robot_metadata = copy.deepcopy(payload["_robot_metadata"])
    seen = []

    def select(snapshot):
        pixels = decode_host_image(snapshot.images["forward"])
        seen.append(
            (
                int(pixels[0, 0, 0]),
                snapshot.payload["arm_left_shoulder_pan.pos"],
                snapshot.request_started_s,
                snapshot.received_s,
            )
        )
        return {name: snapshot.payload[name] for name in state_names("alohamini2pro")}

    policy.select_action.side_effect = select
    with HostClient(
        "test-only",
        expected_model="alohamini2pro",
        timeout_s=0.2,
        request_window=3,
        prefetch_before_decode=True,
    ) as client:
        transport = DelayedResponseSocket(clock, delay)
        transport.image_factory = lambda number: cv2.imencode(
            ".jpg", np.full((8, 8, 3), number, np.uint8)
        )[1].tobytes()
        client._socket = transport
        client._command_socket = Mock()
        with (
            patch.object(client, "_connect"),
            patch.object(client, "_connect_commands"),
            patch.object(client, "refresh", side_effect=AssertionError("Unexpected refresh")),
            patch("test_client.state_payload", side_effect=lambda: copy.deepcopy(payload)),
        ):
            count = run_evaluation(client, policy, "alohamini2pro", duration_s=1.8)
        assert count >= 3
        commands = [json.loads(call.args[0]) for call in client._command_socket.send.call_args_list]
        assert len(commands) == count
        assert len(client._pending) <= 3
        assert all(pixel == state for pixel, state, _, _ in seen)
        assert seen[-1][0] > seen[0][0]
        assert any(first == second for first, second in zip(seen, seen[1:], strict=False))
        assert all(command["_command"]["host_session_id"] == "session" for command in commands)
        assert policy.reset.call_count == 2  # Initialization and first image after startup wait.


def test_cached_read_does_not_renew_feedback_age_and_recovery_resets_once(setup):
    clock, client, policy, _ = setup
    original_select = policy.select_action.side_effect
    original_read = client.read
    seen_times = []

    def select(snapshot):
        seen_times.append(snapshot.request_started_s)
        clock.sleep(0.04)
        return original_select(snapshot)

    def read(**kwargs):
        if seen_times and clock.now < 1.5:
            clock.sleep(0.2)
            raise ResponseTimeoutError("offline")
        return original_read(**kwargs)

    client.read = read
    policy.select_action.side_effect = select
    assert run_evaluation(client, policy, "alohamini2pro", duration_s=1.9) > 2
    assert len(set(seen_times[:3])) == 1
    assert any(sent_at < 1.0 for sent_at, _, _ in client.sent)
    assert not any(1.0 <= sent_at < 1.5 for sent_at, _, _ in client.sent)
    assert any(sent_at >= 1.5 for sent_at, _, _ in client.sent)
    assert policy.reset.call_count == 2


def test_cached_feedback_continues_commands_without_duplicate_dataset_frames(setup, tmp_path):
    clock, client, policy, _ = setup
    original_read = client.read
    original_select = policy.select_action.side_effect
    input_times = []

    def select(snapshot):
        input_times.append(snapshot.request_started_s)
        clock.sleep(0.04)
        return original_select(snapshot)

    def read(**kwargs):
        if input_times:
            clock.sleep(0.02)
            raise ResponseTimeoutError("timeout")
        return original_read(**kwargs)

    client.read = read
    policy.select_action.side_effect = select
    root = tmp_path / "evaluation"
    dataset = LocalDataset(root, fps=30, task="pick", robot_metadata=policy.robot_metadata)
    dataset.begin_episode()
    try:
        assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.3, dataset=dataset) > 1
    finally:
        dataset.close()
    assert len(set(input_times)) == 1
    assert pq.read_table(root / "episodes/episode_000000/frames.parquet").num_rows == 1
    policy.reset.assert_called_once()


def test_runtime_summary_distinguishes_constant_targets_from_no_commands(setup):
    _, client, policy, _ = setup
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert run_evaluation(client, policy, "alohamini2pro", duration_s=2.1) > 50
    assert "[EVAL] select_hz=" in output.getvalue()
    assert "sent_hz=" in output.getvalue()
    assert "target_changes=0" in output.getvalue()


def test_contact_warning_and_release_do_not_pause_evaluation(setup, caplog):
    clock, client, policy, _ = setup
    read = client.read

    def protected(**kwargs):
        client.state.payload["_safety"]["joint_holds"] = (
            {"arm_left_elbow_flex": {}} if clock.now < 0.1 else {}
        )
        return read(**kwargs)

    client.read = protected
    assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.2) > 0
    assert client.sent[0][0] < 0.1
    policy.reset.assert_called_once()
    assert (
        len([r for r in caplog.records if "Host 报告关节保持：arm_left_elbow_flex" in r.message])
        == 1
    )
    assert any("Host 关节保持报告已清除" in r.message for r in caplog.records)
    assert not any("暂停发送" in r.message for r in caplog.records)


@pytest.mark.parametrize("drive_mode", [0, 1])
@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("arm_left_gripper.pos", -0.01, 0.0),
        ("arm_right_gripper.pos", 100.01, 100.0),
        ("arm_left_elbow_flex.pos", -100.01, -100.0),
        ("arm_right_wrist_flex.pos", 1000, 100.0),
        ("lift_axis.height_mm", -0.01, 0.0),
        ("lift_axis.height_mm", 600.01, 600.0),
    ],
)
def test_policy_targets_saturate_like_motor_bus_and_lift(setup, drive_mode, key, value, expected):
    _, client, policy, _ = setup
    snapshot = client.state
    for motor in snapshot.payload["_robot_metadata"]["motors"].values():
        motor["drive_mode"] = drive_mode
    names = state_names("alohamini2pro")
    raw = {name: snapshot.payload[name] for name in names}
    raw[key] = value
    bounded = _action(raw, names, snapshot)
    assert bounded[key] == expected
    assert raw[key] == value
    assert all(bounded[name] == raw[name] for name in names if name != key)


def test_degree_joint_retains_strict_host_calibration_limit(setup):
    _, client, _, _ = setup
    client.state.payload["_robot_metadata"]["motors"]["arm_left_elbow_flex"]["normalization"] = (
        "degrees"
    )
    names = state_names("alohamini2pro")
    raw = {name: client.state.payload[name] for name in names}
    raw["arm_left_elbow_flex.pos"] = 1000
    with pytest.raises(ValueError, match="joint range"):
        _action(raw, names, client.state)


def test_clipped_action_and_raw_prediction_are_recorded_without_claiming_ack(
    setup, tmp_path, caplog
):
    _, client, policy, _ = setup
    client.acknowledge = False
    original = policy.select_action.side_effect
    policy.select_action.side_effect = lambda s: {**original(s), "arm_left_gripper.pos": -0.01}
    root = tmp_path / "eval"
    dataset = LocalDataset(root, fps=30, task="pick", robot_metadata=policy.robot_metadata)
    dataset.begin_episode()
    try:
        count = run_evaluation(client, policy, "alohamini2pro", duration_s=0.15, dataset=dataset)
    finally:
        dataset.close()
    rows = pq.read_table(root / "episodes/episode_000000/frames.parquet").to_pylist()
    records = [
        json.loads(line)
        for line in (root / "episodes/episode_000000/safety.jsonl").read_text().splitlines()
    ]
    records = [r for r in records if r.get("frame_index") is not None]
    assert len(rows) == len(records) == count
    index = state_names("alohamini2pro").index("arm_left_gripper.pos")
    for row, record in zip(rows, records, strict=True):
        assert row["action"][index] == record["requested_action"]["arm_left_gripper.pos"] == 0
        assert record["policy_action"]["arm_left_gripper.pos"] == -0.01
        assert "accepted_safety" not in record
    assert len([r for r in caplog.records if "targets clipped" in r.message]) == 1


def test_duplicate_camera_frames_do_not_block_chunk_execution_or_create_duplicate_rows(
    setup, tmp_path
):
    clock, client, policy, _ = setup
    from test_dataset import jpeg

    client.state.payload["_robot_metadata"]["cameras"] = ["forward"]
    policy.robot_metadata = copy.deepcopy(client.state.payload["_robot_metadata"])
    client.state.images["forward"] = jpeg()
    client.state.payload["_host_timing"] = {
        "camera_capture_monotonic_s": {"forward": 0.0},
        "state_sample_finished_monotonic_s": 0.0,
    }
    root = tmp_path / "eval"
    dataset = LocalDataset(root, fps=30, task="pick", robot_metadata=policy.robot_metadata)
    dataset.begin_episode()
    try:
        count = run_evaluation(client, policy, "alohamini2pro", duration_s=0.15, dataset=dataset)
    finally:
        dataset.close()
    assert count >= 4
    assert len(pq.read_table(root / "episodes/episode_000000/frames.parquet")) == 1


def test_extra_host_camera_and_recording_skew_do_not_gate_policy(setup, tmp_path):
    _, client, policy, _ = setup
    from test_dataset import jpeg

    client.state.payload["_robot_metadata"]["cameras"] = ["forward", "wrist_right"]
    policy.robot_metadata = copy.deepcopy(client.state.payload["_robot_metadata"])
    policy.manifest = {"cameras": ["forward"]}
    client.state.images = {"forward": jpeg(), "wrist_right": jpeg()}
    client.state.payload["_host_timing"] = {
        "camera_capture_monotonic_s": {"forward": 1.0, "wrist_right": 0.5},
        "state_sample_finished_monotonic_s": 1.0,
    }
    dataset = Mock(robot_metadata=policy.robot_metadata, fps=30)
    assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.15, dataset=dataset) >= 4
    dataset.add_frame.assert_not_called()


def test_entry_restores_three_prefetched_requests(setup):
    _, client, policy, _ = setup
    connection = Mock()
    connection.__enter__ = Mock(return_value=client)
    connection.__exit__ = Mock(return_value=False)
    with patch("alohamini.apps.evaluation.HostClient", return_value=connection) as factory:
        evaluate("pi", "alohamini2pro", policy_factory=lambda: policy, episode_time_s=0.1)
    factory.assert_called_once_with(
        "pi",
        expected_model="alohamini2pro",
        timeout_s=0.2,
        request_window=3,
        prefetch_before_decode=True,
    )


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
