import csv
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from test_replay import replay_snapshot

from alohamini.apps.tracking import TrackingAnalysis, TrackingLog, target_tick
from alohamini.calibration.encoder import HostPositionUnits
from alohamini.cli import main


def payload(sequence=1, *, command_at=0.021, sample_at=0.020, command_tick=2000):
    value = replay_snapshot().payload
    value["_host_timing"] = {"state_sequence": sequence}
    safety = value["_safety"]
    safety.update(
        accepted_at_monotonic_s=command_at,
        accepted_targets={},
        requested_targets={},
        joint_holds={},
        command={"sequence": sequence},
    )
    motors = {}
    for name, info in value["_robot_metadata"]["motors"].items():
        if not name.startswith("arm_"):
            continue
        info.update(range_min=1000, range_max=3000, drive_mode=0)
        target = (command_tick - 1000) / 10 - 100
        safety["accepted_targets"][f"{name}.pos"] = target
        safety["requested_targets"][f"{name}.pos"] = target
        motors[name] = dict(
            position_raw=1900,
            velocity_raw=300,
            current_ma=650,
            sample_started_s=sample_at,
            sample_finished_s=sample_at + 0.001,
            packet_error=0,
            field_errors={},
        )
    value["_motor_feedback"] = {"motors": motors}
    return value


def test_new_target_after_feedback_is_not_scored_against_that_feedback():
    analysis = TrackingAnalysis()
    first = analysis.rows(payload())
    assert len(first) == 12
    assert all(row["pairing"] == "unavailable" and row["error_deg"] is None for row in first)
    second = analysis.rows(payload(2, sample_at=0.04, command_at=0.042, command_tick=2100))
    assert all(row["pairing"] == "previous_cycle" for row in second)
    assert all(row["command_tick"] == 2000 and row["command_sequence"] == 1 for row in second)
    assert all(row["error_deg"] == pytest.approx(100 * 360 / 4096) for row in second)


def test_gap_does_not_guess_intervening_commands_but_old_confirmed_target_is_usable():
    analysis = TrackingAnalysis()
    analysis.rows(payload())
    unknown = analysis.rows(payload(4, sample_at=0.08, command_at=0.082))
    assert all(row["error_deg"] is None for row in unknown)
    known = analysis.rows(payload(8, sample_at=0.16, command_at=0.10))
    assert all(row["pairing"] == "current" and row["error_deg"] is not None for row in known)


@pytest.mark.parametrize("change", ["session", "epoch", "calibration", "owner"])
def test_context_change_never_reuses_previous_target(change):
    analysis = TrackingAnalysis()
    analysis.rows(payload())
    current = payload(2, sample_at=0.04, command_at=0.042)
    if change == "calibration":
        current["_robot_metadata"]["motors"]["arm_left_elbow_flex"]["range_min"] += 1
    else:
        field = {"session": "host_session_id", "epoch": "control_epoch", "owner": "control_owner"}[
            change
        ]
        current["_safety"][field] = 1 if change == "epoch" else "changed"
    assert all(row["error_deg"] is None for row in analysis.rows(current))


def test_duplicate_samples_are_not_counted_twice():
    analysis = TrackingAnalysis()
    first = payload(command_at=0)
    analysis.rows(first)
    assert analysis.rows(deepcopy(first)) == []
    assert all(row["samples"] == 1 for row in analysis.summary())


@pytest.mark.parametrize("normalization", ["range_m100_100", "degrees"])
@pytest.mark.parametrize("direction", [0, 1])
def test_reported_ticks_round_trip_without_false_stationary_error(normalization, direction):
    units = HostPositionUnits(normalization, 456, 3210, direction)
    for tick in (456, 999, 1199, 2000, 2999, 3210):
        assert target_tick(units, units.from_tick(tick)) == tick


def test_current_masks_holds_and_calibrated_units_are_preserved():
    analysis = TrackingAnalysis()
    value = payload(command_at=0)
    name = "arm_left_elbow_flex"
    value["_safety"]["joint_holds"] = {name: 0}
    value["_safety"]["requested_targets"][f"{name}.pos"] = 10
    value["_motor_feedback"]["motors"][name]["field_errors"] = {"current_ma": "missing"}
    row = next(row for row in analysis.rows(value) if row["joint"] == name)
    assert row["joint_hold"]
    assert row["current_a"] is None
    assert row["target_reduction_deg"] == pytest.approx(100 * 360 / 4096)
    assert row["error_deg"] == pytest.approx(100 * 360 / 4096)
    summary = next(row for row in analysis.summary() if row["joint"] == name)
    assert summary["hold_samples"] == 1
    assert summary["mean_current_a"] is None


@pytest.mark.parametrize("invalid", ["packet", "position", "feedback", "wrap"])
def test_invalid_feedback_and_ambiguous_wrap_do_not_enter_error_statistics(invalid):
    value = payload(command_at=0)
    sample = value["_motor_feedback"]["motors"]["arm_left_elbow_flex"]
    if invalid == "packet":
        sample["packet_error"] = 1
    elif invalid == "position":
        sample["field_errors"] = {"position_raw": "missing"}
    elif invalid == "wrap":
        sample["position_raw"] = 4095
    else:
        value["_safety"]["feedback_valid"] = False
    rows = TrackingAnalysis().rows(value)
    assert next(row for row in rows if row["joint"] == "arm_left_elbow_flex")["error_deg"] is None


def test_writer_preserves_samples_and_summary_without_robot_access(tmp_path):
    with patch("alohamini.client.HostClient") as client:
        log = TrackingLog(tmp_path / "tracking")
        log.submit(payload(command_at=0))
        log.close()
    client.assert_not_called()
    assert log.error is None
    with (log.root / "samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (log.root / "summary.csv").open() as stream:
        summaries = list(csv.DictReader(stream))
    assert len(rows) == len(summaries) == 12
    assert all(float(row["mae_deg"]) == pytest.approx(100 * 360 / 4096) for row in summaries)
    assert all(row["scored_samples"] == "1" for row in summaries)
    with pytest.raises(FileExistsError):
        TrackingLog(log.root)


def test_full_queue_drops_diagnostics_without_blocking():
    with patch("alohamini.apps.tracking.threading.Thread"):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            log = TrackingLog(Path(directory) / "tracking")
            sample = payload()
            for _ in range(130):
                log.submit(sample)
            assert log.queue.qsize() == 128
            assert log.dropped == 2


def test_tracking_option_is_forwarded_without_changing_control_defaults():
    with patch("alohamini.apps.teleoperation.teleoperate") as teleop:
        assert main(["teleoperate", "--robot_model", "alohamini2pro", "--tracking"]) == 0
    assert teleop.call_args.kwargs["tracking"] is True
    assert teleop.call_args.kwargs["fps"] == 50


def test_loop_uses_existing_read_and_does_not_send_diagnostic_commands():
    from alohamini.apps.teleoperation import run_loop

    client = Mock(client_id="pc")
    client.read.return_value = replay_snapshot()
    keyboard = Mock()
    keyboard.read.return_value = None  # Exit after one existing observation.
    tracking = Mock()
    run_loop(client, "alohamini2pro", None, keyboard, tracking=tracking)
    client.read.assert_called_once()
    client.send_command.assert_not_called()
    tracking.submit.assert_called_once_with(client.read.return_value.payload)


def test_diagnostic_failure_does_not_touch_robot_and_is_reported(tmp_path):
    log = TrackingLog(tmp_path / "tracking")
    log.submit({"invalid": "snapshot"})
    log.close()
    assert log.error is not None
    assert (log.root / "samples.csv").exists()
    assert not (log.root / "summary.csv").exists()
