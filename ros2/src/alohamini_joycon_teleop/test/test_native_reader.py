import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from alohamini_joycon_teleop import input_log

SCRIPT = Path(__file__).parents[1] / "scripts" / "joycon_native_reader.py"
SPEC = importlib.util.spec_from_file_location("joycon_native_reader", SCRIPT)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def test_stationary_gravity_initializes_level_attitude():
    estimator = reader.ComplementaryAttitudeEstimator()
    orientation = estimator.update(
        [[0.0, 0.0, 0.0]] * 3,
        [[0.0, 0.0, -1.0]] * 3,
        1.0,
    )
    assert orientation == pytest.approx([0.0, 0.0, 0.0])


def test_yaw_uses_measured_elapsed_time_and_remains_relative():
    estimator = reader.ComplementaryAttitudeEstimator()
    estimator.update(
        [[0.0, 0.0, 1.0]] * 3,
        [[0.0, 0.0, -1.0]] * 3,
        1.0,
    )
    first_yaw = estimator.rpy[2]
    estimator.update(
        [[0.0, 0.0, 1.0]] * 3,
        [[0.0, 0.0, -1.0]] * 3,
        1.02,
    )
    assert estimator.rpy[2] - first_yaw == pytest.approx(-0.02)


def test_orientation_quaternion_is_normalized():
    quaternion = reader.euler_xyz_quaternion(0.3, -0.2, 0.7)
    assert sum(value * value for value in quaternion) == pytest.approx(1.0)


def test_frozen_hid_reports_are_not_republished_as_fresh_input(tmp_path):
    publisher = Mock()
    context = Mock()
    context.socket.return_value = publisher
    payloads = [
        {
            "side": "left",
            "report_counter": count,
            "stick": [2000.0, 2000.0],
            "buttons": {"shoulder": False, "sl": False, "sr": False},
        }
        for count in (1, 1, 1, 2)
    ]
    args = SimpleNamespace(
        endpoint="tcp://127.0.0.1:5568",
        rate_hz=50.0,
        sides=["left"],
        skip_imu_calibration=True,
        stick_log=tmp_path / "stick.log",
    )
    controller = Mock()
    with (
        patch.object(reader, "parse_args", return_value=args),
        patch.object(reader.zmq, "Context", return_value=context),
        patch.object(reader, "SingleHidJoyCon", return_value=controller),
        patch.object(reader, "sample", side_effect=payloads),
        patch.object(reader.signal, "signal"),
        patch.object(reader.time, "sleep", side_effect=[None, None, None, KeyboardInterrupt]),
    ):
        with pytest.raises(KeyboardInterrupt):
            reader.main()
    assert publisher.send_string.call_count == 2
    publisher.close.assert_called_once()
    controller.disconnnect.assert_called_once()


def test_replay_marks_samples_and_releases_transport_on_invalid_file(tmp_path):
    source = tmp_path / "input.ndjson"
    source.write_text(json.dumps({"monotonic_ns": 1, "side": "left"}) + "\ninvalid\n")
    context = MagicMock()
    publisher = context.__enter__.return_value.socket.return_value.__enter__.return_value
    with (
        patch.object(input_log.zmq, "Context", return_value=context),
        patch.object(input_log.time, "sleep"),
    ):
        with pytest.raises(json.JSONDecodeError):
            input_log.replay("tcp://127.0.0.1:5568", source, 1.0)
    payload = json.loads(publisher.send_string.call_args.args[0])
    assert payload["replay"] is True and payload["monotonic_ns"] > 1
    context.__exit__.assert_called_once()
