import contextlib
import importlib.util
import io
import json
import os
import tempfile
import threading
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

from test_feetech_device import RegisterSerial

from alohamini.apps.teleoperation import (
    KeyboardInput,
    KeyboardTargets,
    ready_units,
    run_loop,
    stop_owned_robot,
    teleoperate,
)
from alohamini.calibration.servo import MotorCalibration
from alohamini.cli import main
from alohamini.errors import ResponseTimeoutError
from alohamini.hardware.leader import BimanualLeader
from alohamini.model import get_robot_model
from alohamini.protocol import HostSnapshot
from alohamini.schema import CommandIdentity


def snapshot(model="alohamini2pro"):
    motors = {}
    payload = {}
    for motor in get_robot_model(model).actuators:
        if motor.name.startswith("arm_"):
            payload[f"{motor.name}.pos"] = 50.0 if motor.name.endswith("gripper") else 0.0
            motors[motor.name] = {
                "normalization": "range_0_100"
                if motor.name.endswith("gripper")
                else "range_m100_100"
            }
    payload.update(
        {
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
            "lift_axis.height_mm": 100.0,
            "_images": [],
            "_robot_metadata": {
                "schema_version": 1,
                "robot_model": model,
                "motors": motors,
                "lift_axis": {"soft_min_mm": 0.0, "soft_max_mm": 600.0},
            },
            "_safety": {
                "version": 1,
                "feedback_valid": True,
                "lift_reference_valid": True,
                "phase": "ready",
                "fault": None,
                "host_session_id": "session",
                "control_epoch": 0,
                "control_owner": None,
                "joint_holds": {},
            },
        }
    )
    return HostSnapshot(payload, {}, time.monotonic(), time.monotonic())


@unittest.skipUnless(importlib.util.find_spec("scservo_sdk"), "optional STS SDK not installed")
class LeaderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.prepare_model("alohamini2pro")
        self.factory = patch("serial.Serial", side_effect=self.serials.values()).start()
        self.addCleanup(patch.stopall)

    def prepare_model(self, model_name):
        self.serials = {}
        model = get_robot_model(model_name)
        for side in ("left", "right"):
            serial = RegisterSerial()
            template = serial.registers[1].copy()
            serial.registers = {}
            calibration = {}
            for motor in model.actuators:
                if not motor.name.startswith(f"arm_{side}_"):
                    continue
                name = motor.name.removeprefix(f"arm_{side}_")
                i = motor.motor_id
                serial.registers[i] = template.copy()
                entry = MotorCalibration(i, int(side == "right"), -123, 1000, 3000)
                calibration[name] = asdict(entry)
                for address, width, value in (
                    (3, 2, 777),
                    (9, 2, 1000),
                    (11, 2, 3000),
                    (31, 2, entry.offset_register),
                    (33, 1, 0),
                    (56, 2, 1500),
                ):
                    serial.set(i, address, width, value)
            leader_id = "so101_leader_bi" if model_name == "alohamini1" else "am_leader_bi"
            # JSON insertion order is deliberately different from motor/model order.
            (self.directory / f"{leader_id}_{side}.json").write_text(
                json.dumps(dict(reversed(calibration.items())))
            )
            self.serials[side] = serial

    def leader(self, model="alohamini2pro"):
        return BimanualLeader(
            model,
            calibration_dir=self.directory,
            left_port="/dev/test-left",
            right_port="/dev/test-right",
        )

    def test_reuses_calibration_and_reads_only_position_in_model_order(self):
        with self.leader() as leader:
            for serial in self.serials.values():
                serial.requests.clear()
            units = ready_units(snapshot(), "alohamini2pro", "client")
            action = leader.read(units)
            self.assertEqual(list(action), list(units))
            self.assertEqual(action["arm_left_shoulder_pan.pos"], -50.0)
            self.assertEqual(action["arm_right_shoulder_pan.pos"], 50.0)
            self.assertEqual(action["arm_left_gripper.pos"], 25.0)
            self.assertEqual(action["arm_right_gripper.pos"], 75.0)
            for serial in self.serials.values():
                self.assertEqual(len(serial.requests), 1)
                self.assertEqual(serial.requests[0][4:7], bytes([0x82, 56, 2]))

    def test_host_degree_units_are_used_without_double_offset(self):
        with self.leader() as leader:
            units = ready_units(snapshot(), "alohamini2pro", "client")
            units = {k: v if k.endswith("gripper.pos") else "degrees" for k, v in units.items()}
            action = leader.read(units)
            self.assertAlmostEqual(action["arm_left_shoulder_pan.pos"], -500 * 360 / 4095)
            self.assertEqual(
                action["arm_left_shoulder_pan.pos"], action["arm_right_shoulder_pan.pos"]
            )

    def test_all_three_models_choose_expected_leader_ids_and_channels(self):
        for model, count in (("alohamini1", 12), ("alohamini2", 14), ("alohamini2pro", 14)):
            self.prepare_model(model)
            self.factory.side_effect = iter(self.serials.values())
            with self.subTest(model=model), self.leader(model) as leader:
                self.assertEqual(
                    len(leader.read(ready_units(snapshot(model), model, "client"))), count
                )

    def test_connection_and_cleanup_only_write_torque_off(self):
        with self.leader():
            pass
        for serial in self.serials.values():
            writes = [p for p in serial.requests if p[4] in (3, 0x83)]
            self.assertTrue(writes)
            self.assertTrue(all(p[4:7] == bytes([3, 40, 0]) for p in writes))
            self.assertFalse(serial.is_open)

    def test_missing_right_calibration_opens_neither_port(self):
        (self.directory / "am_leader_bi_right.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.leader()
        self.factory.assert_not_called()

    def test_wrong_right_model_closes_both_without_writing_unknown_motor(self):
        self.serials["right"].set(1, 3, 2, 2825)
        with self.assertRaisesRegex(ConnectionError, "model mismatch"):
            with self.leader():
                pass
        self.assertFalse(any(s.is_open for s in self.serials.values()))
        self.assertFalse(any(p[4] == 3 for p in self.serials["right"].requests))

    def test_calibration_mismatch_is_not_silently_written_back(self):
        self.serials["right"].set(3, 31, 2, 0)
        with self.assertRaisesRegex(ValueError, "calibration mismatch"):
            with self.leader():
                pass
        self.assertEqual(self.serials["right"].get(3, 31, 2), 0)
        self.assertFalse(any(s.is_open for s in self.serials.values()))

    def test_missing_corrupt_or_faulted_position_never_becomes_raw_action(self):
        with self.leader() as leader:
            serial = self.serials["right"]
            units = ready_units(snapshot(), "alohamini2pro", "client")
            serial.drop.add((3, 0x82, 56))
            with self.assertRaises(ConnectionError):
                leader.read(units)
            serial.drop.clear()
            serial.errors[(3, 0x82, 56)] = 1
            with self.assertRaises(ConnectionError):
                leader.read(units)
            serial.errors.clear()
            serial.set(3, 56, 2, 0x8001)
            with self.assertRaises(ConnectionError):
                leader.read(units)


class KeyboardTests(unittest.TestCase):
    def test_axis_signs_and_speed_levels(self):
        mapper = KeyboardTargets()
        payload = snapshot().payload
        self.assertEqual(
            mapper.targets(set("wza"), payload, now=0),
            {"x.vel": 0.15, "y.vel": 0.15, "theta.vel": 45, "lift_axis.height_mm": 100},
        )
        action = mapper.targets(set("sxdt"), payload, now=0.02)
        self.assertEqual((action["x.vel"], action["y.vel"], action["theta.vel"]), (-0.2, -0.2, -60))
        mapper.targets({"t"}, payload, now=0.04)
        self.assertEqual(mapper.speed_index, 2)
        self.assertEqual(mapper.targets(set("wszxaduj"), payload, now=0.06)["x.vel"], 0)

    def test_lift_release_reversal_lead_and_limits(self):
        mapper = KeyboardTargets()
        payload = snapshot().payload
        self.assertEqual(mapper.targets({"u"}, payload, now=0)["lift_axis.height_mm"], 103)
        for t in range(1, 50):
            action = mapper.targets({"u"}, payload, now=t * 0.1)
        self.assertEqual(action["lift_axis.height_mm"], 150)
        self.assertEqual(mapper.targets({"j"}, payload, now=5)["lift_axis.height_mm"], 97)
        payload["lift_axis.height_mm"] = 102
        self.assertEqual(mapper.targets(set(), payload, now=5.1)["lift_axis.height_mm"], 102)
        payload["lift_axis.height_mm"] = 599
        self.assertEqual(mapper.targets({"u"}, payload, now=5.2)["lift_axis.height_mm"], 600)
        payload.pop("lift_axis.height_mm")
        with self.assertRaises(KeyError):
            mapper.targets({"j"}, payload, now=5.3)

    def test_press_release_quit_and_dead_listener(self):
        keyboard = KeyboardInput()
        keyboard._listener = Mock(is_alive=Mock(return_value=True))
        keyboard.update("w", True)
        keyboard.update("w", True)
        self.assertEqual(keyboard.read(), {"w"})
        keyboard.update("w", False)
        self.assertEqual(keyboard.read(), set())
        keyboard._listener.is_alive.return_value = False
        with self.assertRaises(RuntimeError):
            keyboard.read()
        keyboard.update("q", True)
        self.assertIsNone(keyboard.read())

    def test_wayland_fails_before_import_or_hardware(self):
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}),
            patch("sys.platform", "linux"),
        ):
            with self.assertRaisesRegex(RuntimeError, "X11"):
                with KeyboardInput():
                    pass


class TeleoperationTests(unittest.TestCase):
    def test_cli_accepts_prior_model_and_id_spelling(self):
        with patch("alohamini.apps.teleoperation.teleoperate") as run:
            self.assertEqual(
                main(
                    [
                        "teleoperate",
                        "--robot.remote_ip",
                        "127.0.0.1",
                        "--robot.robot_model",
                        "alohamini2pro",
                        "--teleop.id",
                        "custom",
                        "--no_keyboard",
                    ]
                ),
                0,
            )
        self.assertEqual(run.call_args.args, ("127.0.0.1", "alohamini2pro"))
        self.assertEqual(run.call_args.kwargs["leader_id"], "custom")

    def test_gate_waits_for_lift_and_owner_but_allows_contact_retreat(self):
        snap = snapshot()
        safety = snap.payload["_safety"]
        safety["joint_holds"] = {"arm_left_shoulder_pan": True}
        self.assertIsNotNone(ready_units(snap, "alohamini2pro", "client"))
        for field, value in (
            ("control_owner", "other"),
            ("lift_reference_valid", False),
            ("feedback_valid", False),
        ):
            old = safety[field]
            safety[field] = value
            self.assertIsNone(ready_units(snap, "alohamini2pro", "client"))
            safety[field] = old
        snap.payload.pop("arm_right_wrist_yaw.pos")
        with self.assertRaises(KeyError):
            ready_units(snap, "alohamini2pro", "client")

    def test_loop_reads_state_then_leader_then_sends_no_images(self):
        stop = threading.Event()
        client, leader = Mock(client_id="client"), Mock()
        client.read.return_value = snapshot()
        leader.read.return_value = {"arm_left_shoulder_pan.pos": 12.0}
        events = []
        client.read.side_effect = lambda: events.append("state") or snapshot()
        leader.read.side_effect = lambda _units: (
            events.append("leader") or {"arm_left_shoulder_pan.pos": 12.0}
        )

        def send(*_args, **_kwargs):
            events.append("send")
            stop.set()
            return CommandIdentity("client", 1, "session", 0)

        client.send_command.side_effect = send
        with patch("alohamini.apps.teleoperation.stop_owned_robot") as cleanup:
            run_loop(client, "alohamini2pro", leader, None, stop_event=stop)
        self.assertEqual(events, ["state", "leader", "send"])
        self.assertEqual(client.send_command.call_args.args[0]["x.vel"], 0)
        cleanup.assert_called_once()

    def test_failed_state_or_leader_read_sends_no_command(self):
        for failure in ("state", "leader"):
            client, leader = Mock(client_id="client"), Mock()
            client.read.return_value = snapshot()
            if failure == "state":
                client.read.side_effect = ResponseTimeoutError("offline")
            else:
                leader.read.side_effect = ConnectionError("no positions")
            error = ResponseTimeoutError if failure == "state" else ConnectionError
            with patch("alohamini.apps.teleoperation.stop_owned_robot"), self.assertRaises(error):
                run_loop(client, "alohamini2pro", leader, None)
            client.send_command.assert_not_called()

    def test_epoch_change_never_relabels_old_actions(self):
        client, leader = Mock(client_id="client"), Mock()
        leader.read.return_value = {}
        changed = snapshot()
        changed.payload["_safety"]["control_epoch"] = 1
        client.read.side_effect = [snapshot(), changed]
        client.send_command.return_value = CommandIdentity("client", 1, "session", 0)
        with (
            patch("alohamini.apps.teleoperation.stop_owned_robot"),
            self.assertRaisesRegex(RuntimeError, "控制权"),
        ):
            run_loop(client, "alohamini2pro", leader, None)
        client.send_command.assert_called_once()
        leader.read.assert_called_once()

    def test_quit_while_waiting_for_host_sends_nothing(self):
        client, keyboard = Mock(client_id="client"), Mock()
        client.read.return_value = snapshot()
        client.read.return_value.payload["_safety"]["lift_reference_valid"] = False
        keyboard.read.return_value = None
        run_loop(client, "alohamini2pro", None, keyboard)
        client.send_command.assert_not_called()

    def test_stop_uses_measured_targets_and_confirms_same_owner(self):
        client = Mock(client_id="client")
        first, confirmed = snapshot(), snapshot()
        first.payload["_safety"]["control_owner"] = "client"
        confirmed.payload["_safety"]["command"] = {"client_id": "client", "sequence": 8}
        client.read.side_effect = [first, confirmed]
        client.send_command.return_value = CommandIdentity("client", 8, "session", 0)
        stop_owned_robot(client, "alohamini2pro", CommandIdentity("client", 7, "session", 0))
        targets = client.send_command.call_args.args[0]
        self.assertEqual(len(targets), 18)
        self.assertEqual(targets["arm_left_gripper.pos"], 50)
        self.assertEqual(targets["x.vel"], 0)
        self.assertEqual(targets["lift_axis.height_mm"], 100)

    def test_stop_does_not_claim_free_or_other_or_restarted_host(self):
        for owner, session, epoch in (
            (None, "session", 0),
            ("other", "session", 0),
            ("client", "restarted", 0),
            ("client", "session", 1),
        ):
            client = Mock(client_id="client")
            snap = snapshot()
            snap.payload["_safety"].update(
                control_owner=owner, host_session_id=session, control_epoch=epoch
            )
            client.read.return_value = snap
            stop_owned_robot(client, "alohamini2pro", CommandIdentity("client", 1, "session", 0))
            client.send_command.assert_not_called()

    def test_invalid_app_settings_fail_before_opening_devices(self):
        with patch("alohamini.apps.teleoperation.BimanualLeader") as leader:
            for options in ({"fps": 0}, {"fps": 60}, {"no_leader": True, "no_keyboard": True}):
                with self.assertRaises(ValueError):
                    teleoperate("127.0.0.1", "alohamini2pro", **options)
            leader.assert_not_called()

    def test_camera_cadence_preview_order_and_no_request_backlog(self):
        from types import SimpleNamespace

        clock = SimpleNamespace(now=0.0, loops=0)
        events, requests = [], []

        def wait(duration):
            clock.now += duration
            clock.loops += 1

        stop = SimpleNamespace(is_set=lambda: clock.loops >= 10, wait=wait)
        client, leader = Mock(client_id="client"), Mock()

        def read(**kwargs):
            requests.append((clock.now, kwargs.get("include_images", False)))
            events.append("read")
            return snapshot()

        client.read.side_effect = read
        leader.read.side_effect = lambda _units: events.append("input") or {}
        client.send_command.side_effect = lambda *_args, **_kwargs: (
            events.append("send") or CommandIdentity("client", 1, "session", 0)
        )

        def preview(*_args):
            events.append("preview")

        with (
            patch("time.perf_counter", side_effect=lambda: clock.now),
            patch("alohamini.apps.teleoperation.stop_owned_robot"),
        ):
            run_loop(
                client,
                "alohamini2pro",
                leader,
                None,
                fps=50,
                camera_fps=30,
                on_frame=preview,
                stop_event=stop,
            )
        self.assertEqual(events, ["read", "input", "send", "preview"] * 10)
        # This is the original teleoperate_bi.py accumulator, not integer decimation.
        next_camera = 0.0
        expected = []
        for stamp, _ in requests:
            request = stamp >= next_camera
            if request:
                while next_camera <= stamp:
                    next_camera += 1 / 30
            expected.append(request)
        self.assertEqual([mode for _, mode in requests], expected)
        self.assertEqual(sum(expected), 6)

    def test_dry_run_never_creates_host_or_fabricates_lift_feedback(self):
        client, keys = Mock(), Mock()
        keys.read.side_effect = [{"u", "w"}, None]
        preview = Mock()
        with (
            patch("alohamini.apps.teleoperation.HostClient", client),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            run_loop(None, "alohamini2pro", None, keys, on_frame=preview)
        client.assert_not_called()
        self.assertIn("[TELEOP ACTION]", output.getvalue())
        self.assertIsNone(preview.call_args.args[0])
        action = preview.call_args.args[1]
        self.assertEqual(action["x.vel"], 0.15)
        self.assertNotIn("lift_axis.height_mm", action)

    def test_no_robot_no_preview_app_skips_host_and_viewer(self):
        with (
            patch("alohamini.apps.teleoperation.HostClient") as client,
            patch("alohamini.apps.teleoperation.KeyboardInput") as keyboard,
            patch("alohamini.apps.teleoperation.run_loop") as loop,
            patch("alohamini.apps.visualization.init_rerun") as viewer,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            teleoperate(
                "127.0.0.1", "alohamini2pro", no_robot=True, no_leader=True, no_preview=True
            )
        client.assert_not_called()
        viewer.assert_not_called()
        self.assertIsNone(loop.call_args.args[0])
        self.assertIsNone(loop.call_args.kwargs["on_frame"])
        keyboard.return_value.__exit__.assert_called_once()

    def test_viewer_cleanup_runs_when_leader_connection_fails(self):
        with (
            patch("alohamini.apps.teleoperation.HostClient") as client,
            patch("alohamini.apps.teleoperation.BimanualLeader") as leader,
            patch("alohamini.apps.visualization.init_rerun"),
            patch("alohamini.apps.visualization.shutdown_rerun") as shutdown,
        ):
            leader.return_value.__enter__.side_effect = OSError("right port failed")
            with self.assertRaisesRegex(OSError, "right port"):
                teleoperate("127.0.0.1", "alohamini2pro", no_keyboard=True)
        shutdown.assert_called_once()
        client.return_value.__exit__.assert_called_once()

    def test_legacy_optional_profile_is_validated_before_io(self):
        with patch("alohamini.apps.teleoperation.BimanualLeader") as leader:
            with self.assertRaisesRegex(ValueError, "profile"):
                teleoperate("127.0.0.1", "alohamini2pro", arm_profile="so-arm-5dof")
            with self.assertRaisesRegex(ValueError, "camera-fps"):
                teleoperate("127.0.0.1", "alohamini2pro", camera_fps=51)
        leader.assert_not_called()

    def test_preview_failure_still_runs_owned_stop(self):
        client, leader = Mock(client_id="client"), Mock()
        client.read.return_value = snapshot()
        leader.read.return_value = {}
        identity = CommandIdentity("client", 1, "session", 0)
        client.send_command.return_value = identity
        with patch("alohamini.apps.teleoperation.stop_owned_robot") as stop:
            with self.assertRaisesRegex(RuntimeError, "preview"):
                run_loop(
                    client,
                    "alohamini2pro",
                    leader,
                    None,
                    on_frame=Mock(side_effect=RuntimeError("preview failed")),
                )
        stop.assert_called_once_with(client, "alohamini2pro", identity)

    def test_cli_reports_missing_calibration_without_traceback(self):
        with (
            patch(
                "alohamini.apps.teleoperation.teleoperate",
                side_effect=FileNotFoundError("calibration"),
            ),
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            self.assertEqual(
                main(["teleoperate", "--host", "127.0.0.1", "--robot_model", "alohamini2pro"]), 1
            )
        self.assertIn("calibration", errors.getvalue())
