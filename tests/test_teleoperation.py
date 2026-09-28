import contextlib
import importlib.util
import io
import json
import math
import os
import tempfile
import threading
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

from test_feetech_device import DelayedRegisterSerial, RegisterSerial, SimulatedClock

from alohamini.apps.teleoperation import (
    KeyboardInput,
    KeyboardTargets,
    _same_control_session,
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

    def test_connection_and_cleanup_never_write_motion_or_calibration_targets(self):
        with self.leader():
            pass
        for serial in self.serials.values():
            writes = [p for p in serial.requests if p[4] in (3, 0x83)]
            self.assertTrue(writes)
            self.assertTrue(all(p[4] == 3 and p[5] in (40, 41, 85) for p in writes))
            self.assertTrue(all(p[6] == 0 for p in writes if p[5] == 40))
            self.assertFalse(serial.is_open)

    def test_passive_connection_does_not_read_follower_motion_profile(self):
        for serial in self.serials.values():
            for motor_id in serial.registers:
                for address in (21, 22, 23, 46):
                    serial.errors[(motor_id, 2, address)] = 0x01
        with self.leader():
            for serial in self.serials.values():
                reads = {p[5] for p in serial.requests if p[4] == 2}
                self.assertTrue(reads.isdisjoint({21, 22, 23, 46}))

    def test_passive_preparation_restores_source_sampling_configuration(self):
        for serial in self.serials.values():
            for motor_id in serial.registers:
                serial.set(motor_id, 7, 1, 250)
                serial.set(motor_id, 18, 1, 0x1C)
                serial.set(motor_id, 33, 1, 1)
        with self.leader():
            for serial in self.serials.values():
                for motor_id in serial.registers:
                    for address, expected in ((7, 0), (18, 0x0C), (33, 0), (85, 254), (41, 254)):
                        self.assertEqual(serial.get(motor_id, address, 1), expected)
                    self.assertEqual(serial.get(motor_id, 40, 1), 0)

    def test_passive_acceleration_uses_acknowledged_writes_without_readback(self):
        for address in (85, 41):
            self.serials["right"].errors[(7, 2, address)] = 0x01
        with self.leader() as leader:
            self.assertEqual(
                len(leader.read(ready_units(snapshot(), "alohamini2pro", "client"))), 14
            )
            for serial in self.serials.values():
                profile_packets = [p for p in serial.requests if p[5] in (85, 41)]
                self.assertEqual(len(profile_packets), 2 * len(serial.registers))
                self.assertTrue(all(p[4] == 3 and p[6] == 254 for p in profile_packets))
                for motor_id in serial.registers:
                    self.assertEqual(serial.get(motor_id, 40, 1), 0)

    def test_passive_acceleration_write_alarms_still_abort_and_close_both_ports(self):
        for address in (85, 41):
            with self.subTest(address=address):
                self.prepare_model("alohamini2pro")
                self.factory.side_effect = iter(self.serials.values())
                self.serials["right"].errors[(7, 3, address)] = 0x01
                with self.assertRaisesRegex(
                    ConnectionError, f"gripper.*Write {address}=254.*servo_error=0x01"
                ):
                    with self.leader():
                        self.fail("Faulted write must not establish a leader connection")
                self.assertFalse(any(s.is_open for s in self.serials.values()))
                for serial in self.serials.values():
                    for motor_id in serial.registers:
                        self.assertEqual(serial.get(motor_id, 40, 1), 0)

    def test_passive_firmware_mismatch_fails_before_configuration(self):
        self.serials["right"].set(6, 0, 1, 99)
        with self.assertRaisesRegex(ConnectionError, "firmware"):
            with self.leader():
                pass
        self.assertFalse(any(s.is_open for s in self.serials.values()))
        self.assertFalse(any(p[4] == 3 for p in self.serials["right"].requests))

    def test_startup_voltage_alarm_on_firmware_and_phase_remains_explicit(self):
        for address in (1, 18):
            with self.subTest(address=address):
                self.prepare_model("alohamini2pro")
                self.factory.side_effect = iter(self.serials.values())
                self.serials["right"].errors[(7, 2, address)] = 0x01
                with self.assertRaisesRegex(
                    ConnectionError, f"gripper.*Read {address}.*Input voltage error"
                ):
                    with self.leader():
                        self.fail("A startup register alarm must not be hidden")
                self.assertFalse(any(s.is_open for s in self.serials.values()))

    def test_voltage_only_torque_off_readback_allows_passive_leader_connection(self):
        self.serials["right"].errors[(6, 2, 40)] = 0x01
        with self.leader() as leader:
            self.assertEqual(
                len(leader.read(ready_units(snapshot(), "alohamini2pro", "client"))), 14
            )
            self.assertEqual(self.serials["right"].get(6, 40, 1), 0)
        self.assertFalse(any(s.is_open for s in self.serials.values()))

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

    def test_leader_enter_restores_selected_files_without_enabling_torque(self):
        originals = {p: p.read_bytes() for p in self.directory.glob("*.json")}
        for serial in self.serials.values():
            serial.set(1, 31, 2, 999)
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="") as prompt,
        ):
            with self.leader() as leader:
                self.assertEqual(prompt.call_count, 2)
                self.assertEqual(
                    len(leader.read(ready_units(snapshot(), "alohamini2pro", "client"))), 14
                )
                for serial in self.serials.values():
                    self.assertEqual(serial.get(1, 31, 2), 0x800 | 123)
                    self.assertTrue(all(serial.get(i, 40, 1) == 0 for i in serial.registers))
                    self.assertFalse(any(p[4] == 3 and p[5] in (42, 46) for p in serial.requests))
        self.assertEqual({p: p.read_bytes() for p in originals}, originals)

    def test_leader_c_recalibrates_only_selected_arm_and_reloads_ranges(self):
        for serial in self.serials.values():
            serial.set(1, 31, 2, 999)
        right_path = self.directory / "am_leader_bi_right.json"
        before = right_path.read_bytes()
        self.factory.side_effect = list(self.serials.values()) * 2
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=["c", "", ""]) as prompt,
            patch(
                "alohamini.calibration.procedure.record_ranges_of_motion",
                side_effect=lambda bus, names: (
                    dict.fromkeys(names, 800),
                    dict.fromkeys(names, 3200),
                ),
            ) as ranges,
        ):
            with self.leader() as leader:
                self.assertEqual(prompt.call_count, 3)
                ranges.assert_called_once()
                self.assertEqual(leader.calibrations["left"]["shoulder_pan"].range_min, 800)
                for side, device in leader.devices.items():
                    for name, encoder in device.position_calibrations.items():
                        self.assertEqual(
                            encoder, leader.calibrations[side][name].encoder_calibration()
                        )
                values = leader.read(ready_units(snapshot(), "alohamini2pro", "client"))
                self.assertAlmostEqual(values["arm_left_shoulder_pan.pos"], (1500 - 800) / 12 - 100)
        self.assertEqual(right_path.read_bytes(), before)
        for serial in self.serials.values():
            self.assertFalse(serial.is_open)
            self.assertTrue(all(serial.get(i, 40, 1) == 0 for i in serial.registers))
            self.assertFalse(any(p[4] == 3 and p[5] in (42, 46) for p in serial.requests))

    def test_leader_cancel_second_file_does_not_restore_first_bus(self):
        for serial in self.serials.values():
            serial.set(1, 31, 2, 999)
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=["", "q"]),
            self.assertRaises(InterruptedError),
        ):
            with self.leader():
                pass
        for serial in self.serials.values():
            self.assertEqual(serial.get(1, 31, 2), 999)
            self.assertFalse(serial.is_open)
            self.assertFalse(any(p[4] == 3 and p[5] in (9, 11, 31) for p in serial.requests))

    def test_missing_corrupt_or_faulted_position_never_becomes_raw_action(self):
        with self.leader() as leader:
            serial = self.serials["right"]
            units = ready_units(snapshot(), "alohamini2pro", "client")
            serial.drop.add((3, 0x82, 56))
            with self.assertRaises(ConnectionError):
                leader.read(units)
            serial.drop.clear()
            expected = leader.read(units)
            serial.errors[(3, 0x82, 56)] = 1
            self.assertEqual(leader.read(units), expected)
            serial.errors[(3, 0x82, 56)] = 2
            with self.assertRaises(ConnectionError):
                leader.read(units)
            serial.errors.clear()
            serial.set(3, 56, 2, 0x8001)
            with self.assertRaises(ConnectionError):
                leader.read(units)

    def test_passive_query_allows_delayed_write_and_restores_serial_timeout(self):
        with self.leader() as leader:
            serial = self.serials["left"]
            write = serial.write
            clock = SimulatedClock()

            def delayed_write(packet):
                self.assertEqual(packet[4:7], bytes([0x82, 56, 2]))
                self.assertGreater(serial.write_timeout, 0.012)
                clock.now += 0.012
                return write(packet)

            with (
                patch.object(serial, "write", side_effect=delayed_write),
                patch("time.monotonic", side_effect=clock),
            ):
                self.assertEqual(len(leader.devices["left"].read_positions()), 7)
            self.assertEqual(serial.write_timeout, 0.005)

    def test_passive_query_accepts_delayed_replies(self):
        with self.leader() as leader:
            serial = self.serials["left"]
            delayed = DelayedRegisterSerial()
            delayed.registers = serial.registers
            with (
                patch.object(serial, "write", side_effect=delayed.write),
                patch.object(serial, "read", side_effect=delayed.read),
                patch.object(serial, "reset_input_buffer", side_effect=delayed.reset_input_buffer),
                patch("time.monotonic", side_effect=delayed.clock),
            ):
                self.assertEqual(len(leader.devices["left"].read_positions()), 7)
            self.assertGreater(delayed.clock(), 0.012)
            self.assertLess(delayed.clock(), 0.050)
            self.assertEqual(len(delayed.requests), 1)

    def test_passive_query_retries_write_timeout_without_motion_packets(self):
        from serial import SerialTimeoutException

        with self.leader() as leader:
            serial = self.serials["left"]
            write = serial.write
            attempts = []

            def transient_timeout(packet):
                attempts.append(packet)
                if len(attempts) == 1:
                    raise SerialTimeoutException("Write timeout")
                return write(packet)

            with patch.object(serial, "write", side_effect=transient_timeout):
                self.assertEqual(len(leader.devices["left"].read_positions()), 7)
            self.assertEqual(len(attempts), 2)
            self.assertTrue(all(p[4:7] == bytes([0x82, 56, 2]) for p in attempts))
            self.assertEqual(serial.write_timeout, 0.005)

    def test_passive_query_retries_entire_group_without_merging_partial_samples(self):
        with self.leader() as leader:
            serial = self.serials["left"]
            write = serial.write
            attempts = []

            def partial_then_fresh(packet):
                attempts.append(packet)
                serial.drop.clear()
                if len(attempts) == 1:
                    serial.drop.add((3, 0x82, 56))
                else:
                    for motor_id in serial.registers:
                        serial.set(motor_id, 56, 2, 1600)
                return write(packet)

            with patch.object(serial, "write", side_effect=partial_then_fresh):
                positions = leader.devices["left"].read_positions()
            self.assertEqual(set(positions.values()), {1600})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0], attempts[1])

    def test_passive_query_bounds_timeout_retries_and_reports_bus_context(self):
        from serial import SerialTimeoutException

        with self.leader() as leader:
            serial = self.serials["left"]
            clock = SimulatedClock()

            def timeout(_packet):
                clock.now += serial.write_timeout
                raise SerialTimeoutException("Write timeout")

            with (
                patch.object(serial, "write", side_effect=timeout) as write,
                patch("time.monotonic", side_effect=clock),
                self.assertRaisesRegex(
                    ConnectionError,
                    r"/dev/test-left \[left\] SyncRead Present_Position.*IDs.*after 4 attempt",
                ),
            ):
                leader.devices["left"].read_positions()
            self.assertEqual(write.call_count, 4)
            self.assertAlmostEqual(clock(), 0.2)
            self.assertEqual(serial.write_timeout, 0.005)

    def test_passive_servo_fault_is_not_retried(self):
        with self.leader() as leader:
            serial = self.serials["left"]
            serial.requests.clear()
            serial.errors[(3, 0x82, 56)] = 0x02
            with self.assertRaises(ConnectionError):
                leader.devices["left"].read_positions()
            self.assertEqual(len(serial.requests), 1)

    def test_cleanup_failure_preserves_original_error_and_closes_both_ports(self):
        leader = self.leader()
        with self.assertRaisesRegex(RuntimeError, "original read failure; Leader cleanup failed"):
            with leader:
                leader._resources.callback(Mock(side_effect=ConnectionError("cleanup failure")))
                raise ConnectionError("original read failure")
        self.assertFalse(any(s.is_open for s in self.serials.values()))

    def test_cleanup_failure_does_not_replace_keyboard_interrupt(self):
        leader = self.leader()
        with self.assertLogs(level="ERROR"), self.assertRaises(KeyboardInterrupt):
            with leader:
                leader._resources.callback(Mock(side_effect=ConnectionError("cleanup failure")))
                raise KeyboardInterrupt
        self.assertFalse(any(s.is_open for s in self.serials.values()))


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
    def test_skipped_send_resamples_input_and_preserves_last_identity_for_cleanup(self):
        client, leader = Mock(client_id="client"), Mock()
        stop, on_frame = threading.Event(), Mock()
        client.read.side_effect = lambda: snapshot()
        leader.read.side_effect = [{"arm_left_gripper.pos": v} for v in (10, 20, 30, 40)]
        first = CommandIdentity("client", 1, "session", 0)
        last = CommandIdentity("client", 3, "session", 0)
        results = iter([first, None, last, None])

        def send(*args, **kwargs):
            result = next(results)
            if client.send_command.call_count == 4:
                stop.set()
            return result

        client.send_command.side_effect = send
        with patch("alohamini.apps.teleoperation.stop_owned_robot") as cleanup:
            run_loop(client, "alohamini2pro", leader, None, stop_event=stop, on_frame=on_frame)
        self.assertEqual(
            [call.args[0]["arm_left_gripper.pos"] for call in client.send_command.call_args_list],
            [10, 20, 30, 40],
        )
        self.assertEqual(on_frame.call_count, 2)
        cleanup.assert_called_once_with(client, "alohamini2pro", last)

    def watchdog_states(self):
        before, stopped = snapshot(), snapshot()
        before.payload["_safety"].update(
            phase="active", control_owner="client", watchdog_events=2, joint_hold_events=0
        )
        stopped.payload["_safety"].update(
            control_epoch=1, watchdog_events=3, watchdog_active=True, joint_hold_events=0
        )
        return before, stopped

    def test_same_host_watchdog_changes_are_recoverable_without_counting_epochs(self):
        before, stopped = self.watchdog_states()
        self.assertTrue(_same_control_session(before, stopped, "client"))
        for field, value in (
            ("control_epoch", 2),
            ("watchdog_events", 4),
            ("watchdog_active", False),
            ("joint_hold_events", 1),
            ("joint_holds", {"joint": {}}),
        ):
            with self.subTest(field=field):
                _, candidate = self.watchdog_states()
                candidate.payload["_safety"][field] = value
                self.assertTrue(_same_control_session(before, candidate, "client"))
        for field, value in (("host_session_id", "restarted"), ("control_owner", "other")):
            _, candidate = self.watchdog_states()
            candidate.payload["_safety"][field] = value
            self.assertFalse(_same_control_session(before, candidate, "client"))
        before.payload["_safety"]["control_owner"] = None
        self.assertTrue(_same_control_session(before, stopped, "client"))
        before.payload["_safety"]["control_owner"] = "other"
        self.assertFalse(_same_control_session(before, stopped, "client"))

    def test_loop_resamples_leader_after_watchdog_without_confirmation(self):
        before, stopped = self.watchdog_states()
        client, keyboard, leader = Mock(client_id="client"), Mock(), Mock()
        event = threading.Event()
        client.read.side_effect = [before, ResponseTimeoutError("late"), stopped]
        keyboard.read.return_value = set()
        leader.read.side_effect = [{"arm_left_gripper.pos": v} for v in (10, 15, 20)]

        def send(_action, *, based_on):
            if based_on is stopped:
                event.set()
            return CommandIdentity(
                "client",
                client.send_command.call_count,
                "session",
                based_on.payload["_safety"]["control_epoch"],
            )

        client.send_command.side_effect = send
        with patch("alohamini.apps.teleoperation.stop_owned_robot"):
            run_loop(client, "alohamini2pro", leader, keyboard, stop_event=event)
        self.assertEqual(client.send_command.call_count, 3)
        self.assertEqual(leader.read.call_count, 3)
        client.refresh.assert_not_called()
        self.assertEqual(client.send_command.call_args.args[0]["arm_left_gripper.pos"], 20)
        self.assertEqual(client.send_command.call_args.args[0]["x.vel"], 0)

    def test_watchdog_recovery_does_not_require_keyboard(self):
        before, stopped = self.watchdog_states()
        client, leader = Mock(client_id="client"), Mock()
        event = threading.Event()
        client.read.side_effect = [before, ResponseTimeoutError("late"), stopped]
        leader.read.return_value = {"arm_left_gripper.pos": 10}

        def send(_action, *, based_on):
            if based_on is stopped:
                event.set()
            return CommandIdentity(
                "client",
                client.send_command.call_count,
                "session",
                based_on.payload["_safety"]["control_epoch"],
            )

        client.send_command.side_effect = send
        with patch("alohamini.apps.teleoperation.stop_owned_robot"):
            run_loop(client, "alohamini2pro", leader, None, stop_event=event)
        self.assertEqual(client.send_command.call_count, 3)
        self.assertEqual(leader.read.call_count, 3)

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
        client.prefetch.side_effect = lambda **kwargs: events.append("prefetch")
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
        self.assertEqual(events, ["state", "prefetch", "leader", "send"])
        self.assertEqual(client.send_command.call_args.args[0]["x.vel"], 0)
        cleanup.assert_called_once()

    def test_prefetch_overlaps_leader_read_with_next_host_cycle(self):
        from types import SimpleNamespace

        clock = SimulatedClock()
        sent = []
        pending = None
        client, leader = Mock(client_id="client"), Mock()

        def prefetch(**kwargs):
            nonlocal pending
            self.assertFalse(kwargs.get("include_images", False))
            if pending is None:
                # A 50 Hz Host polls at cycle start and replies after 6 ms of I/O.
                reply_at = (math.floor((clock.now + 1e-9) / 0.02) + 1) * 0.02 + 0.006
                pending = (clock.now, reply_at)

        def read():
            nonlocal pending
            prefetch()
            started, reply_at = pending
            clock.now = max(clock.now, reply_at)
            pending = None
            result = snapshot()
            result.request_started_s = started
            return result

        def read_leader(_units):
            clock.now += 0.016
            return {"arm_left_shoulder_pan.pos": 12.0}

        def send(*args, **kwargs):
            sent.append(clock.now)
            return CommandIdentity("client", len(sent), "session", 0)

        stop = SimpleNamespace(
            is_set=lambda: len(sent) >= 5,
            wait=lambda duration: setattr(clock, "now", clock.now + duration),
        )
        client.read.side_effect = read
        client.prefetch.side_effect = prefetch
        client.send_command.side_effect = send
        leader.read.side_effect = read_leader
        with (
            patch("time.monotonic", side_effect=clock),
            patch("time.perf_counter", side_effect=clock),
            patch("alohamini.apps.teleoperation.stop_owned_robot"),
        ):
            run_loop(client, "alohamini2pro", leader, None, stop_event=stop)
        for previous, current in zip(sent, sent[1:], strict=False):
            self.assertAlmostEqual(current - previous, 0.02)

    def test_preview_never_requests_images_on_control_connection(self):
        from types import SimpleNamespace

        clock = SimulatedClock()
        pending = None
        modes, sent = [], []
        client, leader = Mock(client_id="client"), Mock()

        def prefetch(*, include_images=False):
            nonlocal pending
            self.assertIsNone(pending)
            pending = include_images

        def read(*, include_images=False):
            nonlocal pending
            if pending is not None:
                self.assertEqual(include_images, pending)
            modes.append(include_images)
            pending = None
            return snapshot()

        def read_leader(_units):
            clock.now += 0.04 if len(modes) == 1 else 0.001
            return {}

        def send(*args, **kwargs):
            sent.append(clock.now)
            return CommandIdentity("client", len(sent), "session", 0)

        client.read.side_effect = read
        client.prefetch.side_effect = prefetch
        client.send_command.side_effect = send
        leader.read.side_effect = read_leader
        stop = SimpleNamespace(
            is_set=lambda: len(sent) >= 4,
            wait=lambda duration: setattr(clock, "now", clock.now + duration),
        )
        with (
            patch("time.monotonic", side_effect=clock),
            patch("time.perf_counter", side_effect=clock),
            patch("alohamini.apps.teleoperation.stop_owned_robot"),
        ):
            run_loop(client, "alohamini2pro", leader, None, on_frame=Mock(), stop_event=stop)
        self.assertEqual(modes, [False] * 4)

    def test_failed_state_or_leader_read_sends_no_command(self):
        from alohamini.errors import ProtocolError

        for failure in ("state", "leader"):
            client, leader = Mock(client_id="client"), Mock()
            client.read.return_value = snapshot()
            if failure == "state":
                client.read.side_effect = ProtocolError("invalid response")
            else:
                leader.read.side_effect = ConnectionError("no positions")
            error = ProtocolError if failure == "state" else ConnectionError
            with patch("alohamini.apps.teleoperation.stop_owned_robot"), self.assertRaises(error):
                run_loop(client, "alohamini2pro", leader, None)
            client.send_command.assert_not_called()

    def test_leader_read_longer_than_watchdog_drops_action_and_resamples(self):
        clock = SimulatedClock()
        stop = threading.Event()
        client, leader = Mock(client_id="client"), Mock()
        inputs = []
        client.read.side_effect = lambda **kwargs: snapshot()

        def read(_units):
            inputs.append(clock())
            if len(inputs) == 1:
                clock.now += 1.1
            return {"arm_left_gripper.pos": len(inputs) * 10}

        def send(*_args, **_kwargs):
            stop.set()
            return CommandIdentity("client", 1, "session", 0)

        leader.read.side_effect = read
        client.send_command.side_effect = send
        with (
            patch("time.monotonic", side_effect=clock),
            patch("alohamini.apps.teleoperation.stop_owned_robot"),
        ):
            run_loop(client, "alohamini2pro", leader, None, stop_event=stop)
        self.assertEqual(leader.read.call_count, 2)
        client.send_command.assert_called_once()
        self.assertEqual(client.send_command.call_args.args[0]["arm_left_gripper.pos"], 20)

    def test_brief_timeout_reads_new_targets_using_bounded_cached_feedback(self):
        stop = threading.Event()
        client, leader, preview = Mock(client_id="client"), Mock(), Mock()
        first, recovered = snapshot(), snapshot()
        client.read.side_effect = [first, ResponseTimeoutError("late"), recovered]
        leader.read.side_effect = [{"arm_left_gripper.pos": v} for v in (10, 15, 20)]

        def send(*_args, **kwargs):
            if kwargs["based_on"] is recovered:
                stop.set()
            return CommandIdentity("client", 1, "session", 0)

        client.send_command.side_effect = send
        with patch("alohamini.apps.teleoperation.stop_owned_robot"):
            run_loop(client, "alohamini2pro", leader, None, on_frame=preview, stop_event=stop)
        self.assertEqual(client.read.call_count, 3)
        self.assertEqual(client.read.call_args_list[-1].kwargs, {})
        self.assertEqual(leader.read.call_count, 3)
        self.assertEqual(client.send_command.call_count, 3)
        self.assertIs(client.send_command.call_args.kwargs["based_on"], recovered)
        self.assertEqual(client.send_command.call_args.args[0]["arm_left_gripper.pos"], 20)
        self.assertEqual(preview.call_count, 3)

    def test_quit_during_repeated_timeouts_sends_nothing(self):
        client, leader, keyboard = Mock(client_id="client"), Mock(), Mock()
        client.read.side_effect = ResponseTimeoutError("offline")
        keyboard.read.side_effect = [set(), set(), None]
        with patch("alohamini.apps.teleoperation.stop_owned_robot"):
            run_loop(client, "alohamini2pro", leader, keyboard)
        self.assertEqual(client.read.call_count, 3)
        leader.read.assert_not_called()
        client.send_command.assert_not_called()

    def test_transient_unready_feedback_waits_and_resamples_instead_of_exiting(self):
        client, leader = Mock(client_id="client"), Mock()
        first, unavailable, recovered = snapshot(), snapshot(), snapshot()
        unavailable.payload["_safety"]["feedback_valid"] = False
        client.read.side_effect = [first, unavailable, recovered]
        leader.read.side_effect = [{"arm_left_gripper.pos": 10}, {"arm_left_gripper.pos": 20}]
        stop = threading.Event()

        def send(_action, *, based_on):
            if based_on is recovered:
                stop.set()
            return CommandIdentity("client", client.send_command.call_count, "session", 0)

        client.send_command.side_effect = send
        with patch("alohamini.apps.teleoperation.stop_owned_robot"):
            run_loop(client, "alohamini2pro", leader, None, stop_event=stop)
        self.assertEqual(client.send_command.call_count, 2)
        self.assertEqual(leader.read.call_count, 2)

    def test_single_timeout_does_not_print_host_stopped_warning(self):
        client, leader = Mock(client_id="client"), Mock()
        client.read.side_effect = [ResponseTimeoutError("temporary"), snapshot()]
        leader.read.return_value = {}
        stop = threading.Event()

        def send(*args, **kwargs):
            stop.set()
            return CommandIdentity("client", 1, "session", 0)

        client.send_command.side_effect = send
        with (
            patch("alohamini.apps.teleoperation.stop_owned_robot"),
            patch("alohamini.apps.teleoperation.logger.warning") as warning,
        ):
            run_loop(client, "alohamini2pro", leader, None, stop_event=stop)
        warning.assert_not_called()
        client.send_command.assert_called_once()

    def test_foreign_owner_never_relabels_old_actions(self):
        client, leader = Mock(client_id="client"), Mock()
        leader.read.return_value = {}
        changed = snapshot()
        changed.payload["_safety"]["control_epoch"] = 1
        changed.payload["_safety"]["control_owner"] = "other"
        client.read.side_effect = [snapshot(), ResponseTimeoutError("late"), changed]
        client.send_command.return_value = CommandIdentity("client", 1, "session", 0)
        with (
            patch("alohamini.apps.teleoperation.stop_owned_robot"),
            self.assertRaisesRegex(RuntimeError, "控制状态"),
        ):
            run_loop(client, "alohamini2pro", leader, None)
        self.assertEqual(client.send_command.call_count, 2)
        self.assertEqual(leader.read.call_count, 2)

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

    def test_stop_supersedes_inflight_first_command_on_same_lease(self):
        client = Mock(client_id="client")
        idle, confirmed = snapshot(), snapshot()
        idle.payload["_safety"]["control_owner"] = None
        confirmed.payload["_safety"]["command"] = {"client_id": "client", "sequence": 2}
        client.read.side_effect = [idle, confirmed]
        client.send_command.return_value = CommandIdentity("client", 2, "session", 0)
        stop_owned_robot(client, "alohamini2pro", CommandIdentity("client", 1, "session", 0))
        client.send_command.assert_called_once()
        self.assertEqual(client.send_command.call_args.args[0]["x.vel"], 0)

    def test_stop_without_prior_command_does_not_claim_idle_host(self):
        client = Mock(client_id="client")
        stop_owned_robot(client, "alohamini2pro", None)
        client.read.assert_not_called()
        client.send_command.assert_not_called()

    def test_unsent_stop_does_not_wait_for_a_nonexistent_ack(self):
        client = Mock(client_id="client")
        client.read.return_value = snapshot()
        client.send_command.return_value = None
        with self.assertLogs("alohamini.apps.teleoperation", level="WARNING") as logs:
            stop_owned_robot(client, "alohamini2pro", CommandIdentity("client", 1, "session", 0))
        client.read.assert_called_once()
        self.assertIn("停止目标未发送", logs.output[0])

    def test_stop_does_not_claim_other_or_restarted_host(self):
        for owner, session, epoch in (
            (None, "restarted", 0),
            (None, "session", 1),
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

    def test_control_cadence_and_preview_submission_order(self):
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
        self.assertEqual([mode for _, mode in requests], [False] * 10)

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

    def test_control_handshake_failure_precedes_leader_connection_and_preview(self):
        with (
            patch("alohamini.apps.teleoperation.HostClient") as client,
            patch("alohamini.apps.teleoperation.BimanualLeader") as leader,
            patch("alohamini.apps.visualization.init_rerun") as preview,
        ):
            client.return_value.__enter__.return_value.connect_control.side_effect = (
                ResponseTimeoutError("offline")
            )
            with self.assertRaises(ResponseTimeoutError):
                teleoperate("127.0.0.1", "alohamini2pro", no_keyboard=True)
        leader.return_value.__enter__.assert_not_called()
        preview.assert_not_called()

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
