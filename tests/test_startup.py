import importlib.util
import json
import math
import socket
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_feetech_device import DelayedRegisterSerial, RegisterSerial, SimulatedClock

from alohamini.calibration.servo import MotorCalibration, load_motor_calibration
from alohamini.errors import CommandRejectedError
from alohamini.model import get_robot_model
from alohamini.runtime.lifecycle import CommandSubmission, HostPhase
from alohamini.runtime.startup import open_host
from alohamini.schema import CommandIdentity, RobotCommand


def installed_calibration(model):
    return {
        m.name: MotorCalibration(
            m.motor_id,
            int(m.name.startswith("arm_right_")),
            -123 if m.name.startswith("arm_") else 0,
            1000 if m.name.startswith("arm_") and not m.name.endswith("wrist_roll") else 0,
            3000 if m.name.startswith("arm_") and not m.name.endswith("wrist_roll") else 4095,
        )
        for m in model.actuators
    }


class CalibrationFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "robot.json"
        self.model = get_robot_model("alohamini2pro")
        self.values = installed_calibration(self.model)

    def write(self, raw):
        self.path.write_text(json.dumps(raw), encoding="utf-8")

    def test_old_json_format_reuses_names_ids_offsets_and_ranges(self):
        self.write({name: asdict(value) for name, value in reversed(self.values.items())})
        actual = load_motor_calibration(self.path, self.model.actuators)
        self.assertEqual(actual, self.values)
        self.assertEqual(actual["arm_left_shoulder_pan"].offset_register, 0x800 | 123)
        self.assertEqual(actual["arm_right_shoulder_pan"].id, 1)  # IDs are bus-local.

    def test_invalid_fields_types_ranges_ids_and_incomplete_model_rejected(self):
        base = {name: asdict(value) for name, value in self.values.items()}
        for field, value in (
            ("id", True),
            ("id", 3),
            ("drive_mode", 2),
            ("drive_mode", False),
            ("homing_offset", 2048),
            ("homing_offset", -2048),
            ("range_min", 3000),
            ("range_max", 4096),
            ("range_max", 3000.0),
            ("extra", 1),
        ):
            raw = {name: dict(values) for name, values in base.items()}
            raw["arm_left_shoulder_pan"][field] = value
            self.write(raw)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                load_motor_calibration(self.path, self.model.actuators)
        for missing in ("lift_axis", "arm_right_gripper"):
            self.write({name: values for name, values in base.items() if name != missing})
            with self.assertRaises(ValueError):
                load_motor_calibration(self.path, self.model.actuators)

    def test_duplicate_keys_and_oversize_input_rejected(self):
        self.path.write_text('{"id":1,"id":2}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_motor_calibration(self.path, self.model.actuators)
        self.path.write_bytes(b" " * (128 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, "128 KiB"):
            load_motor_calibration(self.path, self.model.actuators)

    def test_wire_encoder_roundtrip_preserves_old_units_without_double_offset(self):
        for drive in (0, 1):
            for lower, upper in ((0, 4095), (1000, 3000)):
                calibration = MotorCalibration(1, drive, -1000, lower, upper)
                encoder = calibration.encoder_calibration()
                for normalization in ("range_m100_100", "range_0_100", "degrees"):
                    units = calibration.position_units(normalization)
                    for tick in (lower, lower + 100, 2048, upper):
                        position = encoder.position_from_tick(tick)
                        self.assertEqual(encoder.position_to_tick(position), tick)
                        self.assertAlmostEqual(position, (tick - 2048) * math.tau / 4096)
                        # Deployed degree / normalized wire conversion truncates to int.
                        self.assertLessEqual(abs(units.to_tick(units.from_tick(tick)) - tick), 1)


@unittest.skipUnless(
    importlib.util.find_spec("scservo_sdk") and importlib.util.find_spec("zmq"),
    "Host dependencies unavailable",
)
class StartupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "AlohaMiniRobot.json"
        self.model = get_robot_model("alohamini2pro")
        self.calibrations = installed_calibration(self.model)
        self.path.write_text(
            json.dumps({name: asdict(c) for name, c in self.calibrations.items()}), encoding="utf-8"
        )
        self.serials, self.events = {}, []
        for bus in ("left", "right"):
            serial = RegisterSerial()
            template = serial.registers[1].copy()
            serial.registers = {}
            for motor in self.model.actuators:
                if motor.bus != bus:
                    continue
                i = motor.motor_id
                serial.registers[i] = template.copy()
                calibration = self.calibrations[motor.name]
                for address, width, value in (
                    (3, 2, {"sts3250": 2825, "sts3095": 2569}[motor.motor_model]),
                    (9, 2, calibration.range_min),
                    (11, 2, calibration.range_max),
                    (31, 2, calibration.offset_register),
                    (33, 1, 0),
                    (55, 1, 1),
                    (7, 1, 250),
                    (21, 1, 32),
                    (41, 1, 254),
                    (46, 2, 999),
                    (44, 2, 99),
                    (56, 2, 2048),
                    (58, 2, 0),
                    (66, 1, 0),
                    (19, 1, 60),
                    (20, 1, 47),
                    (28, 2, 300),
                ):
                    serial.set(i, address, width, value)
            original_write = serial.write

            def record(packet, source=bus, write=original_write):
                self.events.append((source, packet))
                return write(packet)

            serial.write = record
            self.serials[f"/dev/am_arm_follower_{bus}"] = serial
        self.factory = patch("serial.Serial", side_effect=self.make_serial).start()
        self.addCleanup(patch.stopall)
        # Reserve both ports together, then release them for loopback-only ZMQ.
        with socket.socket() as cmd, socket.socket() as state:
            cmd.bind(("127.0.0.1", 0))
            state.bind(("127.0.0.1", 0))
            self.ports = (cmd.getsockname()[1], state.getsockname()[1])

    def make_serial(self, **_kwargs):
        serial = RegisterSerial()

        def open_serial():
            actual = self.serials[serial.port]
            actual.open()
            serial.__dict__ = actual.__dict__

        serial.open = open_serial
        return serial

    def open(self, **kwargs):
        return open_host(
            "alohamini2pro",
            calibration_file=self.path,
            cameras={},
            bind_host="127.0.0.1",
            command_port=self.ports[0],
            state_port=self.ports[1],
            **kwargs,
        )

    def writes(self):
        return [(bus, p) for bus, p in self.events if p[4] in (3, 0x83)]

    def assert_closed(self):
        for serial in self.serials.values():
            self.assertFalse(serial.is_open)
            self.assertTrue(all(serial.get(i, 40, 1) == 0 for i in serial.registers))

    def test_full_startup_initializes_both_buses_and_preserves_protection_and_calibration(self):
        host = self.open()
        try:
            writes = self.writes()
            first_enable = next(n for n, (_, p) in enumerate(writes) if p[5:7] == b"\x28\x01")
            disabled = [(bus, p[2]) for bus, p in writes[:first_enable] if p[5:7] == b"\x28\x00"]
            self.assertEqual(len(disabled), 18)
            self.assertTrue(all(p[5] == 40 for _, p in writes[:18]))
            self.assertFalse(any(p[5] in (9, 11, 19, 20, 28, 31) for _, p in writes))
            for motor in self.model.actuators:
                serial = self.serials[f"/dev/am_arm_follower_{motor.bus}"]
                position = motor.name.startswith("arm_")
                self.assertEqual(serial.get(motor.motor_id, 40, 1), 1)
                self.assertEqual(serial.get(motor.motor_id, 55, 1), 1)
                self.assertEqual(serial.get(motor.motor_id, 33, 1), 0 if position else 1)
                self.assertEqual(serial.get(motor.motor_id, 46, 2), 2000 if position else 0)
                self.assertEqual(serial.get(motor.motor_id, 41, 1), 100 if position else 254)
                self.assertEqual(serial.get(motor.motor_id, 44, 2), 0)
                if position:
                    self.assertEqual(serial.get(motor.motor_id, 42, 2), 2048)
            self.assertEqual(host.control.base_lift.lift_homing_phase, "seeking")
            self.assertIsNone(host.control.base_lift.lift_height_m)
            result = host.step()
            payload = host._payload(result)
            timing = payload["_host_timing"]
            self.assertEqual(
                timing["state_sample_started_monotonic_s"], result.feedback[0].request_started_s
            )
            self.assertEqual(
                timing["state_sample_finished_monotonic_s"], result.feedback[-1].received_s
            )
            self.assertIsInstance(timing["state_sample_unix_ns"], int)
            self.assertIn("host_clock_reference", timing)
            metadata = payload["_robot_metadata"]["motors"]
            self.assertEqual(len(metadata), 18)
            for name, calibration in self.calibrations.items():
                self.assertEqual(metadata[name]["homing_offset"], calibration.homing_offset)
                self.assertEqual(metadata[name]["drive_mode"], calibration.drive_mode)
            names = [key for key in payload if not key.startswith("_")]
            expected = [f"{m.name}.pos" for m in self.model.actuators if m.name.startswith("arm_")]
            self.assertEqual(
                names,
                expected
                + [
                    "x.vel",
                    "y.vel",
                    "theta.vel",
                    "lift_axis.homed",
                    "lift_axis.reference_sequence",
                ],
            )
            self.assertFalse(payload["lift_axis.homed"])
            self.assertEqual(payload["lift_axis.reference_sequence"], 0)
            with self.assertRaisesRegex(CommandRejectedError, "homing"):
                host.control.validate_targets(
                    RobotCommand(positions_rad={"arm_left_shoulder_pan": 0})
                )
        finally:
            host.close()
        self.assert_closed()

    def test_startup_lift_and_cleanup_wrist_recover_from_one_missing_ack(self):
        serial = self.serials["/dev/am_arm_follower_left"]
        reply = serial.reply
        dropped = set()
        targets = {(11, 3, 44)}

        def drop_once(motor_id, instruction, address, data=b""):
            key = (motor_id, instruction, address)
            if key in targets and key not in dropped:
                dropped.add(key)
                return
            reply(motor_id, instruction, address, data)

        with patch.object(serial, "reply", side_effect=drop_once):
            host = self.open()
            try:
                self.assertEqual(serial.get(11, 44, 2), 0)
            finally:
                targets.add((6, 3, 42))
                host.close()
        self.assertEqual(dropped, targets)
        self.assert_closed()

    def test_watchdog_keeps_lift_reference_without_per_joint_register_roundtrips(self):
        from test_host import host_fixture

        clock = SimulatedClock()
        host, serials = host_fixture(
            command_port=self.ports[0], state_port=self.ports[1], lift_encoder_speed=32767
        )
        for serial in serials.values():
            write = serial.write

            def delayed_write(packet, write=write):
                clock.now += 0.004 if packet[4] in (2, 3) else 0.001
                return write(packet)

            serial.write = delayed_write
        with (
            patch("serial.Serial", side_effect=list(serials.values())),
            patch("time.monotonic", side_effect=clock),
        ):
            host.start()
            try:
                identity = CommandIdentity("pc", 0, host.supervisor.status.host_session_id, 0)
                host.supervisor.cycle(
                    CommandSubmission(
                        identity, lambda: host.control.base_lift.establish_lift_reference(0.1)
                    )
                )
                for _ in range(60):
                    clock.now += 0.02
                    for serial in serials.values():
                        serial.requests.clear()
                    result = host.supervisor.cycle()
                    if result.status.watchdog_events:
                        break
                self.assertEqual(result.status.watchdog_events, 1)
                self.assertEqual(result.status.phase, HostPhase.READY)
                self.assertEqual(result.status.control_epoch, 1)
                self.assertAlmostEqual(host.control.base_lift.lift_height_m, 0.1)
                singles = [
                    (p[2], p[4], p[5])
                    for serial in serials.values()
                    for p in serial.requests
                    if p[4] in (2, 3)
                ]
                self.assertEqual(singles, [(11, 3, 46)])
                clock.now += 0.15  # No new command; stationary encoder after a scheduling pause.
                self.assertEqual(host.supervisor.cycle().status.phase, HostPhase.READY)
                self.assertAlmostEqual(host.control.base_lift.lift_height_m, 0.1)
            finally:
                host.close()

    def test_full_startup_and_cleanup_allow_slow_usb_writes_and_fragmented_replies(self):
        from serial import SerialTimeoutException

        clock = SimulatedClock()

        class SlowUSBSerial(DelayedRegisterSerial):
            def write(self, packet):
                if self.write_timeout < 0.012:
                    raise SerialTimeoutException("Write timeout")
                self.clock.now += 0.012
                return super().write(packet)

        def factory(**_kwargs):
            serial = SlowUSBSerial()
            serial.clock = clock

            def open_serial():
                serial.registers = self.serials[serial.port].registers
                self.serials[serial.port] = serial
                serial.is_open = True

            serial.open = open_serial
            return serial

        with (
            patch("serial.Serial", side_effect=factory),
            patch("time.monotonic", side_effect=clock),
        ):
            host = self.open()
            try:
                for serial in self.serials.values():
                    self.assertEqual(serial.write_timeout, 0.005)
                    self.assertTrue(all(serial.get(i, 40, 1) == 1 for i in serial.registers))
                self.assertEqual(host.control.base_lift.lift_homing_phase, "seeking")
            finally:
                host.close()
        self.assert_closed()
        self.assertTrue(all(s.write_timeout == 0.005 for s in self.serials.values()))

    def test_homing_completes_only_after_stop_and_fresh_stationary_samples(self):
        host = self.open()
        serial = self.serials["/dev/am_arm_follower_left"]
        serial.set(11, 69, 2, 50)  # 0.325 A current-contact feedback, not overload.
        try:
            clock = time.monotonic()
            for cycle in range(70):
                with patch("time.monotonic", return_value=clock + cycle * 0.02):
                    result = host.step()
                self.assertIsNot(result.status.phase, HostPhase.FAULT, result.status.fault)
                if host.control.base_lift.lift_homing_phase == "complete":
                    break
            self.assertEqual(host.control.base_lift.lift_homing_phase, "complete")
            self.assertEqual(host.control.base_lift.lift_height_m, 0)
            self.assertEqual(serial.get(11, 46, 2), 0)
            self.assertEqual(serial.get(11, 40, 1), 0)
        finally:
            host.close()
        self.assert_closed()

    def test_mismatched_installed_calibration_never_enables_or_overwrites_it(self):
        self.serials["/dev/am_arm_follower_right"].set(1, 31, 2, 999)
        with self.assertRaisesRegex(ValueError, "calibration mismatch") as failure:
            self.open()
        message = str(failure.exception)
        self.assertIn("actual=999", message)
        self.assertIn("expected=2171", message)
        self.assertIn(str(self.path), message)
        self.assertIn("alohamini calibrate robot", message)
        self.assertFalse(any(p[5:7] == b"\x28\x01" for _, p in self.writes()))
        self.assertEqual(self.serials["/dev/am_arm_follower_right"].get(1, 31, 2), 999)
        self.assert_closed()

    def test_startup_enter_restores_both_buses_before_any_torque_enable(self):
        for serial in self.serials.values():
            serial.set(1, 31, 2, 0x800 | 592)
            serial.set(1, 9, 2, 582)
            serial.set(1, 11, 2, 3454)
        before = self.path.read_bytes()

        def confirm(_prompt):
            for serial in self.serials.values():
                self.assertTrue(all(serial.get(i, 40, 1) == 0 for i in serial.registers))
            self.assertFalse(any(p[4] == 3 and p[5] in (9, 11, 31) for _, p in self.events))
            return ""

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=confirm) as prompt,
        ):
            host = self.open()
        try:
            prompt.assert_called_once()
            self.assertEqual(self.path.read_bytes(), before)
            for serial in self.serials.values():
                self.assertEqual(serial.get(1, 31, 2), 0x800 | 123)
                self.assertEqual(serial.get(1, 9, 2), 1000)
                self.assertEqual(serial.get(1, 11, 2), 3000)
            writes = self.writes()
            enables = [i for i, (_, p) in enumerate(writes) if p[5:7] == b"\x28\x01"]
            restores = [i for i, (_, p) in enumerate(writes) if p[5] in (9, 11, 31)]
            self.assertTrue(enables and restores)
            self.assertLess(max(restores), min(enables))
        finally:
            host.close()

    def test_startup_c_recalibrates_and_rebuilds_host_coordinates(self):
        self.serials["/dev/am_arm_follower_left"].set(1, 31, 2, 999)

        def ranges(bus, names):
            self.assertFalse(any(p[5:7] == b"\x28\x01" for _, p in self.writes()))
            return dict.fromkeys(names, 800), dict.fromkeys(names, 3200)

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=["c", "", ""]) as prompt,
            patch("alohamini.calibration.procedure.record_ranges_of_motion", side_effect=ranges),
        ):
            host = self.open()
        try:
            self.assertEqual(prompt.call_count, 3)
            saved = load_motor_calibration(self.path, self.model.actuators)
            for name, units in host._units.items():
                self.assertEqual(units, saved[name].position_units(units.normalization))
            self.assertEqual(saved["arm_left_shoulder_pan"].range_min, 800)
            self.assertEqual(saved["arm_right_elbow_flex"].range_max, 3200)
            self.assertTrue(list(self.path.parent.glob("*.backup.json")))
        finally:
            host.close()

    def test_startup_c_interruption_never_enables_or_saves(self):
        self.serials["/dev/am_arm_follower_left"].set(1, 31, 2, 999)
        before = self.path.read_bytes()
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=["c", KeyboardInterrupt()]),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.open()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(any(p[5:7] == b"\x28\x01" for _, p in self.writes()))
        self.assert_closed()

    def test_startup_cancel_or_disconnected_terminal_never_restores_or_enables(self):
        for response in ("q", EOFError(), KeyboardInterrupt()):
            self.serials["/dev/am_arm_follower_right"].set(1, 31, 2, 999)
            with (
                self.subTest(response=response),
                patch("sys.stdin.isatty", return_value=True),
                patch(
                    "builtins.input",
                    **(
                        {"return_value": response}
                        if isinstance(response, str)
                        else {"side_effect": response}
                    ),
                ),
                self.assertRaises((InterruptedError, EOFError, KeyboardInterrupt)),
            ):
                self.open()
            self.assertEqual(self.serials["/dev/am_arm_follower_right"].get(1, 31, 2), 999)
            self.assertFalse(
                any(p[5] in (9, 11, 31) or p[5:7] == b"\x28\x01" for _, p in self.writes())
            )
            self.assert_closed()

    def test_startup_right_restore_failure_rolls_back_left_without_enabling(self):
        for serial in self.serials.values():
            serial.set(1, 31, 2, 999)
        self.serials["/dev/am_arm_follower_right"].ignore_writes.add((1, 31))
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value=""),
            self.assertRaisesRegex(ConnectionError, "Register write not accepted"),
        ):
            self.open()
        for serial in self.serials.values():
            self.assertEqual(serial.get(1, 31, 2), 999)
        self.assertFalse(any(p[5:7] == b"\x28\x01" for _, p in self.writes()))
        self.assert_closed()

    def test_host_startup_restores_original_operator_messages(self):
        clock = SimpleNamespace(now=10.0)

        def wait(duration):
            clock.now += max(0.02, duration)

        stop = SimpleNamespace(is_set=lambda: clock.now >= 12.0, wait=wait)
        with (
            patch("time.monotonic", side_effect=lambda: clock.now),
            patch("time.perf_counter", side_effect=lambda: clock.now),
            self.assertLogs(level="INFO") as output,
        ):
            host = self.open()
            self.serials["/dev/am_arm_follower_left"].set(11, 69, 2, 50)
            host.run(stop)
        messages = [record.getMessage() for record in output.records]
        self.assertEqual(
            messages,
            [
                "Configuring AlohaMini",
                "Connecting AlohaMini",
                "Disable torque output (motor will be released)",
                "Lift axis homed to 0mm.",
                "Waiting for commands...",
            ],
        )
        self.assert_closed()

    def test_lost_setting_write_and_partial_enable_both_cleanup_every_bus(self):
        for address in (41, 40):
            with self.subTest(address=address):
                serial = self.serials["/dev/am_arm_follower_right"]
                if address == 41:
                    serial.ignore_writes.add((1, address))
                else:
                    # Accept writes but reject only an enable ACK, not cleanup.
                    original = serial.write

                    def fail_enable(packet, write=original):
                        if packet[2] == 1 and packet[4:7] == b"\x03\x28\x01":
                            raise OSError("enable failed")
                        return write(packet)

                    serial.write = fail_enable
                with self.assertRaises((OSError, RuntimeError)):
                    self.open()
                self.assert_closed()
                serial.ignore_writes.clear()

    def test_missing_file_and_duplicate_bus_are_rejected_before_serial_access(self):
        with self.assertRaisesRegex(ValueError, "different serial"):
            self.open(right_port="/dev/am_arm_follower_left")
        self.path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "--calibration"):
            self.open()
        self.factory.assert_not_called()

    def test_prepared_gripper_and_joint_margins_preserve_wire_direction_and_units(self):
        for degrees in (False, True):
            host = self.open(use_degrees=degrees)
            try:
                for name, spec in host._joints.items():
                    wire = host._units[name]
                    if name.endswith("gripper"):
                        self.assertEqual(wire.normalization, "range_0_100")
                        self.assertEqual(
                            spec.calibration.position_to_tick(spec.contact.closed_position_rad),
                            wire.to_tick(0),
                        )
                        self.assertEqual(
                            spec.calibration.position_to_tick(spec.contact.open_position_rad),
                            wire.to_tick(100),
                        )
                    else:
                        ticks = 4095 / 360 if degrees else (wire.range_max - wire.range_min) / 200
                        self.assertAlmostEqual(
                            spec.contact.release_margin_rad, ticks * math.tau / 4096
                        )
            finally:
                host.close()


if __name__ == "__main__":
    unittest.main()
