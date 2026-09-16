import contextlib
import importlib.util
import io
import json
import socket
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

from test_feetech_device import RegisterSerial

from alohamini.calibration.procedure import calibrate, enter_pressed, record_ranges_of_motion
from alohamini.calibration.servo import (
    MotorCalibration,
    load_motor_calibration,
    save_motor_calibration,
)
from alohamini.hardware.feetech_device import DeviceOperationError, FeetechBusDevice
from alohamini.hardware.leader import BimanualLeader
from alohamini.model import ActuatorSpec, get_robot_model


class CalibrationStorageTests(unittest.TestCase):
    def test_atomic_save_retains_old_json_and_loads_with_existing_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.json"
            motors = [ActuatorSpec("joint", "left", 1, "sts3215")]
            old = {"joint": MotorCalibration(1, 1, -123, 100, 3000)}
            new = {"joint": MotorCalibration(1, 0, 456, 200, 3500)}
            save_motor_calibration(path, old, motors)
            original = path.read_bytes()
            save_motor_calibration(path, new, motors)
            self.assertEqual(load_motor_calibration(path, motors), new)
            backups = list(path.parent.glob("*.backup.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            with patch("alohamini.calibration.servo.os.replace", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    save_motor_calibration(path, old, motors)
            self.assertEqual(load_motor_calibration(path, motors), new)
            self.assertFalse(list(path.parent.glob(".robot.json.*")))
            with self.assertRaises(ValueError):
                save_motor_calibration(path, {}, motors)
            self.assertEqual(load_motor_calibration(path, motors), new)

    def test_terminal_eof_aborts_instead_of_accepting_ranges(self):
        with (
            patch("sys.stdin", io.StringIO("")),
            patch("select.select", return_value=([1], [], [])),
        ):
            with self.assertRaises(EOFError):
                enter_pressed()

    def test_range_table_tracks_actual_extrema_and_rejects_unmoved_joint(self):
        bus = Mock()
        bus.read_positions.side_effect = ({"joint": 2047}, {"joint": 1000}, {"joint": 3000})
        with (
            patch("alohamini.calibration.procedure.enter_pressed", side_effect=(False, True)),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(
                record_ranges_of_motion(bus, ["joint"]), ({"joint": 1000}, {"joint": 3000})
            )
        self.assertIn("NAME            |    MIN |    POS |    MAX", output.getvalue())
        self.assertIn("joint           |   1000 |   3000 |   3000", output.getvalue())
        bus.read_positions.side_effect = None
        bus.read_positions.return_value = {"joint": 2047}
        with patch("alohamini.calibration.procedure.enter_pressed", return_value=True):
            with self.assertRaisesRegex(ValueError, "same min and max"):
                record_ranges_of_motion(bus, ["joint"], display_values=False)


class CalibrationSerial(RegisterSerial):
    """Retain mechanical position when the EEPROM homing offset changes."""

    def write(self, packet):
        motor_id, instruction, address = packet[2], packet[4], packet[5]
        if instruction == 3 and address == 31:
            previous = self.get(motor_id, 31, 2)
            offset = -(previous & 0x7FF) if previous & 0x800 else previous
            physical = (self.get(motor_id, 56, 2) + offset) % 4096
            result = super().write(packet)
            current = self.get(motor_id, 31, 2)
            offset = -(current & 0x7FF) if current & 0x800 else current
            self.set(motor_id, 56, 2, (physical - offset) % 4096)
            return result
        return super().write(packet)


@unittest.skipUnless(importlib.util.find_spec("scservo_sdk"), "Feetech SDK unavailable")
class CalibrationProcedureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.serials = {}
        factory_patch = patch("serial.Serial", side_effect=self.serial_factory)
        factory_patch.start()
        self.addCleanup(factory_patch.stop)
        tty_patch = patch("sys.stdin.isatty", return_value=True)
        tty_patch.start()
        self.addCleanup(tty_patch.stop)

    def serial_factory(self, **_kwargs):
        serials = self.serials

        class PortSelection:
            def __setattr__(self, name, value):
                if name == "port":
                    object.__setattr__(self, "selected", serials[value])
                else:
                    setattr(self.selected, name, value)

            def __getattr__(self, name):
                return getattr(self.selected, name)

        return PortSelection()

    def configure(self, target="leader", model_name="alohamini2pro"):
        self.target, self.model_name = target, model_name
        self.actuators = {}
        self.original = {}
        for side in ("left", "right"):
            motors = [m for m in get_robot_model(model_name).actuators if m.bus == side]
            if target == "leader":
                motors = [
                    ActuatorSpec(m.name.removeprefix(f"arm_{side}_"), side, m.motor_id, "sts3215")
                    for m in motors
                    if m.name.startswith("arm_")
                ]
            self.actuators[side] = motors
            serial = CalibrationSerial()
            template = serial.registers[1].copy()
            serial.registers = {m.motor_id: template.copy() for m in motors}
            for motor in motors:
                for address, width, value in (
                    (3, 2, {"sts3215": 777, "sts3250": 2825, "sts3095": 2569}[motor.motor_model]),
                    (7, 1, 250),
                    (9, 2, 200),
                    (11, 2, 3500),
                    (18, 1, 0x1C),
                    (31, 2, 0x800 | 123),
                    (33, 1, 0),
                    (55, 1, 1),
                    (56, 2, 1700),
                ):
                    serial.set(motor.motor_id, address, width, value)
            self.serials[f"/dev/test-{side}"] = serial
            self.original[side] = {i: bytes(values) for i, values in serial.registers.items()}

    def run_calibration(self, *, response="", ranges=None):
        if ranges is None:

            def ranges(_bus, names):
                return ({name: 1000 for name in names}, {name: 3000 for name in names})

        with (
            patch("builtins.input", return_value=response),
            patch("alohamini.calibration.procedure.record_ranges_of_motion", side_effect=ranges),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            calibrate(
                self.target,
                self.model_name,
                calibration_dir=self.root,
                left_port="/dev/test-left",
                right_port="/dev/test-right",
            )

    def assert_passive_and_closed(self):
        for serial in self.serials.values():
            self.assertFalse(serial.is_open)
            for motor_id in serial.registers:
                self.assertEqual(serial.get(motor_id, 40, 1), 0)
                self.assertEqual(serial.get(motor_id, 55, 1), 1)
            for packet in serial.requests:
                if packet[4] == 3:
                    self.assertNotIn(packet[5], (42, 46))  # No motion targets, including homing.
                    if packet[5] == 40:
                        self.assertEqual(packet[6], 0)

    def assert_restored(self, sides=("left", "right")):
        for side in sides:
            serial = self.serials[f"/dev/test-{side}"]
            for motor_id, expected in self.original[side].items():
                for address, width in ((7, 1), (9, 2), (11, 2), (18, 1), (31, 2), (33, 1)):
                    self.assertEqual(
                        bytes(serial.registers[motor_id][address : address + width]),
                        expected[address : address + width],
                        (side, motor_id, address),
                    )

    def test_leader_files_are_directly_usable_by_native_teleoperation_for_all_models(self):
        for model in ("alohamini1", "alohamini2", "alohamini2pro"):
            self.configure(model_name=model)
            self.run_calibration(response="c")
            leader = BimanualLeader(
                model,
                calibration_dir=self.root,
                left_port="/dev/test-left",
                right_port="/dev/test-right",
            )
            with leader:
                units = {
                    f"arm_{side}_{name}.pos": "range_0_100"
                    if name == "gripper"
                    else "range_m100_100"
                    for side, values in leader.calibrations.items()
                    for name in values
                }
                positions = leader.read(units)
                self.assertEqual(len(positions), 12 if model == "alohamini1" else 14)
            for values in leader.calibrations.values():
                self.assertEqual(values["wrist_roll"].range_min, 0)
                self.assertEqual(values["wrist_roll"].range_max, 4095)
                for entry in values.values():
                    self.assertEqual(entry.homing_offset, 1700 - 123 - 2047)
                    self.assertEqual(entry.drive_mode, 0)
            self.assert_passive_and_closed()

    def test_whole_robot_file_contains_both_arms_and_fixed_base_lift_convention(self):
        self.configure("robot")
        self.run_calibration()
        path = self.root / "AlohaMiniRobot.json"
        values = load_motor_calibration(path, get_robot_model(self.model_name).actuators)
        self.assertEqual(len(values), 18)
        for name, value in values.items():
            if not name.startswith("arm_"):
                self.assertEqual(
                    (value.homing_offset, value.range_min, value.range_max), (0, 0, 4095)
                )
        self.assert_passive_and_closed()

    @unittest.skipUnless(importlib.util.find_spec("zmq"), "ZMQ unavailable")
    def test_saved_robot_calibration_starts_existing_native_host_without_rewriting_it(self):
        from alohamini.runtime.startup import open_host

        self.configure("robot")
        self.run_calibration()
        path = self.root / "AlohaMiniRobot.json"
        before = path.read_bytes()
        with socket.socket() as first, socket.socket() as second:
            first.bind(("127.0.0.1", 0))
            second.bind(("127.0.0.1", 0))
            ports = first.getsockname()[1], second.getsockname()[1]
        for serial in self.serials.values():
            serial.requests.clear()
        host = open_host(
            "alohamini2pro",
            calibration_file=path,
            left_port="/dev/test-left",
            right_port="/dev/test-right",
            cameras={},
            bind_host="127.0.0.1",
            command_port=ports[0],
            state_port=ports[1],
        )
        try:
            payload = host._payload(host.step())
            self.assertTrue(payload["_safety"]["feedback_valid"])
            self.assertFalse(payload["_safety"]["lift_reference_valid"])
            self.assertEqual(len(payload["_robot_metadata"]["motors"]), 18)
            self.assertAlmostEqual(payload["arm_left_shoulder_pan.pos"], 4.7)
            self.assertEqual(path.read_bytes(), before)
            for serial in self.serials.values():
                self.assertFalse(any(p[4] == 3 and p[5] in (9, 11, 31) for p in serial.requests))
        finally:
            host.close()

    def test_invalid_model_profile_ports_or_terminal_fail_before_serial_access(self):
        with patch("alohamini.calibration.procedure.FeetechBusDevice") as device:
            for options in (
                {"arm_profile": "so-arm-5dof"},
                {"left_port": "/dev/same", "right_port": "/dev/same"},
                {"device_id": "../wrong"},
            ):
                with self.assertRaises(ValueError):
                    calibrate("leader", "alohamini2pro", **options)
            with patch("sys.stdin.isatty", return_value=False):
                with self.assertRaises(RuntimeError):
                    calibrate("leader", "alohamini2pro")
            device.assert_not_called()

    def test_reuse_preserves_software_inversion_and_does_not_rewrite_json(self):
        self.configure("robot")
        values = {
            m.name: MotorCalibration(m.motor_id, 1, -200, 300, 3400)
            for motors in self.actuators.values()
            for m in motors
        }
        path = self.root / "AlohaMiniRobot.json"
        original = json.dumps({k: asdict(v) for k, v in values.items()}).encode()
        path.write_bytes(original)
        self.run_calibration(ranges=Mock(side_effect=AssertionError("Must not collect new ranges")))
        self.assertEqual(path.read_bytes(), original)
        for serial in self.serials.values():
            self.assertTrue(all(serial.get(i, 31, 2) == (0x800 | 200) for i in serial.registers))
        self.assert_passive_and_closed()

    def test_abort_on_right_arm_restores_both_buses_and_creates_no_robot_file(self):
        self.configure("robot")

        def ranges(bus, names):
            if bus.actuators[0].bus == "right":
                raise KeyboardInterrupt
            return ({name: 1000 for name in names}, {name: 3000 for name in names})

        with self.assertRaises(KeyboardInterrupt):
            self.run_calibration(ranges=ranges)
        self.assert_restored()
        self.assertFalse(list(self.root.iterdir()))
        self.assert_passive_and_closed()

    def test_save_failure_restores_robot_eeprom_and_keeps_existing_file(self):
        self.configure("robot")
        values = {
            m.name: MotorCalibration(m.motor_id, 1, -123, 200, 3500)
            for motors in self.actuators.values()
            for m in motors
        }
        path = self.root / "AlohaMiniRobot.json"
        save_motor_calibration(path, values, get_robot_model(self.model_name).actuators)
        original = path.read_bytes()
        with patch("alohamini.calibration.servo.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.run_calibration(response="c")
        self.assertEqual(path.read_bytes(), original)
        self.assert_restored()
        self.assert_passive_and_closed()

    def test_ignored_write_aborts_and_relocks_every_motor(self):
        self.configure("robot")
        self.serials["/dev/test-right"].ignore_writes.add((1, 31))
        with self.assertRaises(ConnectionError):
            self.run_calibration()
        self.assert_restored()
        self.assert_passive_and_closed()

    def test_overflowing_half_turn_offset_is_rejected_not_sign_flipped(self):
        self.configure("robot")
        self.serials["/dev/test-left"].set(
            1, 56, 2, 122
        )  # Actual position 4095 after offset reset.
        with self.assertRaisesRegex(ValueError, "homing_offset"):
            self.run_calibration()
        self.assert_restored()
        self.assert_passive_and_closed()

    def test_missing_right_motor_does_not_write_any_calibration(self):
        self.configure("robot")
        self.serials["/dev/test-right"].set(1, 3, 2, 777)
        with self.assertRaises(ConnectionError):
            self.run_calibration()
        for serial in self.serials.values():
            self.assertFalse(serial.is_open)
            self.assertFalse(any(p[4] == 3 and p[5] != 40 for p in serial.requests))
        self.assertFalse(list(self.root.iterdir()))

    def test_configured_host_bus_cannot_open_a_calibration_transaction(self):
        self.configure()
        bus = FeetechBusDevice(
            "/dev/test-left",
            self.actuators["left"],
            position_calibrations={},
            velocity_limits={m.name: 1 for m in self.actuators["left"]},
        )
        with self.assertRaises(RuntimeError):
            with bus.calibration_session():
                self.fail("Host bus must not enter calibration")

    def test_unconfigured_bus_has_no_motion_capability_even_after_calibration(self):
        self.configure()
        bus = FeetechBusDevice(
            "/dev/test-left",
            self.actuators["left"],
            position_calibrations={},
            velocity_limits={},
        )
        try:
            bus.connect("test-session")
            bus.disable_torque()
            with bus.calibration_session():
                bus.configure_calibration([m.name for m in bus.actuators])
            with self.assertRaises(ConnectionError):
                bus.write_targets(positions_rad={"shoulder_pan": 0}, velocities_raw={})
            with self.assertRaises(RuntimeError):
                bus.enable_torque()
            with self.assertRaises(RuntimeError):
                bus.prepare({})
            with self.assertRaises(RuntimeError):
                bus.write_calibration({})
        finally:
            bus.close()

    def test_restore_failure_is_reported_and_does_not_skip_other_motors(self):
        self.configure("robot")

        def ranges(bus, _names):
            if bus.actuators[0].bus == "left":
                self.serials["/dev/test-left"].ignore_writes.add((1, 31))
                raise OSError("feedback lost")

        with self.assertRaisesRegex(DeviceOperationError, "restore arm_left_shoulder_pan"):
            self.run_calibration(ranges=ranges)
        self.assert_passive_and_closed()
        self.assert_restored(("right",))
        serial = self.serials["/dev/test-left"]
        self.assertEqual(serial.get(2, 31, 2), 0x800 | 123)
