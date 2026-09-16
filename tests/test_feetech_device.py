import importlib.util
import math
import threading
import time
import unittest
from unittest.mock import patch

from alohamini.calibration import EncoderCalibration
from alohamini.hardware.feetech_device import DeviceOperationError, FeetechBusDevice
from alohamini.model import ActuatorSpec
from alohamini.runtime.lifecycle import CommandSubmission, HostPhase, HostSupervisor
from alohamini.schema import CommandIdentity


class RegisterSerial:
    """In-memory STS register device; SDK handles all actual packet framing."""

    def __init__(self):
        self.is_open = False
        self.baudrate, self.bytesize, self.parity, self.stopbits = 1_000_000, 8, "N", 1
        self.timeout, self.write_timeout = 0, 0.005
        self.pending = b""
        self.requests = []
        self.errors = {}
        self.drop = set()
        self.ignore_writes = set()
        self.registers = {motor_id: bytearray(96) for motor_id in (1, 2)}
        for motor_id in self.registers:
            for address, width, value in (
                (3, 2, 2825),
                (33, 1, 0 if motor_id == 1 else 1),
                (46, 2, 2000 if motor_id == 1 else 1000),
                (41, 1, 100),
                (21, 1, 16),
                (23, 1, 0),
                (22, 1, 32),
                (40, 1, 1),
                (42, 2, 3000),
                (56, 2, 1234),
                (69, 2, 10),
            ):
                self.set(motor_id, address, width, value)

    def set(self, motor_id, address, width, value):
        self.registers[motor_id][address : address + width] = value.to_bytes(width, "little")

    def get(self, motor_id, address, width):
        return int.from_bytes(self.registers[motor_id][address : address + width], "little")

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def reset_input_buffer(self):
        self.pending = b""

    def read(self, size):
        result, self.pending = self.pending[:size], self.pending[size:]
        return result

    def reply(self, motor_id, instruction, address, data=b""):
        key = (motor_id, instruction, address)
        if key in self.drop:
            return
        error = self.errors.get(key, 0)
        content = bytes([motor_id, len(data) + 2, error]) + bytes(data)
        self.pending += b"\xff\xff" + content + bytes([~sum(content) & 255])

    def write(self, packet):
        assert packet[:2] == b"\xff\xff"
        assert len(packet) == packet[3] + 4
        assert packet[-1] == ~sum(packet[2:-1]) & 255
        self.requests.append(packet)
        motor_id, instruction, address = packet[2], packet[4], packet[5]
        if instruction == 2:
            width = packet[6]
            self.reply(
                motor_id, instruction, address, self.registers[motor_id][address : address + width]
            )
        elif instruction == 3:
            data = packet[6:-1]
            if (motor_id, address) not in self.ignore_writes:
                self.registers[motor_id][address : address + len(data)] = data
            self.reply(motor_id, instruction, address)
        elif instruction == 0x82:
            width = packet[6]
            for motor_id in reversed(packet[7:-1]):
                self.reply(
                    motor_id,
                    instruction,
                    address,
                    self.registers[motor_id][address : address + width],
                )
        elif instruction == 0x83:
            width, data = packet[6], packet[7:-1]
            for offset in range(0, len(data), width + 1):
                motor_id = data[offset]
                self.registers[motor_id][address : address + width] = data[
                    offset + 1 : offset + 1 + width
                ]
        else:
            raise AssertionError(f"Unexpected instruction: {instruction}")
        return len(packet)


@unittest.skipUnless(importlib.util.find_spec("scservo_sdk"), "optional Feetech SDK not installed")
class FeetechDeviceTests(unittest.TestCase):
    def setUp(self):
        self.serial = RegisterSerial()
        self.factory = patch("serial.Serial", return_value=self.serial).start()
        self.addCleanup(patch.stopall)
        self.actuators = [
            ActuatorSpec("joint", "left", 1, "sts3250"),
            ActuatorSpec("wheel", "left", 2, "sts3250"),
        ]
        self.calibration = EncoderCalibration(
            4096, 2048, 0, 1, position_min_rad=-1, position_max_rad=1
        )
        self.device = self.new_device()

    def new_device(self, **kwargs):
        return FeetechBusDevice(
            "/dev/test-only",
            self.actuators,
            position_calibrations=kwargs.get("position_calibrations", {"joint": self.calibration}),
            velocity_limits=kwargs.get("velocity_limits", {"wheel": 3000}),
        )

    def connect(self):
        self.device.connect("session")
        self.serial.requests.clear()

    def writes(self):
        return [p for p in self.serial.requests if p[4] in (3, 0x83)]

    def test_construction_has_no_device_access_and_connect_is_read_only(self):
        self.factory.assert_not_called()
        self.device.connect("session")
        self.assertTrue(self.serial.is_open)
        self.assertFalse(self.writes())
        self.assertTrue(self.factory.call_args.kwargs["exclusive"])
        self.assertEqual(self.factory.call_args.kwargs["timeout"], 0)
        self.assertEqual(self.factory.call_args.kwargs["write_timeout"], 0.005)

    def test_read_feedback_uses_own_session_and_bus(self):
        self.connect()
        batch = self.device.read_feedback()
        self.assertEqual((batch.clock_id, batch.source_id), ("session", "left"))
        self.assertEqual(set(batch.samples), {"joint", "wheel"})

    def test_position_and_signed_velocity_packets_match_existing_register_format(self):
        self.connect()
        self.device.write_targets(
            positions_rad={"joint": math.pi / 8}, velocities_raw={"wheel": -1300}
        )
        self.assertEqual(self.serial.get(1, 42, 2), 2304)
        self.assertEqual(self.serial.get(2, 46, 2), 0x8000 | 1300)
        writes = self.writes()
        self.assertEqual([p[4:7] for p in writes], [bytes([0x83, 42, 2]), bytes([0x83, 46, 2])])
        self.assertEqual(writes[0][7:-1], bytes([1, 0, 9]))
        self.assertEqual(writes[1][7:-1], bytes([2, 0x14, 0x85]))

    def test_invalid_velocity_does_not_allow_partial_position_write(self):
        self.connect()
        for velocity in (3001, -3001, True, 1.5):
            with self.assertRaises(ValueError):
                self.device.write_targets(
                    positions_rad={"joint": 0}, velocities_raw={"wheel": velocity}
                )
        self.assertFalse(self.writes())

    def test_unknown_or_out_of_bounds_targets_rejected_before_io(self):
        self.connect()
        for positions, velocities in (
            ({"joint": 2}, {}),
            ({"joint": float("nan")}, {}),
            ({"wheel": 0}, {}),
            ({}, {"joint": 1}),
            ({}, {}),
        ):
            with self.assertRaises(ValueError):
                self.device.write_targets(positions_rad=positions, velocities_raw=velocities)
        self.assertFalse(self.serial.requests)

    def test_stop_zeroes_velocity_first_and_holds_new_feedback_without_clamping(self):
        self.connect()
        self.serial.set(
            1, 56, 2, 500
        )  # Outside command limits; actual hold must not jump to a bound.
        self.device.stop()
        self.assertEqual(self.serial.get(2, 46, 2), 0)
        self.assertEqual(self.serial.get(1, 42, 2), 500)
        self.assertEqual([(p[2], p[5]) for p in self.writes()], [(2, 46), (1, 42)])
        self.assertEqual(self.serial.get(1, 46, 2), 2000)  # Zero must never replace arm Profile.
        self.assertEqual(self.serial.get(1, 40, 1), 1)  # Stop never enables or disables torque.

    def test_position_feedback_loss_never_reuses_last_target(self):
        self.connect()
        self.device.read_feedback()
        self.serial.drop.add((1, 0x82, 56))
        self.serial.drop.add((1, 2, 56))
        with self.assertRaises(DeviceOperationError):
            self.device.stop()
        self.assertEqual(self.serial.get(2, 46, 2), 0)
        self.assertFalse(any(p[5] == 42 for p in self.writes()))

    def test_optional_block_failure_falls_back_to_fresh_required_registers(self):
        self.connect()
        self.serial.drop.add((1, 0x82, 56))
        first = self.device.read_feedback()
        self.assertFalse(first.failures)
        self.assertEqual(set(first.samples["joint"].registers), {"position_raw", "current_raw"})
        self.assertIn("temperature_raw", first.samples["joint"].field_errors)
        self.serial.set(1, 56, 2, 1500)
        self.serial.set(1, 69, 2, 200)
        self.serial.requests.clear()
        second = self.device.read_feedback()
        self.assertEqual(second.samples["joint"].registers["position_raw"], 1500)
        self.assertEqual(second.samples["joint"].current_a, 1.3)
        requests = [p for p in self.serial.requests if p[4] == 0x82]
        self.assertEqual([list(p[7:-1]) for p in requests], [[2]])
        self.assertFalse(self.writes())

    def test_critical_fallback_failure_is_not_an_optional_telemetry_error(self):
        self.connect()
        self.serial.drop.update({(1, 0x82, 56), (1, 2, 69)})
        start = time.monotonic()
        feedback = self.device.read_feedback()
        self.assertLess(time.monotonic() - start, 0.1)
        self.assertIn("joint", feedback.failures)
        self.assertNotIn("joint", feedback.samples)
        self.assertIn("wheel", feedback.samples)

    def test_block_probe_recovers_extra_fields_after_backoff(self):
        self.connect()
        self.serial.drop.add((1, 0x82, 56))
        first = self.device.read_feedback()
        self.assertNotIn("temperature_raw", first.samples["joint"].registers)
        self.serial.drop.clear()
        due = self.device._feedback_retry["joint"][0]
        with patch("time.monotonic", return_value=due + 0.001):
            recovered = self.device.read_feedback()
        self.assertIn("temperature_raw", recovered.samples["joint"].registers)
        self.assertFalse(self.device._feedback_retry)

    def test_servo_fault_packet_never_becomes_healthy_via_fallback(self):
        self.connect()
        self.serial.errors[(1, 0x82, 56)] = 1
        batch = self.device.read_feedback()
        self.assertIn("joint", batch.failures)
        self.assertEqual(batch.samples["joint"].packet_error, 1)
        self.assertFalse(any(p[4] == 2 and p[5] == 69 for p in self.serial.requests))

    def test_faulted_feedback_never_becomes_position_hold(self):
        self.connect()
        self.serial.errors[(1, 0x82, 56)] = 1
        with self.assertRaises(DeviceOperationError):
            self.device.stop()
        self.assertFalse(any(p[5] == 42 for p in self.writes()))

    def test_stop_write_ack_is_not_enough_without_target_readback(self):
        self.connect()
        self.serial.ignore_writes.add((2, 46))
        with self.assertRaises(DeviceOperationError) as error:
            self.device.stop()
        self.assertIn("wheel", error.exception.errors)
        self.assertEqual(self.serial.get(1, 42, 2), 1234)

    def test_disable_attempts_every_motor_even_after_error(self):
        self.connect()
        self.serial.errors[(1, 3, 40)] = 1
        with self.assertRaises(DeviceOperationError):
            self.device.disable_torque()
        self.assertEqual([(p[2], p[5]) for p in self.writes()], [(1, 40), (2, 40)])
        self.assertEqual(self.serial.get(2, 40, 1), 0)
        self.assertFalse(any(p[5] == 55 for p in self.writes()))
        with self.assertRaises(ConnectionError):
            self.device.write_targets(positions_rad={"joint": 0}, velocities_raw={})

    def test_disable_checks_readback(self):
        self.connect()
        self.serial.ignore_writes.add((1, 40))
        with self.assertRaises(DeviceOperationError):
            self.device.disable_torque()
        self.assertEqual(self.serial.get(2, 40, 1), 0)

    def test_interrupt_during_disable_still_attempts_next_motor(self):
        self.connect()
        serial_write = self.serial.write

        def interrupted_write(packet):
            if packet[2] == 1 and packet[4] == 3 and packet[5] == 40:
                raise KeyboardInterrupt()
            return serial_write(packet)

        with patch.object(self.serial, "write", side_effect=interrupted_write):
            with self.assertRaises(KeyboardInterrupt):
                self.device.disable_torque()
        self.assertEqual(self.serial.get(2, 40, 1), 0)

    def test_interrupt_during_velocity_stop_still_attempts_position_hold(self):
        self.connect()
        serial_write = self.serial.write

        def interrupted_write(packet):
            if packet[2] == 2 and packet[4] == 3 and packet[5] == 46:
                raise KeyboardInterrupt()
            return serial_write(packet)

        with patch.object(self.serial, "write", side_effect=interrupted_write):
            with self.assertRaises(KeyboardInterrupt):
                self.device.stop()
        self.assertEqual(self.serial.get(1, 42, 2), 1234)

    def test_wrong_model_is_never_written_even_during_cleanup(self):
        self.serial.set(1, 3, 2, 123)
        host = HostSupervisor({"left": self.device}, self.actuators)
        with self.assertRaises(ConnectionError):
            host.start()
        self.assertFalse(self.writes())
        self.assertFalse(self.serial.is_open)
        self.assertEqual(host.status.phase, HostPhase.FAULT)

    def test_wrong_mode_is_not_given_a_stop_target(self):
        self.serial.set(1, 33, 1, 1)
        host = HostSupervisor({"left": self.device}, self.actuators)
        host.start()
        self.assertFalse(self.writes())
        with self.assertRaises(ConnectionError):
            self.device.write_targets(positions_rad={"joint": 0}, velocities_raw={})
        host.close()
        self.assertTrue(all(p[5] == 40 for p in self.writes() if p[2] == 1))
        self.assertFalse(self.serial.is_open)

    def test_profile_mismatch_does_not_reconfigure_motor(self):
        for address, width, value in (
            (46, 2, 1000),
            (41, 1, 254),
            (21, 1, 32),
            (23, 1, 1),
            (22, 1, 64),
        ):
            self.serial = RegisterSerial()
            self.factory.return_value = self.serial
            self.serial.set(1, address, width, value)
            device = self.new_device()
            device.connect("session")
            with self.assertRaises(ConnectionError):
                device.write_targets(positions_rad={"joint": 0}, velocities_raw={})
            self.assertFalse(self.writes())
            device.close()

    def test_all_supported_motor_model_numbers(self):
        for model, number in (("sts3215", 777), ("sts3250", 2825), ("sts3095", 2569)):
            self.serial = RegisterSerial()
            self.factory.return_value = self.serial
            for motor_id in (1, 2):
                self.serial.set(motor_id, 3, 2, number)
            self.actuators = [
                ActuatorSpec("joint", "left", 1, model),
                ActuatorSpec("wheel", "left", 2, model),
            ]
            device = self.new_device()
            device.connect("session")
            self.assertFalse(self.writes())
            self.assertFalse(device.read_feedback().failures)
            device.close()

    def test_short_serial_write_is_reported(self):
        self.connect()
        with patch.object(self.serial, "write", return_value=0):
            with self.assertRaises(ConnectionError):
                self.device.write_targets(positions_rad={"joint": 0}, velocities_raw={})

    def test_partial_open_is_closed_by_supervisor(self):
        host = HostSupervisor({"left": self.device}, self.actuators)

        def partial_open():
            self.serial.is_open = True
            raise OSError("open failed after allocation")

        with patch.object(self.serial, "open", side_effect=partial_open):
            with self.assertRaises(OSError):
                host.start()
        self.assertFalse(self.serial.is_open)
        self.assertFalse(self.writes())

    def test_cross_thread_writes_rejected(self):
        self.connect()
        errors = []

        def write():
            try:
                self.device.write_targets(positions_rad={"joint": 0}, velocities_raw={})
            except RuntimeError as exc:
                errors.append(exc)

        thread = threading.Thread(target=write)
        thread.start()
        thread.join(timeout=1)
        self.assertEqual(len(errors), 1)
        self.assertFalse(self.writes())

    def test_reconnect_requires_new_device_and_close_is_idempotent(self):
        self.connect()
        self.device.close()
        self.device.close()
        with self.assertRaises(RuntimeError):
            self.device.connect("new-session")

    def test_configuration_validated_before_serial_access(self):
        for kwargs in (
            {"velocity_limits": {"wheel": 0}},
            {"velocity_limits": {}},
            {"position_calibrations": {"joint": EncoderCalibration(4096, 0, 0, 1)}},
        ):
            with self.assertRaises(ValueError):
                self.new_device(**kwargs)
        self.factory.assert_not_called()

    def test_supervisor_uses_concrete_backend_for_command_stop_disable_and_close(self):
        host = HostSupervisor({"left": self.device}, self.actuators)
        with host:
            identity = CommandIdentity("pc", 0, host.status.host_session_id, 0)
            submission = CommandSubmission(
                identity,
                lambda: self.device.write_targets(
                    positions_rad={"joint": 0}, velocities_raw={"wheel": 200}
                ),
            )
            result = host.cycle(submission)
            self.assertTrue(result.command_applied)
            self.assertEqual(self.serial.get(2, 46, 2), 200)
        self.assertEqual(host.status.phase, HostPhase.CLOSED)
        self.assertEqual(self.serial.get(2, 46, 2), 0)
        self.assertEqual(self.serial.get(1, 42, 2), 1234)
        self.assertTrue(all(self.serial.get(i, 40, 1) == 0 for i in (1, 2)))
        self.assertFalse(self.serial.is_open)

    def test_watchdog_uses_concrete_stop_and_rejects_old_command(self):
        host = HostSupervisor({"left": self.device}, self.actuators)
        with host:
            identity = CommandIdentity("pc", 0, host.status.host_session_id, 0)
            write = lambda: self.device.write_targets(  # noqa: E731
                positions_rad={"joint": 0}, velocities_raw={"wheel": 200}
            )
            host.cycle(CommandSubmission(identity, write))
            clock = time.monotonic
            with patch(
                "alohamini.runtime.lifecycle.time.monotonic", side_effect=lambda: clock() + 2
            ):
                result = host.cycle(CommandSubmission(identity, write))
            self.assertFalse(result.command_applied)
            self.assertEqual(result.status.control_epoch, 1)
            self.assertEqual(result.status.phase, HostPhase.READY)
            self.assertEqual(self.serial.get(1, 42, 2), 1234)
            self.assertEqual(self.serial.get(1, 46, 2), 2000)
            self.assertEqual(self.serial.get(2, 46, 2), 0)

    def test_partial_group_write_failure_reaches_concrete_cleanup(self):
        host = HostSupervisor({"left": self.device}, self.actuators)
        host.start()
        identity = CommandIdentity("pc", 0, host.status.host_session_id, 0)
        serial_write = self.serial.write

        def fail_velocity_group(packet):
            if packet[4] == 0x83 and packet[5] == 46:
                raise OSError("USB write failed")
            return serial_write(packet)

        with patch.object(self.serial, "write", side_effect=fail_velocity_group):
            result = host.cycle(
                CommandSubmission(
                    identity,
                    lambda: self.device.write_targets(
                        positions_rad={"joint": 0}, velocities_raw={"wheel": 200}
                    ),
                )
            )
        self.assertFalse(result.command_applied)
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertEqual(self.serial.get(1, 42, 2), 1234)
        self.assertEqual(self.serial.get(2, 46, 2), 0)
        self.assertTrue(all(self.serial.get(i, 40, 1) == 0 for i in (1, 2)))
        self.assertFalse(self.serial.is_open)

    def test_missing_ack_is_bounded_and_does_not_skip_next_motor(self):
        self.connect()
        self.serial.drop.add((1, 3, 40))
        start = time.monotonic()
        with self.assertRaises(DeviceOperationError):
            self.device.disable_torque()
        self.assertLess(time.monotonic() - start, 0.2)
        self.assertEqual(self.serial.get(2, 40, 1), 0)


if __name__ == "__main__":
    unittest.main()
