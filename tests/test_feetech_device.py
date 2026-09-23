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


class SimulatedClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class DelayedRegisterSerial(RegisterSerial):
    """Actual SDK frames, delayed and delivered in separate USB-sized reads."""

    def __init__(self):
        super().__init__()
        self.clock = SimulatedClock()
        self.delay = 0.012
        self.scheduled = []

    def reply(self, motor_id, instruction, address, data=b""):
        pending = self.pending
        self.pending = b""
        super().reply(motor_id, instruction, address, data)
        self.scheduled.append((self.clock() + self.delay, self.pending))
        self.pending = pending

    def _deliver(self):
        ready = [item for item in self.scheduled if item[0] <= self.clock()]
        self.scheduled = [item for item in self.scheduled if item[0] > self.clock()]
        self.pending += b"".join(packet for _, packet in ready)

    def read(self, size):
        self.clock.now += 0.0005
        self._deliver()
        return super().read(min(size, 2))

    def reset_input_buffer(self):
        self._deliver()
        super().reset_input_buffer()  # Future replies survive clearing current input.


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

    def test_read_calibration_is_read_only_and_decodes_live_offsets(self):
        with self.assertRaises(ConnectionError):
            self.device.read_calibration()
        self.connect()
        for motor_id, offset in ((1, 456), (2, 0x800 | 123)):
            self.serial.set(motor_id, 31, 2, offset)
            self.serial.set(motor_id, 9, 2, 200)
            self.serial.set(motor_id, 11, 2, 3500)
        values = self.device.read_calibration()
        self.assertEqual(values["joint"].homing_offset, 456)
        self.assertEqual(values["wheel"].homing_offset, -123)
        for name, motor_id in (("joint", 1), ("wheel", 2)):
            entry = values[name]
            self.assertEqual((entry.id, entry.drive_mode), (motor_id, 0))
            self.assertEqual((entry.range_min, entry.range_max), (200, 3500))
        self.serial.set(1, 31, 2, 0x1000)
        with self.assertRaisesRegex(ValueError, "homing offset"):
            self.device.read_calibration()
        self.assertFalse(self.writes())

    def test_read_feedback_uses_own_session_and_bus(self):
        self.connect()
        batch = self.device.read_feedback()
        self.assertEqual((batch.clock_id, batch.source_id), ("session", "left"))
        self.assertEqual(set(batch.samples), {"joint", "wheel"})

    def test_active_position_reads_still_reject_voltage_flags(self):
        self.connect()
        self.serial.errors[(1, 0x82, 56)] = 1
        with self.assertRaisesRegex(DeviceOperationError, "Input voltage error"):
            self.device.read_positions()
        self.assertIn("joint", self.device.read_feedback().failures)

    def test_voltage_only_torque_read_requires_complete_disabled_value(self):
        self.connect()
        self.serial.errors[(1, 2, 40)] = 0x01
        self.serial.set(1, 40, 1, 0)
        self.assertEqual(self.device._read_register("joint", 40, 1), 0)
        for value in (1, 2):
            self.serial.set(1, 40, 1, value)
            with self.subTest(value=value), self.assertRaises(ConnectionError):
                self.device._read_register("joint", 40, 1)
        self.serial.set(1, 40, 1, 0)
        for error in (0x02, 0x03, 0x04, 0x08, 0x20):
            self.serial.errors[(1, 2, 40)] = error
            with self.subTest(error=error), self.assertRaises(ConnectionError):
                self.device._read_register("joint", 40, 1)

    def test_voltage_alarm_does_not_relax_other_registers_or_torque_writes(self):
        self.connect()
        for address, width in ((3, 2), (33, 1), (40, 2), (56, 2)):
            self.serial.errors[(1, 2, address)] = 0x01
            with self.subTest(address=address), self.assertRaises(ConnectionError):
                self.device._read_register("joint", address, width)
        self.serial.errors[(1, 3, 40)] = 0x01
        with self.assertRaises(ConnectionError):
            self.device._write_register("joint", 40, 1, 1)

    def test_voltage_only_torque_read_does_not_accept_empty_write_ack(self):
        self.connect()
        self.serial.errors[(1, 2, 40)] = 0x01
        reply = self.serial.reply

        def empty_reply(motor_id, instruction, address, data=b""):
            reply(motor_id, instruction, address)

        with patch.object(self.serial, "reply", side_effect=empty_reply):
            with self.assertRaises(ConnectionError):
                self.device._read_register("joint", 40, 1)

    def test_active_position_read_retains_short_budget_and_single_attempt(self):
        self.connect()
        with patch.object(
            self.device._reader, "read_positions", wraps=self.device._reader.read_positions
        ) as read:
            with patch.object(self.serial, "write", side_effect=TimeoutError("Write timeout")):
                with self.assertRaises(DeviceOperationError):
                    self.device.read_positions()
        read.assert_called_once_with(["joint", "wheel"], timeout_s=0.008)
        self.assertEqual(self.serial.write_timeout, 0.005)

    def test_passive_calibration_keeps_position_and_fault_validation(self):
        device = self.new_device(position_calibrations={}, velocity_limits={})
        device.connect("calibration")
        device.disable_torque()
        with device.calibration_session():
            self.serial.errors[(1, 0x82, 56)] = 1
            self.assertEqual(device.read_positions()["joint"], 1234)
            for error in (0x02, 0x03, 0x04, 0x08, 0x20):
                self.serial.errors[(1, 0x82, 56)] = error
                with self.subTest(error=error), self.assertRaises(DeviceOperationError):
                    device.read_positions()
            self.serial.errors[(1, 0x82, 56)] = 1
            self.serial.set(1, 56, 2, 4096)
            with self.assertRaisesRegex(ConnectionError, "invalid position"):
                device.read_positions()
            self.serial.set(1, 56, 2, 1234)
            self.serial.drop.add((1, 0x82, 56))
            with self.assertRaises(DeviceOperationError):
                device.read_positions()
            self.serial.drop.clear()
        # Leaving the verified torque-off transaction removes the exception.
        with self.assertRaises(DeviceOperationError):
            device.read_positions()

    def test_passive_sampling_never_ignores_register_write_errors(self):
        device = self.new_device(position_calibrations={}, velocity_limits={})
        device.connect("calibration")
        device.disable_torque()
        with device.calibration_session():
            self.serial.errors[(1, 3, 55)] = 1
            with self.assertRaisesRegex(ConnectionError, "servo_error=0x01"):
                device._write_verified("joint", 55, 1, 0)

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

    def test_group_writes_allow_short_usb_delay_without_changing_feedback_budget(self):
        from serial import SerialTimeoutException

        self.connect()
        clock = SimulatedClock()
        original = self.serial.write
        budgets = []

        def delayed_write(packet):
            budgets.append(self.serial.write_timeout)
            if self.serial.write_timeout < 0.012:
                raise SerialTimeoutException("Write timeout")
            clock.now += 0.012
            return original(packet)

        with (
            patch("time.monotonic", side_effect=clock),
            patch.object(self.serial, "write", side_effect=delayed_write),
        ):
            self.device.write_targets(positions_rad={"joint": 0}, velocities_raw={"wheel": 200})
            self.assertEqual(len(self.writes()), 2)
            self.assertAlmostEqual(clock.now, 0.024)
            for budget in budgets:
                self.assertAlmostEqual(budget, 0.050)
            self.assertAlmostEqual(self.device._packet_port().deadline - clock.now, 0.008)
        self.assertEqual(self.serial.write_timeout, 0.005)

    def test_group_write_timeout_is_bounded_contextual_and_never_retried(self):
        from serial import SerialTimeoutException

        self.connect()
        clock = SimulatedClock()

        def blocked_write(packet):
            self.assertLessEqual(self.serial.write_timeout, 0.050)
            clock.now += self.serial.write_timeout
            raise SerialTimeoutException("Write timeout")

        with (
            patch("time.monotonic", side_effect=clock),
            patch.object(self.serial, "write", side_effect=blocked_write) as write,
        ):
            with self.assertRaisesRegex(
                ConnectionError,
                r"/dev/test-only SyncWrite Goal_Velocity \(46\) \[wheel \(ID 2\)\]: "
                r"SerialTimeoutException: Write timeout",
            ):
                self.device.write_targets(positions_rad={}, velocities_raw={"wheel": 1300})
            write.assert_called_once()
            self.assertAlmostEqual(clock.now, 0.050)
        self.assertEqual(self.serial.write_timeout, 0.005)

    def test_short_group_write_is_not_reported_as_success_or_retried(self):
        self.connect()
        with patch.object(self.serial, "write", return_value=4) as write:
            with self.assertRaisesRegex(
                ConnectionError, r"/dev/test-only SyncWrite Goal_Position \(42\).*ID 1.*transport="
            ):
                self.device.write_targets(positions_rad={"joint": 0}, velocities_raw={})
            write.assert_called_once()
        self.assertEqual(self.serial.write_timeout, 0.005)

    def test_invalid_velocity_does_not_allow_partial_position_write(self):
        self.connect()
        for velocity in (3001, -3001, True, 1.5):
            with self.assertRaises(ValueError):
                self.device.write_targets(
                    positions_rad={"joint": 0}, velocities_raw={"wheel": velocity}
                )
        self.assertFalse(self.writes())

    def test_released_velocity_stays_off_at_idle_and_resumes_only_for_motion(self):
        self.connect()
        self.device.release_velocity("wheel")
        self.assertEqual(self.serial.get(2, 46, 2), 0)
        self.assertEqual(self.serial.get(2, 40, 1), 0)
        self.assertEqual(self.serial.get(1, 40, 1), 1)
        self.serial.requests.clear()
        self.device.write_targets(positions_rad={}, velocities_raw={"wheel": 0})
        self.assertFalse(self.writes())
        self.device.stop()
        self.assertEqual(self.serial.get(2, 40, 1), 0)
        self.serial.requests.clear()
        self.device.write_targets(positions_rad={}, velocities_raw={"wheel": -100})
        self.assertEqual(self.serial.get(2, 40, 1), 1)
        self.assertEqual(self.serial.get(2, 46, 2), 0x8000 | 100)
        writes = self.writes()
        self.assertEqual([packet[5] for packet in writes], [46, 40, 46])
        self.assertEqual(writes[0][6:-1], bytes([0, 0]))

    def test_release_attempts_torque_off_even_when_zero_velocity_is_not_accepted(self):
        self.connect()
        self.serial.ignore_writes.add((2, 46))
        with self.assertRaises(DeviceOperationError):
            self.device.release_velocity("wheel")
        self.assertEqual(self.serial.get(2, 40, 1), 0)
        self.assertEqual(self.serial.get(1, 40, 1), 1)

    def test_release_never_disables_an_arm_or_unverified_motor(self):
        self.connect()
        for name in ("joint", "unknown"):
            with self.assertRaises(ValueError):
                self.device.release_velocity(name)
        self.assertFalse(self.writes())

    def test_failed_resume_does_not_send_a_nonzero_velocity(self):
        self.connect()
        self.device.release_velocity("wheel")
        self.serial.ignore_writes.add((2, 40))
        self.serial.requests.clear()
        with self.assertRaises(ConnectionError):
            self.device.write_targets(positions_rad={}, velocities_raw={"wheel": 100})
        self.assertEqual(self.serial.get(2, 46, 2), 0)
        self.assertFalse(any(p[4] == 0x83 for p in self.serial.requests))

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

    def test_unprepared_stop_accepts_only_fresh_verified_torque_off(self):
        self.connect()
        self.device._modes.clear()
        for motor_id in (1, 2):
            self.serial.set(motor_id, 40, 1, 0)
        self.device.stop()
        self.assertFalse(self.writes())
        self.serial.set(2, 40, 1, 1)
        with self.assertRaisesRegex(DeviceOperationError, "Operating mode not verified"):
            self.device.stop()
        self.serial.set(2, 40, 1, 0)
        self.serial.errors[(2, 2, 40)] = 0x04
        with self.assertRaises(DeviceOperationError):
            self.device.stop()
        self.assertFalse(self.writes())

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

    def test_watchdog_reuses_current_feedback_and_only_group_writes_arm_and_base(self):
        self.connect()
        batch = self.device.read_feedback()
        self.serial.requests.clear()
        self.device.stop_motion(batch)
        self.assertEqual([(p[4], p[5]) for p in self.serial.requests], [(0x83, 42), (0x83, 46)])
        self.assertEqual(self.serial.get(1, 42, 2), 1234)
        self.assertEqual(self.serial.get(2, 46, 2), 0)
        self.assertEqual(self.serial.get(1, 46, 2), 2000)

    def test_watchdog_faulted_feedback_never_becomes_a_hold_target(self):
        self.connect()
        self.serial.errors[(1, 0x82, 56)] = 0x20
        batch = self.device.read_feedback()
        self.serial.requests.clear()
        with self.assertRaises(ConnectionError):
            self.device.stop_motion(batch)
        self.assertFalse(self.writes())

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
        self.assertLess(time.monotonic() - start, 0.3)
        self.assertEqual(
            sum(p[2:3] == b"\x01" and p[4:6] == b"\x03\x28" for p in self.serial.requests), 4
        )
        self.assertEqual(self.serial.get(2, 40, 1), 0)

    def test_transient_register_reply_loss_recovers_with_source_retry_limit(self):
        self.connect()
        reply = self.serial.reply
        counts = {}

        def drop_first_three(motor_id, instruction, address, data=b""):
            key = (motor_id, instruction, address)
            counts[key] = counts.get(key, 0) + 1
            if counts[key] >= 4:
                reply(motor_id, instruction, address, data)

        with patch.object(self.serial, "reply", side_effect=drop_first_three):
            self.device._write_verified("wheel", 44, 2, 0)
        self.assertEqual(counts, {(2, 3, 44): 4, (2, 2, 44): 4})

    def test_servo_errors_and_unsafe_writes_are_not_retried(self):
        self.connect()
        for address, width, value in ((31, 2, 100), (55, 1, 0), (40, 1, 1)):
            self.serial.requests.clear()
            self.serial.drop.add((1, 3, address))
            with self.assertRaisesRegex(ConnectionError, "attempts=1"):
                self.device._write_register("joint", address, width, value)
            self.assertEqual(len(self.serial.requests), 1)
            self.serial.drop.clear()
        self.serial.errors[(2, 3, 46)] = 0x20
        self.serial.requests.clear()
        with self.assertRaisesRegex(ConnectionError, "servo_error=0x20.*attempts=1"):
            self.device._write_register("wheel", 46, 2, 0)
        self.assertEqual(len(self.serial.requests), 1)

    def test_cyclic_register_fallback_does_not_retry_or_extend_deadline(self):
        self.connect()
        self.serial.drop.add((1, 2, 69))
        started = time.monotonic()
        with self.assertRaisesRegex(ConnectionError, "attempts=1"):
            self.device._read_register("joint", 69, 2, deadline=started + 0.003)
        self.assertEqual(len(self.serial.requests), 1)
        self.assertLess(time.monotonic() - started, 0.025)

    def test_late_write_ack_is_not_a_register_value(self):
        self.connect()
        reply = self.serial.reply

        def ack_before_read(motor_id, instruction, address, data=b""):
            if instruction == 2:
                reply(motor_id, 3, address)
            reply(motor_id, instruction, address, data)

        with patch.object(self.serial, "reply", side_effect=ack_before_read):
            self.assertEqual(self.device._read_register("joint", 40, 1), 1)

    def test_delayed_register_replies_work_without_extending_cyclic_feedback_budget(self):
        serial = DelayedRegisterSerial()
        self.factory.return_value = serial
        with patch("time.monotonic", side_effect=serial.clock):
            self.device.connect("delayed-session")
            self.assertGreater(serial.clock(), 0.008)
            # Normal feedback retains its short, shared runtime budget.
            started = serial.clock()
            self.assertTrue(self.device.read_feedback().failures)
            # Full-block read plus the existing bounded critical-feedback fallback.
            self.assertLess(serial.clock() - started, 0.025)
            # Let old replies arrive, then drain before this independent check.
            serial.clock.now += 0.050
            serial.reset_input_buffer()
            started = serial.clock()
            self.device.disable_torque()
            self.assertLess(serial.clock() - started, 0.080)
            self.device.stop()
            self.assertEqual(serial.get(2, 46, 2), 0)
            self.assertEqual(serial.get(1, 42, 2), 1234)
            self.assertEqual(serial.write_timeout, 0.005)

    def test_failed_connection_leaves_cyclic_write_timeout_unchanged(self):
        self.serial.set(1, 3, 2, 777)
        with self.assertRaisesRegex(ConnectionError, "model mismatch"):
            self.device.connect("session")
        self.assertEqual(self.serial.write_timeout, 0.005)
        with patch("time.monotonic", return_value=1.0):
            self.assertEqual(self.device._packet_port().deadline, 1.008)

    def test_active_connection_rejects_mixed_firmware_before_configuration(self):
        self.serial.set(2, 1, 1, 99)
        with self.assertRaisesRegex(ConnectionError, "firmware"):
            self.device.connect("session")
        self.assertFalse(self.writes())

    def test_overcurrent_cleanup_does_not_hold_overloaded_arm_before_torque_off(self):
        host = HostSupervisor({"left": self.device}, self.actuators)
        host.start()
        self.serial.set(1, 69, 2, 600)  # STS3250 near-stall threshold exceeded.
        host.cycle()
        self.serial.requests.clear()
        now = time.monotonic()
        with patch("time.monotonic", return_value=now + 0.1):
            result = host.cycle()
        self.assertIs(result.status.phase, HostPhase.FAULT)
        self.assertIsNotNone(result.status.current_trip)
        self.assertFalse(any(p[4] in (3, 0x83) and p[5] == 42 for p in self.serial.requests))
        self.assertEqual(self.serial.get(2, 46, 2), 0)
        self.assertTrue(all(self.serial.get(i, 40, 1) == 0 for i in (1, 2)))

    def test_usb_write_timeout_keeps_motor_register_context_and_restores_port(self):
        from serial import SerialTimeoutException

        self.connect()
        with patch.object(
            self.serial, "write", side_effect=SerialTimeoutException("Write timeout")
        ):
            with self.assertRaisesRegex(
                ConnectionError, r"/dev/test-only wheel \(ID 2\): Write 46=0: Write timeout"
            ):
                self.device._write_register("wheel", 46, 2, 0)
        self.assertEqual(self.serial.write_timeout, 0.005)

    def test_register_fallback_write_cannot_exceed_its_cyclic_deadline(self):
        self.connect()
        with patch("time.monotonic", return_value=1.0):
            port = self.device._packet_port(1.003, register=True)
            observed = []
            with patch.object(
                self.serial,
                "write",
                side_effect=lambda packet: (
                    observed.append(self.serial.write_timeout) or len(packet)
                ),
            ):
                port.writePort([1])
        self.assertAlmostEqual(observed[0], 0.003)
        self.assertEqual(self.serial.write_timeout, 0.005)

    def test_eeprom_primary_error_is_not_replaced_by_relock_error(self):
        self.connect()
        original_write = self.serial.write

        def fail_write_and_lock(packet):
            if packet[4] == 3 and packet[5] == 31:
                self.serial.drop.update({(1, 3, 31), (1, 3, 55)})
            return original_write(packet)

        with patch.object(self.serial, "write", side_effect=fail_write_and_lock):
            with self.assertRaises(DeviceOperationError) as caught:
                self.device._calibration_write("joint", [(31, 2, 123)])
        message = str(caught.exception)
        for text in ("Write 31=123", "Write 55=1", "/dev/test-only joint (ID 1)"):
            self.assertIn(text, message)
        # A missing ACK never triggers a blind repeat of the EEPROM write.
        self.assertEqual(sum(p[4] == 3 and p[5] == 31 for p in self.serial.requests), 1)

    def test_eeprom_read_failure_still_attempts_relock(self):
        self.connect()
        self.serial.set(1, 55, 1, 0)
        self.serial.drop.add((1, 2, 11))
        with self.assertRaises(OSError):
            self.device._calibration_write("joint", [(11, 2, 3500)])
        self.assertEqual(self.serial.get(1, 55, 1), 1)

    def test_late_write_ack_does_not_hide_failed_eeprom_transaction(self):
        serial = DelayedRegisterSerial()
        self.factory.return_value = serial
        device = self.new_device(position_calibrations={}, velocity_limits={})
        reply = serial.reply

        def late_ack(motor_id, instruction, address, data=b""):
            previous = serial.delay
            if instruction == 3 and address == 31:
                serial.delay = 0.060
            try:
                reply(motor_id, instruction, address, data)
            finally:
                serial.delay = previous

        with patch("time.monotonic", side_effect=serial.clock):
            device.connect("calibration")
            with patch.object(serial, "reply", side_effect=late_ack):
                with self.assertRaisesRegex(OSError, "Write 31=123"):
                    device._calibration_write("joint", [(31, 2, 123)])
        self.assertEqual(sum(p[4] == 3 and p[5] == 31 for p in serial.requests), 1)


if __name__ == "__main__":
    unittest.main()
