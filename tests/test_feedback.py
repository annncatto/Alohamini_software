import importlib.util
import math
import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from alohamini.calibration import EncoderCalibration
from alohamini.hardware.feedback import FeedbackBatch, MotorFeedback
from alohamini.hardware.feetech import FeetechFeedbackReader, decode_feedback
from alohamini.model import ActuatorSpec, get_robot_model
from alohamini.runtime.feedback import FeedbackProjector


def block(position=1234, velocity=0x8005, current=18):
    result = bytearray(15)
    for offset, value in ((0, position), (2, velocity), (4, 0x0407), (13, current)):
        result[offset : offset + 2] = value.to_bytes(2, "little")
    result[6], result[7], result[10] = 124, 32, 1
    return bytes(result)


def packet(motor_id, *, error=0, data=None):
    data = block() if data is None else data
    content = bytes([motor_id, len(data) + 2, error]) + data
    return b"\xff\xff" + content + bytes([~sum(content) & 0xFF])


class MemorySerial:
    """In-memory wire; actual SDK generates and parses every packet."""

    baudrate = 1_000_000
    bytesize = 8
    parity = "N"
    stopbits = 1
    timeout = 0
    write_timeout = 0.001
    is_open = True

    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.pending = b""
        self.writes = []

    def reset_input_buffer(self):
        self.pending = b""

    def write(self, data):
        self.writes.append(data)
        self.pending = next(self.responses, b"")
        return len(data)

    def read(self, size):
        data, self.pending = self.pending[:size], self.pending[size:]
        return data


def motors(*ids):
    return [ActuatorSpec(f"joint_{i}", "left", i, "sts3250") for i in ids]


def batch(sequence, *, position=1234):
    return FeedbackBatch(
        "left",
        "session-1",
        sequence,
        sequence + 1.0,
        sequence + 1.001,
        {"joint_1": decode_feedback(block(position=position))},
        {},
    )


class FeedbackValueTests(unittest.TestCase):
    def test_sign_magnitude_and_current_match_deployed_driver(self):
        sample = decode_feedback(block())
        self.assertEqual(
            dict(sample.registers),
            {
                "position_raw": 1234,
                "velocity_raw": -5,
                "load_raw": -7,
                "voltage_raw": 124,
                "temperature_raw": 32,
                "status_raw": 0,
                "moving": 1,
                "current_raw": 18,
            },
        )
        self.assertAlmostEqual(sample.current_a, 0.117)
        self.assertIsNone(sample.position_rad)
        self.assertIsNone(sample.velocity_rad_s)

    def test_current_is_unsigned_and_negative_zero_is_zero(self):
        sample = decode_feedback(block(position=0x8000, velocity=0x8000, current=0x8001))
        self.assertEqual(sample.registers["position_raw"], 0)
        self.assertEqual(sample.registers["velocity_raw"], 0)
        self.assertEqual(sample.registers["current_raw"], 32769)

    def test_block_and_error_validation(self):
        for value in (b"", bytes(14), bytes(16), [0] * 15):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_feedback(value)
        for error in (-1, 128, True):
            with self.assertRaises(ValueError):
                decode_feedback(block(), packet_error=error)

    def test_nested_values_are_immutable_copies(self):
        values = {"current_raw": 0}
        sample = MotorFeedback(values)
        values["current_raw"] = 100
        self.assertEqual(sample.registers["current_raw"], 0)
        with self.assertRaises(TypeError):
            sample.registers["current_raw"] = 1
        value = batch(0)
        with self.assertRaises(TypeError):
            value.samples["joint_1"] = sample

    def test_invalid_batch_metadata(self):
        for values in (
            {"sequence": True},
            {"sequence": -1},
            {"received_s": 0},
            {"request_started_s": float("nan")},
            {"source_id": ""},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(batch(0), **values)


@unittest.skipUnless(importlib.util.find_spec("scservo_sdk"), "optional Feetech SDK not installed")
class FeedbackReaderTests(unittest.TestCase):
    def reader(self, serial, *ids):
        return FeetechFeedbackReader(serial, motors(*(ids or (1,))), clock_id="session-1")

    def test_actual_sdk_only_sends_sync_read(self):
        serial = MemorySerial([packet(2) + packet(1)])
        result = self.reader(serial, 1, 2).read()
        self.assertEqual(set(result.samples), {"joint_1", "joint_2"})
        self.assertFalse(result.failures)
        content = bytes([254, 6, 130, 56, 15, 1, 2])
        self.assertEqual(serial.writes, [b"\xff\xff" + content + bytes([~sum(content) & 255])])
        self.assertGreaterEqual(result.received_s, result.request_started_s)

    def test_wire_to_calibrated_feedback_preserves_partial_failures(self):
        raw = self.reader(MemorySerial([packet(2, error=1)]), 1, 2).read()
        projector = FeedbackProjector(
            {"joint_2": EncoderCalibration(4096, 0, 0, 1)},
            source_id="left",
            clock_id="session-1",
        )
        result = projector.project(raw)
        self.assertEqual(set(result.failures), {"joint_1", "joint_2"})
        self.assertEqual(result.request_started_s, raw.request_started_s)
        self.assertEqual(result.received_s, raw.received_s)
        self.assertAlmostEqual(result.samples["joint_2"].position_rad, 1234 * math.tau / 4096)
        self.assertAlmostEqual(result.samples["joint_2"].current_a, 0.117)
        self.assertEqual(result.samples["joint_2"].packet_error, 1)

    def test_invalid_serial_framing_is_rejected(self):
        serial = MemorySerial()
        serial.parity = "E"
        with self.assertRaises(ValueError):
            self.reader(serial)
        self.assertFalse(serial.writes)

    def test_missing_first_id_does_not_hide_later_motor(self):
        result = self.reader(MemorySerial([packet(2)]), 1, 2).read()
        self.assertEqual(set(result.samples), {"joint_2"})
        self.assertEqual(set(result.failures), {"joint_1"})

    def test_missing_feedback_is_not_cached_or_zero_filled(self):
        serial = MemorySerial([packet(1, data=bytes(15)), b""])
        reader = self.reader(serial)
        self.assertEqual(reader.read().samples["joint_1"].current_a, 0)
        result = reader.read()
        self.assertEqual(result.sequence, 1)
        self.assertFalse(result.samples)
        self.assertIn("joint_1", result.failures)

    def test_pending_input_is_cleared(self):
        serial = MemorySerial()
        serial.pending = packet(1)
        self.assertFalse(self.reader(serial).read().samples)

    def test_packet_error_retains_measurements_and_marks_failure(self):
        result = self.reader(MemorySerial([packet(1, error=0x29)])).read()
        self.assertEqual(result.samples["joint_1"].packet_error, 0x29)
        self.assertEqual(result.samples["joint_1"].registers["status_raw"], 0)
        self.assertIn("0x29", result.failures["joint_1"])

    def test_corrupt_wrong_and_duplicate_packets_do_not_replace_samples(self):
        corrupt = packet(1)[:-1] + bytes([packet(1)[-1] ^ 1])
        wire = corrupt + packet(20) + packet(1) + packet(1, data=bytes(15)) + packet(2)
        result = self.reader(MemorySerial([wire]), 1, 2).read()
        self.assertFalse(result.failures)
        self.assertEqual(result.samples["joint_1"].registers["position_raw"], 1234)

    def test_short_packet_is_not_complete_feedback(self):
        result = self.reader(MemorySerial([packet(1, data=bytes(2)) + packet(2)]), 1, 2).read()
        self.assertEqual(set(result.samples), {"joint_2"})
        self.assertEqual(set(result.failures), {"joint_1"})

    def test_truncated_reply_keeps_preceding_complete_reply(self):
        result = self.reader(MemorySerial([packet(2) + packet(1)[:10]]), 1, 2).read()
        self.assertEqual(set(result.samples), {"joint_2"})
        self.assertEqual(set(result.failures), {"joint_1"})

    def test_write_failure_is_one_failed_batch_without_retry(self):
        serial = MemorySerial()
        with patch.object(serial, "write", side_effect=OSError("disconnected")) as write:
            result = self.reader(serial).read()
        self.assertEqual(write.call_count, 1)
        self.assertEqual(result.failures["joint_1"], "disconnected")

    def test_read_failure_keeps_partial_feedback(self):
        serial = MemorySerial([packet(1)])
        read = serial.read

        def fail_after_packet(size):
            if not serial.pending:
                raise OSError("disconnected")
            return read(size)

        with patch.object(serial, "read", side_effect=fail_after_packet):
            result = self.reader(serial, 1, 2).read()
        self.assertEqual(set(result.samples), {"joint_1"})
        self.assertEqual(result.failures["joint_2"], "disconnected")

    def test_continuous_noise_cannot_extend_deadline(self):
        serial = MemorySerial()
        reader = self.reader(serial)
        ticks = iter(i / 10000 for i in range(1000))
        with patch("alohamini.hardware.feetech.time.monotonic", side_effect=lambda: next(ticks)):
            with patch.object(serial, "read", side_effect=lambda n: b"\x00" * n):
                result = reader.read()
        self.assertFalse(result.samples)
        self.assertLess(result.received_s - result.request_started_s, 0.009)

    def test_empty_read_returns_within_software_timeout(self):
        started = time.monotonic()
        result = self.reader(MemorySerial()).read()
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertFalse(result.samples)

    def test_port_requirements_checked_again_at_read(self):
        for attr, value in (
            ("timeout", None),
            ("timeout", 1),
            ("write_timeout", None),
            ("write_timeout", 0),
            ("write_timeout", 10),
            ("is_open", False),
        ):
            serial = MemorySerial()
            reader = self.reader(serial)
            setattr(serial, attr, value)
            with self.subTest(attr=attr, value=value), self.assertRaises(ValueError):
                reader.read()
            self.assertFalse(serial.writes)

    def test_other_thread_cannot_read(self):
        serial, errors = MemorySerial(), []
        reader = self.reader(serial)

        def other_thread():
            try:
                reader.read()
            except RuntimeError as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(timeout=1)
        self.assertEqual(len(errors), 1)
        self.assertFalse(serial.writes)

    def test_model_validation_and_all_builtin_buses(self):
        for model_id in ("alohamini1", "alohamini2", "alohamini2pro"):
            model = get_robot_model(model_id)
            for bus in {motor.bus for motor in model.actuators}:
                entries = [motor for motor in model.actuators if motor.bus == bus]
                serial = MemorySerial([b"".join(packet(m.motor_id) for m in entries)])
                result = FeetechFeedbackReader(serial, entries, clock_id="test").read()
                self.assertEqual(set(result.samples), {m.name for m in entries})
        for entries in (
            [],
            motors(1, 1),
            [*motors(1), ActuatorSpec("right", "right", 2, "sts3250")],
            [ActuatorSpec("wrong", "left", 1, "unknown")],
        ):
            with self.assertRaises(ValueError):
                FeetechFeedbackReader(MemorySerial(), entries, clock_id="test")


class FeedbackProjectionTests(unittest.TestCase):
    def setUp(self):
        self.calibration = EncoderCalibration(4096, 0, 0, -1)
        self.projector = FeedbackProjector(
            {"joint_1": self.calibration}, source_id="left", clock_id="session-1"
        )

    def test_position_velocity_and_raw_current_preserved(self):
        value = self.projector.project(batch(0)).samples["joint_1"]
        self.assertAlmostEqual(value.position_rad, -1234 * math.tau / 4096)
        self.assertAlmostEqual(value.velocity_rad_s, 5 * math.tau / 4096)
        self.assertAlmostEqual(value.current_a, 0.117)
        self.assertEqual(value.registers["position_raw"], 1234)

    def test_invalid_position_does_not_drop_current_or_other_motor(self):
        raw = batch(0, position=0x8005)
        raw = replace(raw, samples={**raw.samples, "uncalibrated": decode_feedback(block())})
        value = self.projector.project(raw)
        sample = value.samples["joint_1"]
        self.assertIsNone(sample.position_rad)
        self.assertIn("position_rad", sample.field_errors)
        self.assertAlmostEqual(sample.current_a, 0.117)
        self.assertIsNotNone(sample.velocity_rad_s)
        self.assertIsNone(value.samples["uncalibrated"].position_rad)

    def test_missing_registers_are_not_zero(self):
        raw = replace(batch(0), samples={"joint_1": MotorFeedback({"current_raw": 0})})
        value = self.projector.project(raw).samples["joint_1"]
        self.assertIsNone(value.position_rad)
        self.assertIsNone(value.velocity_rad_s)
        self.assertEqual(set(value.field_errors), {"position_rad", "velocity_rad_s"})

    def test_continuity_and_skipped_sequence(self):
        self.projector.project(batch(0, position=2000))
        value = self.projector.project(batch(1, position=2100)).samples["joint_1"]
        self.assertLess(value.position_rad, 0)
        value = self.projector.project(batch(3, position=2100)).samples["joint_1"]
        self.assertGreater(value.position_rad, 0)

    def test_missing_motor_breaks_continuity(self):
        self.projector.project(batch(0, position=2000))
        self.projector.project(replace(batch(1), samples={}, failures={"joint_1": "missing"}))
        value = self.projector.project(batch(2, position=2100)).samples["joint_1"]
        self.assertGreater(value.position_rad, 0)

    def test_servo_error_breaks_continuity_but_preserves_fault(self):
        self.projector.project(batch(0, position=2000))
        raw = batch(1, position=2100)
        raw = replace(raw, samples={"joint_1": replace(raw.samples["joint_1"], packet_error=1)})
        value = self.projector.project(raw).samples["joint_1"]
        self.assertGreater(value.position_rad, 0)
        self.assertEqual(value.packet_error, 1)

    def test_explicit_reset_does_not_allow_replay(self):
        self.projector.project(batch(0))
        self.projector.reset()
        with self.assertRaises(ValueError):
            self.projector.project(batch(0))

    def test_wrong_bus_session_and_overlapping_times_are_rejected(self):
        self.projector.project(batch(0))
        for values in (
            {"source_id": "right"},
            {"clock_id": "new-session"},
            {"request_started_s": 1.0005},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.projector.project(replace(batch(1), **values))


if __name__ == "__main__":
    unittest.main()
