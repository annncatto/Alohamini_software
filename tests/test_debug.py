import ast
import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from test_dataset import jpeg
from test_feetech_device import RegisterSerial

from alohamini.cli import main
from alohamini.hardware import find_cameras, find_port

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    os.environ.get("ALOHAMINI_SOURCE_REPO", "/home/anncatto/lerobot_alohamini_pr_50hz_fix")
)


def script(name):
    spec = importlib.util.spec_from_file_location(f"debug_{name}", ROOT / "examples/debug" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PortDebugTests(unittest.TestCase):
    def setUp(self):
        # Any accidental device open or network operation must fail the test.
        patch("serial.Serial", side_effect=AssertionError("serial device opened")).start()
        patch("socket.socket", side_effect=AssertionError("network access")).start()
        self.addCleanup(patch.stopall)

    def test_enumeration_is_sorted_and_excludes_aliases_without_opening_ports(self):
        devices = [SimpleNamespace(device=name) for name in ("/dev/ttyACM1", "/dev/ttyACM0")]
        with patch("serial.tools.list_ports.comports", return_value=devices) as enumerate_ports:
            self.assertEqual(find_port.find_available_ports(), ["/dev/ttyACM0", "/dev/ttyACM1"])
        enumerate_ports.assert_called_once_with(include_links=False)

    def test_cli_retains_unplug_enter_compare_and_reconnect_workflow(self):
        with (
            patch.object(
                find_port,
                "find_available_ports",
                side_effect=[["/dev/ttyACM0", "/dev/ttyACM1"], ["/dev/ttyACM1"]],
            ) as enumerate_ports,
            patch("builtins.input", return_value="") as prompt,
            patch.object(find_port.time, "sleep") as sleep,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(main(["find-port"]), 0)
        self.assertEqual(enumerate_ports.call_count, 2)
        prompt.assert_called_once_with()
        sleep.assert_called_once_with(0.5)
        self.assertIn("The port of this MotorsBus is '/dev/ttyACM0'", output.getvalue())
        self.assertTrue(output.getvalue().endswith("Reconnect the USB cable.\n"))

    def test_missing_and_ambiguous_changes_are_errors_not_guessed_ports(self):
        for after, message in (
            (["/dev/ttyACM0", "/dev/ttyACM1"], "No difference"),
            ([], "More than one port"),
        ):
            with (
                self.subTest(after=after),
                patch.object(
                    find_port,
                    "find_available_ports",
                    side_effect=[["/dev/ttyACM0", "/dev/ttyACM1"], after],
                ),
                patch("builtins.input", return_value=""),
                patch.object(find_port.time, "sleep"),
                contextlib.redirect_stdout(io.StringIO()) as output,
                contextlib.redirect_stderr(io.StringIO()) as errors,
            ):
                self.assertEqual(main(["find-port"]), 1)
                self.assertIn(message, errors.getvalue())
                self.assertNotIn("The port of this MotorsBus is", output.getvalue())
                self.assertIn("Reconnect the USB cable.", output.getvalue())

    def test_no_ports_fails_before_waiting_for_input(self):
        with (
            patch.object(find_port, "find_available_ports", return_value=[]),
            patch("builtins.input") as prompt,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            self.assertEqual(main(["find-port"]), 1)
        prompt.assert_not_called()
        self.assertIn("No serial ports found", errors.getvalue())

    def test_cancel_and_eof_remind_operator_to_reconnect(self):
        for exception, code in ((KeyboardInterrupt, 130), (EOFError, 1)):
            with (
                self.subTest(exception=exception),
                patch.object(find_port, "find_available_ports", return_value=["/dev/ttyACM0"]),
                patch("builtins.input", side_effect=exception),
                contextlib.redirect_stdout(io.StringIO()) as output,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(["find-port"]), code)
                self.assertIn("Reconnect the USB cable.", output.getvalue())

    def test_enumeration_failure_after_unplug_reminds_operator_to_reconnect(self):
        with (
            patch.object(
                find_port, "find_available_ports", side_effect=[["/dev/ttyACM0"], OSError("failed")]
            ),
            patch("builtins.input", return_value=""),
            patch.object(find_port.time, "sleep"),
            contextlib.redirect_stdout(io.StringIO()) as output,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(["find-port"]), 1)
        self.assertIn("Reconnect the USB cable.", output.getvalue())

    def test_help_does_not_enumerate_or_wait_for_input(self):
        with (
            patch.object(find_port, "find_available_ports") as enumerate_ports,
            patch("builtins.input") as prompt,
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as exited,
        ):
            main(["find-port", "--help"])
        self.assertEqual(exited.exception.code, 0)
        enumerate_ports.assert_not_called()
        prompt.assert_not_called()


class DiagnosticSerial(RegisterSerial):
    def write(self, packet):
        if packet[2] not in self.registers:
            self.requests.append(packet)
            return len(packet)
        return super().write(packet)


class MotorDebugTests(unittest.TestCase):
    def setUp(self):
        self.module = script("motors.py")
        self.serial = DiagnosticSerial()
        self.serial.set(1, 3, 2, 777)
        self.serial.set(2, 3, 2, 2569)
        self.serial.set(1, 31, 2, 0x800 | 123)
        self.serial.set(1, 62, 1, 124)
        self.factory = patch("serial.Serial", return_value=self.serial).start()
        self.addCleanup(patch.stopall)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_scan_table_and_close_only_read_registers(self):
        with self.module.MotorStateReader("/dev/test-only") as bus:
            motors = self.module.probe_scan_ids(bus)
            self.assertEqual([motor.model for motor in motors.values()], ["sts3215", "sts3095"])
            rows, errors = self.module.collect_states(bus, motors)
        self.assertFalse(errors)
        first = rows[0][1]
        self.assertEqual(first["Offset"], -123)
        self.assertEqual(first["Current(mA)"], 65)
        self.assertEqual(first["Voltage"], 124)
        self.assertEqual(first["Angle"], round(1234 / 4096 * 360, 1))
        self.assertTrue(self.factory.call_args.kwargs["exclusive"])
        self.assertEqual(self.factory.call_args.kwargs["timeout"], 0)
        self.assertTrue(all(packet[4] == 2 for packet in self.serial.requests))
        self.assertEqual(self.serial.get(1, 40, 1), 1)
        self.assertFalse(self.serial.is_open)

    def test_read_accepts_reply_after_ten_ms_with_source_timeout(self):
        clock = SimpleNamespace(now=0.0)
        original = self.serial.read

        def delayed(size):
            clock.now += 0.001
            return b"" if clock.now < 0.010 else original(size)

        with (
            patch.object(self.serial, "read", side_effect=delayed),
            patch.object(self.module.time, "monotonic", side_effect=lambda: clock.now),
            self.module.MotorStateReader("/dev/test-only") as bus,
        ):
            self.assertEqual(bus.read(1, "Model_Number"), (777, 0))
        self.assertGreaterEqual(clock.now, 0.010)

    def test_maintenance_retries_transport_not_fault_or_id_change(self):
        bus = self.module.MotorMaintenance("/dev/test-only")
        bus.serial = self.serial
        bus.handler = Mock()
        bus.handler.writeTxRx.side_effect = [(-6, 0), (0, 0)]
        bus.write(1, "Lock", 1)
        self.assertEqual(bus.handler.writeTxRx.call_count, 2)
        for register, reply in (("Lock", (0, 1)), ("ID", (-6, 0))):
            bus.handler.reset_mock()
            bus.handler.writeTxRx.side_effect = None
            bus.handler.writeTxRx.return_value = reply
            with self.assertRaises(OSError):
                bus.write(1, register, 2)
            bus.handler.writeTxRx.assert_called_once()

    def test_missing_feedback_is_not_zero_and_servo_fault_flags_remain_visible(self):
        self.serial.drop.add((1, 2, 69))
        self.serial.errors[(2, 2, 56)] = 1
        with self.module.MotorStateReader("/dev/test-only") as bus:
            rows, errors = self.module.collect_states(bus, self.module.probe_scan_ids(bus))
        self.assertIsNone(rows[0][1]["Current(mA)"])
        self.assertEqual(rows[1][1]["Position"], 1234)
        self.assertTrue(any("servo_error=0x01" in error for error in errors))

    def test_busy_port_fails_without_sending_packets(self):
        with patch.object(self.serial, "open", side_effect=OSError("exclusive lock busy")):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    self.module.main(["get_motors_states", "--port", "/dev/test-only"]), 1
                )
        self.assertFalse(self.serial.requests)
        self.assertFalse(self.serial.is_open)

    def test_live_cli_keeps_headers_and_releases_port_on_interrupt(self):
        with (
            patch("shutil.get_terminal_size", return_value=os.terminal_size((180, 50))),
            patch.object(self.module.time, "sleep", side_effect=KeyboardInterrupt),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(self.module.main(["get_motors_states", "--port", "/dev/test-only"]), 0)
        for label in (
            "NAME",
            "MODEL",
            "POS",
            "OFF",
            "ANG",
            "ACC",
            "VOLT",
            "CURR(MA)",
            "TEMP",
            "PHASE",
        ):
            self.assertIn(label, output.getvalue())
        self.assertIn("\x1b[?25h", output.getvalue())
        self.assertFalse(self.serial.is_open)
        self.assertTrue(all(packet[4] == 2 for packet in self.serial.requests))

    @unittest.skipUnless((SOURCE / "examples/debug/motors.py").is_file(), "source unavailable")
    def test_original_row_formatter_and_angle_calculation_are_retained(self):
        def load_function(path, name):
            node = next(
                n
                for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            namespace = {"HALF_TURN_DEGREE": 180}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
            return namespace[name]

        old = SOURCE / "examples/debug/motors.py"
        new = ROOT / "examples/debug/motors.py"
        state = {
            "ID": 11,
            "Model": "sts3095",
            "Position": 3110,
            "Offset": -1733,
            "Angle": 273.3,
            "Acceleration": 254,
            "Voltage": 124,
            "Current(mA)": 65.0,
            "Temperature": 33,
            "Phase": 12,
        }
        for width in (60, 100, 140, 200):
            self.assertEqual(
                load_function(old, "_format_row")("motor_11", state, width),
                load_function(new, "_format_row")("motor_11", state, width),
            )
        for position in (-1, 0, 2048, 4095):
            self.assertEqual(
                load_function(old, "_motor_angle_from_position")(position),
                self.module._motor_angle_from_position(position),
            )


class MaintenanceSerial(DiagnosticSerial):
    def write(self, packet):
        result = super().write(packet)
        if packet[4:6] == bytes([3, 5]) and (packet[2], 5) not in self.ignore_writes:
            self.registers[packet[6]] = self.registers.pop(packet[2])
        return result


class MotorMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.module = script("motors.py")
        self.serial = MaintenanceSerial()
        for motor_id in self.serial.registers:
            self.serial.set(motor_id, 5, 1, motor_id)
            self.serial.set(motor_id, 55, 1, 1)
        self.factory = patch("serial.Serial", return_value=self.serial).start()
        patch.object(self.module.time, "sleep").start()
        self.addCleanup(patch.stopall)
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def writes(self):
        return [p for p in self.serial.requests if p[4] == 3]

    def test_phase_change_only_writes_selected_phase_lock_and_torque_off(self):
        self.module.configure_motor_phase("/dev/test-only", 1, 12)
        self.assertEqual(self.serial.get(1, 18, 1), 12)
        self.assertEqual(self.serial.get(1, 55, 1), 1)
        self.assertEqual(self.serial.get(1, 40, 1), 0)
        self.assertTrue(all(p[2] == 1 and p[5] in (18, 40, 55) for p in self.writes()))
        self.assertFalse(self.serial.is_open)

    def test_unchanged_phase_does_not_rewrite_or_unlock_eeprom(self):
        self.module.configure_motor_phase("/dev/test-only", 1, 0)
        self.assertFalse(any(p[5] == 18 or p[5:7] == bytes([55, 0]) for p in self.writes()))

    def test_failed_phase_verification_still_relocks(self):
        self.serial.ignore_writes.add((1, 18))
        with self.assertRaisesRegex(OSError, "verification failed"):
            self.module.configure_motor_phase("/dev/test-only", 1, 12)
        self.assertEqual(self.serial.get(1, 55, 1), 1)
        self.assertEqual(self.serial.get(1, 40, 1), 0)
        self.assertFalse(self.serial.is_open)

    def test_fault_flags_block_configuration_before_writing(self):
        self.serial.errors[(1, 2, 3)] = 1
        with self.assertRaisesRegex(OSError, "servo_error"):
            self.module.configure_motor_phase("/dev/test-only", 1, 12)
        self.assertFalse(self.writes())

    def test_id_collision_does_not_write(self):
        with self.assertRaisesRegex(ValueError, "occupied"):
            self.module.configure_motor_id("/dev/test-only", 1, 2)
        self.assertFalse(self.writes())

    def test_id_change_uses_one_port_then_locks_the_new_id(self):
        self.module.configure_motor_id("/dev/test-only", 1, 3)
        self.factory.assert_called_once()
        self.assertNotIn(1, self.serial.registers)
        self.assertEqual(self.serial.get(3, 5, 1), 3)
        self.assertEqual(self.serial.get(3, 40, 1), 0)
        self.assertEqual(self.serial.get(3, 55, 1), 1)
        self.assertFalse(self.serial.is_open)

    def test_lost_id_write_ack_still_finds_and_relocks_changed_motor(self):
        self.serial.drop.add((1, 3, 5))
        with self.assertRaises(OSError):
            self.module.configure_motor_id("/dev/test-only", 1, 3)
        self.assertEqual(self.serial.get(3, 55, 1), 1)
        self.assertEqual(self.serial.get(3, 40, 1), 0)
        self.assertFalse(self.serial.is_open)

    def test_ignored_id_write_relocks_original_id(self):
        self.serial.ignore_writes.add((1, 5))
        with self.assertRaises(OSError):
            self.module.configure_motor_id("/dev/test-only", 1, 3)
        self.assertEqual(self.serial.get(1, 55, 1), 1)
        self.assertEqual(self.serial.get(1, 40, 1), 0)

    def test_lock_failure_is_not_reported_as_success(self):
        self.serial.ignore_writes.add((1, 55))
        self.serial.set(1, 55, 1, 0)
        with self.assertRaisesRegex(OSError, "lock unconfirmed"):
            self.module.configure_motor_phase("/dev/test-only", 1, 12)

    def test_invalid_arguments_open_no_serial_port(self):
        for before, after in ((0, 1), (1, 254), (1, 1)):
            with self.assertRaises(ValueError):
                self.module.configure_motor_id("/dev/test-only", before, after)
        with self.assertRaises(ValueError):
            self.module.configure_motor_phase("/dev/test-only", 1, 256)
        self.factory.assert_not_called()

    def test_torque_off_attempts_every_discovered_motor_after_a_failure(self):
        self.serial.ignore_writes.add((1, 40))
        with self.assertRaisesRegex(OSError, "torque-off unconfirmed"):
            self.module.reset_motors_torque("/dev/test-only")
        self.assertEqual(self.serial.get(2, 40, 1), 0)
        self.assertTrue(all(p[5:7] == bytes([40, 0]) for p in self.writes()))
        self.assertFalse(self.serial.is_open)

    def test_torque_off_can_be_confirmed_while_servo_fault_remains_set(self):
        self.serial.errors[(1, 2, 3)] = 1
        self.serial.errors[(1, 3, 40)] = 1
        self.serial.errors[(1, 2, 40)] = 1
        self.module.reset_motors_torque("/dev/test-only", 1)
        self.assertEqual(self.serial.get(1, 40, 1), 0)
        self.assertTrue(all(p[2] == 1 and p[5:7] == bytes([40, 0]) for p in self.writes()))
        self.assertEqual(self.serial.errors[(1, 2, 40)], 1)

    def test_cli_dispatch_and_required_id(self):
        with patch.object(self.module, "configure_motor_phase") as configure:
            self.assertEqual(
                self.module.main(["configure_motor_phase", "--id", "1", "--set_phase", "12"]),
                0,
            )
        configure.assert_called_once_with("/dev/ttyACM0", 1, 12)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.module.main(["configure_motor_phase", "--set_phase", "12"])
        self.factory.assert_not_called()


class CameraDebugTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "snapshots"
        self.metadata = [{"type": "OpenCV", "id": "/dev/test-only"}]

    def test_cli_retains_camera_filter_output_and_duration_arguments(self):
        with patch.object(find_cameras, "save_images_from_all_cameras") as capture:
            self.assertEqual(
                main(
                    [
                        "find-cameras",
                        "opencv",
                        "--output-dir",
                        str(self.output),
                        "--record-time-s",
                        "0",
                    ]
                ),
                0,
            )
        capture.assert_called_once_with(str(self.output), 0, "opencv")

    def test_list_only_does_not_capture_or_create_output(self):
        with (
            patch.object(find_cameras, "find_and_print_cameras", return_value=self.metadata),
            patch.object(find_cameras, "OpenCVCamera") as camera,
        ):
            self.assertIsNone(find_cameras.save_images_from_all_cameras(self.output, 0))
        camera.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_capture_preserves_rgb_and_bounded_number_of_snapshot_files(self):
        camera = Mock()
        camera.read_frame_history.return_value = ((1.0, jpeg()),)
        with (
            patch.object(find_cameras, "OpenCVCamera", return_value=camera),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            find_cameras.save_images_from_cameras(self.metadata, self.output, 0.05)
        camera.start.assert_called_once()
        camera.close.assert_called_once()
        files = list(self.output.iterdir())
        self.assertEqual([p.name for p in files], ["opencv__dev_test-only.png"])
        bgr = cv2.imread(str(files[0]))
        self.assertGreater(bgr[0, 0, 2], 240)
        self.assertLess(bgr[0, 0, 0], 10)

    def test_missing_frames_are_reported_and_camera_is_closed(self):
        camera = Mock()
        camera.read_frame_history.return_value = ()
        with (
            patch.object(find_cameras, "OpenCVCamera", return_value=camera),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(OSError, "without a healthy snapshot"):
                find_cameras.save_images_from_cameras(self.metadata, self.output, 0.01)
        camera.close.assert_called_once()

    def test_existing_output_is_not_overwritten_and_invalid_input_does_not_start_capture(self):
        self.output.mkdir()
        with patch.object(find_cameras, "OpenCVCamera") as camera:
            with self.assertRaises(FileExistsError):
                find_cameras.save_images_from_cameras(self.metadata, self.output, 0.01)
            for duration in (-1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    find_cameras.save_images_from_all_cameras(self.output, duration)
            with self.assertRaises(ValueError):
                find_cameras.save_images_from_cameras([{"id": "http://camera"}], self.output, 1)
        camera.assert_not_called()

    def test_partial_start_failure_closes_all_started_cameras(self):
        first, second, third = Mock(), Mock(), Mock()
        first.read_frame_history.return_value = third.read_frame_history.return_value = (
            (1.0, jpeg()),
        )
        second.start.side_effect = OSError("start failed")
        metadata = [{"type": "OpenCV", "id": f"/dev/test-{i}"} for i in range(3)]
        with patch.object(find_cameras, "OpenCVCamera", side_effect=[first, second, third]):
            with self.assertRaisesRegex(OSError, "without a healthy snapshot.*test-1"):
                find_cameras.save_images_from_cameras(metadata, self.output, 0.01)
        first.close.assert_called_once()
        second.close.assert_called_once()
        third.start.assert_called_once()
        third.close.assert_called_once()
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {"opencv__dev_test-0.png", "opencv__dev_test-2.png"},
        )

    def test_stuck_discovery_worker_is_terminated(self):
        receiver, sender, process = Mock(), Mock(), Mock()
        receiver.poll.return_value = False
        process.pid = 123
        process.is_alive.side_effect = [True, False, False]
        context = Mock()
        context.Pipe.return_value = receiver, sender
        context.Process.return_value = process
        with patch.object(find_cameras.multiprocessing, "get_context", return_value=context):
            self.assertIsNone(find_cameras._probe_one("/dev/test-only"))
        process.terminate.assert_called_once()
        process.close.assert_called_once()
        receiver.close.assert_called_once()

    def test_no_started_camera_does_not_wait_for_capture_duration(self):
        camera = Mock()
        camera.start.side_effect = OSError("unavailable")
        with (
            patch.object(find_cameras, "OpenCVCamera", return_value=camera),
            patch.object(find_cameras.time, "sleep") as sleep,
        ):
            with self.assertRaisesRegex(OSError, "without a healthy snapshot"):
                find_cameras.save_images_from_cameras(self.metadata, self.output, 6)
        sleep.assert_not_called()
        camera.close.assert_called_once()

    def test_failed_camera_cleanup_is_not_treated_as_successful_skip(self):
        camera = Mock()
        camera.start.side_effect = OSError("unavailable")
        camera.close.side_effect = RuntimeError("Camera process did not exit")
        with patch.object(find_cameras, "OpenCVCamera", return_value=camera) as factory:
            with self.assertRaisesRegex(RuntimeError, "did not exit"):
                find_cameras.save_images_from_cameras(self.metadata * 2, self.output, 0.01)
        factory.assert_called_once()


class OtherDebugTests(unittest.TestCase):
    def test_every_script_has_inert_help_without_optional_frameworks(self):
        for path in (ROOT / "examples/debug").glob("*.py"):
            with self.subTest(script=path.name):
                result = subprocess.run(
                    [sys.executable, str(path), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_dataset_inspects_real_parquet_without_huggingface(self):
        module = script("test_dataset.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.parquet"
            pq.write_table(
                pa.table({"timestamp": [1.0, 1.001, 2.0], "action": [[0], [1], [2]]}), path
            )
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(
                    module.main([str(path), "--timestamp", "1", "--tolerance", "0.002"]), 0
                )
            self.assertIn("Matched rows: 2", output.getvalue())

    def test_network_uses_only_explicit_urls_with_timeout(self):
        module = script("test_network.py")
        response = Mock()
        response.__enter__ = Mock(return_value=SimpleNamespace(status=200))
        response.__exit__ = Mock(return_value=False)
        with (
            patch.object(module, "urlopen", return_value=response) as request,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(module.main(["https://example.test"]), 0)
        request.assert_called_once_with("https://example.test", timeout=5)

    def test_audio_output_is_visible_and_does_not_overwrite_existing_files(self):
        module = script("test_mic.py")
        sound = Mock()
        sound.query_devices.return_value = {"default_samplerate": 16000}
        sound.rec.return_value = np.zeros((48000, 1), np.int16)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"sounddevice": sound}),
            patch.dict(os.environ, {"ALOHAMINI_WORKSPACE": directory}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            path = Path(module.record_3s_wav(0))
            self.assertTrue(path.is_relative_to(Path(directory) / "logs/debug"))
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                module.record_3s_wav(0, output=path)
            self.assertEqual(path.read_bytes(), original)

    def test_asr_requires_local_model_and_never_downloads_by_name(self):
        module = script("test_mic.py")
        model = Mock()
        model.transcribe.return_value = ([SimpleNamespace(text="你好")], {})
        factory = Mock(return_value=model)
        with (
            patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=factory)}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertTrue(module.try_asr("test.wav", Path("/local/model")))
        factory.assert_called_once_with("/local/model", device="cpu", local_files_only=True)


if __name__ == "__main__":
    unittest.main()
