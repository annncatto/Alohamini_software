import contextlib
import importlib.util
import io
import json
import math
import socket
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from queue import Empty, Full, Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_feetech_device import RegisterSerial

from alohamini.calibration import EncoderCalibration
from alohamini.calibration.encoder import HostPositionUnits
from alohamini.client import HostClient
from alohamini.datasets.images import image_shape
from alohamini.hardware.camera import CameraConfig, OpenCVCamera, _capture
from alohamini.hardware.feetech_device import FeetechBusDevice
from alohamini.model import get_robot_model
from alohamini.runtime.arm_contact import (
    ArmJointSpec,
    GripperContactCalibration,
    JointContactCalibration,
)
from alohamini.runtime.host import NativeHost, RecordingCameraBuffer
from alohamini.runtime.lifecycle import CleanupFailure, CycleResult, HostPhase
from alohamini.runtime.lift_control import LiftAxisSpec


def host_fixture(*, cameras=None, command_port=5555, state_port=5556, camera_port=5557):
    """Synthetic calibration and in-memory SDK registers, never physical device files."""
    model = get_robot_model("alohamini2pro")
    calibration = EncoderCalibration(4096, 2048, 0, 1, position_min_rad=-2, position_max_rad=2)
    joints, units = [], {}
    for motor in model.actuators:
        if motor.name.startswith("arm_"):
            gripper = motor.name.endswith("gripper")
            contact = (
                GripperContactCalibration(
                    calibration.position_from_tick(1000), calibration.position_from_tick(3000)
                )
                if gripper
                else JointContactCalibration(math.radians(1))
            )
            joints.append(ArmJointSpec(motor, calibration, contact))
            units[motor.name] = HostPositionUnits(
                "range_0_100" if gripper else "range_m100_100", 1000, 3000, 0
            )
    serials, devices = {}, {}
    for source in ("left", "right"):
        motors = [motor for motor in model.actuators if motor.bus == source]
        serial = RegisterSerial()
        template = serial.registers[1].copy()
        serial.registers = {motor.motor_id: template.copy() for motor in motors}
        for motor in motors:
            motor_id = motor.motor_id
            position = motor.name.startswith("arm_")
            serial.set(motor_id, 3, 2, {"sts3250": 2825, "sts3095": 2569}[motor.motor_model])
            serial.set(motor_id, 33, 1, 0 if position else 1)
            serial.set(motor_id, 46, 2, 2000 if position else 0)
            serial.set(motor_id, 56, 2, 2048)
            serial.set(motor_id, 58, 2, 0)
            serial.set(motor_id, 66, 1, 0)
        serials[f"/dev/test-{source}"] = serial
        devices[source] = FeetechBusDevice(
            f"/dev/test-{source}",
            motors,
            position_calibrations={m.name: calibration for m in motors if m.name in units},
            velocity_limits={
                m.name: 1300 if m.name == "lift_axis" else 3000
                for m in motors
                if m.name not in units
            },
        )
    lift_motor = next(m for m in model.actuators if m.name == "lift_axis")
    host = NativeHost(
        model,
        devices,
        joints,
        LiftAxisSpec(lift_motor, 0.131, -1, 4096),
        units,
        cameras=cameras,
        command_port=command_port,
        state_port=state_port,
        camera_port=camera_port,
    )
    return host, serials


class CachedCamera:
    def __init__(self):
        self.started = self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def set_stream_enabled(self, enabled):
        self.streaming = enabled

    def read_stream_snapshot(self):
        return None

    def read_frame_history(self):
        # Cached bytes stand in for the camera worker, not a device read.
        return ((time.monotonic() - 0.001, b"jpeg"),)


def synthetic_capture(_config, output, stop, _stream_enabled):
    """Spawned test producer; deliberately never opens VideoCapture or a device."""
    output.cancel_join_thread()
    while not stop.is_set():
        try:
            output.put_nowait((time.monotonic(), b"jpeg", None, None))
        except Full:
            pass
        stop.wait(0.002)


class CameraWorkerTests(unittest.TestCase):
    def test_config_rejects_network_paths_and_invalid_capture_parameters(self):
        for fields in (
            {"device": "http://camera"},
            {"device": "/dev/video0", "fps": 60},
            {"device": "/dev/video0", "width": 0},
            {"device": "/dev/video0", "rotation": True},
        ):
            with self.assertRaises(ValueError):
                CameraConfig(**fields)

    def test_spawned_producer_populates_bounded_cache_and_closes(self):
        camera = OpenCVCamera(CameraConfig("/dev/test-only"))
        with patch("alohamini.hardware.camera._capture", synthetic_capture):
            camera.start()
        try:
            deadline = time.monotonic() + 3
            while len(camera.read_frame_history()) < 8 and time.monotonic() < deadline:
                threading.Event().wait(0.02)
            history = camera.read_frame_history()
            self.assertEqual(len(history), 8)
            self.assertTrue(all(a[0] < b[0] for a, b in zip(history, history[1:], strict=False)))
            self.assertTrue(all(jpeg == b"jpeg" for _, jpeg in history))
        finally:
            camera.close()
        self.assertEqual(camera.read_frame_history(), ())
        camera.close()

    def test_stuck_capture_is_terminated_without_unbounded_join(self):
        camera = OpenCVCamera(CameraConfig("/dev/test-only"))
        camera._stop = Mock()
        camera._process = Mock(pid=1)
        camera._process.is_alive.side_effect = (True, True, False)
        camera.close()
        camera._process.terminate.assert_called_once()
        camera._process.kill.assert_called_once()
        self.assertTrue(
            all(call.kwargs["timeout"] <= 0.3 for call in camera._process.join.call_args_list)
        )

    def test_broken_capture_channel_invalidates_cached_frames(self):
        camera = OpenCVCamera(CameraConfig("/dev/test-only"))
        camera._history.append((time.monotonic(), b"old"))
        camera._queue = Mock()
        camera._queue.get.side_effect = EOFError
        camera._receive()
        with self.assertRaisesRegex(OSError, "channel"):
            camera.read_frame_history()
        self.assertFalse(camera._history)

    def test_close_handles_receiver_that_failed_to_start(self):
        camera = OpenCVCamera(CameraConfig("/dev/test-only"))
        camera._receiver = threading.Thread(target=lambda: None)
        camera.close()

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV unavailable")
    def test_capture_preserves_deployed_rgb_decode_and_timestamp_pair(self):
        import cv2
        import numpy as np

        stop = threading.Event()
        output = Queue(maxsize=8)
        output.cancel_join_thread = lambda: None
        bgr = np.zeros((8, 16, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255
        capture = Mock()
        capture.isOpened.return_value = capture.set.return_value = True

        def read():
            stop.set()
            return True, bgr

        capture.read.side_effect = read
        before = time.monotonic()
        with patch("cv2.VideoCapture", return_value=capture):
            _capture(
                CameraConfig("/dev/test-only", width=16, height=8),
                output,
                stop,
                SimpleNamespace(value=False),
            )
        stamp, jpeg, timing, stream = output.get_nowait()
        self.assertIsNone(stream)
        self.assertEqual(timing["frames"], 1)
        self.assertGreaterEqual(timing["capture_ms"], 0)
        self.assertGreaterEqual(timing["encode_ms"], 0)
        self.assertGreaterEqual(stamp, before)
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(int(decoded[0, 0, 0]), 240)
        self.assertLess(int(decoded[0, 0, 2]), 10)
        capture.release.assert_called_once()


class HostLifecycleTests(unittest.TestCase):
    def test_camera_port_is_bound_before_any_hardware_is_opened(self):
        import zmq

        with zmq.Context() as context, context.socket(zmq.PUB) as occupied:
            occupied.linger = 0
            camera_port = occupied.bind_to_random_port("tcp://127.0.0.1")
            with socket.socket() as command, socket.socket() as state:
                command.bind(("127.0.0.1", 0))
                state.bind(("127.0.0.1", 0))
                ports = command.getsockname()[1], state.getsockname()[1]
            camera = CachedCamera()
            host, _ = host_fixture(
                cameras={"forward": camera},
                command_port=ports[0],
                state_port=ports[1],
                camera_port=camera_port,
            )
            with patch("serial.Serial") as serial:
                with self.assertRaisesRegex(RuntimeError, "publisher failed to start"):
                    host.start()
                serial.assert_not_called()
            self.assertFalse(camera.started)
            self.assertFalse(host._camera_stream._thread.is_alive())

    def test_camera_stream_cleanup_failure_does_not_skip_motor_or_camera_cleanup(self):
        host, _ = host_fixture()
        order = Mock()
        host.supervisor = order.supervisor
        host.supervisor.close.return_value.cleanup_failures = ()
        host._camera_stream = order.stream
        host._camera_stream.close.side_effect = RuntimeError("publisher close failed")
        host._attempted_cameras = [order.camera]
        with self.assertRaisesRegex(RuntimeError, "publisher close failed"):
            host.close()
        self.assertEqual(
            [item[0] for item in order.mock_calls],
            ["supervisor.close", "stream.close", "camera.close"],
        )

    def test_camera_free_host_has_no_camera_publisher(self):
        host, _ = host_fixture()
        self.assertIsNone(host._camera_stream)
        host.close()

    def test_cleanup_failure_is_reported_after_all_resources_are_closed(self):
        host, _ = host_fixture()
        host.supervisor = Mock()
        host.supervisor.close.return_value.cleanup_failures = (
            CleanupFailure("left", "disable_torque", "readback failed"),
        )
        first, second = Mock(), Mock()
        first.close.side_effect = OSError("camera cleanup failed")
        host._attempted_cameras = [first, second]
        host._commands, host._states, host._context = Mock(), Mock(), Mock()
        with self.assertRaisesRegex(RuntimeError, "readback failed"):
            host.close()
        second.close.assert_called_once()
        host._commands.close.assert_called_once_with(linger=0)
        host._states.close.assert_called_once_with(linger=0)
        host._context.term.assert_called_once()
        host.close()
        host.supervisor.close.assert_called_once()

    def test_camera_interrupt_does_not_skip_other_cleanup(self):
        host, _ = host_fixture()
        first, second = Mock(), Mock()
        first.close.side_effect = KeyboardInterrupt
        host._attempted_cameras = [first, second]
        host._states, host._context = Mock(), Mock()
        with self.assertRaises(KeyboardInterrupt):
            host.close()
        second.close.assert_called_once()
        host._states.close.assert_called_once_with(linger=0)
        host._context.term.assert_called_once()

    def test_fault_payload_does_not_publish_retained_height_as_valid_feedback(self):
        host, _ = host_fixture()
        host.control.base_lift = Mock(lift_height_m=0.2, measured_base_velocity=None)
        status = replace(host.supervisor.status, phase=HostPhase.FAULT, fault="feedback lost")
        payload = host._payload(CycleResult((), False, status))
        self.assertFalse(payload["_safety"]["feedback_valid"])
        self.assertFalse(payload["_safety"]["lift_reference_valid"])
        self.assertNotIn("lift_axis.height_mm", payload)


class HostTimingTests(unittest.TestCase):
    def run_host(self, enabled):
        camera = Mock()
        camera.timing_stats.return_value = {"frames": 30, "capture_ms": 600.0, "encode_ms": 60.0}
        host, _ = host_fixture(cameras={"forward": camera})
        host._started = True
        host.timing_ms = {
            "command": 0.1,
            "robot_observation": 7.0,
            "robot_action": 1.0,
            "left_bus": 4.0,
            "right_bus": 3.0,
            "response_pack": 0.2,
            "response_send": 0.1,
            "camera_cache_forward": 0.05,
        }
        host.control.timing_ms = {"prepare": 0.1, "arms": 0.6, "base_lift": 0.3}
        result = CycleResult((), True, replace(host.supervisor.status, phase=HostPhase.READY))
        clock = SimpleNamespace(now=0.0, loops=0)

        def wait(_duration):
            clock.now += 0.1
            clock.loops += 1

        stop = SimpleNamespace(is_set=lambda: clock.loops >= 22, wait=wait)
        with (
            patch.object(host, "step", return_value=result),
            patch.object(host, "close") as close,
            patch("time.perf_counter", side_effect=lambda: clock.now),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            host.run(stop, profile_timing=enabled)
        close.assert_called_once()
        return output.getvalue(), camera

    def test_profile_output_uses_measured_stages_and_original_prefixes(self):
        output, _ = self.run_host(True)
        lines = output.splitlines()
        self.assertEqual(sum(line.startswith("[HOST TIMING avg ms/loop]") for line in lines), 2)
        self.assertEqual(
            sum(line.startswith("[HOST ACTION avg ms/control-cycle]") for line in lines), 2
        )
        self.assertIn("robot_obs=7.0 robot_action=1.0 left_bus=4.0 right_bus=3.0", output)
        self.assertIn("prepare=0.1 arms=0.6 base_lift=0.3 total=1.0", output)
        self.assertIn("[HOST CAMERA avg ms/frame][forward] n=30 capture=20.0 encode=2.0", output)
        self.assertEqual(
            output.count("[HOST CAMERA"), 1
        )  # No repeated accounting of cached frames.
        self.assertNotIn("currents=0.0", output)
        self.assertNotIn("jpeg=", output)  # Worker cost is not Host-thread latency.

    def test_timing_is_opt_in_as_in_source_host(self):
        output, camera = self.run_host(False)
        self.assertEqual(output, "")
        camera.timing_stats.assert_not_called()


class CameraPairingTests(unittest.TestCase):
    def test_episode_does_not_backfill_or_reuse_images(self):
        buffer = RecordingCameraBuffer()
        token = b"request:episode:record"
        histories = {"front": ((1.0, b"old"),), "wrist": ((1.0, b"old"),)}
        self.assertEqual(buffer.select(histories, b"pc", token, now=1.01), {})
        histories = {"front": ((1.02, b"a"),), "wrist": ((1.024, b"b"),)}
        selected = buffer.select(histories, b"pc", token, now=1.03)
        self.assertEqual(selected, {"front": (1.02, b"a"), "wrist": (1.024, b"b")})
        self.assertEqual(buffer.select(histories, b"pc", token, now=1.04), {})
        self.assertEqual(buffer.select(histories, b"pc", b"q:new:record", now=1.04), {})

    def test_clients_have_independent_cursors_and_sessions_are_bounded(self):
        buffer = RecordingCameraBuffer()
        for client in (b"a", b"b"):
            buffer.select({"front": ()}, client, b"q:episode:record", now=1)
        histories = {"front": ((1.02, b"jpeg"),)}
        for client in (b"a", b"b"):
            self.assertTrue(buffer.select(histories, client, b"q:episode:record", now=1.03))
        for i in range(20):
            buffer.select(histories, b"pc", f"q:{i}:record".encode(), now=1.04)
        self.assertEqual(len(buffer._cursors), 8)

    def test_missing_stale_and_skewed_images_do_not_form_a_group(self):
        for histories in (
            {"front": ((1.02, b"a"),), "wrist": ()},
            {"front": ((1.02, b"a"),), "wrist": ((1.06, b"b"),)},
            {"front": ((0.5, b"a"),), "wrist": ((0.5, b"b"),)},
        ):
            buffer = RecordingCameraBuffer()
            buffer.select({name: () for name in histories}, b"pc", b"q:ep:record", now=1)
            self.assertEqual(buffer.select(histories, b"pc", b"q:ep:record", now=1.07), {})


@unittest.skipUnless(
    importlib.util.find_spec("zmq")
    and importlib.util.find_spec("serial")
    and importlib.util.find_spec("scservo_sdk"),
    "Host dependencies unavailable",
)
class HostIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "PC dataset dependencies unavailable")
    def test_replay_uses_native_acknowledgements_and_stops_owned_base(self):
        import numpy as np
        from test_replay import replay_snapshot

        from alohamini.apps.replay import ReplayEpisode, run_replay
        from alohamini.datasets.native import state_names

        # This fixture normally omits the startup calibration metadata. Supply
        # the synthetic installed ranges, never a physical calibration file.
        def prepare():
            self.host._metadata["motors"] = replay_snapshot().payload["_robot_metadata"]["motors"]
            self.host.begin_lift_homing()

        self.serials["/dev/test-left"].set(11, 69, 2, 50)
        self.hooks.put(prepare)
        state = self.wait_for(lambda p: p["_safety"]["lift_homing_phase"] == "complete")
        names = state_names("alohamini2pro")
        actions = np.asarray([[state.payload[key] for key in names]] * 3)
        actions[:, 0] += np.arange(3)
        actions[:, names.index("x.vel")] = 0.05
        episode = ReplayEpisode(
            Path("/test-only"), 0, 30, state.payload["_robot_metadata"], tuple(names), actions
        )
        with (
            patch.object(self.client, "send_command", wraps=self.client.send_command) as send,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run_replay(self.client, episode)
        self.assertEqual(send.call_count, 4)  # Three frames plus measured hold/zero.
        for call, expected in zip(send.call_args_list[:3], actions, strict=True):
            self.assertEqual(call.args[0], dict(zip(names, expected, strict=True)))
        final = self.client.read().payload["_safety"]
        self.assertEqual(final["command"]["sequence"], 4)
        self.assertEqual(final["watchdog_events"], 0)
        self.assertEqual(final["joint_hold_events"], 0)
        self.assertTrue(
            all(final["requested_targets"][key] == 0 for key in ("x.vel", "y.vel", "theta.vel"))
        )
        self.assertTrue(
            all(self.serials["/dev/test-left"].get(i, 46, 2) == 0 for i in (8, 9, 10, 11))
        )

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "PC dataset dependencies unavailable")
    def test_native_recording_pairs_real_host_feedback_images_and_commands_into_parquet(self):
        import pyarrow.parquet as pq
        from test_dataset import jpeg

        from alohamini.apps.recording import record_loop
        from alohamini.datasets.native import LocalDataset
        from alohamini.datasets.tools import IntegrityChecker

        image = jpeg()
        self.camera.read_frame_history = lambda: ((time.monotonic() - 0.001, image),)
        self.serials["/dev/test-left"].set(11, 69, 2, 50)
        self.hooks.put(self.host.begin_lift_homing)
        state = self.wait_for(lambda p: p["_safety"]["lift_homing_phase"] == "complete")
        metadata = state.payload["_robot_metadata"]
        leader, keyboard = Mock(), Mock()
        leader.read.side_effect = lambda units: {
            key: 50.0 if key.endswith("gripper.pos") else 20.0 for key in units
        }
        keyboard.read.return_value = set()
        keyboard.events = dict(exit_early=False, rerecord_episode=False, stop_recording=False)
        with tempfile.TemporaryDirectory() as directory:
            dataset = LocalDataset(
                Path(directory) / "capture", fps=30, task="pick", robot_metadata=metadata
            )
            try:
                dataset.begin_episode()
                with contextlib.redirect_stdout(io.StringIO()):
                    record_loop(
                        self.client,
                        "alohamini2pro",
                        leader,
                        keyboard,
                        fps=30,
                        duration_s=0.35,
                        metadata=metadata,
                        dataset=dataset,
                    )
                dataset.save_episode()
                episode = dataset.root / "episodes/episode_000000"
                rows = pq.read_table(episode / "frames.parquet").to_pylist()
                self.assertGreater(len(rows), 0)
                self.assertLessEqual(len(rows), 11)
                for row in rows:
                    self.assertEqual(row["action"][0], 20.0)
                    self.assertTrue(all(row["motor_feedback.current_raw_valid"]))
                    self.assertEqual(len(row["observation.motor_current_raw"]), 18)
                    self.assertEqual(
                        image_shape(episode, "forward", row["observation.images.forward"])[2], 3
                    )
                logs = [
                    json.loads(line) for line in (episode / "safety.jsonl").read_text().splitlines()
                ]
                captures = [row for row in logs if row["frame_index"] is not None]
                self.assertEqual(len(captures), len(rows))
                for row in captures:
                    timing = row["client_timing"]
                    self.assertLessEqual(
                        timing["observation_received_monotonic_s"],
                        timing["action_sample_started_monotonic_s"],
                    )
                    self.assertEqual(row["issued_command"]["client_id"], self.client.client_id)
                self.assertEqual(self.serials["/dev/test-left"].get(1, 42, 2), 2048)
                self.assertTrue(
                    all(self.serials["/dev/test-left"].get(i, 46, 2) == 0 for i in (8, 9, 10, 11))
                )
            finally:
                dataset.close()
            report = IntegrityChecker(dataset.root, decode_images=True).run()
            self.assertTrue(report["valid"], report)
            self.assertEqual(report["summary"]["frames"], len(rows))

    def setUp(self):
        self.stop = threading.Event()
        self.ready, self.errors = Queue(), Queue()
        self.camera = CachedCamera()
        self.hooks = Queue()
        # Reserve distinct loopback ports for state, commands and camera stream.
        reservations = [socket.socket(), socket.socket(), socket.socket()]
        for reservation in reservations:
            reservation.bind(("127.0.0.1", 0))
        self.command_port, self.state_port, self.camera_port = [
            s.getsockname()[1] for s in reservations
        ]
        for reservation in reservations:
            reservation.close()

        def serve():
            host, serials = host_fixture(
                cameras={"forward": self.camera},
                command_port=self.command_port,
                state_port=self.state_port,
                camera_port=self.camera_port,
            )

            def serial_factory(**_kwargs):
                # FeetechBusDevice assigns port immediately after construction.
                class PortSelection:
                    def __setattr__(self, name, value):
                        if name == "port":
                            object.__setattr__(self, "selected", serials[value])
                        else:
                            setattr(self.selected, name, value)

                    def __getattr__(self, name):
                        return getattr(self.selected, name)

                return PortSelection()

            try:
                with patch("serial.Serial", side_effect=serial_factory):
                    host.start()
                    original_step = host.step

                    def step():
                        try:
                            task = self.hooks.get_nowait()
                        except Empty:
                            pass
                        else:
                            task()
                        return original_step()

                    host.step = step
                    self.ready.put((host, serials))
                    host.run(self.stop)
            except BaseException as exc:
                self.errors.put(exc)
                self.ready.put(None)

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
        info = self.ready.get(timeout=3)
        if info is None:
            raise self.errors.get()
        self.host, self.serials = info
        self.client = HostClient(
            "127.0.0.1",
            port=self.state_port,
            command_port=self.command_port,
            expected_model="alohamini2pro",
            timeout_s=1,
            request_window=1,
        )

    def tearDown(self):
        self.client.close()
        self.stop.set()
        self.thread.join(timeout=3)
        self.assertFalse(self.thread.is_alive())
        self.assertTrue(self.camera.closed)
        self.assertTrue(all(not serial.is_open for serial in self.serials.values()))
        if not self.errors.empty():
            raise self.errors.get()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            state = self.client.read()
            if predicate(state.payload):
                return state
        self.fail("Host did not publish expected state")

    def send_raw(self, data):
        import zmq

        before = self.client.read().payload["_safety"]["rejected_commands"]
        with zmq.Context() as context:
            with context.socket(zmq.PUSH) as writer:
                writer.setsockopt(zmq.IMMEDIATE, 1)
                writer.setsockopt(zmq.LINGER, 100)
                writer.connect(f"tcp://127.0.0.1:{self.command_port}")
                self.assertTrue(writer.poll(1000, zmq.POLLOUT))
                writer.send(data)
                # The deployed CONFLATE receiver may drop its final unread message
                # when a peer disconnects. Keep the peer alive until observed.
                self.wait_for(lambda p: p["_safety"]["rejected_commands"] > before)

    def test_full_model_feedback_has_all_raw_fields_without_fake_lift_height(self):
        state = self.client.read()
        self.assertEqual(state.payload["_safety"]["phase"], "ready")
        self.assertNotIn("lift_axis.height_mm", state.payload)
        self.assertFalse(state.payload["_safety"]["lift_reference_valid"])
        motors = state.payload["_motor_feedback"]["motors"]
        self.assertEqual(len(motors), 18)
        for values in motors.values():
            self.assertEqual(values["position_raw"], 2048)
            self.assertEqual(values["velocity_raw"], 0)
            self.assertIn("temperature_raw", values)
            self.assertIn("current_ma", values)
            self.assertLessEqual(values["sample_started_s"], values["sample_finished_s"])

    def test_client_command_reaches_actual_sdk_registers_and_acknowledged_state(self):
        state = self.client.read()
        identity = self.client.send_command(
            {"arm_left_shoulder_pan.pos": 20, "x.vel": 0.1}, based_on=state
        )
        actual = self.wait_for(
            lambda p: p["_safety"]["command"].get("sequence") == identity.sequence
        )
        self.assertEqual(actual.payload["_safety"]["control_owner"], self.client.client_id)
        self.assertEqual(self.serials["/dev/test-left"].get(1, 42, 2), 2200)
        self.assertNotEqual(self.serials["/dev/test-left"].get(8, 46, 2), 0)

    def test_native_recording_cursor_survives_state_image_switches(self):
        self.client.set_recording_cameras(True)
        first = self.client.read(include_images=True)
        self.assertTrue(first.payload["_camera_buffer"]["pending"])
        self.client.read()
        second = self.client.read(include_images=True)
        self.assertFalse(second.payload["_camera_buffer"]["pending"])
        self.assertTrue(second.images)
        self.client.set_recording_cameras(True)
        reset = self.client.read(include_images=True)
        self.assertTrue(reset.payload["_camera_buffer"]["pending"])

    def test_optional_motor_telemetry_loss_does_not_stop_host_or_fake_extra_fields(self):
        self.hooks.put(lambda: self.serials["/dev/test-left"].drop.add((1, 0x82, 56)))
        state = self.wait_for(
            lambda p: (
                "temperature_raw" not in p["_motor_feedback"]["motors"]["arm_left_shoulder_pan"]
            )
        )
        feedback = state.payload["_motor_feedback"]["motors"]["arm_left_shoulder_pan"]
        self.assertTrue(state.payload["_safety"]["feedback_valid"])
        self.assertIn("position_raw", feedback)
        self.assertIn("current_raw", feedback)
        identity = self.client.send_command({"arm_left_shoulder_pan.pos": 20}, based_on=state)
        self.wait_for(lambda p: p["_safety"]["command"].get("sequence") == identity.sequence)
        self.assertEqual(self.serials["/dev/test-left"].get(1, 42, 2), 2200)

    def test_arm_only_wire_command_zeros_previous_base_motion(self):
        state = self.client.read()
        self.client.send_command({"x.vel": 0.1}, based_on=state)
        state = self.wait_for(lambda p: bool(p["_safety"]["command"]))
        identity = self.client.send_command({"arm_left_shoulder_pan.pos": 10}, based_on=state)
        self.wait_for(lambda p: p["_safety"]["command"].get("sequence") == identity.sequence)
        self.assertTrue(all(self.serials["/dev/test-left"].get(i, 46, 2) == 0 for i in (8, 9, 10)))

    def test_unknown_height_rejects_entire_target_without_claiming_control(self):
        state = self.client.read()
        self.client.send_command(
            {"arm_left_shoulder_pan.pos": 10, "lift_axis.height_mm": 200}, based_on=state
        )
        rejected = self.wait_for(lambda p: p["_safety"]["rejected_commands"] > 0)
        self.assertIsNone(rejected.payload["_safety"]["control_owner"])
        self.assertEqual(rejected.payload["_safety"]["phase"], "ready")
        self.assertEqual(self.serials["/dev/test-left"].get(1, 42, 2), 3000)

    def test_invalid_json_is_rejected_but_control_sampling_continues(self):
        before = self.client.read()
        self.send_raw(b'{"x.vel":NaN}')
        after = self.wait_for(lambda p: p["_safety"]["rejected_commands"] > 0)
        self.assertGreater(
            after.payload["_host_timing"]["state_sequence"],
            before.payload["_host_timing"]["state_sequence"],
        )
        self.assertEqual(after.payload["_safety"]["phase"], "ready")

    def test_stale_session_is_not_applied(self):
        self.send_raw(
            json.dumps(
                {
                    "x.vel": 0.1,
                    "_command": {
                        "client_id": "another",
                        "sequence": 1,
                        "host_session_id": "old-session",
                        "control_epoch": 0,
                    },
                }
            ).encode()
        )
        state = self.wait_for(lambda p: p["_safety"]["rejected_commands"] > 0)
        self.assertIsNone(state.payload["_safety"]["control_owner"])
        self.assertTrue(all(self.serials["/dev/test-left"].get(i, 46, 2) == 0 for i in (8, 9, 10)))

    def test_camera_cache_does_not_change_state_only_wire_reply(self):
        state = self.client.read()
        self.assertEqual(state.images, {})
        full = self.client.read(include_images=True)
        self.assertEqual(full.images, {"forward": b"jpeg"})
        self.assertIn("forward", full.payload["_host_timing"]["camera_capture_monotonic_s"])

    def test_camera_subscription_does_not_own_or_block_robot_requests(self):
        import zmq
        from test_dataset import jpeg

        from alohamini.hardware.camera import JpegSnapshot

        stamp = time.monotonic()
        native, standard = jpeg(), jpeg(color=(0, 0, 255))
        self.camera.read_frame_history = lambda: ((stamp, native),)
        self.camera.read_stream_snapshot = lambda: JpegSnapshot(stamp, standard, 24, 16)
        with zmq.Context() as context, context.socket(zmq.SUB) as subscriber:
            subscriber.linger = 0
            subscriber.setsockopt(zmq.SUBSCRIBE, b"camera/forward")
            subscriber.connect(f"tcp://127.0.0.1:{self.camera_port}")
            self.assertTrue(subscriber.poll(1000))
            topic, metadata, image = subscriber.recv_multipart()
            full = self.client.read(include_images=True)
            self.assertEqual(topic, b"camera/forward")
            self.assertEqual(image, standard)
            self.assertEqual(full.images["forward"], native)
            self.assertEqual(
                json.loads(metadata)["capture_monotonic_s"],
                full.payload["_host_timing"]["camera_capture_monotonic_s"]["forward"],
            )
            self.assertEqual(
                json.loads(metadata)["host_session_id"], full.payload["_safety"]["host_session_id"]
            )
            self.assertIsNone(full.payload["_safety"]["control_owner"])
            identity = self.client.send_command({"x.vel": 0.05}, based_on=full)
            current = self.wait_for(
                lambda p: p["_safety"]["command"].get("sequence") == identity.sequence
            )
            self.assertGreater(
                current.payload["_host_timing"]["state_sequence"],
                full.payload["_host_timing"]["state_sequence"],
            )

    def test_camera_failure_reports_missing_images_without_stopping_control(self):
        with patch.object(
            self.camera, "read_frame_history", side_effect=OSError("camera unplugged")
        ):
            state = self.client.read(include_images=True)
            self.assertEqual(state.images, {})
            self.assertEqual(state.payload["_camera_status"]["unavailable"], ["forward"])
            self.assertEqual(state.payload["_safety"]["phase"], "ready")
            self.client.send_command({"x.vel": 0}, based_on=state)
            self.wait_for(lambda p: bool(p["_safety"]["command"]))

    def test_oversized_image_response_does_not_stop_control(self):
        # Lower the message cap rather than allocating oversized real image frames.
        with (
            patch("alohamini.protocol.MAX_MESSAGE_BYTES", 32768),
            patch.object(
                self.camera,
                "read_frame_history",
                side_effect=lambda: ((time.monotonic() - 0.001, b"j" * 32768),),
            ),
        ):
            state = self.client.read(include_images=True)
            self.assertEqual(state.images, {})
            self.assertIn("limit", state.payload["_camera_status"]["error"])
            self.assertEqual(state.payload["_host_timing"]["camera_capture_monotonic_s"], {})
            self.assertEqual(state.payload["_safety"]["phase"], "ready")
            identity = self.client.send_command({"x.vel": 0.1}, based_on=state)
            self.wait_for(lambda p: p["_safety"]["command"].get("sequence") == identity.sequence)

    def test_gripper_contact_does_not_become_a_joint_stall_event(self):
        self.serials["/dev/test-left"].set(7, 69, 2, 100)
        state = self.client.read()
        self.client.send_command({"arm_left_gripper.pos": 0}, based_on=state)
        state = self.wait_for(lambda p: bool(p["_safety"]["gripper_holds"]))
        self.assertIn("arm_left_gripper", state.payload["_safety"]["gripper_holds"])
        self.assertEqual(state.payload["_safety"]["joint_holds"], {})
        self.assertEqual(state.payload["_safety"]["joint_hold_events"], 0)

    def test_joint_stall_remains_visible_and_allows_retreat(self):
        self.serials["/dev/test-left"].set(1, 69, 2, 340)
        state = self.client.read()
        self.client.send_command({"arm_left_shoulder_pan.pos": 100}, based_on=state)
        state = self.wait_for(lambda p: bool(p["_safety"]["joint_holds"]))
        self.assertIn("arm_left_shoulder_pan", state.payload["_safety"]["joint_holds"])
        self.assertEqual(state.payload["_safety"]["joint_hold_events"], 1)
        self.client.send_command({"arm_left_shoulder_pan.pos": 0}, based_on=state)
        self.wait_for(lambda p: not p["_safety"]["joint_holds"])

    def test_watchdog_runs_even_without_observation_requests(self):
        state = self.client.read()
        self.client.send_command({"x.vel": 0.1}, based_on=state)
        self.wait_for(lambda p: bool(p["_safety"]["command"]))
        threading.Event().wait(1.1)
        stopped = self.client.read()
        self.assertEqual(stopped.payload["_safety"]["watchdog_events"], 1)
        self.assertIsNone(stopped.payload["_safety"]["control_owner"])
        self.assertEqual(stopped.payload["_safety"]["command"], {})
        self.assertEqual(self.serials["/dev/test-left"].get(8, 46, 2), 0)

    def test_local_homing_keeps_state_and_images_available_but_rejects_motion(self):
        self.serials["/dev/test-left"].set(11, 69, 2, 50)
        self.hooks.put(self.host.begin_lift_homing)
        state = self.wait_for(lambda p: p["_safety"]["lift_homing_phase"] == "seeking")
        self.assertFalse(state.payload["_safety"]["lift_reference_valid"])
        self.client.send_command({"arm_left_shoulder_pan.pos": 30}, based_on=state)
        rejected = self.wait_for(lambda p: p["_safety"]["rejected_commands"] > 0)
        self.assertIsNone(rejected.payload["_safety"]["control_owner"])
        self.assertIn("homing", rejected.payload["_safety"]["last_command_error"])
        full = self.client.read(include_images=True)
        self.assertEqual(full.images, {"forward": b"jpeg"})
        self.assertGreater(
            full.payload["_host_timing"]["state_sequence"],
            state.payload["_host_timing"]["state_sequence"],
        )
        homed = self.wait_for(lambda p: p["_safety"]["lift_homing_phase"] == "complete")
        self.assertEqual(homed.payload["lift_axis.height_mm"], 0)
        self.assertEqual(homed.payload["_safety"]["lift_reference_source"], "current_contact")
        self.assertEqual(self.serials["/dev/test-left"].get(11, 46, 2), 0)
        self.assertEqual(self.serials["/dev/test-left"].get(1, 42, 2), 2048)

    def test_teleoperation_preview_and_exit_share_the_live_host_control_lease(self):
        from alohamini.apps.teleoperation import run_loop

        self.serials["/dev/test-left"].set(11, 69, 2, 50)
        self.hooks.put(self.host.begin_lift_homing)
        self.wait_for(lambda p: p["_safety"]["lift_homing_phase"] == "complete")
        stop = threading.Event()
        frames, register_targets = [], []
        leader, keyboard = Mock(), Mock()
        leader.read.side_effect = lambda units: {
            key: 50.0 if key.endswith("gripper.pos") else 20.0 for key in units
        }
        keyboard.read.return_value = {"w"}

        def preview(snapshot, action):
            frames.append((snapshot, action))
            register_targets.append(self.serials["/dev/test-left"].get(1, 42, 2))
            if len(frames) >= 12:
                stop.set()

        run_loop(
            self.client,
            "alohamini2pro",
            leader,
            keyboard,
            on_frame=preview,
            stop_event=stop,
        )
        self.assertEqual(len(frames), 12)
        self.assertTrue(any(s.images == {"forward": b"jpeg"} for s, _ in frames))
        self.assertTrue(any(not s.images for s, _ in frames))
        self.assertTrue(all(len(action) == 18 for _, action in frames))
        self.assertTrue(all(action["x.vel"] == 0.15 for _, action in frames))
        self.assertIn(2200, register_targets)
        stopped = self.client.read().payload["_safety"]
        self.assertEqual(stopped["rejected_commands"], 0)
        self.assertEqual(stopped["watchdog_events"], 0)
        self.assertEqual(stopped["control_owner"], self.client.client_id)
        self.assertEqual(stopped["command"]["sequence"], 13)
        self.assertEqual(self.serials["/dev/test-left"].get(1, 42, 2), 2048)
        self.assertTrue(
            all(self.serials["/dev/test-left"].get(i, 46, 2) == 0 for i in (8, 9, 10, 11))
        )

    def test_stop_event_closes_devices_without_waiting_for_client(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.host.supervisor.status.phase, HostPhase.CLOSED)
        for serial in self.serials.values():
            self.assertTrue(all(serial.get(i, 40, 1) == 0 for i in serial.registers))


if __name__ == "__main__":
    unittest.main()
