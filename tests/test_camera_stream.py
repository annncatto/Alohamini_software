import json
import socket
import threading
import time
import unittest
from queue import Full, Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import zmq

from alohamini.hardware.camera import CameraConfig, JpegSnapshot, OpenCVCamera, _capture
from alohamini.runtime.camera_stream import CameraStreamPublisher, encode_camera_stream_message


def synthetic_stream_capture(_config, output, stop, stream_enabled):
    """Spawn-test producer; no VideoCapture or physical device access."""
    output.cancel_join_thread()
    while not stop.is_set():
        stamp = time.monotonic()
        stream = JpegSnapshot(stamp, b"standard", 16, 8) if stream_enabled.value else None
        try:
            output.put_nowait((stamp, b"native", {"frames": 1}, stream))
        except Full:
            pass
        stop.wait(0.005)


def capture_frame(enabled=True, rotation=0, fail_stream=False):
    """One synthetic V4L2 read; no device is opened."""
    stop, stream_enabled = threading.Event(), SimpleNamespace(value=enabled)
    output = Queue(maxsize=8)
    output.cancel_join_thread = lambda: None
    bgr = np.zeros((8, 16, 3), np.uint8)
    bgr[:, :, 2] = 255
    bgr[:, :, 1] = np.arange(8, dtype=np.uint8)[:, None] * 12
    bgr[:, :, 0] = np.arange(16, dtype=np.uint8)[None, :] * 6
    camera = Mock()
    camera.isOpened.return_value = camera.set.return_value = True
    camera.get.return_value = 30.0

    def read():
        stop.set()
        return True, bgr

    camera.read.side_effect = read
    encode = cv2.imencode
    calls = []

    def encoding(*args):
        calls.append(args)
        return (False, None) if fail_stream and len(calls) == 2 else encode(*args)

    with (
        patch("cv2.VideoCapture", return_value=camera),
        patch("cv2.imencode", side_effect=encoding),
    ):
        _capture(
            CameraConfig("/dev/test-only", width=16, height=8, rotation=rotation),
            output,
            stop,
            stream_enabled,
        )
    camera.release.assert_called_once()
    return output.get_nowait(), len(calls)


class StreamCaptureTests(unittest.TestCase):
    def test_spawned_capture_receives_subscription_demand_and_returns_paired_bytes(self):
        camera = OpenCVCamera(CameraConfig("/dev/test-only"))
        self.addCleanup(camera.close)
        camera.set_stream_enabled(True)  # A subscriber may precede camera startup.
        with patch("alohamini.hardware.camera._capture", synthetic_stream_capture):
            camera.start()
        deadline = time.monotonic() + 3
        snapshot = None
        while snapshot is None and time.monotonic() < deadline:
            snapshot = camera.read_stream_snapshot()
            threading.Event().wait(0.005)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.jpeg, b"standard")
        with camera._lock:
            self.assertIn((camera._stream_snapshot.capture_monotonic_s, b"native"), camera._history)
        camera.set_stream_enabled(False)
        self.assertFalse(camera._stream_enabled.value)
        self.assertIsNone(camera.read_stream_snapshot())

    def test_standard_stream_preserves_source_color_and_native_jpeg_bytes(self):
        native, native_encodes = capture_frame(False)
        for rotation in (0, 90, 180, 270):
            (stamp, jpeg, timing, stream), encodes = capture_frame(rotation=rotation)
            self.assertEqual(encodes, 2)
            self.assertEqual(stream.capture_monotonic_s, stamp)
            self.assertEqual(
                (stream.width, stream.height), (8, 16) if rotation in (90, 270) else (16, 8)
            )
            decoded_native = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            decoded_ros = cv2.imdecode(np.frombuffer(stream.jpeg, np.uint8), cv2.IMREAD_COLOR)
            self.assertGreater(decoded_native[0, 0, 0], 240)  # Deployed RGB convention.
            self.assertGreater(decoded_ros[0, 0, 2], 240)  # Standard JPEG -> BGR.
            self.assertEqual(timing["stream_frames"], 1)
            self.assertEqual(timing["stream_errors"], 0)
            self.assertGreaterEqual(timing["stream_encode_ms"], 0)
            if rotation == 0:
                self.assertEqual(jpeg, native[1])
        self.assertEqual(native_encodes, 1)
        self.assertIsNone(native[3])
        self.assertEqual(native[2]["stream_encode_ms"], 0)

    def test_stream_encoding_failure_does_not_discard_native_capture(self):
        (stamp, jpeg, timing, stream), _ = capture_frame(fail_stream=True)
        self.assertGreater(stamp, 0)
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))
        self.assertIsNone(stream)
        self.assertEqual(timing["frames"], 1)
        self.assertEqual(timing["stream_errors"], 1)

    def test_receiver_keeps_one_stream_frame_and_native_eight_frame_history(self):
        camera = OpenCVCamera(CameraConfig("/dev/test-only"))
        self.addCleanup(camera.close)
        camera.set_stream_enabled(True)
        packets = [
            (float(i), b"native", {"frames": i}, JpegSnapshot(float(i), b"standard", 16, 8))
            for i in range(1, 21)
        ]

        def get(**_kwargs):
            packet = packets.pop(0)
            if not packets:
                camera._receiver_stop.set()
            return packet

        camera._queue = Mock()
        camera._queue.get.side_effect = get
        camera._receive()
        self.assertEqual(len(camera.read_frame_history()), 8)
        self.assertEqual(camera.read_stream_snapshot().capture_monotonic_s, 20)
        camera.set_stream_enabled(False)
        self.assertIsNone(camera.read_stream_snapshot())
        self.assertEqual(len(camera.read_frame_history()), 8)

    def test_closed_or_failed_capture_cannot_keep_streaming_stale_frames(self):
        camera = OpenCVCamera(CameraConfig("/dev/test-only"))
        camera._stream_enabled = SimpleNamespace(value=False)
        camera.set_stream_enabled(True)
        self.assertTrue(camera._stream_enabled.value)
        camera._stream_snapshot = JpegSnapshot(1, b"jpeg", 16, 8)
        camera._queue = Mock()
        camera._queue.get.side_effect = EOFError
        camera._receive()
        with self.assertRaisesRegex(OSError, "channel closed"):
            camera.read_stream_snapshot()
        self.assertIsNone(camera._stream_snapshot)
        camera.close()
        camera.set_stream_enabled(True)
        self.assertFalse(camera._stream_enabled.value)


class CachedStream:
    def __init__(self):
        self.enabled = threading.Event()
        self.snapshot = None
        self.reads = 0

    def set_stream_enabled(self, enabled):
        self.enabled.set() if enabled else self.enabled.clear()

    def read_stream_snapshot(self):
        self.reads += 1
        return self.snapshot

    def update(self, stamp=None):
        self.snapshot = JpegSnapshot(
            time.monotonic() if stamp is None else stamp, b"\xff\xd8jpeg", 16, 8
        )


class CameraStreamTests(unittest.TestCase):
    def setUp(self):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            self.port = reservation.getsockname()[1]
        self.context = zmq.Context()
        self.addCleanup(self.context.term)
        self.front, self.wrist = CachedStream(), CachedStream()
        self.publisher = CameraStreamPublisher(
            {"forward": self.front, "wrist_right": self.wrist},
            port=self.port,
            host_session_id="test-session",
        )
        self.addCleanup(self.publisher.close)
        self.publisher.start()

    def subscriber(self, topic=b"camera/forward"):
        reader = self.context.socket(zmq.SUB)
        reader.linger = 0
        reader.setsockopt(zmq.SUBSCRIBE, topic)
        reader.connect(f"tcp://127.0.0.1:{self.port}")
        self.addCleanup(reader.close)
        return reader

    def wait_for(self, predicate):
        deadline = time.monotonic() + 1
        while not predicate() and time.monotonic() < deadline:
            threading.Event().wait(0.005)
        self.assertTrue(predicate(), "publisher did not reach expected state")

    def receive(self, reader):
        self.assertTrue(reader.poll(1000))
        parts = reader.recv_multipart()
        return parts, json.loads(parts[1])

    def test_no_subscriber_or_unknown_topic_does_no_camera_work(self):
        self.subscriber(b"camera/not_configured")
        threading.Event().wait(0.03)
        self.assertFalse(self.front.enabled.is_set())
        self.assertFalse(self.wrist.enabled.is_set())
        self.assertEqual(self.front.reads, 0)
        self.assertEqual(self.wrist.reads, 0)

    def test_only_requested_camera_publishes_once_per_capture(self):
        reader = self.subscriber()
        self.wait_for(self.front.enabled.is_set)
        self.front.update()
        parts, meta = self.receive(reader)
        self.assertEqual(parts[0], b"camera/forward")
        self.assertEqual(parts[2], self.front.snapshot.jpeg)
        self.assertEqual(meta["sequence"], 1)
        self.assertEqual(meta["host_session_id"], "test-session")
        self.assertEqual(meta["capture_monotonic_s"], self.front.snapshot.capture_monotonic_s)
        self.assertGreater(meta["capture_unix_ns"], 0)
        self.assertFalse(reader.poll(30))
        self.assertFalse(self.wrist.enabled.is_set())
        self.front.update()
        self.assertEqual(self.receive(reader)[1]["sequence"], 2)
        reader.setsockopt(zmq.UNSUBSCRIBE, b"camera/forward")
        self.wait_for(lambda: not self.front.enabled.is_set())

    def test_two_subscribers_keep_encoding_until_the_last_disconnects(self):
        first = self.subscriber()
        second = self.subscriber()
        self.wait_for(self.front.enabled.is_set)
        self.front.update()
        self.receive(first)
        self.receive(second)
        first.close()
        threading.Event().wait(0.03)
        self.assertTrue(self.front.enabled.is_set())
        self.front.update()
        self.receive(second)
        second.close()
        self.wait_for(lambda: not self.front.enabled.is_set())

    def test_wildcard_and_specific_subscriptions_have_independent_demand(self):
        reader = self.subscriber(b"camera/")
        self.wait_for(self.wrist.enabled.is_set)
        self.assertTrue(self.front.enabled.is_set())
        reader.setsockopt(zmq.SUBSCRIBE, b"camera/forward")
        reader.setsockopt(zmq.UNSUBSCRIBE, b"camera/")
        self.wait_for(lambda: not self.wrist.enabled.is_set())
        self.assertTrue(self.front.enabled.is_set())

    def test_stale_frames_are_not_published_and_new_frames_recover(self):
        reader = self.subscriber()
        self.wait_for(self.front.enabled.is_set)
        with self.assertLogs(level="WARNING"):
            self.front.update(time.monotonic() - 2)
            self.wait_for(lambda: self.publisher.stats()["errors"] > 0)
        self.assertFalse(reader.poll(20))
        self.front.update()
        self.receive(reader)

    def test_slow_subscriber_cannot_block_shutdown(self):
        reader = self.subscriber()
        self.wait_for(self.front.enabled.is_set)
        for _ in range(20):
            self.front.update()
            threading.Event().wait(0.002)
        self.publisher.close()
        self.assertFalse(self.front.enabled.is_set())
        self.assertFalse(self.publisher._thread.is_alive())
        reader.close()
        with self.assertRaises(RuntimeError):
            self.publisher.start()

    def test_busy_port_is_reported_and_thread_resources_are_closed(self):
        other = CameraStreamPublisher({}, port=self.port)
        with self.assertRaisesRegex(RuntimeError, "failed to start"):
            other.start()
        other.close()
        self.assertFalse(other._thread.is_alive())

    def test_stop_deadline_retains_thread_until_it_can_be_joined(self):
        entered, release = threading.Event(), threading.Event()

        def stalled():
            entered.set()
            release.wait(2)
            return None

        self.front.read_stream_snapshot = stalled
        self.subscriber()
        self.assertTrue(entered.wait(1))
        try:
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                self.publisher.close(timeout_s=0.001)
            self.assertTrue(self.publisher._thread.is_alive())
        finally:
            release.set()
            self.publisher.close()
        self.assertFalse(self.front.enabled.is_set())


class CameraMessageTests(unittest.TestCase):
    def test_metadata_has_original_clock_conversion_without_reencoding(self):
        snapshot = JpegSnapshot(10.25, b"\xff\xd8jpeg", 16, 8)
        parts = encode_camera_stream_message(
            "forward",
            snapshot,
            3,
            host_session_id="session",
            host_monotonic_s=10.5,
            host_unix_ns=1_000_000_000,
        )
        meta = json.loads(parts[1])
        self.assertEqual(meta["capture_unix_ns"], 750_000_000)
        self.assertEqual(
            meta["host_clock_reference"], {"monotonic_s": 10.5, "unix_ns": 1_000_000_000}
        )
        self.assertIs(parts[2], snapshot.jpeg)

    def test_invalid_metadata_is_rejected(self):
        for snapshot in (
            JpegSnapshot(float("nan"), b"\xff\xd8jpeg", 16, 8),
            JpegSnapshot(2, b"\xff\xd8jpeg", 16, 8),
            JpegSnapshot(0.5, b"bad", 16, 8),
            JpegSnapshot(0.5, b"\xff\xd8jpeg", 0, 8),
        ):
            with self.assertRaises(ValueError):
                encode_camera_stream_message(
                    "forward", snapshot, 1, host_session_id="session", host_monotonic_s=1
                )
