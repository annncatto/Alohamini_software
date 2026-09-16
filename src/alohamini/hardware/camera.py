# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Capture and image handling adapted from LeRobot OpenCVCamera and AlohaMini Host.
"""V4L2 capture and JPEG production outside the motor-control process/thread."""

from __future__ import annotations

import multiprocessing
import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Full


@dataclass(frozen=True)
class CameraConfig:
    device: str
    width: int = 640
    height: int = 480
    fps: int = 30
    rotation: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.device, str)
            or not self.device.startswith("/dev/")
            or "\x00" in self.device
        ):
            raise ValueError("Camera device must be an explicit /dev/ path")
        for name, maximum in (("width", 4096), ("height", 2160), ("fps", 30)):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        if type(self.rotation) is not int or self.rotation not in (0, 90, 180, 270):
            raise ValueError("rotation must be 0, 90, 180 or 270 degrees clockwise")


@dataclass(frozen=True)
class JpegSnapshot:
    """Standard JPEG and dimensions from one capture, for the camera-only stream."""

    capture_monotonic_s: float
    jpeg: bytes
    width: int
    height: int


def _capture(config: CameraConfig, output, stop, stream_enabled) -> None:
    """Child owns V4L2 and OpenCV; a stuck driver can be terminated on shutdown."""
    capture = None
    # Do not let an unread IPC queue keep the child alive during shutdown.
    output.cancel_join_thread()
    try:
        import cv2

        cv2.setNumThreads(1)
        capture = cv2.VideoCapture(config.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise OSError(f"Cannot open camera: {config.device}")
        for prop, value in (
            (cv2.CAP_PROP_FRAME_WIDTH, config.width),
            (cv2.CAP_PROP_FRAME_HEIGHT, config.height),
            (cv2.CAP_PROP_FPS, config.fps),
        ):
            if not capture.set(prop, value):
                raise OSError(f"Camera rejected property {prop}={value}")
        failures = 0
        timing = {
            "frames": 0,
            "capture_ms": 0.0,
            "encode_ms": 0.0,
            "stream_frames": 0,
            "stream_encode_ms": 0.0,
            "stream_errors": 0,
        }
        while not stop.is_set():
            read_started = time.perf_counter()
            ok, bgr = capture.read()
            capture_done = time.perf_counter()
            stamp = time.monotonic()
            if not ok or bgr is None:
                failures += 1
                if failures > 10:
                    raise OSError("Camera exceeded ten consecutive read failures")
                stop.wait(0.01)
                continue
            failures = 0
            if bgr.shape[:2] != (config.height, config.width):
                raise OSError("Camera resolution does not match configuration")
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if config.rotation:
                frame = cv2.rotate(
                    frame,
                    {
                        90: cv2.ROTATE_90_CLOCKWISE,
                        180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                    }[config.rotation],
                )
            # Preserve deployed 5556 image semantics: its clients use imdecode
            # directly as an RGB array. Standard-JPEG ROS streaming is a separate
            # protocol boundary and must not reinterpret these bytes as BGR input.
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                raise OSError("Camera JPEG encoding failed")
            jpeg = buffer.tobytes()
            if len(jpeg) > 8 * 1024 * 1024:
                raise OSError("Encoded camera frame exceeds 8 MiB")
            timing["frames"] += 1
            timing["capture_ms"] += (capture_done - read_started) * 1e3
            timing["encode_ms"] += (time.perf_counter() - capture_done) * 1e3
            stream = None
            if stream_enabled.value:
                # Migrated from camera_stream.encode_camera_stream_message:
                # encode the same RGB capture, not the already compressed 5556 JPEG.
                started = time.perf_counter()
                try:
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 70],
                    )
                    if not ok or len(encoded) > 8 * 1024 * 1024:
                        raise OSError("Camera stream JPEG encoding failed or exceeded 8 MiB")
                    stream = JpegSnapshot(stamp, encoded.tobytes(), frame.shape[1], frame.shape[0])
                    timing["stream_frames"] += 1
                except Exception:
                    # Optional streaming must not invalidate the native capture.
                    timing["stream_errors"] += 1
                finally:
                    timing["stream_encode_ms"] += (time.perf_counter() - started) * 1e3
            try:
                output.put_nowait((stamp, jpeg, dict(timing), stream))
            except Full:
                pass  # Bounded delivery; never delay capture for a slow consumer.
    except Exception as exc:
        try:
            output.put_nowait((None, None, f"{type(exc).__name__}: {exc}", None))
        except Full:
            pass
    finally:
        if capture is not None:
            capture.release()


class OpenCVCamera:
    """Eight-frame Host-clock cache, populated by an independent capture worker.

    read_frame_history never waits for hardware or IPC. Capture timestamps mark
    read completion, not exposure. JPEG color semantics match the deployed 5556
    Host. close is bounded even if VideoCapture.read never returns.
    """

    def __init__(self, config: CameraConfig) -> None:
        if not isinstance(config, CameraConfig):
            raise TypeError("Expected CameraConfig")
        self.config = config
        self._history = deque(maxlen=8)
        self._lock = threading.Lock()
        self._error: str | None = None
        self._timing = {"frames": 0, "capture_ms": 0.0, "encode_ms": 0.0}
        self._process = self._receiver = self._queue = self._stop = None
        self._stream_enabled = None
        self._stream_requested = False
        self._stream_snapshot = None
        self._receiver_stop = threading.Event()
        self._used = self._closed = False

    def start(self) -> None:
        if self._used or self._closed:
            raise RuntimeError("Create a new camera for each Host session")
        self._used = True
        context = multiprocessing.get_context("spawn")
        self._queue = context.Queue(maxsize=8)
        self._stop = context.Event()
        with self._lock:
            # The capture child only reads this flag. No process-shared lock
            # may be acquired while holding the Host's camera-cache lock.
            self._stream_enabled = context.Value("b", self._stream_requested, lock=False)
        self._process = context.Process(
            target=_capture,
            args=(self.config, self._queue, self._stop, self._stream_enabled),
            daemon=True,
        )
        try:
            self._process.start()
            self._receiver = threading.Thread(
                target=self._receive, daemon=True, name="alohamini-camera-cache"
            )
            self._receiver.start()
        except BaseException:
            self.close()
            raise

    def _receive(self) -> None:
        while not self._receiver_stop.is_set():
            try:
                stamp, jpeg, error, stream = self._queue.get(timeout=0.1)
            except Empty:
                if self._receiver_stop.is_set():
                    return
                if not self._process.is_alive():
                    with self._lock:
                        self._error = self._error or "Camera capture worker stopped"
                        self._history.clear()
                        self._stream_snapshot = None
                    return
                continue
            except (EOFError, OSError, ValueError):
                if not self._receiver_stop.is_set():
                    with self._lock:
                        self._error = "Camera capture channel closed"
                        self._history.clear()
                        self._stream_snapshot = None
                return
            with self._lock:
                if isinstance(error, dict):
                    self._timing = error
                    error = None
                if error is not None:
                    self._error = error
                    self._history.clear()
                    self._stream_snapshot = None
                    return
                if not self._history or stamp > self._history[-1][0]:
                    self._history.append((stamp, jpeg))
                    self._stream_snapshot = stream if self._stream_requested else None

    def read_frame_history(self) -> tuple:
        with self._lock:
            error, history = self._error, tuple(self._history)
        if error:
            raise OSError(error)
        return history

    def timing_stats(self) -> dict:
        """Cumulative capture/encoding cost from the worker, without device I/O."""
        with self._lock:
            return dict(self._timing)

    def set_stream_enabled(self, enabled: bool) -> None:
        """Subscription demand only; never opens, reads or encodes a camera here."""
        if type(enabled) is not bool:
            raise ValueError("Camera stream demand must be a bool")
        with self._lock:
            self._stream_requested = enabled and not self._closed
            if self._stream_enabled is not None:
                self._stream_enabled.value = self._stream_requested
            if not self._stream_requested:
                self._stream_snapshot = None

    def read_stream_snapshot(self) -> JpegSnapshot | None:
        with self._lock:
            error, snapshot = self._error, self._stream_snapshot
            if snapshot is None and self._timing.get("stream_errors", 0):
                error = error or "Standard JPEG encoding unavailable"
        if error:
            raise OSError(error)
        return snapshot

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.set_stream_enabled(False)
        self._receiver_stop.set()
        if self._stop is not None:
            self._stop.set()
        if self._process is not None and self._process.pid is not None:
            self._process.join(timeout=0.3)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=0.3)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.3)
            if self._process.is_alive():
                raise RuntimeError("Camera process did not exit")
        if self._receiver is not None and self._receiver.ident is not None:
            self._receiver.join(timeout=0.2)
        if self._process is not None and self._process.pid is not None:
            self._process.close()
        if self._queue is not None:
            self._queue.cancel_join_thread()
            self._queue.close()
        with self._lock:
            self._history.clear()
            self._stream_snapshot = None
