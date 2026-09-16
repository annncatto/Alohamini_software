# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Migrated from lerobot/robots/alohamini/camera_stream.py.
"""Camera-only ZMQ stream kept outside the AlohaMini Host control loop."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Mapping
from uuid import uuid4

from alohamini._validation import finite_number, identifier
from alohamini.hardware.camera import JpegSnapshot

CAMERA_STREAM_SCHEMA_VERSION = 1


def encode_camera_stream_message(
    camera_name: str,
    snapshot: JpegSnapshot,
    sequence: int,
    *,
    host_session_id: str,
    host_monotonic_s: float | None = None,
    host_unix_ns: int | None = None,
) -> list[bytes]:
    """Build ``[topic, metadata_json, jpeg]`` for one newly captured frame."""
    host_monotonic_s = time.monotonic() if host_monotonic_s is None else host_monotonic_s
    host_unix_ns = time.time_ns() if host_unix_ns is None else host_unix_ns
    identifier(camera_name, "camera_name")
    identifier(host_session_id, "host_session_id")
    finite_number(host_monotonic_s, "host_monotonic_s")
    if not isinstance(snapshot, JpegSnapshot):
        raise ValueError("Expected a JPEG capture snapshot")
    finite_number(snapshot.capture_monotonic_s, "capture_monotonic_s")
    if (
        snapshot.capture_monotonic_s < 0
        or snapshot.capture_monotonic_s > host_monotonic_s
        or type(sequence) is not int
        or sequence < 1
        or type(host_unix_ns) is not int
        or host_unix_ns <= 0
        or type(snapshot.width) is not int
        or not 1 <= snapshot.width <= 4096
        or type(snapshot.height) is not int
        or not 1 <= snapshot.height <= 4096
        or not isinstance(snapshot.jpeg, bytes)
        or not snapshot.jpeg.startswith(b"\xff\xd8")
        or len(snapshot.jpeg) > 8 * 1024 * 1024
    ):
        raise ValueError("Invalid camera stream snapshot or metadata")
    capture_unix_ns = host_unix_ns - round((host_monotonic_s - snapshot.capture_monotonic_s) * 1e9)
    if capture_unix_ns <= 0:
        raise ValueError("Camera capture wall-clock timestamp must be positive")
    metadata = {
        "schema_version": CAMERA_STREAM_SCHEMA_VERSION,
        "host_session_id": host_session_id,
        "camera_name": camera_name,
        "sequence": int(sequence),
        "encoding": "jpeg",
        "width": snapshot.width,
        "height": snapshot.height,
        "capture_monotonic_s": snapshot.capture_monotonic_s,
        "capture_unix_ns": int(capture_unix_ns),
        "host_clock_reference": {
            "monotonic_s": float(host_monotonic_s),
            "unix_ns": int(host_unix_ns),
        },
    }
    return [
        f"camera/{camera_name}".encode(),
        json.dumps(metadata, separators=(",", ":")).encode("utf-8"),
        snapshot.jpeg,
    ]


class CameraStreamPublisher:
    """Publish each unique camera capture from a socket-owning background thread."""

    def __init__(
        self,
        cameras: Mapping,
        *,
        port: int = 5557,
        bind_host: str = "127.0.0.1",
        host_session_id: str | None = None,
        max_age_ms: int = 500,
        poll_interval_s: float = 0.002,
    ) -> None:
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Camera stream port must be an integer in [1, 65535]")
        if (
            not isinstance(bind_host, str)
            or not bind_host
            or any(c.isspace() or c in "/:[]" for c in bind_host)
        ):
            raise ValueError("Provide a bind IPv4 address or hostname")
        finite_number(max_age_ms, "max_age_ms")
        finite_number(poll_interval_s, "poll_interval_s")
        if max_age_ms <= 0 or poll_interval_s <= 0.0:
            raise ValueError("camera stream port, max age, and poll interval must be positive")
        self.cameras = dict(cameras)
        if len(self.cameras) > 16:
            raise ValueError("At most 16 camera streams are supported")
        for name in self.cameras:
            identifier(name, "camera_name")
        self.port = int(port)
        self.bind_host = bind_host
        self.host_session_id = host_session_id if host_session_id is not None else uuid4().hex
        identifier(self.host_session_id, "host_session_id")
        self.max_age_ms = float(max_age_ms)
        self.poll_interval_s = float(poll_interval_s)
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None
        self._used = False
        self._stats_lock = threading.Lock()
        self._stats: dict[str, float] = {
            "published": 0.0,
            "dropped": 0.0,
            "errors": 0.0,
        }

    def start(self, timeout_s: float = 2.0) -> None:
        if self._used or self._stop_event.is_set():
            raise RuntimeError("Create a new camera publisher for each Host session")
        self._used = True
        self._thread = threading.Thread(
            target=self._run,
            name="alohamini-camera-stream",
            daemon=True,
        )
        self._thread.start()
        if not self._started_event.wait(timeout_s):
            self._stop_event.set()
            raise TimeoutError("camera stream publisher did not start")
        if self._startup_error is not None:
            raise RuntimeError("camera stream publisher failed to start") from self._startup_error

    def close(self, timeout_s: float = 0.5) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise RuntimeError("Camera stream publisher did not stop within the deadline")

    def stats(self) -> dict[str, float]:
        with self._stats_lock:
            result = dict(self._stats)
        return result

    def _increment_stats(self, **values: float) -> None:
        with self._stats_lock:
            for name, value in values.items():
                self._stats[name] += value

    def _run(self) -> None:
        context = socket = None
        active = set()
        try:
            import zmq

            context = zmq.Context()
            # Aggregate first-subscribe/last-unsubscribe notifications retain
            # demand when several ROS viewers subscribe to the same topic.
            socket = context.socket(zmq.XPUB)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDHWM, max(4, len(self.cameras) * 2))
            socket.setsockopt(zmq.RCVHWM, 32)
            socket.setsockopt(zmq.MAXMSGSIZE, 256)
            socket.bind(f"tcp://{self.bind_host}:{self.port}")
            self._started_event.set()
            sequences = dict.fromkeys(self.cameras, 0)
            last_timestamps: dict[str, float] = {}
            last_warning: dict[str, float] = {}
            subscriptions: set[bytes] = set()
            topics = {name: f"camera/{name}".encode() for name in self.cameras}
            while not self._stop_event.is_set():
                # A subscription flood cannot starve frame delivery or shutdown.
                for _ in range(64):
                    try:
                        subscription = socket.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if not subscription:
                        continue
                    prefix = subscription[1:]
                    if not any(topic.startswith(prefix) for topic in topics.values()):
                        continue
                    if subscription[0] == 1:
                        subscriptions.add(prefix)
                    else:
                        subscriptions.discard(prefix)
                published_any = False
                for camera_name, camera in self.cameras.items():
                    wanted = any(topics[camera_name].startswith(prefix) for prefix in subscriptions)
                    if wanted != (camera_name in active):
                        camera.set_stream_enabled(wanted)
                        if wanted:
                            active.add(camera_name)
                        else:
                            active.discard(camera_name)
                    if not wanted:
                        continue
                    try:
                        snapshot = camera.read_stream_snapshot()
                        if snapshot is None:
                            continue
                        if time.monotonic() - snapshot.capture_monotonic_s > self.max_age_ms / 1000:
                            raise RuntimeError("camera capture is stale")
                        if snapshot.capture_monotonic_s <= last_timestamps.get(
                            camera_name, -math.inf
                        ):
                            continue
                        sequences[camera_name] += 1
                        parts = encode_camera_stream_message(
                            camera_name,
                            snapshot,
                            sequences[camera_name],
                            host_session_id=self.host_session_id,
                        )
                        socket.send_multipart(parts, flags=zmq.NOBLOCK)
                        last_timestamps[camera_name] = snapshot.capture_monotonic_s
                        self._increment_stats(published=1.0)
                        published_any = True
                    except zmq.Again:
                        self._increment_stats(dropped=1.0)
                    except Exception as error:
                        self._increment_stats(errors=1.0)
                        now = time.monotonic()
                        if now - last_warning.get(camera_name, 0.0) >= 1.0:
                            logging.warning("Camera stream %s unavailable: %s", camera_name, error)
                            last_warning[camera_name] = now
                if not published_any:
                    self._stop_event.wait(self.poll_interval_s)
        except BaseException as error:
            if not self._started_event.is_set():
                self._startup_error = error
            else:
                logging.error("Camera stream publisher stopped: %s", error)
        finally:
            self._started_event.set()
            for camera_name in active:
                try:
                    self.cameras[camera_name].set_stream_enabled(False)
                except Exception:
                    logging.exception("Could not disable camera stream %s", camera_name)
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()
