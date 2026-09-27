# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rerun live visualization, migrated from LeRobot's rerun_visualization.py."""

import logging
import numbers
import os
import threading
import time

import numpy as np

from alohamini._validation import finite_number
from alohamini.protocol import HostSnapshot, parse_camera_message

# Wire/view names and depth-unit inference retained from the source constants/config.
OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
ACTION = "action"
ACTION_PREFIX = ACTION + "."
DEPTH_METER_UNIT = "m"
DEPTH_MILLIMETER_UNIT = "mm"


def infer_depth_unit(dtype: np.dtype | type) -> str:
    """Floating-point depth is in metres; integer depth is in millimetres."""
    return (
        DEPTH_METER_UNIT if np.issubdtype(np.dtype(dtype), np.floating) else DEPTH_MILLIMETER_UNIT
    )


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def init_rerun(
    session_name: str = "alohamini_teleop", ip: str | None = None, port: int | None = None
) -> None:
    """
    Initializes the Rerun SDK for visualizing the control loop.

    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server.
        port: Optional port for connecting to a Rerun server.
    """

    import rerun as rr

    log_rerun_data.blueprint = None  # Reset blueprint cache for new session
    log_rerun_data.paths = (set(), set(), set())

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("ALOHAMINI_RERUN_MEMORY_LIMIT", "10%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.spawn(memory_limit=memory_limit)


def shutdown_rerun() -> None:
    """Shuts down the Rerun SDK gracefully."""

    import rerun as rr

    rr.rerun_shutdown()


def _build_blueprint(observation_paths: set[str], action_paths: set[str], image_paths: set[str]):
    """Lay out camera images, observation and action scalars in separate views.

    Camera images, observation and action scalars are arranged in a grid.
    """

    import rerun.blueprint as rrb

    views = [rrb.Spatial2DView(origin=path, name=path) for path in sorted(image_paths)]

    if observation_paths:
        views.append(rrb.TimeSeriesView(name="observation", contents=sorted(observation_paths)))
    if action_paths:
        views.append(rrb.TimeSeriesView(name="action", contents=sorted(action_paths)))

    return rrb.Blueprint(rrb.Grid(*views))


def _ensure_blueprint(
    observation_paths: set[str], action_paths: set[str], image_paths: set[str]
) -> None:
    """Retain existing views and add cameras that arrive after the first state."""
    previous = getattr(log_rerun_data, "paths", (set(), set(), set()))
    paths = tuple(
        old | new
        for old, new in zip(previous, (observation_paths, action_paths, image_paths), strict=True)
    )
    if paths == previous:
        return

    import rerun as rr

    blueprint = _build_blueprint(*paths)
    log_rerun_data.blueprint = blueprint
    log_rerun_data.paths = paths
    rr.send_blueprint(blueprint)


def log_rerun_data(
    observation: dict | None = None,
    action: dict | None = None,
    compress_images: bool = False,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    Send the observation and action dictionaries to the Rerun viewer:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, optionally compressed to JPEG and logged as images.
    - 1D NumPy arrays are logged as a single `rr.Scalars` batch under one entity path, so that every
      dimension shares the same view instead of being split across one view per element.
    - Multi-dimensional **action** arrays are flattened and logged as a single `rr.Scalars` batch.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Observation/action scalars have separate time-series views and each image its own spatial view.
    Later camera arrivals extend the layout without removing existing views.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Compress images to trade CPU/quality for bandwidth and memory.
    """

    import rerun as rr

    observation_paths: set[str] = set()
    action_paths: set[str] = set()
    image_paths: set[str] = set()

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
                observation_paths.add(key)
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    rr.log(key, rr.Scalars(arr.astype(float)))
                    observation_paths.add(key)
                else:
                    if arr.shape[-1] == 1:
                        # At record time, the depth unit is inferred from the frame type.
                        depth_unit = infer_depth_unit(arr.dtype)
                        img_entity = rr.DepthImage(
                            arr,
                            meter=1000.0 if depth_unit == DEPTH_MILLIMETER_UNIT else 1.0,
                            colormap=rr.components.Colormap.Viridis,
                        )
                    else:
                        img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity)
                    image_paths.add(key)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
                action_paths.add(key)
            elif isinstance(v, np.ndarray):
                # Flatten any (incl. higher-dimensional) array into a single batched Scalars
                rr.log(key, rr.Scalars(v.reshape(-1).astype(float)))
                action_paths.add(key)

    _ensure_blueprint(observation_paths, action_paths, image_paths)


def _decode_images(images: dict) -> dict:
    import cv2

    observation = {}
    for name, jpeg in images.items():
        # Deployed 5556 JPEGs decode directly to RGB; no second channel swap.
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            observation[name] = frame
    return observation


def log_snapshot(snapshot: HostSnapshot | None, action: dict) -> None:
    """Display Host state and JPEG frames without synthesizing images."""
    observation = {}
    if snapshot is not None:
        observation.update({k: v for k, v in snapshot.payload.items() if not k.startswith("_")})
        if snapshot.images:
            observation.update(_decode_images(snapshot.images))
    log_rerun_data(observation, action)


class TeleopPreview:
    """Latest-only viewer worker; the camera subscriber never supplies control state.

    Images are displayed independently, not paired into recording samples. The
    socket and Rerun submissions stay in this worker; submit only replaces one
    pending state/action pair, so a slow viewer cannot accumulate old commands.
    """

    def __init__(self, host: str | None, robot_model: str, *, fps: float):
        finite_number(fps, "preview fps")
        if not 0 < fps <= 50:
            raise ValueError("Preview fps must be in (0, 50]")
        self._host, self._model = host, robot_model
        self._interval = 1.0 / fps
        self._lock = threading.Lock()
        self._pending = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="teleop-preview", daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            logging.warning("遥操预览未及时退出；控制发送已停止。")

    def submit(self, snapshot: HostSnapshot | None, action: dict) -> None:
        with self._lock:
            self._pending = (snapshot, dict(action))

    def _run(self):
        context = socket = None
        report_time = -float("inf")
        sequences = {}
        session = None
        try:
            if self._host is not None:
                import zmq

                context = zmq.Context()
                socket = context.socket(zmq.SUB)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.MAXMSGSIZE, 8 * 1024 * 1024)
                socket.setsockopt(zmq.RCVHWM, 32)
                socket.setsockopt(zmq.SUBSCRIBE, b"camera/")
                socket.connect(f"tcp://{self._host}:5557")
            while not self._stop.is_set():
                started = time.monotonic()
                with self._lock:
                    pending, self._pending = self._pending, None
                if pending is not None:
                    snapshot, action = pending
                    if snapshot is not None and snapshot.robot_model != self._model:
                        raise ValueError("Preview robot_model does not match control state")
                    log_snapshot(snapshot, action)
                    if socket is not None and snapshot is not None:
                        expected_session = snapshot.payload["_safety"]["host_session_id"]
                        if session != expected_session:
                            session = expected_session
                            sequences.clear()
                        names = snapshot.payload["_robot_metadata"].get("cameras", ())
                        latest = {}
                        # Bounded draining retains only each camera's newest frame.
                        for _ in range(32):
                            try:
                                parts = socket.recv_multipart(flags=zmq.NOBLOCK)
                            except zmq.Again:
                                break
                            try:
                                frame = parse_camera_message(parts)
                            except ValueError as exc:
                                if started - report_time >= 5.0:
                                    logging.warning("忽略无效预览图像：%s", exc)
                                    report_time = started
                                continue
                            if (
                                frame.host_session_id != session
                                or frame.camera_name not in names
                                or frame.sequence <= sequences.get(frame.camera_name, 0)
                                or snapshot.payload.get("_host_timing", {}).get(
                                    "state_sample_monotonic_s", frame.capture_monotonic_s
                                )
                                - frame.capture_monotonic_s
                                > 0.5
                            ):
                                continue
                            sequences[frame.camera_name] = frame.sequence
                            latest[frame.camera_name] = frame.jpeg
                        if latest and not self._stop.is_set():
                            # 5557 uses standard JPEG (unlike the legacy 5556 wire).
                            rgb = {
                                name: pixels[:, :, ::-1]
                                for name, pixels in _decode_images(latest).items()
                            }
                            log_rerun_data(rgb)
                self._stop.wait(max(0, self._interval - (time.monotonic() - started)))
        except Exception:
            logging.exception("遥操预览已停止；控制连接不受影响。")
        finally:
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()
