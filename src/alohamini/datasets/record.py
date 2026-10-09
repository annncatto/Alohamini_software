# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Feedback schema copied from AlohaMini motor_feedback.py.
"""Local v3 recording and dataset fields for motor feedback."""

import json
import logging
import math
import os
import re
import shutil
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from queue import Empty, Queue

import numpy as np

from alohamini.datasets.images import (
    IMAGE_FORMAT,
    VIDEO_FORMAT,
    encode_recording_image,
    image_type,
)
from alohamini.model import get_robot_model

FEEDBACK_FIELDS = (
    "position_raw",
    "velocity_raw",
    "current_raw",
    "current_ma",
    "load_raw",
    "voltage_raw",
    "temperature_raw",
    "status_raw",
    "moving",
)
FEEDBACK_TIMES = ("sample_started_s", "sample_finished_s")


def motor_feedback_features(state_names: Sequence[str]) -> dict:
    """Keep physical wheel feedback distinct from the three body-velocity dimensions."""
    motors = [name.removesuffix(".pos") for name in state_names if name.endswith(".pos")]
    motors += ["base_left_wheel", "base_back_wheel", "base_right_wheel", "lift_axis"]
    features = {}
    for field in FEEDBACK_FIELDS:
        for key in (f"observation.motor_{field}", f"motor_feedback.{field}_valid"):
            features[key] = {"dtype": "float32", "shape": (len(motors),), "names": list(motors)}
    for field in FEEDBACK_TIMES:
        features[f"motor_feedback.{field}"] = {
            "dtype": "float64",
            "shape": (len(motors),),
            "names": list(motors),
        }
    return features


def is_motor_feedback_feature(key: str) -> bool:
    return key.startswith("motor_feedback.") or key in {
        f"observation.motor_{field}" for field in FEEDBACK_FIELDS
    }


def motor_feedback_frame(features: Mapping, snapshot: Mapping) -> dict[str, np.ndarray]:
    """Encode fresh feedback only; finite zero placeholders always have a zero validity mask."""
    motors = snapshot.get("motors", {}) if snapshot.get("version") == 1 else {}
    if not isinstance(motors, Mapping):
        motors = {}
    frame = {}
    for field in FEEDBACK_FIELDS:
        key = f"observation.motor_{field}"
        if key not in features:
            continue
        values, valid = [], []
        for motor in features[key]["names"]:
            sample = motors.get(motor, {})
            sample = sample if isinstance(sample, Mapping) else {}
            value = sample.get(field)
            started, finished = (sample.get(name) for name in FEEDBACK_TIMES)
            numbers = (value, started, finished)
            usable = all(
                isinstance(number, (int, float))
                and not isinstance(number, bool)
                and math.isfinite(number)
                for number in numbers
            )
            usable = usable and 0 <= started <= finished
            # Position/velocity/current feedback fits comfortably in float32.
            usable = usable and abs(value) <= np.finfo(np.float32).max
            values.append(value if usable else 0.0)
            valid.append(float(usable))
        frame[key] = np.asarray(values, dtype=np.float32)
        frame[f"motor_feedback.{field}_valid"] = np.asarray(valid, dtype=np.float32)
    for field in FEEDBACK_TIMES:
        key = f"motor_feedback.{field}"
        if key not in features:
            continue
        values = []
        for motor in features[key]["names"]:
            sample = motors.get(motor, {})
            sample = sample if isinstance(sample, Mapping) else {}
            started, finished = (sample.get(name) for name in FEEDBACK_TIMES)
            usable = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in (started, finished)
            )
            usable = usable and 0 <= started <= finished
            values.append(sample[field] if usable else 0.0)
        frame[key] = np.asarray(values, dtype=np.float64)
    return frame


def state_names(robot_model: str) -> list[str]:
    return [
        f"{m.name}.pos" for m in get_robot_model(robot_model).actuators if m.name.startswith("arm_")
    ] + ["x.vel", "y.vel", "theta.vel", "lift_axis.height_mm"]


def dataset_features(robot_model: str) -> dict:
    names = state_names(robot_model)
    vector = {"dtype": "float32", "shape": [len(names)], "names": names}
    return {
        "observation.state": vector,
        "action": dict(vector),
        **motor_feedback_features(names),
    }


def dataset_schema(features: dict, cameras: Sequence[str], image_format=IMAGE_FORMAT):
    import pyarrow as pa

    return pa.schema(
        [
            *[
                pa.field(
                    key,
                    pa.list_(
                        pa.float64() if ft["dtype"] == "float64" else pa.float32(),
                        len(ft["names"]),
                    ),
                )
                for key, ft in features.items()
            ],
            *[pa.field(f"observation.images.{name}", image_type(image_format)) for name in cameras],
            *[
                pa.field(key, pa.int64())
                for key in ("index", "episode_index", "frame_index", "task_index")
            ],
            pa.field("timestamp", pa.float32()),
            pa.field("task", pa.string()),
        ]
    )


class StateSelection:
    """Select training inputs without deleting recorded fields or changing action semantics."""

    DEFAULT = "joint_position,base_velocity,lift_height"
    GROUPS = ("joint_position", "joint_velocity", "joint_current", "base_velocity", "lift_height")

    def __init__(self, info: dict, selection: str = DEFAULT, *, exclude_fixed=False):
        from alohamini.calibration.encoder import HostPositionUnits

        self.groups = tuple(part.strip() for part in selection.split(","))
        if (
            not self.groups
            or len(set(self.groups)) != len(self.groups)
            or any(part not in self.GROUPS for part in self.groups)
        ):
            raise ValueError(f"State groups must be unique members of {self.GROUPS}")
        if "observation.state" not in info["features"] and any(
            group in self.groups for group in ("joint_position", "base_velocity", "lift_height")
        ):
            raise ValueError("Selected state groups require the removed observation.state field")
        self.source_names = info["features"].get(
            "observation.state",
            dataset_features(info["robot_metadata"]["robot_model"])["observation.state"],
        )["names"]
        joints = [name.removesuffix(".pos") for name in self.source_names if name.endswith(".pos")]
        from alohamini.fixed import validate

        fixed = validate(info.get("fixed_dimensions"), info["robot_metadata"]["robot_model"])
        self.columns = []
        names, units = [], []
        for group in self.groups:
            if group in ("base_velocity", "lift_height"):
                keys = (
                    ("x.vel", "y.vel", "theta.vel")
                    if group == "base_velocity"
                    else ("lift_axis.height_mm",)
                )
                unit = ("m/s", "m/s", "deg/s") if group == "base_velocity" else ("mm",)
                for key, quantity in zip(keys, unit, strict=True):
                    if exclude_fixed and key in fixed:
                        continue
                    self.columns.append(
                        ("observation.state", self.source_names.index(key), 1, None)
                    )
                    names.append(key)
                    units.append(quantity)
                continue
            for joint in joints:
                if exclude_fixed and f"{joint}.pos" in fixed:
                    continue
                motor = info["robot_metadata"]["motors"][joint]
                norm = motor["normalization"]
                unit = {
                    "range_m100_100": "normalized[-100,100]",
                    "range_0_100": "percent",
                    "degrees": "deg",
                }.get(norm)
                if unit is None:
                    raise ValueError(f"Unknown Host position units: {joint}")
                if group == "joint_position":
                    key, index, scale, mask = (
                        "observation.state",
                        self.source_names.index(f"{joint}.pos"),
                        1,
                        None,
                    )
                    name = f"{joint}.pos"
                else:
                    field = "velocity_raw" if group == "joint_velocity" else "current_ma"
                    key = f"observation.motor_{field}"
                    if key not in info["features"]:
                        raise ValueError(f"Selected state group requires removed feature: {key}")
                    index = info["features"][key]["names"].index(joint)
                    mask = f"motor_feedback.{field}_valid"
                    if group == "joint_current":
                        scale, name, unit = 0.001, f"{joint}.current_a", "A"
                    else:
                        required = ("normalization", "range_min", "range_max", "drive_mode")
                        if any(item not in motor for item in required):
                            raise ValueError(
                                f"Velocity needs recorded Host ranges/direction: {joint}"
                            )
                        coordinate = HostPositionUnits(**{item: motor[item] for item in required})
                        if norm == "degrees":
                            scale = 360 / 4095
                        else:
                            span = 200 if norm == "range_m100_100" else 100
                            scale = span / (coordinate.range_max - coordinate.range_min)
                            if coordinate.drive_mode:
                                scale = -scale
                        name, unit = f"{joint}.vel", f"{unit}/s"
                self.columns.append((key, index, scale, mask))
                names.append(name)
                units.append(unit)
        if not names:
            raise ValueError("All selected state fields are fixed; use state=none")
        self.feature = {"dtype": "float32", "shape": [len(names)], "names": names}
        self.units = units

    def frame(self, row: dict) -> np.ndarray:
        values = []
        for name, (key, index, scale, mask) in zip(
            self.feature["names"], self.columns, strict=True
        ):
            if mask is not None and row[mask][index] != 1:
                raise ValueError(f"Unavailable selected state feedback: {name}")
            values.append(row[key][index] * scale)
        result = np.asarray(values, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Selected state contains non-finite values")
        return result


def _json(value) -> str:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, default=lambda v: v.tolist()) + "\n"
    )


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def preserve_dataset(dataset):
    """Adapt source safety_utils.preserve_dataset to the dataset close/commit API."""
    try:
        yield dataset
    except BaseException:
        try:
            dataset.close()
        except BaseException:
            logging.exception("Unable to save the partial episode during cleanup")
        raise
    else:
        dataset.close()


class _EpisodeWriter:
    """Bounded temporary episode writer used by v3 recording.

    Bounded, nonblocking submissions keep image/disk work outside robot control.
    A pending directory journals complete frames. Failed/interrupted saves retain
    their journal and temporary images for explicit recovery.
    """

    QUEUE_FRAMES = 64
    QUEUE_BYTES = 64 * 1024 * 1024

    def __init__(
        self,
        root,
        *,
        fps: int,
        task: str,
        robot_metadata: dict,
        resume=False,
        fixed_dimensions=None,
        video_encoding_workers=None,
    ):
        import fcntl

        if type(fps) is not int or not 1 <= fps <= 30:
            raise ValueError("Dataset fps must be an integer in [1, 30]")
        if video_encoding_workers is not None and (
            type(video_encoding_workers) is not int or video_encoding_workers < 1
        ):
            raise ValueError("Video encoding workers must be a positive integer")
        self.video_encoding_workers = video_encoding_workers
        self.last_save_timings = {}
        if not isinstance(task, str) or not task.strip():
            raise ValueError("A nonempty task description is required")
        self.root = Path(root).expanduser()
        if not self.root.is_absolute():
            raise ValueError("Dataset root must be an absolute path")
        self.fps, self.task = fps, task
        self.robot_metadata = deepcopy(robot_metadata)
        self.names = state_names(robot_metadata["robot_model"])
        cameras = robot_metadata.get("cameras", [])
        if (
            not isinstance(cameras, list)
            or len(cameras) != len(set(cameras))
            or any(
                not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name)
                for name in cameras
            )
        ):
            raise ValueError("Invalid Host camera names")
        self.cameras = tuple(cameras)
        self.features = dataset_features(robot_metadata["robot_model"])
        self.schema = dataset_schema(self.features, self.cameras, VIDEO_FORMAT)
        info = json.loads(
            _json(
                {
                    "format": "alohamini-episodes",
                    "version": 2,
                    "fps": fps,
                    "task": task,
                    "robot_metadata": robot_metadata,
                    "features": self.features,
                    "image_format": VIDEO_FORMAT,
                    "image_color": "rgb",
                    "image_paths_relative_to": "episode_directory",
                    "timestamp": "frame_index / fps; physical timestamps are in safety.jsonl",
                    "motor_feedback": {
                        "version": 1,
                        "clock": "host_monotonic_seconds",
                        "current_ma_per_raw": 6.5,
                        "raw_values": (
                            "register units after sign decoding, before calibration normalization"
                        ),
                        "missing_values": "zero with corresponding motor_feedback.<field>_valid=0",
                    },
                }
            )
        )
        from alohamini.fixed import validate

        validate(fixed_dimensions, robot_metadata["robot_model"])
        self.fixed_dimensions = deepcopy(fixed_dimensions)
        if resume:
            if not self.root.is_dir():
                raise FileNotFoundError(self.root)
        else:
            self.root.mkdir(parents=True, exist_ok=False)
        if fixed_dimensions is not None:
            info["fixed_dimensions"] = deepcopy(fixed_dimensions)
        self._file_lock = (self.root / "recording.lock").open("a")
        try:
            fcntl.flock(self._file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            meta = self.root / "meta"
            if resume:
                if json.loads((meta / "info.json").read_text()) != info:
                    raise ValueError(
                        "Dataset schema, task, fps or robot calibration changed; use a new dataset"
                    )
            else:
                meta.mkdir()
                _write_json(meta / "info.json", info)
                (self.root / "episodes").mkdir()
            self.num_episodes = self.total_frames = 0
            self._image_shapes = {}
            for episode in sorted((self.root / "episodes").iterdir()):
                if episode.name.endswith(".pending"):
                    raise RuntimeError(
                        f"Uncommitted episode retained at {episode}; recover before resuming"
                    )
                if episode.name != f"episode_{self.num_episodes:06d}":
                    raise ValueError("Unexpected episode sequence")
                summary = json.loads((episode / "episode.json").read_text())
                import pyarrow.parquet as pq

                parquet = pq.ParquetFile(episode / "frames.parquet")
                if (
                    summary["episode_index"] != self.num_episodes
                    or summary["length"] <= 0
                    or parquet.metadata.num_rows != summary["length"]
                    or not parquet.schema_arrow.equals(self.schema)
                ):
                    raise ValueError(f"Invalid committed episode: {episode}")
                shapes = summary["image_shapes"]
                if set(shapes) != set(self.cameras) or (
                    self.num_episodes and shapes != self._image_shapes
                ):
                    raise ValueError(f"Dataset camera shapes changed: {episode}")
                self._image_shapes = shapes
                self.total_frames += summary["length"]
                self.num_episodes += 1
        except BaseException:
            self._file_lock.close()
            raise
        self._pending = None
        self._thread = None
        self._error = None
        self._queue = Queue(maxsize=self.QUEUE_FRAMES)
        self._closing = threading.Event()
        self._mutex = threading.Lock()
        self._queued_bytes = 0
        self.submitted = self.saved = self.rejected_images = self.queue_overflows = 0
        self.save_failed = False

    def begin_episode(self):
        if self._pending is not None or self._file_lock.closed:
            raise RuntimeError("Finish the current episode before starting another")
        self._pending = self.root / "episodes" / f"episode_{self.num_episodes:06d}.pending"
        self._pending.mkdir(exist_ok=False)
        self._error = None
        self._closing.clear()
        self.submitted = self.saved = self.rejected_images = self.queue_overflows = 0
        self._thread = threading.Thread(
            target=self._write_loop, daemon=True, name="AlohaMiniDataset"
        )
        self._thread.start()

    def check_writer(self):
        if self._error is not None:
            raise OSError(
                f"Dataset writer failed; recovery files retained at {self._pending}"
            ) from self._error

    def add_frame(self, frame: dict, images: dict[str, bytes], record: dict) -> bool:
        self.check_writer()
        if self._pending is None or self._closing.is_set():
            raise RuntimeError("Start an episode before submitting frames")
        if frame.keys() != self.features.keys() or images.keys() != set(self.cameras):
            raise ValueError("Frame fields or cameras do not match the dataset")
        clean = {}
        for key, ft in self.features.items():
            value = np.asarray(frame[key], dtype=ft["dtype"])
            if value.shape != (len(ft["names"]),) or not np.isfinite(value).all():
                raise ValueError(f"Invalid dataset field: {key}")
            clean[key] = value.tolist()
        if any(not isinstance(jpeg, bytes) or not jpeg for jpeg in images.values()):
            raise ValueError("Expected encoded Host JPEG bytes")
        # Take ownership before returning: later control samples cannot mutate this row.
        record = json.loads(_json(record))
        size = (
            sum(len(jpeg) for jpeg in images.values())
            + len(_json(clean).encode("utf-8"))
            + len(_json(record).encode("utf-8"))
        )
        with self._mutex:
            if self._queued_bytes + size > self.QUEUE_BYTES or self._queue.full():
                self.queue_overflows += 1
                return False
            self._queued_bytes += size
            self._queue.put_nowait((self.submitted, clean, dict(images), record, size))
            self.submitted += 1
        return True

    def event(self, event: dict, safety: dict | None = None):
        self.check_writer()
        if self._pending is None or self._closing.is_set():
            return
        row = _json(
            {
                "episode_index": self.num_episodes,
                "frame_index": None,
                "event": event,
                "safety": safety,
                "client_monotonic_s": time.monotonic(),
            }
        )
        size = len(row.encode("utf-8"))
        with self._mutex:
            if self._queued_bytes + size > self.QUEUE_BYTES or self._queue.full():
                self.queue_overflows += 1
                return
            self._queued_bytes += size
            self._queue.put_nowait((None, None, None, row, size))

    def _write_loop(self):
        shapes = self._image_shapes
        try:
            with (
                (self._pending / "journal.jsonl").open(
                    "x", encoding="utf-8", buffering=1
                ) as journal,
                ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="AlohaMiniImage"
                ) as images_pool,
            ):
                while not self._closing.is_set() or not self._queue.empty():
                    try:
                        capture, row, images, record, size = self._queue.get(timeout=0.05)
                    except Empty:
                        continue
                    try:
                        if row is None:
                            journal.write(_json({"record": json.loads(record)}))
                            continue
                        futures = {
                            name: images_pool.submit(encode_recording_image, jpeg)
                            for name, jpeg in images.items()
                        }
                        invalid = []
                        frame_shapes = {}
                        encoded_images = {}
                        for name, future in futures.items():
                            try:
                                encoded, shape = future.result()
                                shape = list(shape)
                                if name in shapes and shape != shapes[name]:
                                    raise ValueError(f"Camera resolution changed: {name}")
                                frame_shapes[name] = shape
                                encoded_images[name] = encoded
                            except ValueError as exc:
                                invalid.append(str(exc))
                        if invalid:
                            self.rejected_images += 1
                            journal.write(
                                _json(
                                    {
                                        "record": {
                                            "episode_index": self.num_episodes,
                                            "frame_index": None,
                                            "event": {
                                                "type": "image_rejected",
                                                "capture_index": capture,
                                                "reason": invalid,
                                            },
                                        }
                                    }
                                )
                            )
                            continue
                        references = {}
                        for name, encoded in encoded_images.items():
                            relative = f"images/{name}/frame_{capture:06d}.png"
                            path = self._pending / relative
                            path.parent.mkdir(exist_ok=True, parents=True)
                            with path.open("xb") as image_file:
                                image_file.write(encoded)
                            references[name] = relative
                        # Journal only complete temporary RGB images, as in fork recording.
                        shapes.update(frame_shapes)
                        row.update(
                            {
                                "index": self.total_frames + self.saved,
                                "episode_index": self.num_episodes,
                                "frame_index": self.saved,
                                "task_index": 0,
                                "task": self.task,
                                "timestamp": self.saved / self.fps,
                                **{
                                    f"observation.images.{name}": reference
                                    for name, reference in references.items()
                                },
                            }
                        )
                        record.update(
                            episode_index=self.num_episodes,
                            frame_index=self.saved,
                            feedback_phase="before_issued_command",
                            event=None,
                        )
                        journal.write(_json({"frame": row, "record": record}))
                        self.saved += 1
                    finally:
                        with self._mutex:
                            self._queued_bytes -= size
                        self._queue.task_done()
                journal.flush()
                os.fsync(journal.fileno())
        except BaseException as exc:
            self._error = exc

    def _finish_writer(self):
        if self._pending is None:
            return
        self._closing.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            self.save_failed = True
            raise TimeoutError(f"Dataset writer still running; files retained at {self._pending}")
        self.check_writer()

    def save_episode(self):
        if self.save_failed:
            raise RuntimeError("Previous save failed; recovery files retained, not retrying")
        if self._pending is None:
            return
        try:
            self.last_save_timings = {}
            started = time.perf_counter()
            self._finish_writer()
            self.last_save_timings["queue_drain"] = time.perf_counter() - started
            if not self.saved:
                self.discard_episode()
                return
            import pyarrow as pa
            import pyarrow.parquet as pq

            started = time.perf_counter()
            with (
                (self._pending / "journal.jsonl").open() as journal,
                (self._pending / "safety.jsonl").open("x", encoding="utf-8") as safety,
                pq.ParquetWriter(
                    self._pending / "frames.parquet",
                    dataset_schema(self.features, self.cameras, "png"),
                    compression="zstd",
                ) as writer,
            ):
                rows = []
                for line in journal:
                    item = json.loads(line)
                    safety.write(_json(item["record"]))
                    if "frame" in item:
                        rows.append(item["frame"])
                    if len(rows) >= 128:
                        writer.write_table(pa.Table.from_pylist(rows, schema=writer.schema))
                        rows.clear()
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=writer.schema))
                safety.write(
                    _json(
                        {
                            "episode_index": self.num_episodes,
                            "frame_index": None,
                            "event": {
                                "type": "recorder_closed",
                                "frame_count": self.saved,
                                "queue_overflows": self.queue_overflows,
                                "rejected_images": self.rejected_images,
                            },
                        }
                    )
                )
                safety.flush()
                os.fsync(safety.fileno())
            _write_json(
                self._pending / "episode.json",
                {
                    "episode_index": self.num_episodes,
                    "length": self.saved,
                    "submitted": self.submitted,
                    "rejected_images": self.rejected_images,
                    "queue_overflows": self.queue_overflows,
                    "image_shapes": self._image_shapes,
                },
            )
            from alohamini.datasets.video import finalize_recording_video

            self.last_save_timings["journal_to_parquet"] = time.perf_counter() - started
            finalize_recording_video(
                self._pending,
                self.fps,
                self.features,
                self.cameras,
                workers=self.video_encoding_workers,
                timings=self.last_save_timings,
            )
            destination = self.root / "episodes" / f"episode_{self.num_episodes:06d}"
            if destination.exists():
                raise FileExistsError(destination)
            self._pending.rename(destination)
            self._pending = None
            self.total_frames += self.saved
            self.num_episodes += 1
            # The complete temporary episode is ready for v3 publication.
            started = time.perf_counter()
            try:
                (destination / "journal.jsonl").unlink()
                if (destination / "images").is_dir():
                    shutil.rmtree(destination / "images")
            except OSError:
                logging.warning("Temporary images retained at %s", destination)
            self.last_save_timings["temporary_cleanup"] = time.perf_counter() - started
        except BaseException:
            self.save_failed = True
            raise

    def discard_episode(self):
        """Rerecord without deleting user data: retain this attempt under discarded/."""
        if self._pending is None:
            return
        self._finish_writer()
        discarded = self.root / "discarded"
        discarded.mkdir(exist_ok=True)
        self._pending.rename(discarded / f"{self._pending.stem}_{time.time_ns()}")
        self._pending = None

    def close(self):
        try:
            if self._pending is not None and not self.save_failed:
                self.save_episode()
        finally:
            # A timed-out live writer retains the exclusive lock until process exit.
            if self._thread is None or not self._thread.is_alive():
                self._file_lock.close()


class LocalDataset:
    """Record complete local LeRobot v3 datasets, including raw motor feedback.

    As in record_bi.py, PC-decoded RGB frames are buffered on disk, then encoded
    and published by save_episode. Only MP4, Parquet and metadata remain on success.
    """

    def __init__(
        self,
        root,
        *,
        fps: int,
        task: str,
        robot_metadata: dict,
        resume=False,
        fixed_dimensions=None,
        video_encoding_workers=None,
    ):
        import fcntl

        if video_encoding_workers is not None and (
            type(video_encoding_workers) is not int or video_encoding_workers < 1
        ):
            raise ValueError("Video encoding workers must be a positive integer")
        self.root = Path(root).expanduser()
        if not self.root.is_absolute():
            raise ValueError("Dataset root must be an absolute path")
        from alohamini.fixed import validate

        validate(fixed_dimensions, robot_metadata["robot_model"])
        self.fixed_dimensions = deepcopy(fixed_dimensions)
        if resume:
            if not self.root.is_dir():
                raise FileNotFoundError(self.root)
        else:
            self.root.mkdir(parents=True, exist_ok=False)
        self._lock = (self.root / "recording.lock").open("a")
        self._failed = False
        self._v3_stats = None
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            staging = self.root / ".recording"
            if any((staging / "episodes").glob("*")):
                raise RuntimeError(
                    "Interrupted recording retained; run dataset recover to a new directory"
                )
            self._writer = _EpisodeWriter(
                staging,
                fps=fps,
                task=task,
                robot_metadata=robot_metadata,
                resume=staging.exists(),
                fixed_dimensions=fixed_dimensions,
                video_encoding_workers=video_encoding_workers,
            )
            self._recording_info = json.loads((staging / "meta/info.json").read_text())
            if resume:
                from alohamini.datasets.lerobot_tools import IntegrityChecker
                from alohamini.datasets.lerobotv3 import recording_statistics

                source = json.loads((self.root / "meta/alohamini.json").read_text())
                if source["source_info"] != self._recording_info:
                    raise ValueError(
                        "Dataset schema, task, fps or robot calibration changed; use a new dataset"
                    )
                checker = IntegrityChecker(self.root, decode_videos=True)
                report = checker.run()
                if not report["valid"]:
                    raise ValueError(f"Recording requires repair before resume: {report['issues']}")
                self._writer.num_episodes = checker.info["total_episodes"]
                self._writer.total_frames = checker.info["total_frames"]
                self._writer._image_shapes = {
                    camera: checker.info["features"][f"observation.images.{camera}"]["shape"]
                    for camera in self.cameras
                }
                self._v3_stats = recording_statistics(self.root)
        except BaseException:
            writer = self.__dict__.get("_writer")
            if writer is not None:
                writer.close()
            self._lock.close()
            raise

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def save_episode(self):
        from types import SimpleNamespace

        from alohamini.datasets.lerobotv3 import publish_recorded_episode

        if self._failed:
            raise RuntimeError("Previous save failed; recover before resuming")
        previous = self._writer.num_episodes
        started = time.perf_counter()
        try:
            self._writer.save_episode()
            if self._writer.num_episodes == previous:
                self.discard_episode()
                return
            episode = self._writer.root / "episodes" / f"episode_{previous:06d}"
            transaction = SimpleNamespace(
                root=self.root,
                _pending=episode,
                _recording_info=self._recording_info,
                _v3_stats=self._v3_stats,
                _image_shapes=self._writer._image_shapes,
                cameras=self.cameras,
                num_episodes=previous,
                total_frames=self.total_frames - self.saved,
                saved=self.saved,
                save_timings=self._writer.last_save_timings,
            )
            publish_recorded_episode(transaction)
            self._v3_stats = transaction._v3_stats
            cleanup_started = time.perf_counter()
            shutil.rmtree(episode)
            self._writer.last_save_timings["cleanup"] = time.perf_counter() - cleanup_started
            self._writer.last_save_timings["total"] = time.perf_counter() - started
        except BaseException:
            self._failed = self._writer.save_failed = True
            raise

    def discard_episode(self):
        self._writer.discard_episode()
        source = self._writer.root / "discarded"
        if source.exists():
            destination = self.root / "discarded"
            destination.mkdir(exist_ok=True)
            for path in source.iterdir():
                path.rename(destination / path.name)

    def close(self):
        try:
            if not self._failed and not self._lock.closed:
                self.save_episode()
        finally:
            self._writer.close()
            if self._writer._file_lock.closed:
                self._lock.close()
                if not self._failed and self._writer.root.exists():
                    shutil.rmtree(self._writer.root)
