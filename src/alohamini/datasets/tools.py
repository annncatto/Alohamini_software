# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks and source-preserving exports for native local datasets.

Numeric feedback and capture-time checks derive from the existing
check_lerobot_dataset_integrity.py. Native storage uses episode Parquet/images
and journals, not LeRobot v3 metadata or video ranges.
"""

from __future__ import annotations

import fcntl
import json
import math
import re
import shutil
import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from alohamini.datasets.images import (
    IMAGE_COLOR,
    IMAGE_FORMAT,
    ImageShards,
    image_bytes,
    image_path,
    image_shape,
)
from alohamini.datasets.native import _json, _write_json, dataset_features, dataset_schema


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    message: str


@contextmanager
def _read_lock(root: Path):
    # Do not create files during inspection or read a changing recording.
    with (root / "recording.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Dataset is in use; stop recording before offline operations"
            ) from exc
        yield


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class IntegrityChecker:
    def __init__(
        self, root: Path, *, decode_images=False, decode_videos=False, timestamp_tolerance_s=1e-4
    ):
        self.root = Path(root).expanduser().resolve()
        self.decode_images = decode_images
        self.decode_videos = decode_videos
        self.timestamp_tolerance_s = timestamp_tolerance_s
        self.issues: list[Issue] = []
        self.info: dict[str, Any] = {}
        self.total_frames = self.num_episodes = 0
        self.shapes = {}
        self.pending: list[Path] = []

    def error(self, code: str, message: str) -> None:
        self.issues.append(Issue("error", code, message))

    def warning(self, code: str, message: str) -> None:
        issue = Issue("warning", code, message)
        if issue not in self.issues:
            self.issues.append(issue)

    def run(self) -> dict:
        try:
            with _read_lock(self.root):
                self._run_unlocked()
                self._check_previews()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            self.error("DATASET_UNREADABLE", str(exc))
        return self.report()

    def _run_unlocked(self):
        # Refuse links before copying or reading files outside this dataset.
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Dataset contains a symbolic link: {path.relative_to(self.root)}")
        self.info = _read_json(self.root / "meta/info.json")
        if (
            not isinstance(self.info, dict)
            or self.info.get("format") != "alohamini-episodes"
            or self.info.get("version") not in (1, 2)
        ):
            raise ValueError("Expected native alohamini-episodes format version 1 or 2")
        if type(self.info["fps"]) is not int or not 1 <= self.info["fps"] <= 30:
            raise ValueError("Invalid dataset fps")
        if not isinstance(self.info["task"], str) or not self.info["task"].strip():
            raise ValueError("Missing dataset task")
        metadata = self.info["robot_metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Invalid robot metadata")
        self.cameras = metadata["cameras"]
        if (
            not isinstance(self.cameras, list)
            or any(
                not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name)
                for name in self.cameras
            )
            or len(set(self.cameras)) != len(self.cameras)
        ):
            raise ValueError("Invalid camera names")
        expected = json.loads(_json(dataset_features(metadata["robot_model"])))
        if self.info["features"] != expected:
            raise ValueError(
                "Feature names, order, units or dimensions do not match the robot model"
            )
        if self.info.get("image_paths_relative_to") != "episode_directory":
            raise ValueError("Unsupported image storage format")
        if self.info["version"] == 1:
            supported = self.info.get("image_format") == "png" and self.info.get(
                "image_compression"
            ) in (0, 6)
        else:
            supported = (
                self.info.get("image_format") == IMAGE_FORMAT
                and self.info.get("image_color") == IMAGE_COLOR
            )
        if not supported:
            raise ValueError("Unsupported image storage format or color semantics")
        self.schema = dataset_schema(expected, self.cameras, self.info["image_format"])
        for episode in sorted((self.root / "episodes").iterdir()):
            if episode.name == f"episode_{self.num_episodes:06d}.pending":
                self.pending.append(episode)
                self.error("EPISODE_RECOVERY_PENDING", f"Interrupted save retained at {episode}")
                continue
            if self.pending or episode.name != f"episode_{self.num_episodes:06d}":
                raise ValueError(f"Unexpected episode sequence: {episode.name}")
            try:
                self._check_episode(episode)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.error("EPISODE_INVALID", f"{episode.name}: {exc}")
            self.num_episodes += 1
        if not self.num_episodes and not self.pending:
            self.warning("DATASET_EMPTY", "Dataset has no committed episodes")

    def _check_episode(self, episode: Path):
        summary = _read_json(episode / "episode.json")
        parquet = pq.ParquetFile(episode / "frames.parquet")
        if not parquet.schema_arrow.equals(self.schema):
            raise ValueError("Parquet schema does not match meta/info.json")
        count = parquet.metadata.num_rows
        if (
            summary["episode_index"] != self.num_episodes
            or count <= 0
            or summary["length"] != count
        ):
            raise ValueError("Episode length or index mismatch")
        shapes = summary["image_shapes"]
        if (
            not isinstance(shapes, dict)
            or set(shapes) != set(self.cameras)
            or (self.num_episodes and self.shapes != shapes)
        ):
            raise ValueError("Camera dimensions changed between episodes")
        for shape in shapes.values():
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or shape[2] != 3
                or any(type(n) is not int or n <= 0 for n in shape)
            ):
                raise ValueError("Invalid image shape")
        self.shapes = shapes
        offset = 0
        seen = {name: set() for name in self.cameras}
        for batch in parquet.iter_batches(batch_size=128):
            table = pa.Table.from_batches([batch])
            self._check_rows(episode, table, offset)
            for row in table.to_pylist():
                for camera in self.cameras:
                    try:
                        path = self._check_image(episode, row, camera)
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        self.error(
                            "IMAGE_INVALID",
                            f"{episode.name}, {camera}, frame {row['frame_index']}: {exc}",
                        )
                        continue
                    if path in seen[camera]:
                        self.error("IMAGE_REUSED", f"{episode.name}: repeated {camera} image")
                    seen[camera].add(path)
            offset += table.num_rows
        self._check_safety(episode, count)
        self.total_frames += count

    def _check_previews(self):
        directory = self.root / "previews"
        if not directory.is_dir():
            return
        from alohamini.datasets.video import check_preview

        for path in sorted(directory.iterdir()):
            if path.name == ".lock":
                continue
            if ".pending-" in path.name:
                self.warning("PREVIEW_INCOMPLETE", f"Unfinished preview: {path}")
                continue
            episode = self.root / "episodes" / path.name
            try:
                check_preview(episode, path, self.info["fps"], decode=self.decode_videos)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                self.warning("PREVIEW_INVALID", f"{path}: {exc}")

    def _check_rows(self, episode: Path, table: pa.Table, offset: int):
        n = table.num_rows
        expected = {
            "frame_index": list(range(offset, offset + n)),
            "index": list(range(self.total_frames + offset, self.total_frames + offset + n)),
            "episode_index": [self.num_episodes] * n,
            "task_index": [0] * n,
            "task": [self.info["task"]] * n,
        }
        for key, values in expected.items():
            if table[key].to_pylist() != values:
                self.error("FRAME_INDEX_INVALID", f"{episode.name}: invalid {key} at row {offset}")
        actual = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        desired = np.asarray(expected["frame_index"], dtype=np.float32) / self.info["fps"]
        if not np.allclose(actual, desired, rtol=0, atol=self.timestamp_tolerance_s):
            self.error("FRAME_TIMESTAMP_INVALID", f"{episode.name}: frame index time mismatch")
        for feature in self.info["features"]:
            self._check_finite_feature(episode, table, feature)
        self._check_motor_feedback(episode, table)

    def _check_image(self, episode: Path, row: dict, camera: str) -> tuple:
        reference = row[f"observation.images.{camera}"]
        shape = list(image_shape(episode, camera, reference, decode=self.decode_images))
        if camera in self.shapes and self.shapes[camera] != shape:
            raise ValueError(f"Invalid image shape: {camera}")
        self.shapes[camera] = shape
        return (
            (image_path(episode, camera, reference), reference.get("offset"))
            if isinstance(reference, dict)
            else (reference,)
        )

    def _check_safety(self, episode: Path, count: int):
        rows = []
        with (episode / "safety.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("episode_index") != self.num_episodes:
                    raise ValueError("Safety episode index mismatch")
                for key in (
                    "safety",
                    "event",
                    "issued_command",
                    "requested_action",
                    "host_timing",
                    "client_timing",
                ):
                    if row.get(key) is not None and not isinstance(row[key], dict):
                        raise ValueError(f"Invalid safety field: {key}")
                rows.append(row)
        frames = [row for row in rows if row.get("frame_index") is not None]
        if [row["frame_index"] for row in frames] != list(range(count)):
            self.error("SAFETY_FRAME_COVERAGE", f"{episode.name}: missing/reordered frame records")
        closing = (rows[-1].get("event") or {}) if rows else {}
        if closing.get("type") != "recorder_closed" or closing.get("frame_count") != count:
            self.warning("SAFETY_LOG_INCOMPLETE", f"{episode.name}: missing closing record")
        if closing.get("queue_overflows") or closing.get("rejected_images"):
            self.warning(
                "SAFETY_CAPTURE_GAP", f"{episode.name}: rejected or dropped capture frames"
            )
        if any(
            (row.get("safety") or {}).get("joint_holds")
            or (row.get("safety") or {}).get("watchdog_active")
            or (row.get("safety") or {}).get("fault")
            or (row.get("event") or {}).get("type") not in (None, "recorder_closed")
            for row in rows
        ):
            self.warning("SAFETY_REVIEW_REQUIRED", f"{episode.name}: protection/capture events")
        offset = 0
        for batch in pq.ParquetFile(episode / "frames.parquet").iter_batches(batch_size=128):
            for values, record in zip(
                batch.to_pylist(), frames[offset : offset + len(batch)], strict=False
            ):
                if record.get("feedback_phase") != "before_issued_command":
                    self.error("SAFETY_FEEDBACK_PHASE", f"{episode.name}: unknown feedback phase")
                requested = record.get("requested_action")
                if requested is not None:
                    names = self.info["features"]["action"]["names"]
                    if set(requested) != set(names) or not np.array_equal(
                        np.asarray([requested[name] for name in names], dtype=np.float32),
                        np.asarray(values["action"], dtype=np.float32),
                    ):
                        self.error(
                            "SAFETY_ACTION_MISMATCH", f"{episode.name}: requested action mismatch"
                        )
                if (
                    record.get("robot_metadata", self.info["robot_metadata"])
                    != self.info["robot_metadata"]
                ):
                    self.error(
                        "SAFETY_MODEL_MISMATCH", f"{episode.name}: model/calibration changed"
                    )
            offset += len(batch)
        for record in frames:
            host = record.get("host_timing") or {}
            cameras = host.get("camera_capture_monotonic_s", {})
            if not isinstance(cameras, dict):
                raise ValueError("Invalid camera capture timestamps")
            if set(cameras) != set(self.cameras):
                self.warning(
                    "SAFETY_CAPTURE_CLOCK_MISSING", f"{episode.name}: camera timing missing"
                )
            if cameras and not all(math.isfinite(float(value)) for value in cameras.values()):
                raise ValueError("Non-finite camera capture time")
            if host.get("state_sample_monotonic_s") is not None and not math.isfinite(
                float(host["state_sample_monotonic_s"])
            ):
                raise ValueError("Non-finite state capture time")
            client = record.get("client_timing")
            if client:
                values = [
                    float(client[key])
                    for key in (
                        "observation_received_monotonic_s",
                        "action_sample_started_monotonic_s",
                        "action_sample_finished_monotonic_s",
                        "command_sent_monotonic_s",
                    )
                ]
                if not all(math.isfinite(value) for value in values) or values != sorted(values):
                    raise ValueError("invalid client observation/action chronology")
        # The deployed checker uses only same-machine differences. Native frames
        # without physical timestamps are reported, never filled from wall time.
        timed = all(
            (
                (row.get("host_timing") or {}).get("camera_capture_monotonic_s")
                or (row.get("host_timing") or {}).get("state_sample_monotonic_s") is not None
            )
            for row in frames
        )
        if not all(row.get("client_timing") for row in frames):
            self.warning("SAFETY_CAPTURE_CLOCK_MISSING", f"{episode.name}: client timing missing")
        if not timed:
            self.warning("SAFETY_CAPTURE_CLOCK_MISSING", f"{episode.name}: physical timing missing")
        else:
            self._check_capture_timeline(self.num_episodes, frames)

    def _check_motor_feedback(self, path: Path, table: pa.Table) -> None:
        for key in table.column_names:
            if not key.startswith("observation.motor_"):
                continue
            self._check_finite_feature(path, table, key)
            mask = f"motor_feedback.{key.removeprefix('observation.motor_')}_valid"
            if mask not in table.column_names:
                self.error("MOTOR_FEEDBACK_MASK_MISSING", f"Missing validity mask for {key}")
                continue
            try:
                values = np.asarray(table[key].to_pylist(), dtype=np.float64)
                valid = np.asarray(table[mask].to_pylist(), dtype=np.float64)
            except (ValueError, TypeError):
                self.error("MOTOR_FEEDBACK_INVALID", f"Invalid motor feedback values for {key}")
                continue
            if valid.shape != values.shape or not np.isin(valid, (0.0, 1.0)).all():
                self.error("MOTOR_FEEDBACK_MASK_INVALID", f"Invalid validity mask for {key}")
            elif not valid.all():
                self.warning(
                    "MOTOR_FEEDBACK_INCOMPLETE",
                    f"{key}: {np.count_nonzero(valid == 0)} unavailable values in "
                    f"{path.relative_to(self.root)}; filter invalid feedback before training",
                )
        time_keys = ("motor_feedback.sample_started_s", "motor_feedback.sample_finished_s")
        if any(key in table.column_names for key in time_keys):
            if not all(key in table.column_names for key in time_keys):
                self.error("MOTOR_FEEDBACK_TIMING_INVALID", "Missing motor feedback sample time")
                return
            try:
                started, finished = (
                    np.asarray(table[key].to_pylist(), dtype=np.float64) for key in time_keys
                )
                valid_times = (
                    started.shape == finished.shape
                    and np.isfinite(started).all()
                    and np.isfinite(finished).all()
                    and (started >= 0).all()
                    and (finished >= started).all()
                )
            except (ValueError, TypeError):
                valid_times = False
            if not valid_times:
                self.error(
                    "MOTOR_FEEDBACK_TIMING_INVALID", "Invalid motor feedback sample interval"
                )

    def _check_finite_feature(self, path: Path, table: pa.Table, feature: str) -> None:
        if feature not in table.column_names:
            return
        try:
            values = np.asarray(table[feature].to_pylist(), dtype=np.float64)
        except (TypeError, ValueError):
            self.error(
                "FEATURE_VALUES_INVALID",
                f"Cannot convert {feature} to a numeric array in {path.relative_to(self.root)}",
            )
            return
        if values.size and not np.isfinite(values).all():
            bad_rows = np.flatnonzero(~np.isfinite(values).all(axis=tuple(range(1, values.ndim))))
            self.error(
                "FEATURE_NONFINITE",
                f"{feature} contains NaN/Inf in {path.relative_to(self.root)}; "
                f"rows={bad_rows[:20].tolist()}",
            )

    def _check_capture_timeline(self, episode: int, frames: list[dict]) -> None:
        """Compare physical acquisition times with the fixed-FPS media index.

        Host and client monotonic clocks must never be subtracted from each other.
        Prefer Host camera times; legacy sidecars can only attest client write times.
        """
        if not frames:
            return
        host_stamps = []
        for row in frames:
            timing = row.get("host_timing") or {}
            cameras = timing.get("camera_capture_monotonic_s") or {}
            values = [float(value) for value in cameras.values()]
            if values:
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("non-finite camera capture time")
                host_stamps.append(float(np.median(values)))
            else:
                stamp = timing.get(
                    "state_sample_monotonic_s", timing.get("state_sample_finished_monotonic_s")
                )
                host_stamps.append(None if stamp is None else float(stamp))
            client = row.get("client_timing")
            if client:
                values = [
                    float(client[key])
                    for key in (
                        "observation_received_monotonic_s",
                        "action_sample_started_monotonic_s",
                        "action_sample_finished_monotonic_s",
                        "command_sent_monotonic_s",
                    )
                ]
                if not all(math.isfinite(value) for value in values) or values != sorted(values):
                    raise ValueError("invalid client observation/action chronology")
        if any(stamp is None for stamp in host_stamps):
            self.warning(
                "SAFETY_CAPTURE_CLOCK_MISSING",
                f"Episode {episode}: physical capture timing is incomplete",
            )
            if any(row.get("client_monotonic_s") is None for row in frames):
                self.warning(
                    "SAFETY_CAPTURE_CLOCK_MISSING",
                    f"Episode {episode}: physical and client times are incomplete",
                )
                return
            stamps = [float(row["client_monotonic_s"]) for row in frames]
        else:
            stamps = host_stamps
        if not all(math.isfinite(stamp) for stamp in stamps):
            raise ValueError("non-finite acquisition time")
        interval = 1 / self.info["fps"]
        # Median camera time can hide one stalled or delayed stream.
        cameras = set().union(
            *(
                ((row.get("host_timing") or {}).get("camera_capture_monotonic_s") or {}).keys()
                for row in frames
            )
        )
        for camera in sorted(cameras):
            values = [
                ((row.get("host_timing") or {}).get("camera_capture_monotonic_s") or {}).get(camera)
                for row in frames
            ]
            if any(value is None for value in values):
                self.warning("SAFETY_CAPTURE_CLOCK_MISSING", f"Episode {episode}: {camera}")
                continue
            deltas = np.diff(np.asarray(values, dtype=np.float64))
            if np.any(deltas <= 0) or np.any(deltas > 1.5 * interval + self.timestamp_tolerance_s):
                self.warning(
                    "SAFETY_CAMERA_GAP", f"Episode {episode}: {camera} has gaps/repeated frames"
                )
        for row in frames:
            values = list(
                ((row.get("host_timing") or {}).get("camera_capture_monotonic_s") or {}).values()
            )
            values = [float(value) for value in values]
            if values and max(values) - min(values) > interval + self.timestamp_tolerance_s:
                self.warning(
                    "SAFETY_CAMERA_SKEW", f"Episode {episode}: camera skew exceeds one frame"
                )
                break
        intervals = np.diff(stamps)
        drift = [abs(stamp - stamps[0] - index * interval) for index, stamp in enumerate(stamps)]
        if any(
            delta <= 0 or delta > 1.5 * interval + self.timestamp_tolerance_s for delta in intervals
        ):
            self.warning(
                "SAFETY_CAPTURE_GAP",
                f"Episode {episode}: physical samples have gaps or clock discontinuities",
            )
        if max(drift) > interval + self.timestamp_tolerance_s:
            self.warning(
                "SAFETY_CAPTURE_TIMEBASE",
                f"Episode {episode}: physical acquisition differs from the fixed-FPS timeline "
                f"by up to {max(drift):.3f}s; temporal resampling or segmentation requires review",
            )

    def report(self):
        errors = sum(issue.severity == "error" for issue in self.issues)
        warnings = len(self.issues) - errors
        return {
            "dataset_root": str(self.root),
            "valid": errors == 0,
            "training_review": "required" if errors or warnings else "not_assessed",
            "errors": errors,
            "warnings": warnings,
            "summary": {
                "episodes": self.num_episodes,
                "frames": self.total_frames,
                "pending_episodes": len(self.pending),
            },
            "issues": [asdict(issue) for issue in self.issues],
        }


def _recover_pending(checker: IntegrityChecker, output: Path):
    """Recover the complete journal prefix, without reindexing or fabricating frames."""
    source = checker.pending[0]
    destination = output / "episodes" / source.stem
    destination.mkdir()
    checker.decode_images = True
    count = 0
    reason = "journal ended before episode commit"
    with (
        (source / "journal.jsonl").open("rb") as journal,
        (destination / "safety.jsonl").open("x", encoding="utf-8") as safety,
        pq.ParquetWriter(
            destination / "frames.parquet", checker.schema, compression="zstd"
        ) as writer,
        ImageShards(destination) if checker.info["version"] == 2 else nullcontext() as shards,
    ):
        for line in journal:
            if not line.endswith(b"\n"):
                reason = "incomplete trailing journal record"
                break
            item = json.loads(line)
            record = item["record"]
            if record.get("episode_index") != checker.num_episodes:
                raise ValueError("Ambiguous journal episode identity")
            if "frame" not in item:
                if record.get("frame_index") is not None:
                    raise ValueError("Journal safety frame has no numeric row")
                safety.write(_json(record))
                continue
            row = item["frame"]
            if set(row) != set(checker.schema.names) or record.get("frame_index") != count:
                raise ValueError("Ambiguous journal frame schema or sequence")
            if any(
                type(row[key]) is not int
                for key in ("index", "frame_index", "episode_index", "task_index")
            ):
                raise ValueError("Journal indices must be integers, not coerced numeric values")
            table = pa.Table.from_pylist([row], schema=checker.schema)
            before = len(checker.issues)
            checker._check_rows(source, table, count)
            if any(issue.severity == "error" for issue in checker.issues[before:]):
                raise ValueError("Invalid journal numeric data; refusing ambiguous repair")
            try:
                for camera in checker.cameras:
                    checker._check_image(source, row, camera)
            except (OSError, ValueError) as exc:
                reason = f"incomplete image group at frame {count}: {exc}"
                break
            for camera in checker.cameras:
                key = f"observation.images.{camera}"
                reference = row[key]
                if shards is not None:
                    capture = int(Path(reference["member"]).stem.removeprefix("frame_"))
                    row[key] = shards.append(
                        camera, capture, image_bytes(source, camera, reference)
                    )
                else:
                    image = image_path(source, camera, reference)
                    target = destination / image.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(image, target)
            table = pa.Table.from_pylist([row], schema=checker.schema)
            writer.write_table(table)
            safety.write(_json(record))
            count += 1
        if not count:
            raise ValueError("No complete journal frames to recover; source retained")
        for event in (
            {"type": "recovery_prefix_saved", "frame_count": count, "reason": reason},
            {
                "type": "recorder_closed",
                "frame_count": count,
                "queue_overflows": None,
                "rejected_images": None,
            },
        ):
            safety.write(
                _json({"episode_index": checker.num_episodes, "frame_index": None, "event": event})
            )
    _write_json(
        destination / "episode.json",
        {
            "episode_index": checker.num_episodes,
            "length": count,
            "submitted": None,
            "queue_overflows": None,
            "rejected_images": None,
            "image_shapes": checker.shapes,
        },
    )


def export_dataset(root: Path, output: Path, *, recover=False) -> dict:
    """New-directory-only export preserving JPEG shards or losslessly compressing old PNGs."""
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(root) or root.is_relative_to(output.resolve()):
        raise ValueError("Output must be separate from the source dataset")
    checker = IntegrityChecker(root, decode_images=True)
    with _read_lock(root):
        checker._run_unlocked()
        errors = [
            issue
            for issue in checker.issues
            if issue.severity == "error"
            and not (recover and issue.code == "EPISODE_RECOVERY_PENDING")
        ]
        if errors or (recover and len(checker.pending) != 1):
            raise ValueError(f"Source requires review before export: {checker.report()}")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f"{output.name}.pending-", dir=output.parent))
        try:
            (stage / "recording.lock").touch(exist_ok=False)
            (stage / "episodes").mkdir()
            (stage / "meta").mkdir()
            info = dict(checker.info)
            if not recover and info["version"] == 1:
                info["image_compression"] = 6
            _write_json(stage / "meta/info.json", info)
            for i in range(checker.num_episodes):
                episode = root / "episodes" / f"episode_{i:06d}"
                target = stage / "episodes" / episode.name
                target.mkdir()
                for name in ("frames.parquet", "safety.jsonl", "episode.json"):
                    shutil.copyfile(episode / name, target / name)
                for batch in pq.ParquetFile(episode / "frames.parquet").iter_batches(
                    batch_size=128
                ):
                    for row in batch.to_pylist():
                        for camera in checker.cameras:
                            source = image_path(
                                episode, camera, row[f"observation.images.{camera}"]
                            )
                            image_target = target / source.relative_to(episode)
                            image_target.parent.mkdir(parents=True, exist_ok=True)
                            if info["version"] == 2:
                                if not image_target.exists():
                                    shutil.copyfile(source, image_target)
                            elif recover:
                                shutil.copyfile(source, image_target)
                            else:
                                with Image.open(source) as image:
                                    image.save(image_target, format="PNG", compress_level=6)
                                    with Image.open(image_target) as saved:
                                        if not np.array_equal(np.asarray(image), np.asarray(saved)):
                                            raise ValueError(
                                                f"Lossless export pixel mismatch: {source}"
                                            )
            if recover:
                _recover_pending(checker, stage)
            _write_json(
                stage / "meta/export.json",
                {
                    "operation": "recover" if recover else "native_export",
                    "source": str(root),
                    "pending_source": str(checker.pending[0]) if recover else None,
                },
            )
            report = IntegrityChecker(stage, decode_images=True).run()
            if not report["valid"]:
                raise ValueError(f"Output validation failed: {report}")
            if output.exists():
                raise FileExistsError(output)
            stage.rename(output)
            report["dataset_root"] = str(output)
            return report
        except BaseException as exc:
            raise RuntimeError(
                f"Export unfinished: {exc}; source unchanged, files retained at {stage}"
            ) from exc


def check_dataset(root, *, decode_images=False, decode_videos=False):
    try:
        info = _read_json(Path(root).expanduser() / "meta/info.json")
        if not isinstance(info, dict):
            raise ValueError("meta/info.json must contain an object")
        if info.get("format") == "alohamini-episodes":
            checker = IntegrityChecker(
                root, decode_images=decode_images, decode_videos=decode_videos
            )
        elif info.get("codebase_version") == "v3.0":
            from alohamini.datasets.lerobot_tools import IntegrityChecker as LeRobotChecker

            checker = LeRobotChecker(root, decode_images=decode_images, decode_videos=decode_videos)
        else:
            raise ValueError("Expected an AlohaMini native or LeRobot v3 dataset")
        return checker.run()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        checker = IntegrityChecker(root)
        checker.error("DATASET_UNREADABLE", str(exc))
        return checker.report()


def repair_dataset(root, output):
    root = Path(root).expanduser().resolve()
    info = _read_json(root / "meta/info.json")
    if not isinstance(info, dict):
        raise ValueError("meta/info.json must contain an object")
    if info.get("format") == "alohamini-episodes":
        return export_dataset(root, output, recover=True)
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Expected an AlohaMini native or LeRobot v3 dataset")
    from alohamini.datasets.lerobot_tools import DatasetRepairer
    from alohamini.datasets.lerobot_tools import IntegrityChecker as LeRobotChecker

    checker = LeRobotChecker(root, decode_videos=True, decode_images=True)
    report = checker.run()
    manifest, repaired = DatasetRepairer(checker, Path(output)).run(
        report, timestamp_tolerance_s=checker.timestamp_tolerance_s
    )
    repaired["repair"] = manifest
    return repaired


def print_report(report: dict):
    summary = report["summary"]
    print(f"Dataset: {report['dataset_root']}")
    if "declared_episodes" in summary:
        print(
            "Declared/actual: "
            f"episodes={summary['declared_episodes']}/{summary['actual_episode_metadata_rows']}, "
            f"frames={summary['declared_frames']}/{summary['actual_data_rows']}, "
            f"videos={summary['referenced_video_files']}"
        )
    else:
        print(
            f"Episodes={summary['episodes']} frames={summary['frames']} "
            f"pending={summary['pending_episodes']}"
        )
    for issue in report["issues"]:
        print(f"[{issue['severity'].upper()}] {issue['code']}: {issue['message']}")
    status = "VALID" if report["valid"] else "INVALID"
    print(f"Result: {status} ({report['errors']} errors, {report['warnings']} warnings)")
    print(
        f"Training review: {report['training_review']} "
        "(structural validity is not training approval)"
    )
