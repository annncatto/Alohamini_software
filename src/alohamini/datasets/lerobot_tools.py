# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline LeRobot v3 checking and conservative repair, migrated from the two source scripts."""

from __future__ import annotations

import io
import json
import math
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from alohamini.datasets.lerobot import RunningQuantileStats
from alohamini.datasets.native import _write_json
from alohamini.datasets.tools import IntegrityChecker as NativeChecker
from alohamini.datasets.video import repack_video, video_frame_count


def aggregate_feature_stats(
    stats_ft_list: list[dict[str, dict]],
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregates stats for a single feature."""
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    # Prepare weighted mean by matching number of dimensions
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    # Compute the weighted mean
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    # Compute the variance using the parallel algorithm
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    aggregated = {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }

    if stats_ft_list:
        quantile_keys = [k for k in stats_ft_list[0] if k.startswith("q") and k[1:].isdigit()]

        for q_key in quantile_keys:
            if all(q_key in s for s in stats_ft_list):
                quantile_values = np.stack([s[q_key] for s in stats_ft_list])
                weighted_quantiles = quantile_values * counts
                aggregated[q_key] = weighted_quantiles.sum(axis=0) / total_count

    return aggregated


def aggregate_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary by joining keys with a separator.

    Example:
        >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        >>> print(flatten_dict(dct))
        {'a/b': 1, 'a/c/d': 2, 'e': 3}

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key to prepend to the keys in this level.
        sep (str): The separator to use between keys.

    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    """Unflatten a dictionary with delimited keys into a nested dictionary.

    Example:
        >>> flat_dct = {"a/b": 1, "a/c/d": 2, "e": 3}
        >>> print(unflatten_dict(flat_dct))
        {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}

    Args:
        d (dict): A dictionary with flattened keys.
        sep (str): The separator used in the keys.

    Returns:
        dict: A nested dictionary.
    """
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_inner = outdict
        for part in parts[:-1]:
            if part not in d_inner:
                d_inner[part] = {}
            d_inner = d_inner[part]
        d_inner[parts[-1]] = value
    return outdict


def get_feature_stats(array, *, axis=0, keepdims=False):
    # Only numeric index vectors are recomputed by metadata repair.
    if axis != 0 or keepdims:
        raise ValueError("Index statistics require axis=0 and keepdims=False")
    stats = RunningQuantileStats()
    stats.update(np.asarray(array).reshape(len(array), -1))
    return stats.get_statistics()


def write_stats(stats, root):
    _write_json(
        root / "meta/stats.json",
        {
            key: {name: value.tolist() for name, value in values.items()}
            for key, values in stats.items()
        },
    )


def _dataset_path(root, template, **values):
    relative = Path(template.format(**values))
    path = root / relative
    if relative.is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes its root")
    return path


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    message: str


@dataclass
class DataEpisode:
    count: int = 0
    indices: list[int] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    files: set[Path] = field(default_factory=set)


@dataclass(frozen=True)
class VideoRange:
    episode_index: int
    start_frame: int
    end_frame: int
    episode_length: int
    video_key: str


REPAIRABLE_ERROR_CODES = {
    "DATA_EPISODE_IDS_NONCONTIGUOUS",
    "EPISODE_DATASET_RANGE_MISMATCH",
    "GLOBAL_INDEX_NONCONTIGUOUS",
    "METADATA_EPISODE_IDS_NONCONTIGUOUS",
    "METADATA_LENGTH_SUM_MISMATCH",
    "TOTAL_EPISODES_MISMATCH",
    "TOTAL_FRAMES_MISMATCH",
    "VIDEO_EPISODE_LENGTH_MISMATCH",
    "VIDEO_FRAME_GAP",
    "VIDEO_LEADING_GAP",
    "VIDEO_TOTAL_FRAMES_MISMATCH",
}


class IntegrityChecker:
    def __init__(
        self, root: Path, *, decode_videos=False, decode_images=False, timestamp_tolerance_s=1e-4
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.decode_images = decode_images
        self.decode_videos = decode_videos
        self.timestamp_tolerance_s = timestamp_tolerance_s
        self.issues: list[Issue] = []
        self.info: dict[str, Any] = {}
        self.data_episodes: dict[int, DataEpisode] = defaultdict(DataEpisode)
        self.data_files: set[Path] = set()
        self.episode_rows: list[dict[str, Any]] = []
        self.referenced_videos: set[Path] = set()
        self.task_indices: set[int] = set()
        self.task_count = 0
        self.total_data_rows = 0
        self.referenced_images: set[Path] = set()
        self._last_episode = self._last_index = -1

    def error(self, code: str, message: str) -> None:
        self.issues.append(Issue("error", code, message))

    def warning(self, code: str, message: str) -> None:
        self.issues.append(Issue("warning", code, message))

    def run(self) -> dict[str, Any]:
        if not self.root.is_dir():
            self.error("ROOT_MISSING", f"Dataset root does not exist: {self.root}")
            return self.report()

        try:
            for path in self.root.rglob("*"):
                if path.is_symlink():
                    raise ValueError(f"Dataset contains a symbolic link: {path}")
            if not self._load_info_and_required_files():
                return self.report()
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self.error("DATASET_UNREADABLE", str(exc))
            return self.report()
        for check in (
            self._check_zero_byte_files,
            self._read_data_parquets,
            self._read_episode_parquets,
            self._check_global_counts_and_indices,
            self._check_episode_correspondence,
            self._check_video_correspondence,
            self._check_unreferenced_files,
            self._check_safety_sidecars,
        ):
            try:
                check()
            except (OSError, ValueError, KeyError, TypeError, OverflowError, AttributeError) as exc:
                self.error("DATASET_UNREADABLE", f"{check.__name__}: {exc}")
        return self.report()

    def _check_safety_sidecars(self) -> None:
        if any(self.root.joinpath("meta/recovery").glob("*.json")):
            self.error("EPISODE_RECOVERY_PENDING", "An interrupted episode save requires recovery")
        directory = self.root / "meta/safety"
        if not directory.exists() and not self.info.get("safety_format"):
            return
        seen = set()
        for path in sorted(directory.glob("episode_*.jsonl")):
            try:
                episode = int(path.stem.removeprefix("episode_"))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if any(
                    not isinstance(row, dict) or row.get("episode_index") != episode for row in rows
                ):
                    raise ValueError("episode_index does not match the sidecar filename")
                if episode not in self.data_episodes:
                    raise ValueError("sidecar has no corresponding data episode")
                if any(
                    row.get(key) is not None and not isinstance(row[key], dict)
                    for row in rows
                    for key in (
                        "safety",
                        "event",
                        "issued_command",
                        "requested_action",
                        "host_timing",
                        "client_timing",
                    )
                ):
                    raise ValueError("sidecar metadata fields must be objects")
                seen.add(episode)
                frames = [row for row in rows if row.get("frame_index") is not None]
                if any(type(row["frame_index"]) is not int for row in frames):
                    raise ValueError("frame_index must be an integer")
                expected = list(range(self.data_episodes[episode].count))
                if [row["frame_index"] for row in frames] != expected:
                    self.warning(
                        "SAFETY_FRAME_COVERAGE",
                        f"Episode {episode}: safety frames are missing or out of order",
                    )
                closing = (rows[-1].get("event") or {}) if rows else {}
                dropped = closing.get("dropped_records")
                if dropped is None and "queue_overflows" in closing:
                    dropped = closing["queue_overflows"] + closing.get("rejected_images", 0)
                if (
                    closing.get("type") != "recorder_closed"
                    or dropped != 0
                    or closing.get("frame_count") != len(expected)
                ):
                    self.warning(
                        "SAFETY_LOG_INCOMPLETE",
                        f"Episode {episode}: safety log completeness is unverified",
                    )
                previous = None
                for row in frames:
                    if row.get("client_monotonic_s") is None:
                        self.warning(
                            "SAFETY_CAPTURE_CLOCK_MISSING",
                            f"Episode {episode}: client time missing",
                        )
                        previous = None
                        continue
                    stamp = float(row["client_monotonic_s"])
                    if not math.isfinite(stamp):
                        raise ValueError("non-finite capture time")
                    if previous is not None and (
                        stamp <= previous or stamp - previous > max(0.1, 2 / self.info["fps"])
                    ):
                        self.warning(
                            "SAFETY_TIME_GAP",
                            f"Episode {episode}, frame {row['frame_index']}: "
                            "capture time discontinuity",
                        )
                    previous = stamp
                self._check_capture_timeline(episode, frames)
                if any(
                    (row.get("safety") or {}).get("joint_holds")
                    or (row.get("safety") or {}).get("watchdog_active")
                    or ((row.get("event") or {}).get("type") not in (None, "recorder_closed"))
                    for row in rows
                ):
                    self.warning(
                        "SAFETY_REVIEW_REQUIRED",
                        f"Episode {episode}: protection or capture/recovery events require review",
                    )
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.error("SAFETY_SIDECAR_INVALID", f"{path}: {error}")
        for episode in sorted(set(self.data_episodes) - seen):
            self.warning("SAFETY_SIDECAR_MISSING", f"Episode {episode}: no safety sidecar")

    _check_capture_timeline = NativeChecker._check_capture_timeline

    def report(self) -> dict[str, Any]:
        errors = sum(issue.severity == "error" for issue in self.issues)
        warnings = sum(issue.severity == "warning" for issue in self.issues)
        return {
            "dataset_root": str(self.root),
            "valid": errors == 0,
            "training_review": "required"
            if errors
            or any(issue.code.startswith(("SAFETY_", "MOTOR_FEEDBACK_")) for issue in self.issues)
            else "not_assessed",
            "errors": errors,
            "warnings": warnings,
            "summary": {
                "declared_episodes": self.info.get("total_episodes"),
                "declared_frames": self.info.get("total_frames"),
                "actual_episode_metadata_rows": len(self.episode_rows),
                "actual_data_rows": self.total_data_rows,
                "actual_data_episodes": sorted(self.data_episodes),
                "referenced_video_files": len(self.referenced_videos),
            },
            "issues": [asdict(issue) for issue in self.issues],
        }

    def _load_info_and_required_files(self) -> bool:
        required = (
            self.root / "meta/info.json",
            self.root / "meta/stats.json",
            self.root / "meta/tasks.parquet",
        )
        ok = True
        for path in required:
            if not path.is_file():
                self.error("REQUIRED_FILE_MISSING", f"Missing required file: {path}")
                ok = False
            elif path.stat().st_size == 0:
                self.error("REQUIRED_FILE_EMPTY", f"Required file is empty: {path}")
                ok = False
        if not ok:
            return False

        try:
            self.info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.error("INFO_INVALID", f"Cannot parse meta/info.json: {exc}")
            return False

        required_info = {
            "fps",
            "features",
            "total_episodes",
            "total_frames",
            "total_tasks",
            "data_path",
        }
        missing = sorted(required_info - self.info.keys())
        if missing:
            self.error("INFO_FIELDS_MISSING", f"meta/info.json is missing fields: {missing}")
            return False
        if any(
            type(self.info[key]) is not int or self.info[key] < 0
            for key in ("total_episodes", "total_frames", "total_tasks")
        ):
            self.error("INFO_TYPES_INVALID", "Dataset counts must be non-negative integers")
            return False

        try:
            fps = float(self.info["fps"])
            total_episodes = int(self.info["total_episodes"])
            total_frames = int(self.info["total_frames"])
        except (TypeError, ValueError) as exc:
            self.error("INFO_TYPES_INVALID", f"Invalid count or fps in meta/info.json: {exc}")
            return False
        if not math.isfinite(fps) or fps <= 0 or total_episodes < 0 or total_frames < 0:
            self.error(
                "INFO_VALUES_INVALID",
                f"Invalid fps/episode/frame values: fps={fps}, "
                f"episodes={total_episodes}, frames={total_frames}",
            )
            return False

        if self.info.get("codebase_version") != "v3.0":
            self.error("FORMAT_UNSUPPORTED", "Expected LeRobot v3.0 metadata")
            return False
        if not isinstance(self.info["features"], dict):
            self.error("FEATURES_INVALID", "features must be a mapping")
            return False
        for key, feature in self.info["features"].items():
            if (
                not isinstance(feature, dict)
                or not isinstance(feature.get("dtype"), str)
                or not isinstance(feature.get("shape"), list)
                or not feature["shape"]
                or any(type(value) is not int or value <= 0 for value in feature["shape"])
            ):
                self.error("FEATURES_INVALID", f"Invalid feature metadata: {key}")
                return False

        tasks_path = self.root / "meta/tasks.parquet"
        try:
            self.task_count = pq.read_metadata(tasks_path).num_rows
            if self.task_count != int(self.info["total_tasks"]):
                self.error(
                    "TOTAL_TASKS_MISMATCH",
                    f"info total_tasks={self.info['total_tasks']}, "
                    f"tasks parquet rows={self.task_count}",
                )
        except Exception as exc:
            self.error("TASKS_PARQUET_INVALID", f"Cannot parse {tasks_path}: {exc}")
        stats_path = self.root / "meta/stats.json"
        try:
            json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.error("STATS_JSON_INVALID", f"Cannot parse {stats_path}: {exc}")
        return True

    def _check_zero_byte_files(self) -> None:
        for directory in ("data", "meta", "videos", "images"):
            base = self.root / directory
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_file() and path.stat().st_size == 0:
                    self.error("ZERO_BYTE_FILE", f"Zero-byte file: {path.relative_to(self.root)}")

    @staticmethod
    def _check_schema(
        path: Path,
        schema: pa.Schema,
        reference_path: Path | None,
        reference_schema: pa.Schema | None,
    ) -> tuple[Path, pa.Schema]:
        if reference_schema is not None and not schema.equals(
            reference_schema, check_metadata=False
        ):
            raise ValueError(f"schema differs from {reference_path}")
        return (
            path if reference_path is None else reference_path,
            schema if reference_schema is None else reference_schema,
        )

    def _read_data_parquets(self) -> None:
        data_root = self.root / "data"
        paths = sorted(data_root.rglob("*.parquet")) if data_root.is_dir() else []
        if not paths:
            self.error("DATA_PARQUETS_MISSING", f"No data parquet files found under {data_root}")
            return

        required_columns = {"episode_index", "frame_index", "timestamp", "index", "task_index"}
        required_columns.update(
            key
            for key, feature in self.info.get("features", {}).items()
            if isinstance(feature, dict) and feature.get("dtype") != "video"
        )
        reference_path: Path | None = None
        reference_schema: pa.Schema | None = None

        for path in paths:
            self.data_files.add(path.resolve())
            try:
                parquet = pq.ParquetFile(path)
            except Exception as exc:
                self.error(
                    "DATA_PARQUET_INVALID", f"Cannot read {path.relative_to(self.root)}: {exc}"
                )
                continue

            try:
                reference_path, reference_schema = self._check_schema(
                    path, parquet.schema_arrow, reference_path, reference_schema
                )
            except ValueError as exc:
                self.error("DATA_SCHEMA_MISMATCH", f"{path.relative_to(self.root)}: {exc}")

            missing = sorted(required_columns - set(parquet.schema_arrow.names))
            if missing:
                self.error(
                    "DATA_COLUMNS_MISSING",
                    f"{path.relative_to(self.root)} is missing columns: {missing}",
                )
                continue

            try:
                for batch in parquet.iter_batches(batch_size=8):
                    table = pa.Table.from_batches([batch])
                    self._check_data_batch(path, table)
            except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
                self.error("DATA_PARQUET_INVALID", f"{path}: {exc}")

    def _check_data_batch(self, path: Path, table: pa.Table):
        self.total_data_rows += table.num_rows
        episode_values = table["episode_index"].to_pylist()
        frame_values = table["frame_index"].to_pylist()
        timestamp_values = table["timestamp"].to_pylist()
        index_values = table["index"].to_pylist()
        self.task_indices.update(
            int(value) for value in table["task_index"].to_pylist() if value is not None
        )

        for episode, frame, timestamp, index in zip(
            episode_values, frame_values, timestamp_values, index_values, strict=True
        ):
            if None in (episode, frame, timestamp, index):
                self.error("DATA_NULL_INDEX", f"Null index value in {path.relative_to(self.root)}")
                continue
            record = self.data_episodes[int(episode)]
            if int(episode) < self._last_episode or int(index) <= self._last_index:
                self.error("DATA_ROW_ORDER_INVALID", f"{path}: episode/global indices out of order")
            if record.frame_indices and int(frame) <= record.frame_indices[-1]:
                self.error("DATA_ROW_ORDER_INVALID", f"{path}: frame indices out of order")
            self._last_episode, self._last_index = int(episode), int(index)
            record.count += 1
            record.indices.append(int(index))
            record.frame_indices.append(int(frame))
            record.timestamps.append(float(timestamp))
            record.files.add(path.resolve())

        for key, feature in self.info["features"].items():
            if feature.get("dtype") not in ("video", "image", "string"):
                self._check_finite_feature(path, table, key)
        self._check_images(path, table)
        self._check_motor_feedback(path, table)

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
        shape = self.info["features"].get(feature, {}).get("shape")
        actual_shape = list(values.shape[1:]) if values.ndim > 1 else [1]
        if shape is not None and list(shape) != actual_shape:
            self.error("FEATURE_SHAPE_MISMATCH", f"{feature}: expected {shape}, got {actual_shape}")
        if self.info["features"].get(feature, {}).get("dtype", "").startswith("int"):
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                self.error("FEATURE_VALUES_INVALID", f"{feature}: expected integral values")
        if values.size and not np.isfinite(values).all():
            bad_rows = np.flatnonzero(~np.isfinite(values).all(axis=tuple(range(1, values.ndim))))
            self.error(
                "FEATURE_NONFINITE",
                f"{feature} contains NaN/Inf in {path.relative_to(self.root)}; "
                f"rows={bad_rows[:20].tolist()}",
            )

    def _read_episode_parquets(self) -> None:
        episode_root = self.root / "meta/episodes"
        paths = sorted(episode_root.rglob("*.parquet")) if episode_root.is_dir() else []
        if not paths:
            self.error(
                "EPISODE_PARQUETS_MISSING",
                f"No episode metadata parquet files found under {episode_root}",
            )
            return

        required_columns = {
            "episode_index",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
        }
        reference_path: Path | None = None
        reference_schema: pa.Schema | None = None

        for path in paths:
            try:
                table = pq.read_table(path)
            except Exception as exc:
                self.error(
                    "EPISODE_PARQUET_INVALID",
                    f"Cannot read {path.relative_to(self.root)}: {exc}",
                )
                continue

            try:
                reference_path, reference_schema = self._check_schema(
                    path, table.schema, reference_path, reference_schema
                )
            except ValueError as exc:
                self.error("EPISODE_SCHEMA_MISMATCH", f"{path.relative_to(self.root)}: {exc}")

            missing = sorted(required_columns - set(table.column_names))
            if missing:
                self.error(
                    "EPISODE_COLUMNS_MISSING",
                    f"{path.relative_to(self.root)} is missing columns: {missing}",
                )
                continue
            rows = table.to_pylist()
            for row in rows:
                if any(type(row[key]) is not int or row[key] < 0 for key in required_columns):
                    self.error("EPISODE_VALUES_INVALID", f"{path}: invalid episode indices/counts")
            self.episode_rows.extend(rows)

    def _check_images(self, path: Path, table: pa.Table) -> None:
        for key, feature in self.info["features"].items():
            if feature.get("dtype") != "image" or key not in table.column_names:
                continue
            for index, reference in enumerate(table[key].to_pylist()):
                try:
                    if not isinstance(reference, dict):
                        raise ValueError("Expected an embedded image or local image reference")
                    if reference.get("bytes") is not None:
                        source = io.BytesIO(reference["bytes"])
                    else:
                        source = _dataset_path(self.root, reference["path"])
                        self.referenced_images.add(source.resolve())
                    with Image.open(source) as image:
                        if list(feature["shape"]) != [image.height, image.width, 3]:
                            raise ValueError("Image dimensions do not match feature metadata")
                        if image.mode != "RGB":
                            raise ValueError("Expected an RGB image")
                        if self.decode_images:
                            image.load()
                        else:
                            image.verify()
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.error("IMAGE_INVALID", f"{path.name}, {key}, row {index}: {exc}")

    def _check_global_counts_and_indices(self) -> None:
        declared_frames = int(self.info.get("total_frames", -1))
        declared_episodes = int(self.info.get("total_episodes", -1))
        if self.total_data_rows != declared_frames:
            self.error(
                "TOTAL_FRAMES_MISMATCH",
                f"info total_frames={declared_frames}, data parquet rows={self.total_data_rows}",
            )
        if len(self.episode_rows) != declared_episodes:
            self.error(
                "TOTAL_EPISODES_MISMATCH",
                f"info total_episodes={declared_episodes}, "
                f"episode metadata rows={len(self.episode_rows)}",
            )
        metadata_length_sum = sum(int(row["length"]) for row in self.episode_rows)
        if metadata_length_sum != declared_frames:
            self.error(
                "METADATA_LENGTH_SUM_MISMATCH",
                f"info total_frames={declared_frames}, "
                f"sum of episode metadata lengths={metadata_length_sum}",
            )
        invalid_tasks = sorted(
            index for index in self.task_indices if index < 0 or index >= self.task_count
        )
        if invalid_tasks:
            self.error(
                "TASK_INDEX_OUT_OF_RANGE",
                "Data contains task_index values outside tasks parquet range "
                f"0..{self.task_count - 1}: "
                f"{invalid_tasks}",
            )

        actual_data_episodes = sorted(self.data_episodes)
        expected_episodes = list(range(declared_episodes))
        if actual_data_episodes != expected_episodes:
            missing = sorted(set(expected_episodes) - set(actual_data_episodes))
            unexpected = sorted(set(actual_data_episodes) - set(expected_episodes))
            self.error(
                "DATA_EPISODE_IDS_NONCONTIGUOUS",
                f"Expected episode ids {expected_episodes}; actual={actual_data_episodes}; "
                f"missing={missing}; unexpected={unexpected}",
            )

        metadata_ids = [
            int(row["episode_index"])
            for row in self.episode_rows
            if row.get("episode_index") is not None
        ]
        if sorted(metadata_ids) != expected_episodes:
            self.error(
                "METADATA_EPISODE_IDS_NONCONTIGUOUS",
                f"Expected metadata episode ids {expected_episodes}; actual={sorted(metadata_ids)}",
            )
        if len(metadata_ids) != len(set(metadata_ids)):
            duplicates = sorted(
                {episode for episode in metadata_ids if metadata_ids.count(episode) > 1}
            )
            self.error(
                "METADATA_EPISODE_IDS_DUPLICATE", f"Duplicate metadata episodes: {duplicates}"
            )

        all_indices = [index for record in self.data_episodes.values() for index in record.indices]
        if len(all_indices) != len(set(all_indices)):
            values, counts = np.unique(np.asarray(all_indices, dtype=np.int64), return_counts=True)
            duplicates = values[counts > 1][:20].tolist()
            self.error(
                "GLOBAL_INDEX_DUPLICATE", f"Duplicate global indices (first 20): {duplicates}"
            )
        if sorted(all_indices) != list(range(declared_frames)):
            expected = set(range(declared_frames))
            actual = set(all_indices)
            self.error(
                "GLOBAL_INDEX_NONCONTIGUOUS",
                f"Global index is not 0..{declared_frames - 1}; "
                f"missing(first 20)={sorted(expected - actual)[:20]}, "
                f"unexpected(first 20)={sorted(actual - expected)[:20]}",
            )

    def _check_episode_correspondence(self) -> None:
        fps = float(self.info.get("fps", 0))
        metadata_by_episode: dict[int, dict[str, Any]] = {}
        for row in self.episode_rows:
            if row.get("episode_index") is not None:
                metadata_by_episode[int(row["episode_index"])] = row

        for episode_index, record in sorted(self.data_episodes.items()):
            order = np.argsort(np.asarray(record.frame_indices, dtype=np.int64))
            frame_indices = np.asarray(record.frame_indices, dtype=np.int64)[order]
            indices = np.asarray(record.indices, dtype=np.int64)[order]
            timestamps = np.asarray(record.timestamps, dtype=np.float64)[order]
            expected_frames = np.arange(record.count, dtype=np.int64)
            if not np.array_equal(frame_indices, expected_frames):
                self.error(
                    "FRAME_INDEX_NONCONTIGUOUS",
                    f"Episode {episode_index}: frame_index is not 0..{record.count - 1}",
                )
            if len(set(record.indices)) != record.count:
                self.error(
                    "EPISODE_INDEX_DUPLICATE", f"Episode {episode_index}: duplicate global index"
                )
            expected_timestamps = frame_indices / fps
            if timestamps.size and not np.allclose(
                timestamps, expected_timestamps, rtol=0, atol=self.timestamp_tolerance_s
            ):
                max_error = float(np.max(np.abs(timestamps - expected_timestamps)))
                self.error(
                    "TIMESTAMP_MISMATCH",
                    f"Episode {episode_index}: timestamp differs from frame_index/fps; "
                    f"max_error={max_error:.6g}s",
                )

            row = metadata_by_episode.get(episode_index)
            if row is None:
                self.error(
                    "EPISODE_METADATA_MISSING", f"Episode {episode_index}: metadata row missing"
                )
                continue
            if int(row["length"]) != record.count:
                self.error(
                    "EPISODE_LENGTH_MISMATCH",
                    f"Episode {episode_index}: metadata length={row['length']}, "
                    f"data rows={record.count}",
                )
            expected_from = int(indices.min()) if indices.size else 0
            expected_to = int(indices.max()) + 1 if indices.size else 0
            if (
                int(row["dataset_from_index"]) != expected_from
                or int(row["dataset_to_index"]) != expected_to
            ):
                self.error(
                    "EPISODE_DATASET_RANGE_MISMATCH",
                    f"Episode {episode_index}: metadata range="
                    f"[{row['dataset_from_index']}, {row['dataset_to_index']}), "
                    f"actual=[{expected_from}, {expected_to})",
                )

            data_path = self._format_path(
                self.info["data_path"],
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
            if data_path is None:
                continue
            if not data_path.is_file():
                self.error(
                    "REFERENCED_DATA_MISSING", f"Episode {episode_index}: missing {data_path}"
                )
            elif data_path.resolve() not in record.files:
                actual = sorted(str(path.relative_to(self.root)) for path in record.files)
                self.error(
                    "DATA_FILE_REFERENCE_MISMATCH",
                    f"Episode {episode_index}: metadata points to "
                    f"{data_path.relative_to(self.root)}, "
                    f"but its rows are in {actual}",
                )

    def _video_keys(self) -> list[str]:
        features = self.info.get("features", {})
        if not isinstance(features, dict):
            self.error("FEATURES_INVALID", "meta/info.json features must be a mapping")
            return []
        return [
            key
            for key, feature in features.items()
            if isinstance(feature, dict) and feature.get("dtype") == "video"
        ]

    def _check_video_correspondence(self) -> None:
        video_keys = self._video_keys()
        if not video_keys:
            return
        template = self.info.get("video_path")
        if not template:
            self.error("VIDEO_PATH_MISSING", "Video features exist but info.video_path is missing")
            return

        fps = float(self.info["fps"])
        groups: dict[Path, list[VideoRange]] = defaultdict(list)
        for row in self.episode_rows:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            for key in video_keys:
                prefix = f"videos/{key}"
                required = (
                    f"{prefix}/chunk_index",
                    f"{prefix}/file_index",
                    f"{prefix}/from_timestamp",
                    f"{prefix}/to_timestamp",
                )
                missing = [
                    column for column in required if column not in row or row[column] is None
                ]
                if missing:
                    self.error(
                        "VIDEO_METADATA_MISSING",
                        f"Episode {episode_index}, {key}: missing columns/values {missing}",
                    )
                    continue
                path = self._format_path(
                    template,
                    video_key=key,
                    chunk_index=int(row[f"{prefix}/chunk_index"]),
                    file_index=int(row[f"{prefix}/file_index"]),
                )
                if path is None:
                    continue
                self.referenced_videos.add(path.resolve())
                start = round(float(row[f"{prefix}/from_timestamp"]) * fps)
                end = round(float(row[f"{prefix}/to_timestamp"]) * fps)
                if (
                    abs(start / fps - float(row[f"{prefix}/from_timestamp"]))
                    > self.timestamp_tolerance_s
                    or abs(end / fps - float(row[f"{prefix}/to_timestamp"]))
                    > self.timestamp_tolerance_s
                ):
                    self.error(
                        "VIDEO_TIMESTAMP_MISMATCH",
                        f"Episode {episode_index}: video range is off-grid",
                    )
                if end <= start:
                    self.error(
                        "VIDEO_RANGE_INVALID",
                        f"Episode {episode_index}, {key}: invalid frame range [{start}, {end})",
                    )
                if end - start != length:
                    self.error(
                        "VIDEO_EPISODE_LENGTH_MISMATCH",
                        f"Episode {episode_index}, {key}: video range has {end - start} frames, "
                        f"metadata length={length}",
                    )
                groups[path].append(VideoRange(episode_index, start, end, length, key))

        for path, ranges in sorted(groups.items(), key=lambda item: str(item[0])):
            if not path.is_file():
                self.error("REFERENCED_VIDEO_MISSING", f"Missing referenced video: {path}")
                continue
            if path.stat().st_size == 0:
                self.error("REFERENCED_VIDEO_EMPTY", f"Referenced video is empty: {path}")
                continue
            ranges.sort(key=lambda item: (item.start_frame, item.episode_index))
            if ranges[0].start_frame != 0:
                self.error(
                    "VIDEO_LEADING_GAP",
                    f"{path.relative_to(self.root)}: first referenced frame is "
                    f"{ranges[0].start_frame}, not 0",
                )
            for previous, current in zip(ranges, ranges[1:], strict=False):
                if current.start_frame > previous.end_frame:
                    self.error(
                        "VIDEO_FRAME_GAP",
                        f"{path.relative_to(self.root)}: "
                        f"{current.start_frame - previous.end_frame} unreferenced frames "
                        f"between episodes {previous.episode_index} and {current.episode_index}",
                    )
                elif current.start_frame < previous.end_frame:
                    self.error(
                        "VIDEO_RANGE_OVERLAP",
                        f"{path.relative_to(self.root)}: ranges overlap between episodes "
                        f"{previous.episode_index} and {current.episode_index}",
                    )
            self._inspect_video(path, ranges[-1].end_frame, fps, ranges[0].video_key)

    def _inspect_video(
        self, path: Path, expected_frames: int, expected_fps: float, video_key: str
    ) -> None:
        try:
            with av.open(str(path)) as container:
                if not container.streams.video:
                    self.error("VIDEO_STREAM_MISSING", f"No video stream in {path}")
                    return
                stream = container.streams.video[0]
                header_frames = int(stream.frames or 0)
                actual_fps = float(stream.average_rate) if stream.average_rate else None
                shape = self.info["features"].get(video_key, {}).get("shape")
                if isinstance(shape, list) and len(shape) >= 2:
                    expected_height, expected_width = int(shape[0]), int(shape[1])
                    if stream.height != expected_height or stream.width != expected_width:
                        self.error(
                            "VIDEO_SHAPE_MISMATCH",
                            f"{path.relative_to(self.root)}: stream shape="
                            f"({stream.height}, {stream.width}), info shape="
                            f"({expected_height}, {expected_width})",
                        )
                if actual_fps is not None and not math.isclose(
                    actual_fps, expected_fps, abs_tol=1e-3
                ):
                    self.error(
                        "VIDEO_FPS_MISMATCH",
                        f"{path.relative_to(self.root)}: stream fps={actual_fps}, "
                        f"info fps={expected_fps}",
                    )
                actual_frames = header_frames
                if self.decode_videos:
                    actual_frames = 0
                    for frame in container.decode(stream):
                        if frame.time is None or not math.isclose(
                            frame.time,
                            actual_frames / expected_fps,
                            abs_tol=self.timestamp_tolerance_s,
                        ):
                            self.error("VIDEO_TIMESTAMP_MISMATCH", f"{path}: frame {actual_frames}")
                        actual_frames += 1
                    if header_frames and actual_frames != header_frames:
                        self.error(
                            "VIDEO_DECODE_COUNT_MISMATCH",
                            f"{path.relative_to(self.root)}: header={header_frames}, "
                            f"decoded={actual_frames}",
                        )
                elif actual_frames == 0 and stream.duration is not None:
                    actual_frames = round(float(stream.duration * stream.time_base) * expected_fps)
                    self.warning(
                        "VIDEO_FRAME_COUNT_ESTIMATED",
                        f"{path.relative_to(self.root)}: stream header has no frame count; "
                        f"estimated {actual_frames} from duration",
                    )
                if actual_frames == 0:
                    self.warning(
                        "VIDEO_FRAME_COUNT_UNKNOWN",
                        f"{path.relative_to(self.root)}: "
                        "cannot determine frame count without --decode-videos",
                    )
                elif actual_frames != expected_frames:
                    self.error(
                        "VIDEO_TOTAL_FRAMES_MISMATCH",
                        f"{path.relative_to(self.root)}: video has {actual_frames} frames, "
                        f"metadata covers {expected_frames}",
                    )
        except Exception as exc:
            self.error("VIDEO_OPEN_OR_DECODE_FAILED", f"Cannot inspect {path}: {exc}")

    def _check_unreferenced_files(self) -> None:
        video_root = self.root / "videos"
        if video_root.is_dir():
            actual_videos = {path.resolve() for path in video_root.rglob("*.mp4")}
            for path in sorted(actual_videos - self.referenced_videos):
                self.warning(
                    "UNREFERENCED_VIDEO",
                    f"Video is not referenced by episode metadata: {path.relative_to(self.root)}",
                )

        referenced_data: set[Path] = set()
        template = self.info.get("data_path")
        if template:
            for row in self.episode_rows:
                path = self._format_path(
                    template,
                    chunk_index=int(row["data/chunk_index"]),
                    file_index=int(row["data/file_index"]),
                )
                if path is not None:
                    referenced_data.add(path.resolve())
        for path in sorted(self.data_files - referenced_data):
            self.warning(
                "UNREFERENCED_DATA_PARQUET",
                "Data parquet is not referenced by episode metadata: "
                f"{path.relative_to(self.root)}",
            )

        images_root = self.root / "images"
        raw_images = list(images_root.rglob("*")) if images_root.is_dir() else []
        raw_image_files = [path for path in raw_images if path.is_file()]
        if raw_image_files and self._video_keys():
            episode_dirs = sorted(
                {
                    str(path.parent.relative_to(self.root))
                    for path in raw_image_files
                    if path.parent.name.startswith("episode-")
                }
            )
            self.warning(
                "RAW_IMAGES_REMAIN",
                f"Finalized video dataset still contains {len(raw_image_files)} raw image files; "
                f"episode directories={episode_dirs[:20]}",
            )

    def _format_path(self, template: str, **values: Any) -> Path | None:
        try:
            return _dataset_path(self.root, template, **values)
        except (KeyError, ValueError, TypeError) as exc:
            self.error("PATH_TEMPLATE_INVALID", f"Cannot format path template {template!r}: {exc}")
            return None


def _numpy_stats_from_row(row: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    flat = {
        key.removeprefix("stats/"): np.atleast_1d(np.asarray(value))
        for key, value in row.items()
        if key.startswith("stats/") and value is not None
    }
    return unflatten_dict(flat)


def _to_arrow_value(value: Any) -> Any:
    return value.tolist() if isinstance(value, np.ndarray) else value


class DatasetRepairer:
    """Conservatively normalize an internally consistent subset into a new dataset."""

    def __init__(self, checker: IntegrityChecker, output: Path) -> None:
        self.checker = checker
        self.source = checker.root
        self.output = Path(output).expanduser().absolute()
        self.metadata_stage = self.output.with_name(f"{self.output.name}.pending-metadata")
        self.candidate = self.output.with_name(f"{self.output.name}.pending-candidate")
        self.info = checker.info
        self.old_ids = sorted(checker.data_episodes)
        self.mapping = {old: new for new, old in enumerate(self.old_ids)}
        self.metadata_by_old = {}
        self.new_ranges: dict[int, tuple[int, int]] = {}

    def _validate_preconditions(self, report: dict[str, Any]) -> None:
        if not self.checker.decode_videos or not self.checker.decode_images:
            # Repair always validates the physical media, irrespective of a prior fast check.
            checker = IntegrityChecker(self.source, decode_videos=True, decode_images=True)
            report = checker.run()
            self.checker = checker
        if self.output == self.source:
            raise ValueError(
                "--repair-output must differ from --dataset.root; in-place repair is forbidden"
            )
        if self.output.resolve().is_relative_to(self.source) or self.source.is_relative_to(
            self.output.resolve()
        ):
            raise ValueError("Repair output must be separate from the source dataset")
        work_paths = (self.output, self.metadata_stage, self.candidate)
        if any(path.exists() or path.is_symlink() for path in work_paths):
            raise FileExistsError(f"Refusing to overwrite existing path: {work_paths}")

        blocking = sorted(
            {
                issue["code"]
                for issue in report["issues"]
                if issue["severity"] == "error" and issue["code"] not in REPAIRABLE_ERROR_CODES
            }
        )
        if blocking:
            raise ValueError(
                f"Repair refused because damage is ambiguous or destructive: {blocking}"
            )

        self.info = self.checker.info
        self.old_ids = sorted(self.checker.data_episodes)
        self.mapping = {old: new for new, old in enumerate(self.old_ids)}
        self.metadata_by_old = {int(row["episode_index"]): row for row in self.checker.episode_rows}

        metadata_ids = [int(row["episode_index"]) for row in self.checker.episode_rows]
        if len(metadata_ids) != len(set(metadata_ids)):
            raise ValueError("Repair refused: duplicate episode metadata rows")
        if set(metadata_ids) != set(self.old_ids):
            raise ValueError(
                "Repair refused: data parquet and episode metadata describe different episodes; "
                f"data={self.old_ids}, metadata={sorted(metadata_ids)}"
            )
        if not self.old_ids:
            raise ValueError("Repair refused: no complete episodes were found")

        cursor = 0
        for old_id in self.old_ids:
            record = self.checker.data_episodes[old_id]
            row = self.metadata_by_old[old_id]
            if record.count != int(row["length"]):
                raise ValueError(
                    f"Repair refused: episode {old_id} has {record.count} data rows but "
                    f"metadata length {row['length']}"
                )
            if len(record.files) != 1:
                raise ValueError(
                    f"Repair refused: episode {old_id} spans {len(record.files)} data files"
                )
            frames = sorted(record.frame_indices)
            if frames != list(range(record.count)):
                raise ValueError(f"Repair refused: episode {old_id} frame_index is not contiguous")
            self.new_ranges[old_id] = (cursor, cursor + record.count)
            cursor += record.count

    def _rewrite_data(self) -> None:
        for source_path in sorted(self.source.joinpath("data").rglob("*.parquet")):
            table = pq.read_table(source_path)
            old_episode_ids = [int(value) for value in table["episode_index"].to_pylist()]
            frame_indices = [int(value) for value in table["frame_index"].to_pylist()]
            new_episode_ids = [self.mapping[old_id] for old_id in old_episode_ids]
            new_indices = [
                self.new_ranges[old_id][0] + frame
                for old_id, frame in zip(old_episode_ids, frame_indices, strict=True)
            ]

            ep_pos = table.schema.get_field_index("episode_index")
            index_pos = table.schema.get_field_index("index")
            table = table.set_column(
                ep_pos,
                "episode_index",
                pa.array(new_episode_ids, type=table.schema.field(ep_pos).type),
            )
            table = table.set_column(
                index_pos,
                "index",
                pa.array(new_indices, type=table.schema.field(index_pos).type),
            )
            destination = self.metadata_stage / source_path.relative_to(self.source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination)

    def _link_referenced_videos(self) -> None:
        """Expose source videos to stage 2 without duplicating large files when possible."""

        for source_path in sorted(self.checker.referenced_videos | self.checker.referenced_images):
            destination = self.metadata_stage / source_path.relative_to(self.source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                destination.hardlink_to(source_path)
            except OSError:
                shutil.copy2(source_path, destination)

    def _rewrite_episode_metadata_and_stats(self) -> None:
        all_episode_stats: list[dict[str, dict[str, np.ndarray]]] = []
        for source_path in sorted(self.source.joinpath("meta/episodes").rglob("*.parquet")):
            source_table = pq.read_table(source_path)
            repaired_rows: list[dict[str, Any]] = []
            for row in source_table.to_pylist():
                old_id = int(row["episode_index"])
                if old_id not in self.mapping:
                    continue
                new_id = self.mapping[old_id]
                start, end = self.new_ranges[old_id]
                row["episode_index"] = new_id
                row["dataset_from_index"] = start
                row["dataset_to_index"] = end

                episode_stats = _numpy_stats_from_row(row)
                length = end - start
                episode_stats["episode_index"] = get_feature_stats(
                    np.full((length, 1), new_id, dtype=np.int64), axis=0, keepdims=False
                )
                episode_stats["index"] = get_feature_stats(
                    np.arange(start, end, dtype=np.int64).reshape(-1, 1), axis=0, keepdims=False
                )
                for stat_key, value in flatten_dict({"stats": episode_stats}).items():
                    if stat_key in row:
                        row[stat_key] = _to_arrow_value(value)
                all_episode_stats.append(episode_stats)
                repaired_rows.append(row)

            destination = self.metadata_stage / source_path.relative_to(self.source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            repaired_table = pa.Table.from_pylist(repaired_rows, schema=source_table.schema)
            pq.write_table(repaired_table, destination)

        write_stats(aggregate_stats(all_episode_stats), self.metadata_stage)

    def _write_info_and_tasks(self) -> dict[str, Any]:
        for feedback_metadata in self.source.joinpath("meta").glob("*.json"):
            if feedback_metadata.name in ("info.json", "stats.json"):
                continue
            destination = self.metadata_stage / "meta" / feedback_metadata.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(feedback_metadata, destination)
        for old_id, new_id in self.mapping.items():
            source = self.source / "meta/safety" / f"episode_{old_id:06d}.jsonl"
            if source.exists():
                destination = self.metadata_stage / "meta/safety" / f"episode_{new_id:06d}.jsonl"
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open() as reader, destination.open("w") as writer:
                    for line in reader:
                        row = json.loads(line)
                        row["episode_index"] = new_id
                        # All episode frames are retained; local indices and times stay unchanged.
                        writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        tasks_source = self.source / "meta/tasks.parquet"
        tasks_destination = self.metadata_stage / "meta/tasks.parquet"
        tasks_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tasks_source, tasks_destination)

        repaired_info = json.loads(json.dumps(self.info))
        repaired_info["total_episodes"] = len(self.old_ids)
        repaired_info["total_frames"] = sum(
            self.checker.data_episodes[old].count for old in self.old_ids
        )
        repaired_info["splits"] = {"train": f"0:{len(self.old_ids)}"}
        (self.metadata_stage / "meta/info.json").write_text(
            json.dumps(repaired_info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest = {
            "source": str(self.source),
            "output": str(self.output),
            "episode_mapping": {str(old): new for old, new in self.mapping.items()},
            "kept_episodes": len(self.old_ids),
            "kept_frames": repaired_info["total_frames"],
            "policy": "data parquet and matching episode metadata are authoritative",
            "workflow": [
                "normalize metadata and global indices",
                "compact referenced video ranges",
                "fully decode and validate the final dataset",
            ],
            "dropped": "unreferenced videos and unfinished raw images are not copied",
        }
        return manifest

    def run(
        self, source_report: dict[str, Any], *, timestamp_tolerance_s: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._validate_preconditions(source_report)
        try:
            self.metadata_stage.mkdir(parents=True)
            self._rewrite_data()
            self._rewrite_episode_metadata_and_stats()
            self._link_referenced_videos()
            manifest = self._write_info_and_tasks()

            video_report = repair_video_gaps(self.metadata_stage, self.candidate)
            video_report["output"] = str(self.output)
            manifest["video_repair"] = video_report
            (self.candidate / "repair_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            repaired_checker = IntegrityChecker(
                self.candidate,
                decode_videos=True,
                decode_images=True,
                timestamp_tolerance_s=timestamp_tolerance_s,
            )
            repaired_report = repaired_checker.run()
            if not repaired_report["valid"]:
                raise RuntimeError(
                    f"Repaired candidate dataset failed validation at {self.candidate}: "
                    f"{repaired_report['errors']} errors"
                )
            if self.output.exists() or self.output.is_symlink():
                raise FileExistsError(self.output)
            self.candidate.rename(self.output)
            repaired_report["dataset_root"] = str(self.output)
            shutil.rmtree(self.metadata_stage)
            return manifest, repaired_report
        except BaseException as exc:
            raise RuntimeError(
                f"Repair unfinished; source unchanged; partial files retained at "
                f"{self.metadata_stage} and {self.candidate}: {exc}"
            ) from exc


def repair_video_gaps(source: Path, output: Path) -> dict[str, int | str]:
    """Compact referenced episode ranges and return a machine-readable summary."""

    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing path: {output}")
    if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        raise ValueError("Output must be separate from the source dataset")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Dataset must not contain symbolic links")
    if not (source / "meta/info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {source}")

    info = json.loads((source / "meta/info.json").read_text())
    fps = float(info["fps"])
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    video_path_template = info.get("video_path")
    if video_keys and not video_path_template:
        raise ValueError("info.json declares video features but has no video_path template")

    episode_tables: list[tuple[Path, pa.Schema, list[dict]]] = []
    rows_by_episode: dict[int, dict] = {}
    for path in sorted((source / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        rows = table.to_pylist()
        episode_tables.append((path.relative_to(source), table.schema, rows))
        for row in rows:
            episode_index = int(row["episode_index"])
            if episode_index in rows_by_episode:
                raise ValueError(f"Duplicate episode_index {episode_index}")
            rows_by_episode[episode_index] = row

    if len(rows_by_episode) != int(info["total_episodes"]):
        raise ValueError(
            f"Episode metadata count {len(rows_by_episode)} "
            f"!= info total_episodes {info['total_episodes']}"
        )
    if sum(int(row["length"]) for row in rows_by_episode.values()) != int(info["total_frames"]):
        raise ValueError("Episode lengths do not add up to info total_frames")

    groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows_by_episode.values():
        for key in video_keys:
            prefix = f"videos/{key}"
            groups[
                (key, int(row[f"{prefix}/chunk_index"]), int(row[f"{prefix}/file_index"]))
            ].append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{output.name}.pending-", dir=output.parent))
    try:
        (staging / "meta").mkdir(parents=True)
        shutil.copy2(source / "meta/info.json", staging / "meta/info.json")
        shutil.copy2(source / "meta/stats.json", staging / "meta/stats.json")
        shutil.copy2(source / "meta/tasks.parquet", staging / "meta/tasks.parquet")
        shutil.copytree(source / "data", staging / "data")
        if (source / "meta/safety").exists():
            shutil.copytree(source / "meta/safety", staging / "meta/safety")
        if (source / "meta/recovery").exists():
            shutil.copytree(source / "meta/recovery", staging / "meta/recovery")
        for path in source.joinpath("meta").glob("*.json"):
            if path.name not in ("info.json", "stats.json"):
                shutil.copy2(path, staging / "meta" / path.name)
        # Embedded images travel with Parquet; copy only referenced external images.
        image_keys = [
            key for key, feature in info["features"].items() if feature.get("dtype") == "image"
        ]
        for path in source.joinpath("data").rglob("*.parquet"):
            if not image_keys:
                break
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=image_keys):
                for row in batch.to_pylist():
                    for reference in row.values():
                        if reference.get("bytes") is None:
                            image = _dataset_path(source, reference["path"])
                            target = staging / image.relative_to(source)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            if not target.exists():
                                shutil.copy2(image, target)

        repaired_files = 0
        copied_files = 0
        removed_frames = 0

        for (key, chunk_index, file_index), rows in sorted(groups.items()):
            rows.sort(key=lambda row: float(row[f"videos/{key}/from_timestamp"]))
            source_path = _dataset_path(
                source,
                video_path_template,
                video_key=key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            output_path = _dataset_path(
                staging,
                video_path_template,
                video_key=key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            source_ranges: list[tuple[int, int]] = []
            target_frame = 0
            previous_end = 0
            for row in rows:
                prefix = f"videos/{key}"
                source_start = round(float(row[f"{prefix}/from_timestamp"]) * fps)
                source_end = source_start + int(row["length"])
                if source_start < 0 or source_start < previous_end:
                    raise ValueError(
                        f"{source_path}: invalid or overlapping source range "
                        f"[{source_start}, {source_end}) after frame {previous_end}"
                    )
                source_ranges.append((source_start, source_end))
                previous_end = source_end
                row[f"{prefix}/from_timestamp"] = target_frame / fps
                target_frame += int(row["length"])
                row[f"{prefix}/to_timestamp"] = target_frame / fps

            actual_frames = video_frame_count(source_path)
            if source_ranges[-1][1] > actual_frames:
                raise ValueError(
                    f"{source_path}: required frame {source_ranges[-1][1]}, "
                    f"but video has {actual_frames} frames"
                )

            contiguous = source_ranges == [
                (
                    sum(end - start for start, end in source_ranges[:i]),
                    sum(end - start for start, end in source_ranges[: i + 1]),
                )
                for i in range(len(source_ranges))
            ]
            if contiguous and actual_frames == target_frame:
                shutil.copy2(source_path, output_path)
                copied_files += 1
                print(f"COPY   {key} file-{file_index:03d}: {actual_frames} frames")
            else:
                repack_video(
                    source_path,
                    output_path,
                    source_ranges,
                    fps,
                    info["features"][key].get("info") or {},
                )
                clean_frames = video_frame_count(output_path)
                if clean_frames != target_frame:
                    raise ValueError(
                        f"{output_path}: encoded {clean_frames} frames, expected {target_frame}"
                    )
                repaired_files += 1
                removed_frames += actual_frames - target_frame
                print(
                    f"REPACK {key} file-{file_index:03d}: {actual_frames} -> {clean_frames} frames"
                )

        for relative_path, schema, rows in episode_tables:
            path = staging / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)

        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staging.rename(output)
        return {
            "output": str(output),
            "episodes": len(rows_by_episode),
            "frames": int(info["total_frames"]),
            "repacked_files": repaired_files,
            "copied_files": copied_files,
            "removed_video_frames": removed_frames,
        }
    except BaseException as exc:
        raise RuntimeError(
            f"Video repair unfinished; source unchanged; files at {staging}: {exc}"
        ) from exc
