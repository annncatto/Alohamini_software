# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local LeRobot v3 image datasets; no LeRobot installation or Hub connection.

Metadata follows LeRobotDatasetMetadata and create_empty_dataset_info.
RunningQuantileStats is copied from the source dataset statistics implementation.
Images use the standard datasets.Image Arrow struct (embedded lossless PNG).
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from alohamini.datasets.images import image_bytes, image_png, image_rgb
from alohamini.datasets.native import StateSelection, _write_json
from alohamini.datasets.tools import IntegrityChecker, _read_lock

DEFAULT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    **{
        key: {"dtype": "int64", "shape": [1], "names": None}
        for key in ("frame_index", "episode_index", "index", "task_index")
    },
}
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
EPISODE_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DATA_FILE_BYTES = 100 * 1024**2


class RunningQuantileStats:
    """
    Maintains running statistics for batches of vectors, including mean,
    standard deviation, min, max, and approximate quantiles.

    Statistics are computed per feature dimension and updated incrementally
    as new batches are observed. Quantiles are estimated using histograms,
    which adapt dynamically if the observed data range expands.
    """

    def __init__(self, quantile_list: list[float] | None = None, num_quantile_bins: int = 5000):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = num_quantile_bins

        self._quantile_list = quantile_list
        if self._quantile_list is None:
            self._quantile_list = DEFAULT_QUANTILES
        self._quantile_keys = [f"q{int(q * 100):02d}" for q in self._quantile_list]

    def update(self, batch: np.ndarray) -> None:
        """Update the running statistics with a batch of vectors.

        Args:
            batch: An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        # Promote integer and low-precision inputs before computing squared statistics.
        batch = batch.astype(np.result_type(batch.dtype, np.float32), copy=False)
        num_elements, vector_length = batch.shape

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError(
                    "The length of new vectors does not match the initialized vector length."
                )

            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )

        self._update_histograms(batch)

    def get_statistics(self) -> dict[str, np.ndarray]:
        """Compute and return the statistics of the vectors processed so far.

        Returns:
            Dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2

        stddev = np.sqrt(np.maximum(0, variance))

        stats = {
            "min": self._min.copy(),
            "max": self._max.copy(),
            "mean": self._mean.copy(),
            "std": stddev,
            "count": np.array([self._count]),
        }

        quantile_results = self._compute_quantiles()
        for i, q in enumerate(self._quantile_keys):
            stats[q] = quantile_results[i]

        return stats

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            old_hist = self._histograms[i]

            # Create new edges with small padding to ensure range coverage
            padding = (self._max[i] - self._min[i]) * 1e-10
            new_edges = np.linspace(
                self._min[i] - padding, self._max[i] + padding, self._num_quantile_bins + 1
            )

            # Redistribute existing histogram counts to new bins
            # We need to map each old bin center to the new bins
            old_centers = (old_edges[:-1] + old_edges[1:]) / 2
            new_hist = np.zeros(self._num_quantile_bins)

            for old_center, count in zip(old_centers, old_hist, strict=False):
                if count > 0:
                    # Find which new bin this old center belongs to
                    bin_idx = np.searchsorted(new_edges, old_center) - 1
                    bin_idx = max(0, min(bin_idx, self._num_quantile_bins - 1))
                    new_hist[bin_idx] += count

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self) -> list[np.ndarray]:
        """Compute quantiles based on histograms."""
        results = []
        for q in self._quantile_list:
            target_count = q * self._count
            q_values = []

            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                q_value = self._compute_single_quantile(hist, edges, target_count)
                q_values.append(q_value)

            results.append(np.array(q_values))
        return results

    def _compute_single_quantile(
        self, hist: np.ndarray, edges: np.ndarray, target_count: float
    ) -> float:
        """Compute a single quantile value from histogram and bin edges."""
        cumsum = np.cumsum(hist)
        idx = np.searchsorted(cumsum, target_count)

        if idx == 0:
            return edges[0]
        if idx >= len(cumsum):
            return edges[-1]

        # If not edge case, interpolate within the bin
        count_before = cumsum[idx - 1]
        count_in_bin = cumsum[idx] - count_before

        # If no samples in this bin, use the bin edge
        if count_in_bin == 0:
            return edges[idx]

        # Linear interpolation within the bin
        fraction = (target_count - count_before) / count_in_bin
        return edges[idx] + fraction * (edges[idx + 1] - edges[idx])


class _Stats:
    def __init__(self, features):
        self.features = features
        self.trackers = {key: RunningQuantileStats() for key in features}
        self.frames = 0

    def update(self, rows):
        self.frames += len(rows)
        for key, feature in self.features.items():
            tracker = self.trackers[key]
            if feature["dtype"] == "image":
                for row in rows:
                    with Image.open(io.BytesIO(row[key]["bytes"])) as image:
                        array = np.asarray(image)
                        stride = max(1, max(array.shape[:2]) // 150)
                        tracker.update(array[::stride, ::stride].reshape(-1, 3) / 255.0)
            else:
                values = np.asarray([row[key] for row in rows], dtype=np.float64)
                tracker.update(values.reshape(len(rows), -1))

    def result(self):
        result = {}
        for key, tracker in self.trackers.items():
            if tracker._count == 1:
                value = tracker._mean.copy()
                stats = {
                    name: value.copy() for name in ("min", "max", "mean", *tracker._quantile_keys)
                }
                stats.update(std=np.zeros_like(value), count=np.array([1]))
            else:
                stats = tracker.get_statistics()
            if self.features[key]["dtype"] == "image":
                stats = {
                    name: value.reshape(3, 1, 1) if name != "count" else np.array([self.frames])
                    for name, value in stats.items()
                }
            result[key] = stats
        return result


def _features(info, selection, shapes):
    features = deepcopy(info["features"])
    features["observation.source_state"] = deepcopy(features["observation.state"])
    features["observation.state"] = deepcopy(selection.feature)
    features.update(
        {
            f"observation.images.{name}": {
                "dtype": "image",
                "shape": shape,
                "names": ["height", "width", "channels"],
            }
            for name, shape in shapes.items()
        }
    )
    features.update(deepcopy(DEFAULT_FEATURES))
    return features


def _arrow_schema(features):
    fields = []
    hf_features = {}
    for key, feature in features.items():
        if feature["dtype"] == "image":
            dtype = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
            hf_features[key] = {"_type": "Image"}
        else:
            dtype = pa.type_for_alias(feature["dtype"])
            hf_feature = {"dtype": feature["dtype"], "_type": "Value"}
            if feature["shape"] != [1]:
                dtype = pa.list_(dtype, feature["shape"][0])
                hf_feature = {"feature": hf_feature, "length": feature["shape"][0], "_type": "List"}
            hf_features[key] = hf_feature
        fields.append(pa.field(key, dtype))
    # Standard Hugging Face Arrow feature metadata lets generic datasets readers
    # recognize embedded PNGs without a custom decoder or absolute file paths.
    return pa.schema(
        fields,
        metadata={
            b"huggingface": json.dumps(
                {
                    "info": {"features": hf_features},
                }
            ).encode()
        },
    )


def _selected_rows(episode, info, selection):
    for batch in pq.ParquetFile(episode / "frames.parquet").iter_batches(batch_size=8):
        for row in batch.to_pylist():
            try:
                selected = selection.frame(row)
            except ValueError as exc:
                raise ValueError(f"{episode.name} frame {row['frame_index']}: {exc}") from exc
            result = {key: row[key] for key in info["features"]}
            result["observation.source_state"] = row["observation.state"]
            result["observation.state"] = (
                selected.tolist() if len(selected) > 1 else float(selected[0])
            )
            result.update({key: row[key] for key in DEFAULT_FEATURES})
            yield row, result


def _write_dataset(source, output, checker, selection):
    features = _features(checker.info, selection, checker.shapes)
    schema = _arrow_schema(features)
    meta = output / "meta"
    meta.mkdir()
    (meta / "safety").mkdir()
    global_stats = _Stats(features)
    data_writer = metadata_writer = None
    file_number = file_bytes = offset = 0
    try:
        for index in range(checker.num_episodes):
            episode = source / "episodes" / f"episode_{index:06d}"
            if data_writer is not None and file_bytes >= DATA_FILE_BYTES:
                data_writer.close()
                data_writer = None
                file_number += 1
                file_bytes = 0
            chunk, file = divmod(file_number, 1000)
            if data_writer is None:
                path = output / DATA_PATH.format(chunk_index=chunk, file_index=file)
                path.parent.mkdir(parents=True, exist_ok=True)
                data_writer = pq.ParquetWriter(path, schema, compression="zstd")
            episode_stats, rows = _Stats(features), []
            length = 0
            for original, row in _selected_rows(episode, checker.info, selection):
                for camera in checker.cameras:
                    key = f"observation.images.{camera}"
                    row[key] = {"bytes": image_png(episode, camera, original[key]), "path": None}
                rows.append(row)
                length += 1
                if len(rows) == 8:
                    table = pa.Table.from_pylist(rows, schema=schema)
                    data_writer.write_table(table)
                    file_bytes += table.nbytes
                    episode_stats.update(rows)
                    global_stats.update(rows)
                    rows.clear()
            if rows:
                table = pa.Table.from_pylist(rows, schema=schema)
                data_writer.write_table(table)
                file_bytes += table.nbytes
                episode_stats.update(rows)
                global_stats.update(rows)
            episode_row = {
                "episode_index": index,
                "tasks": [checker.info["task"]],
                "length": length,
                "data/chunk_index": chunk,
                "data/file_index": file,
                "dataset_from_index": offset,
                "dataset_to_index": offset + length,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                **{
                    f"stats/{key}/{stat}": value.tolist()
                    for key, stats in episode_stats.result().items()
                    for stat, value in stats.items()
                },
            }
            table = pa.Table.from_pylist([episode_row])
            if metadata_writer is None:
                path = output / EPISODE_PATH.format(chunk_index=0, file_index=0)
                path.parent.mkdir(parents=True, exist_ok=True)
                metadata_writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            metadata_writer.write_table(table)
            shutil.copyfile(
                episode / "safety.jsonl", meta / "safety" / f"episode_{index:06d}.jsonl"
            )
            offset += length
    finally:
        if data_writer is not None:
            data_writer.close()
        if metadata_writer is not None:
            metadata_writer.close()
    pd.DataFrame({"task_index": [0]}, index=[checker.info["task"]]).to_parquet(
        meta / "tasks.parquet"
    )
    _write_json(
        meta / "info.json",
        {
            "codebase_version": "v3.0",
            "robot_type": checker.info["robot_metadata"]["robot_model"],
            "total_episodes": checker.num_episodes,
            "total_frames": offset,
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": checker.info["fps"],
            "splits": {"train": f"0:{checker.num_episodes}"},
            "data_path": DATA_PATH,
            "video_path": None,
            "features": features,
        },
    )
    _write_json(meta / "stats.json", global_stats.result())
    _write_json(
        meta / "alohamini.json",
        {
            "version": 1,
            "source": str(source),
            "source_info": checker.info,
            "state_groups": selection.groups,
            "state_units": selection.units,
            "action": "unchanged submitted Host targets, not measured motion",
            "joint_velocity": (
                "signed feedback ticks/s scaled to the recorded Host position coordinate per second"
            ),
            "joint_current": "reported current magnitude in amperes; not joint torque",
            "source_state": "observation.source_state retains the original observation.state",
            "feedback_statistics": (
                "extra motor fields retain zero placeholders; apply their validity masks before use"
            ),
            "safety_path": "meta/safety/episode_{episode_index:06d}.jsonl",
            "training_review": checker.report()["training_review"],
        },
    )


def _validate_export(source, output, checker, selection):
    """Read back every written row and embedded image before publishing."""
    info = json.loads((output / "meta/info.json").read_text())
    expected_features = _features(checker.info, selection, checker.shapes)
    if info["features"] != expected_features or info["total_frames"] != checker.total_frames:
        raise ValueError("Export feature metadata or frame count mismatch")
    offset = 0
    active_path, written = None, iter(())
    for batch in pq.ParquetFile(
        output / EPISODE_PATH.format(chunk_index=0, file_index=0)
    ).iter_batches():
        for metadata in batch.to_pylist():
            index = metadata["episode_index"]
            episode = source / "episodes" / f"episode_{index:06d}"
            path = output / DATA_PATH.format(
                chunk_index=metadata["data/chunk_index"], file_index=metadata["data/file_index"]
            )
            if path != active_path:
                if next(written, None) is not None:
                    raise ValueError("Unreferenced rows in exported data shard")
                active_path = path
                written = (
                    row
                    for part in pq.ParquetFile(path).iter_batches(batch_size=8)
                    for row in part.to_pylist()
                )
            count = 0
            for original, expected in _selected_rows(episode, checker.info, selection):
                row = next(written, None)
                if row is None:
                    raise ValueError("Missing row in exported data shard")
                for key, value in expected.items():
                    dtype = expected_features[key]["dtype"]
                    if not np.array_equal(
                        np.asarray(row[key], dtype=dtype), np.asarray(value, dtype=dtype)
                    ):
                        raise ValueError(f"Exported numeric field changed: {key}")
                for camera in checker.cameras:
                    key = f"observation.images.{camera}"
                    if row[key]["path"] is not None:
                        raise ValueError(f"Exported image has an external path: {key}")
                    if isinstance(original[key], str):
                        identical = row[key]["bytes"] == image_bytes(episode, camera, original[key])
                    else:
                        with Image.open(io.BytesIO(row[key]["bytes"])) as image:
                            identical = (
                                image.format == "PNG"
                                and image.mode == "RGB"
                                and np.array_equal(
                                    np.asarray(image), image_rgb(episode, camera, original[key])
                                )
                            )
                    if not identical:
                        raise ValueError(f"Exported image changed: {key}")
                count += 1
            if (
                metadata["length"] != count
                or metadata["dataset_from_index"] != offset
                or metadata["dataset_to_index"] != offset + count
            ):
                raise ValueError("Exported episode boundary mismatch")
            if (output / "meta/safety" / f"episode_{index:06d}.jsonl").read_bytes() != (
                episode / "safety.jsonl"
            ).read_bytes():
                raise ValueError("Exported safety log changed")
            offset += count
    if offset != checker.total_frames:
        raise ValueError("Export is missing recorded frames")
    if next(written, None) is not None:
        raise ValueError("Unreferenced rows at the end of exported data")


def export_lerobot(root, output, *, state=StateSelection.DEFAULT):
    source = Path(root).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        raise ValueError("Output must be separate from the source dataset")
    with _read_lock(source):
        checker = IntegrityChecker(source, decode_images=True)
        checker._run_unlocked()
        report = checker.report()
        if not report["valid"] or not checker.total_frames:
            raise ValueError(f"Source must be complete before LeRobot export: {report}")
        selection = StateSelection(checker.info, state)
        # Validate selected feedback before creating output. Missing optional
        # feedback remains stored, but cannot silently become training input.
        for index in range(checker.num_episodes):
            for _ in _selected_rows(
                source / "episodes" / f"episode_{index:06d}", checker.info, selection
            ):
                pass
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f"{output.name}.pending-", dir=output.parent))
        try:
            _write_dataset(source, stage, checker, selection)
            _validate_export(source, stage, checker, selection)
            if output.exists():
                raise FileExistsError(output)
            stage.rename(output)
        except BaseException as exc:
            raise RuntimeError(
                f"LeRobot export unfinished: {exc}; source unchanged, files retained at {stage}"
            ) from exc
        report.update(
            dataset_root=str(output), format="lerobot-v3", state_names=selection.feature["names"]
        )
        return report
