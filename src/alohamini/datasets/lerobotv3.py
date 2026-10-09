# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LeRobot v3 storage; no LeRobot installation or Hub connection.

Metadata follows LeRobotDatasetMetadata and create_empty_dataset_info.
RunningQuantileStats is copied from the source dataset statistics implementation.
Images use the standard datasets.Image Arrow struct (embedded JPEG or PNG).
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import time
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from alohamini.datasets.image_statistics import ImageStatistics, sample_indices, sample_video_images
from alohamini.datasets.images import (
    VIDEO_FORMAT,
    decode_host_image,
    image_bytes,
    image_rgb,
)
from alohamini.datasets.record import StateSelection, _write_json, dataset_schema
from alohamini.datasets.statistics import ExactQuantileStats, RunningQuantileStats
from alohamini.datasets.tools import IntegrityChecker, _read_lock

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    **{
        key: {"dtype": "int64", "shape": [1], "names": None}
        for key in ("frame_index", "episode_index", "index", "task_index")
    },
}
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
EPISODE_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DATA_FILE_BYTES = 100 * 1024**2


class _Stats:
    def __init__(self, features, *, recording=False):
        self.features = features
        self.image_statistics = ImageStatistics() if recording else None
        self.trackers = {
            key: (
                (None if recording else RunningQuantileStats())
                if f["dtype"] in ("image", "video")
                else ExactQuantileStats()
            )
            for key, f in features.items()
        }
        self.frames = 0

    def update(self, rows, decoded_images=None):
        self.frames += len(rows)
        for key, feature in self.features.items():
            tracker = self.trackers[key]
            if feature["dtype"] in ("image", "video"):
                if decoded_images is None:
                    arrays = []
                    for row in rows:
                        with Image.open(io.BytesIO(row[key]["bytes"])) as image:
                            arrays.append(np.asarray(image))
                else:
                    arrays = (
                        decoded_images.get(key, [])
                        if self.image_statistics is not None
                        else decoded_images[key]
                    )
                for array in arrays:
                    if self.image_statistics is not None:
                        self.image_statistics.update(key, array, source="decoded_video")
                        continue
                    stride = max(1, max(array.shape[:2]) // 150)
                    tracker.update(array[::stride, ::stride].reshape(-1, 3) / 255.0)
            else:
                values = np.asarray([row[key] for row in rows], dtype=np.float64)
                tracker.update(values.reshape(len(rows), -1))

    def result(self):
        result = {}
        for key, tracker in self.trackers.items():
            if tracker is None:
                continue
            if tracker._count == 1:
                value = tracker._mean.copy()
                stats = {
                    name: value.copy() for name in ("min", "max", "mean", *tracker._quantile_keys)
                }
                stats.update(std=np.zeros_like(value), count=np.array([1]))
            else:
                stats = tracker.get_statistics()
            if self.features[key]["dtype"] in ("image", "video"):
                stats = {
                    name: value.reshape(3, 1, 1) if name != "count" else np.array([self.frames])
                    for name, value in stats.items()
                }
            result[key] = stats
        if self.image_statistics is not None:
            result.update(self.image_statistics.result())
        return result


def _features(info, selection, shapes):
    features = deepcopy(info["features"])
    if selection.feature != features["observation.state"]:
        features["observation.source_state"] = deepcopy(features["observation.state"])
    features["observation.state"] = deepcopy(selection.feature)
    features.update(
        {
            f"observation.images.{name}": {
                "dtype": "video" if info["image_format"] == VIDEO_FORMAT else "image",
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
        if feature["dtype"] == "video":
            continue  # LeRobot locates video frames through episode metadata.
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
    # recognize embedded images without a custom decoder or absolute file paths.
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


def _selected_rows(episode, info, selection, *, batch_size=8):
    for batch in pq.ParquetFile(episode / "frames.parquet").iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            try:
                selected = selection.frame(row)
            except ValueError as exc:
                raise ValueError(f"{episode.name} frame {row['frame_index']}: {exc}") from exc
            result = {key: row[key] for key in info["features"]}
            if selection.feature != info["features"]["observation.state"]:
                result["observation.source_state"] = row["observation.state"]
            result["observation.state"] = (
                selected.tolist() if len(selected) > 1 else float(selected[0])
            )
            result.update({key: row[key] for key in DEFAULT_FEATURES})
            yield row, result


def _export_image(episode, camera, reference):
    """Copy standard bytes; only legacy wire JPEGs need color conversion."""
    encoded = image_bytes(episode, camera, reference)
    legacy = isinstance(reference, dict) and reference.get("member", "").endswith(".jpg")
    if legacy:
        rgb = decode_host_image(encoded)
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG", compress_level=1)
        encoded = buffer.getvalue()
    else:
        with Image.open(io.BytesIO(encoded)) as image:
            rgb = np.asarray(image)
    return encoded, rgb


def _sample_video_frames(container, indices):
    """Decode once in time order; convert only selected frames to RGB."""
    stream = container.streams.video[0]
    if not stream.average_rate or not stream.time_base:
        raise ValueError("Video must have a fixed frame rate and timestamps")
    targets = iter(sorted(indices))
    target = next(targets, None)
    if target is None:
        return
    for frame in container.decode(stream):
        if frame.pts is None:
            raise ValueError("Video frame has no timestamp")
        position = frame.pts * frame.time_base * stream.average_rate
        index = round(position)
        if abs(position - index) > 0.01:
            raise ValueError("Video frame timestamp is off the dataset timeline")
        if index == target:
            yield frame.to_ndarray(format="rgb24")
            target = next(targets, None)
            if target is None:
                return
        elif index > target:
            break
    raise ValueError(f"Missing video frame {target}")


def _write_dataset(
    source, output, checker, selection, *, start_episode=0, start_offset=0, stats=None
):
    features = _features(checker.info, selection, checker.shapes)
    schema = _arrow_schema(features)
    meta = output / "meta"
    meta.mkdir()
    (meta / "safety").mkdir()
    recording = getattr(checker, "recording", False)
    global_stats = stats if stats is not None else _Stats(features, recording=recording)
    data_writer = metadata_writer = None
    file_number, file_bytes, offset = start_episode, 0, start_offset
    meta_chunk, meta_file = divmod(start_episode, 1000)
    videos = checker.info["image_format"] == VIDEO_FORMAT
    image_sample_limit = getattr(checker, "image_sample_limit", None)
    timings = getattr(checker, "save_timings", None)
    if timings is not None:
        timings.setdefault("image_decode", 0.0)

    def update_statistics(rows, decoded):
        started = time.perf_counter()
        episode_stats.update(rows, decoded)
        global_stats.update(rows, {} if recording else decoded)
        if timings is not None:
            timings["statistics_update"] = (
                timings.get("statistics_update", 0.0) + time.perf_counter() - started
            )

    # Sampled MP4 exports contain numeric rows, not embedded full-frame payloads.
    batch_size = 512 if videos and (image_sample_limit or recording) else 8
    try:
        for index in range(start_episode, checker.num_episodes):
            episode = getattr(checker, "episode_paths", {}).get(
                index, source / "episodes" / f"episode_{index:06d}"
            )
            video_metadata = {}
            video_paths = {
                c: getattr(checker, "video_paths", {})
                .get(index, {})
                .get(c, episode / "videos" / f"{c}.mp4")
                for c in checker.cameras
            }
            summary = json.loads((episode / "episode.json").read_text())
            episode_length = summary["length"]
            image_stats_path = episode / "image_stats.json"
            image_summary = (
                ImageStatistics(json.loads(image_stats_path.read_text()))
                if recording and image_stats_path.exists()
                else None
            )
            if image_summary is not None and set(image_summary.histograms) != {
                f"observation.images.{c}" for c in checker.cameras
            }:
                raise ValueError("Recording image statistics do not match cameras")
            image_indices = (
                set(
                    np.linspace(
                        0, episode_length - 1, min(image_sample_limit, episode_length), dtype=int
                    )
                )
                if image_sample_limit
                else None
            )
            if recording and image_summary is None:
                image_indices = set(sample_indices(episode_length))
            if videos:
                for camera in checker.cameras:
                    key = f"observation.images.{camera}"
                    video_chunk, video_file = divmod(index, 1000)
                    target = output / VIDEO_PATH.format(
                        video_key=key, chunk_index=video_chunk, file_index=video_file
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(video_paths[camera], target)
                    video_metadata.update(
                        {
                            f"videos/{key}/chunk_index": video_chunk,
                            f"videos/{key}/file_index": video_file,
                            f"videos/{key}/from_timestamp": 0.0,
                            f"videos/{key}/to_timestamp": episode_length / checker.info["fps"],
                        }
                    )
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
            episode_stats, rows = _Stats(features, recording=recording), []
            if image_summary is not None:
                episode_stats.image_statistics = image_summary
            decoded = {f"observation.images.{c}": [] for c in checker.cameras}
            length = 0
            with ExitStack() as stack:
                if videos and image_summary is None:
                    import av

                    streams = {}
                    for camera in checker.cameras:
                        container = stack.enter_context(av.open(str(video_paths[camera])))
                        streams[camera] = (
                            sample_video_images(container, image_indices)
                            if recording
                            else _sample_video_frames(container, image_indices)
                            if image_indices is not None
                            else iter(container.decode(video=0))
                        )
                for original, row in _selected_rows(
                    episode, checker.info, selection, batch_size=batch_size
                ):
                    for camera in () if image_summary is not None else checker.cameras:
                        key = f"observation.images.{camera}"
                        decode_started = time.perf_counter()
                        if videos:
                            if image_indices is not None:
                                if length not in image_indices:
                                    continue
                                rgb = next(streams[camera])
                            else:
                                rgb = next(streams[camera]).to_ndarray(format="rgb24")
                        else:
                            encoded, rgb = _export_image(episode, camera, original[key])
                            row[key] = {"bytes": encoded, "path": None}
                        decoded[key].append(rgb)
                        if timings is not None:
                            timings["image_decode"] = (
                                timings.get("image_decode", 0.0)
                                + time.perf_counter()
                                - decode_started
                            )
                    rows.append(row)
                    length += 1
                    if len(rows) == batch_size:
                        table = pa.Table.from_pylist(rows, schema=schema)
                        data_writer.write_table(table)
                        file_bytes += table.nbytes
                        update_statistics(rows, decoded)
                        rows.clear()
                        for images in decoded.values():
                            images.clear()
            if rows:
                table = pa.Table.from_pylist(rows, schema=schema)
                data_writer.write_table(table)
                file_bytes += table.nbytes
                update_statistics(rows, decoded)
            if recording:
                global_stats.image_statistics.merge(episode_stats.image_statistics)
            stats_started = time.perf_counter()
            episode_result = episode_stats.result()
            if timings is not None:
                timings["episode_statistics"] = time.perf_counter() - stats_started
            episode_row = {
                "episode_index": index,
                "tasks": [summary.get("task", checker.info["task"])],
                "length": length,
                **(
                    {"image_statistics": json.dumps(episode_stats.image_statistics.payload())}
                    if recording
                    else {}
                ),
                "data/chunk_index": chunk,
                "data/file_index": file,
                "dataset_from_index": offset,
                "dataset_to_index": offset + length,
                "meta/episodes/chunk_index": meta_chunk,
                "meta/episodes/file_index": meta_file,
                **video_metadata,
                **{
                    f"stats/{key}/{stat}": value.tolist()
                    for key, stats in episode_result.items()
                    for stat, value in stats.items()
                },
            }
            table = pa.Table.from_pylist([episode_row])
            if metadata_writer is None:
                path = output / EPISODE_PATH.format(chunk_index=meta_chunk, file_index=meta_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                metadata_writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            metadata_writer.write_table(table)
            shutil.copyfile(
                episode / "safety.jsonl", meta / "safety" / f"episode_{index:06d}.jsonl"
            )
            offset += length
            if hasattr(checker, "export_progress"):
                checker.export_progress(index + 1)
            elif not getattr(checker, "recording", False):
                print(
                    f"[EXPORT] Episodes {index + 1}/{checker.num_episodes}; "
                    f"frames {offset}/{checker.total_frames}",
                    flush=True,
                )
    finally:
        if data_writer is not None:
            data_writer.close()
        if metadata_writer is not None:
            metadata_writer.close()
    tasks = checker.info.get("tasks", [checker.info["task"]])
    pd.DataFrame({"task_index": list(range(len(tasks)))}, index=tasks).to_parquet(
        meta / "tasks.parquet"
    )
    _write_json(
        meta / "info.json",
        {
            "codebase_version": "v3.0",
            "robot_type": checker.info["robot_metadata"]["robot_model"],
            "total_episodes": checker.num_episodes,
            "total_frames": offset,
            "total_tasks": len(tasks),
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": checker.info["fps"],
            "splits": {"train": f"0:{checker.num_episodes}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if videos else None,
            "features": features,
        },
    )
    stats_started = time.perf_counter()
    global_result = global_stats.result()
    if timings is not None:
        timings["global_statistics"] = time.perf_counter() - stats_started
    _write_json(meta / "stats.json", global_result)
    _write_json(
        meta / "alohamini.json",
        {
            "version": 1,
            "source": str(source),
            "source_info": getattr(checker, "source_info", checker.info),
            "state_groups": selection.groups,
            "state_units": selection.units,
            "action": "unchanged submitted Host targets, not measured motion",
            "joint_velocity": (
                "signed feedback ticks/s scaled to the recorded Host position coordinate per second"
            ),
            "joint_current": "reported current magnitude in amperes; not joint torque",
            "source_state": (
                "observation.source_state retains the original observation.state"
                if "observation.source_state" in features
                else "observation.state retains the original recorded coordinates"
            ),
            "feedback_statistics": (
                "extra motor fields retain zero placeholders; apply their validity masks before use"
            ),
            "numeric_statistics": "float64 centered moments; exact linear quantiles; version 2",
            "image_quantiles": (
                "linear quantiles of sampled uint8 pixels; merged RGB histograms"
                if recording
                else "approximate sampled histograms"
            ),
            **(
                {
                    "image_statistics": {
                        "sources": sorted(global_stats.image_statistics.sources),
                        "sampling": "per-episode image_statistics metadata",
                        "count": "sampled frames per camera",
                    }
                }
                if recording
                else {}
            ),
            "image_statistics_frame_limit_per_episode": 10_000 if recording else image_sample_limit,
            "safety_path": "meta/safety/episode_{episode_index:06d}.jsonl",
            "training_review": checker.report()["training_review"],
        },
    )
    return global_stats


def recording_statistics(root):
    """Restore image histograms; rebuild exact numeric accumulators from tables."""
    info = json.loads((root / "meta/info.json").read_text())
    stats = _Stats(info["features"], recording=True)
    cameras = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        for meta in pq.read_table(path).to_pylist():
            data = root / DATA_PATH.format(
                chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"]
            )
            for batch in pq.ParquetFile(data).iter_batches(batch_size=512):
                rows = [r for r in batch.to_pylist() if r["episode_index"] == meta["episode_index"]]
                if rows:
                    stats.update(rows, {})
            if meta.get("image_statistics") is not None:
                stats.image_statistics.merge(ImageStatistics(json.loads(meta["image_statistics"])))
            else:
                # Legacy recordings have no retained PNG histogram. Reconstruct
                # from sampled video frames; never relabel these as PNGs.
                import av

                for key in cameras:
                    video = root / VIDEO_PATH.format(
                        video_key=key,
                        chunk_index=meta[f"videos/{key}/chunk_index"],
                        file_index=meta[f"videos/{key}/file_index"],
                    )
                    with av.open(str(video)) as container:
                        offset = round(meta[f"videos/{key}/from_timestamp"] * info["fps"])
                        for rgb in sample_video_images(
                            container, sample_indices(meta["length"]), offset=offset
                        ):
                            stats.image_statistics.update(key, rgb, source="legacy_decoded_video")
    return stats


def publish_recorded_episode(dataset):
    """Publish one v3 episode; metadata counts are the final commit marker."""
    from types import SimpleNamespace

    root, episode = dataset.root, dataset._pending
    stage = episode / "v3"
    stage.mkdir()
    checker = SimpleNamespace(
        info=dataset._recording_info,
        shapes=dataset._image_shapes,
        cameras=dataset.cameras,
        num_episodes=dataset.num_episodes + 1,
        total_frames=dataset.total_frames + dataset.saved,
        episode_paths={dataset.num_episodes: episode},
        recording=True,
        save_timings=getattr(dataset, "save_timings", None),
        report=lambda: {"training_review": "required"},
    )
    started = time.perf_counter()
    dataset._v3_stats = _write_dataset(
        root,
        stage,
        checker,
        StateSelection(checker.info),
        start_episode=dataset.num_episodes,
        start_offset=dataset.total_frames,
        stats=dataset._v3_stats,
    )
    timings = checker.save_timings
    if timings is not None:
        # These result() calls are serial children, subtracted to avoid double counting.
        timings["v3_data_metadata"] = max(
            0.0,
            time.perf_counter()
            - started
            - sum(
                timings.get(key, 0.0)
                for key in (
                    "image_decode",
                    "statistics_update",
                    "episode_statistics",
                    "global_statistics",
                )
            ),
        )
    started = time.perf_counter()
    files = sorted(p.relative_to(stage) for p in stage.rglob("*") if p.is_file())
    # Global metadata is replaceable; data/video/episode shards must be new.
    global_files = {
        Path(f"meta/{name}")
        for name in ("info.json", "stats.json", "tasks.parquet", "alohamini.json")
    }
    for relative in files:
        if (root / relative).exists() and relative not in global_files:
            raise FileExistsError(root / relative)
    backup = episode / "backup"
    backup.mkdir()
    for relative in global_files:
        if (root / relative).exists():
            target = backup / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
    _write_json(
        episode / "publish.json",
        {
            "episode_index": dataset.num_episodes,
            "files": [str(p) for p in files],
        },
    )
    # Incomplete publishes retain their source PNGs, videos and metadata backup.
    # Recovery uses this manifest; never silently resume a partial commit.
    files.remove(Path("meta/info.json"))
    for relative in [*files, Path("meta/info.json")]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        (stage / relative).replace(target)
    if timings is not None:
        timings["v3_publish"] = time.perf_counter() - started


def recover_recording(source, output):
    """Recover a recorder transaction into a new v3 directory; never edit source."""
    from types import SimpleNamespace

    from alohamini.datasets.lerobot_tools import IntegrityChecker as V3Checker
    from alohamini.datasets.tools import _recover_pending

    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Output already exists: {output}; choose a new --output. Nothing overwritten."
        )
    if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        raise ValueError("Recovery output must be separate from the source")
    with _read_lock(source):
        if any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError("Recording contains a symbolic link")
        pending = sorted((source / ".recording/episodes").iterdir())
        if len(pending) != 1:
            raise ValueError("Expected exactly one interrupted recording episode")
        episode = pending[0]
        recording_info = json.loads((source / ".recording/meta/info.json").read_text())
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f"{output.name}.pending-", dir=output.parent))
        try:
            for path in source.iterdir():
                if path.name == ".recording":
                    continue
                if path.is_dir():
                    shutil.copytree(path, stage / path.name)
                else:
                    shutil.copyfile(path, stage / path.name)
            info_path = stage / "meta/info.json"
            info = json.loads(info_path.read_text()) if info_path.exists() else {}
            index = int(episode.name.removeprefix("episode_").removesuffix(".pending"))
            committed = info.get("total_episodes", 0) > index
            if not committed:
                if (episode / "publish.json").exists():
                    manifest = json.loads((episode / "publish.json").read_text())
                    chunk, file = divmod(index, 1000)
                    global_files = {
                        f"meta/{name}"
                        for name in ("info.json", "stats.json", "tasks.parquet", "alohamini.json")
                    }
                    allowed = {
                        DATA_PATH.format(chunk_index=chunk, file_index=file),
                        EPISODE_PATH.format(chunk_index=chunk, file_index=file),
                        f"meta/safety/episode_{index:06d}.jsonl",
                        *(
                            VIDEO_PATH.format(
                                video_key=f"observation.images.{c}",
                                chunk_index=chunk,
                                file_index=file,
                            )
                            for c in recording_info["robot_metadata"]["cameras"]
                        ),
                        *global_files,
                    }
                    if set(manifest["files"]) != allowed or manifest["episode_index"] != index:
                        raise ValueError("Invalid recording publication manifest")
                    for relative in manifest["files"]:
                        backup = episode / "backup" / relative
                        target = stage / relative
                        if relative in global_files and backup.is_file():
                            shutil.copyfile(backup, target)
                        else:
                            target.unlink(missing_ok=True)
                    info = json.loads(info_path.read_text()) if info_path.exists() else {}
                if info.get("total_episodes", 0) != index:
                    raise ValueError("Interrupted episode does not follow committed episodes")
                scratch = stage / ".recovery"
                (scratch / "episodes").mkdir(parents=True)
                checker = IntegrityChecker(source)
                checker.info, checker.pending = recording_info, [episode]
                checker.num_episodes, checker.total_frames = index, info.get("total_frames", 0)
                checker.cameras = recording_info["robot_metadata"]["cameras"]
                checker.schema = dataset_schema(
                    recording_info["features"], checker.cameras, VIDEO_FORMAT
                )
                checker.episode_task, checker.task_index = recording_info["task"], 0
                checker.shapes = (
                    {
                        c: info["features"][f"observation.images.{c}"]["shape"]
                        for c in checker.cameras
                    }
                    if index
                    else {}
                )
                if episode.name.endswith(".pending"):
                    _recover_pending(checker, scratch)
                    recovered = scratch / "episodes" / episode.stem
                else:
                    recovered = scratch / "episodes" / episode.name
                    recovered.mkdir()
                    for name in ("frames.parquet", "episode.json", "safety.jsonl"):
                        shutil.copyfile(episode / name, recovered / name)
                    shutil.copytree(episode / "videos", recovered / "videos")
                    if (episode / "image_stats.json").exists():
                        shutil.copyfile(
                            episode / "image_stats.json", recovered / "image_stats.json"
                        )
                summary = json.loads((recovered / "episode.json").read_text())
                transaction = SimpleNamespace(
                    root=stage,
                    _pending=recovered,
                    _recording_info=recording_info,
                    _v3_stats=recording_statistics(stage) if index else None,
                    _image_shapes=summary["image_shapes"],
                    cameras=checker.cameras,
                    num_episodes=index,
                    total_frames=info.get("total_frames", 0),
                    saved=summary["length"],
                )
                publish_recorded_episode(transaction)
                shutil.rmtree(scratch)
            report = V3Checker(stage, decode_videos=True, decode_images=True).run()
            if not report["valid"]:
                raise ValueError(f"Recovered recording failed validation: {report['issues']}")
            stage.rename(output)
            report["dataset_root"] = str(output)
            return report
        except BaseException as exc:
            raise RuntimeError(
                f"Recovery unfinished; source unchanged; partial output: {stage}: {exc}"
            ) from exc


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
            video_source = checker.info["image_format"] == VIDEO_FORMAT
            if video_source:
                from alohamini.datasets.video import file_sha256

                for camera in checker.cameras:
                    key = f"observation.images.{camera}"
                    target = output / VIDEO_PATH.format(
                        video_key=key,
                        chunk_index=metadata[f"videos/{key}/chunk_index"],
                        file_index=metadata[f"videos/{key}/file_index"],
                    )
                    original_video = (
                        getattr(checker, "video_paths", {})
                        .get(index, {})
                        .get(camera, episode / "videos" / f"{camera}.mp4")
                    )
                    source_hash = getattr(checker, "video_hashes", {}).get(index, {}).get(camera)
                    if source_hash is None:
                        source_hash = file_sha256(original_video)
                    if file_sha256(target) != source_hash:
                        raise ValueError(f"Exported video changed: {camera}")
                    if (
                        metadata[f"videos/{key}/from_timestamp"] != 0.0
                        or metadata[f"videos/{key}/to_timestamp"]
                        != metadata["length"] / checker.info["fps"]
                    ):
                        raise ValueError(f"Exported video interval changed: {camera}")
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
                for camera in () if video_source else checker.cameras:
                    key = f"observation.images.{camera}"
                    if row[key]["path"] is not None:
                        raise ValueError(f"Exported image has an external path: {key}")
                    if isinstance(original[key], str) or (
                        original[key].get("member", "").endswith(".png")
                    ):
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


def export_lerobot(root, output, *, state=StateSelection.DEFAULT, vision_only=False):
    if vision_only:
        if state != StateSelection.DEFAULT:
            raise ValueError("--vision-only cannot be combined with a custom --state")
        return export_visual_lerobot(root, output)
    source = Path(root).expanduser().resolve()
    if json.loads((source / "meta/info.json").read_text()).get("codebase_version") == "v3.0":
        return export_visual_lerobot(source, output, vision_only=False, state=state)
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Output already exists: {output}; choose a new --output. Nothing overwritten."
        )
    if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        raise ValueError("Output must be separate from the source dataset")
    with _read_lock(source):
        print("[EXPORT] Checking source images and records...", flush=True)
        checker = IntegrityChecker(source, decode_images=True)
        checker._run_unlocked()
        report = checker.report()
        if not report["valid"] or not checker.total_frames:
            raise ValueError(f"Source must be complete before LeRobot export: {report}")
        if not {"observation.state", "action"}.issubset(checker.info["features"]):
            raise ValueError("This LeRobot exporter requires the original state and action fields")
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
            print(f"[EXPORT] Writing {checker.total_frames} frames to {stage}", flush=True)
            _write_dataset(source, stage, checker, selection)
            print("[EXPORT] Verifying exported images, records and indices...", flush=True)
            _validate_export(source, stage, checker, selection)
            if output.exists():
                raise FileExistsError(
                    f"Output already exists: {output}; choose a new --output. Nothing overwritten."
                )
            stage.rename(output)
        except BaseException as exc:
            raise RuntimeError(
                f"LeRobot export unfinished: {exc}; source unchanged, files retained at {stage}"
            ) from exc
        report.update(
            dataset_root=str(output), format="lerobot-v3", state_names=selection.feature["names"]
        )
        return report


def export_visual_lerobot(root, output, *, vision_only=True, state=StateSelection.DEFAULT):
    """Project local v3 fields without changing media, frame order or pairing.

    Numeric observation and feedback columns are removed, not zero-filled.
    Calibration, safety sidecars, indices and action coordinates remain available.
    """
    from alohamini.datasets.lerobot_tools import IntegrityChecker as LeRobotChecker

    source = Path(root).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Output already exists: {output}; choose a new --output. Nothing overwritten."
        )
    if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        raise ValueError("Output must be separate from the source dataset")
    with _read_lock(source) if (source / "recording.lock").exists() else nullcontext():
        checker = LeRobotChecker(source, decode_images=True)
        report = checker.run()
        if not report["valid"] or not checker.total_data_rows:
            raise ValueError(f"Expected a complete LeRobot v3 source: {report['issues']}")
        info = deepcopy(checker.info)
        cameras = {k: v for k, v in info["features"].items() if k.startswith("observation.images.")}
        if (vision_only and not cameras) or any(
            v["dtype"] not in ("image", "video") for v in cameras.values()
        ):
            raise ValueError("Vision export requires image or video cameras")
        keep = {"action", *DEFAULT_FEATURES, *cameras} if vision_only else set(info["features"])
        if not keep.issubset(info["features"]):
            raise ValueError("Source is missing action or index features")
        info["features"] = {k: v for k, v in info["features"].items() if k in keep}
        selection = None
        if not vision_only and state != StateSelection.DEFAULT:
            source_info = json.loads((source / "meta/alohamini.json").read_text())["source_info"]
            if info["features"].get("observation.state") != source_info["features"].get(
                "observation.state"
            ):
                raise ValueError("Select state fields from the original recording")
            selection = StateSelection(source_info, state)
            info["features"]["observation.source_state"] = deepcopy(
                info["features"]["observation.state"]
            )
            info["features"]["observation.state"] = selection.feature
            keep.add("observation.source_state")
        schema = _arrow_schema(info["features"])
        state_stats = _Stats({"observation.state": selection.feature}) if selection else None
        episode_stats = {}
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f"{output.name}.pending-", dir=output.parent))
        try:
            shutil.copytree(source / "meta", stage / "meta")
            if any(v["dtype"] == "video" for v in cameras.values()):
                for path in sorted(checker.referenced_videos):
                    target = stage / path.relative_to(source)
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
            _write_json(stage / "meta/info.json", info)
            stats = json.loads((source / "meta/stats.json").read_text())
            if selection:
                stats["observation.source_state"] = stats["observation.state"]
            _write_json(stage / "meta/stats.json", {k: v for k, v in stats.items() if k in keep})
            for path in sorted(checker.data_files):
                target = stage / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                with pq.ParquetWriter(target, schema, compression="zstd") as writer:
                    for batch in pq.ParquetFile(path).iter_batches(
                        batch_size=8, columns=None if selection else schema.names
                    ):
                        if selection:
                            rows = batch.to_pylist()
                            for row in rows:
                                row["observation.source_state"] = row["observation.state"]
                                row["observation.state"] = selection.frame(row).tolist()
                                if row["episode_index"] not in episode_stats:
                                    episode_stats[row["episode_index"]] = _Stats(
                                        {"observation.state": selection.feature}
                                    )
                                tracker = episode_stats[row["episode_index"]]
                                tracker.update([row])
                            state_stats.update(rows)
                            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                        else:
                            writer.write_table(pa.Table.from_batches([batch]).cast(schema))
            if selection:
                stats["observation.state"] = state_stats.result()["observation.state"]
                _write_json(
                    stage / "meta/stats.json", {k: v for k, v in stats.items() if k in keep}
                )
            for path in sorted((stage / "meta/episodes").rglob("*.parquet")):
                table = pq.read_table(path)
                columns = [
                    k
                    for k in table.column_names
                    if not k.startswith("stats/") or k.split("/")[1] in keep
                ]
                if selection:
                    rows = table.to_pylist()
                    for row in rows:
                        for key, value in list(row.items()):
                            if key.startswith("stats/observation.state/"):
                                row[
                                    key.replace("observation.state", "observation.source_state")
                                ] = value
                        for key, value in (
                            episode_stats[row["episode_index"]]
                            .result()["observation.state"]
                            .items()
                        ):
                            row[f"stats/observation.state/{key}"] = value.tolist()
                    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
                else:
                    pq.write_table(table.select(columns), path, compression="zstd")
            metadata_path = stage / "meta/alohamini.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                metadata.update(vision_only=vision_only, derived_from=str(source))
                if vision_only:
                    metadata.update(
                        state_groups=[],
                        state_units=[],
                        source_state="numeric observation columns omitted",
                    )
                elif selection:
                    metadata.update(
                        state_groups=selection.groups,
                        state_units=selection.units,
                        source_state="observation.source_state",
                    )
                _write_json(metadata_path, metadata)
            stats_info_path = stage / "meta/stats_info.json"
            if stats_info_path.exists():
                from alohamini.datasets.video import file_sha256

                provenance = json.loads(stats_info_path.read_text())
                if "diagnostics" in provenance:
                    provenance["diagnostics"] = {
                        k: v for k, v in provenance["diagnostics"].items() if k in keep
                    }
                provenance.update(
                    source_sha256={
                        str(path.relative_to(stage)): file_sha256(path)
                        for path in sorted(stage.glob("data/**/*.parquet"))
                    },
                    source_info_sha256=file_sha256(stage / "meta/info.json"),
                    derived_from_statistics=file_sha256(source / "meta/stats.json"),
                    projection="unchanged cells; source statistics projected, not refitted",
                )
                _write_json(stats_info_path, provenance)
            result = LeRobotChecker(stage, decode_images=True).run()
            if not result["valid"]:
                raise ValueError(f"Vision export validation failed: {result['issues']}")
            # Projection must not change even one retained cell, image byte or boundary.
            unchanged = [
                k
                for k in schema.names
                if not selection or k not in ("observation.state", "observation.source_state")
            ]
            for path in sorted(checker.data_files):
                original = pq.ParquetFile(path).iter_batches(batch_size=8, columns=unchanged)
                copied = pq.ParquetFile(stage / path.relative_to(source)).iter_batches(
                    batch_size=8, columns=unchanged
                )
                for before, after in zip(original, copied, strict=True):
                    if not pa.Table.from_batches([before]).equals(
                        pa.Table.from_batches([after]), check_metadata=False
                    ):
                        raise ValueError("Vision projection changed a retained field")
            if output.exists():
                raise FileExistsError(
                    f"Output already exists: {output}; choose a new --output. Nothing overwritten."
                )
            stage.rename(output)
            result["dataset_root"] = str(output)
            return result
        except BaseException as exc:
            raise RuntimeError(
                f"Vision export unfinished: {exc}; source unchanged, files retained at {stage}"
            ) from exc
