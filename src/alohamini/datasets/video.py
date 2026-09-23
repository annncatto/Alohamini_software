# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline MP4 previews and video range repacking.

Encoding and range selection adapt LeRobot video_utils.encode_video_frames and
dataset_tools._keep_episodes_from_video_with_av. No capture or control work runs here.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alohamini.datasets.images import image_path, image_rgb
from alohamini.datasets.native import _write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_video(path: Path, *, decode=False) -> dict:
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"Expected one video stream: {path}")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        frames = stream.frames
        if decode:
            frames = 0
            for frame in container.decode(stream):
                if (
                    fps is None
                    or frame.time is None
                    or not math.isclose(frame.time, frames / fps, abs_tol=1e-4)
                ):
                    raise ValueError(f"Non-contiguous video timestamps: {path}, frame {frames}")
                frames += 1
            if stream.frames and frames != stream.frames:
                raise ValueError(f"Video decoded/header frame counts differ: {path}")
        return {"frames": frames, "fps": fps, "shape": [stream.height, stream.width, 3]}


def video_frame_count(path: Path) -> int:
    # Fully decode: damaged datasets cannot rely on the container frame count.
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def _encode_frames(frames, path, fps, shape, *, codec="libx264", pix_fmt="yuv420p", options=None):
    rate = Fraction(str(fps)).limit_denominator(100000)
    if rate <= 0:
        raise ValueError("Video fps must be positive")
    count = 0
    with av.open(str(path), "w") as output:
        stream = output.add_stream(codec, rate=rate, options=options or {})
        stream.width, stream.height = shape[1], shape[0]
        stream.pix_fmt = pix_fmt
        stream.thread_count = 2
        stream.time_base = 1 / rate
        for frame in frames:
            frame = frame.reformat(width=stream.width, height=stream.height, format=pix_fmt)
            frame.pts, frame.time_base = count, 1 / rate
            for packet in stream.encode(frame):
                output.mux(packet)
            count += 1
        if not count:
            raise ValueError("No frames to encode")
        for packet in stream.encode():
            output.mux(packet)
    return count


def _repack_encoder_options(codec: str, info: dict) -> dict[str, str]:
    # Adapt VideoEncoderConfig.from_video_info/get_codec_options for the software
    # encoders selected from the source stream. Null metadata uses class defaults.
    defaults = {"g": 2, "crf": 30, "preset": 12 if codec == "libsvtav1" else None}
    options = {
        key: str(info.get(f"video.{key}") if info.get(f"video.{key}") is not None else default)
        for key, default in defaults.items()
        if info.get(f"video.{key}") is not None or default is not None
    }
    fast_decode = info.get("video.fast_decode")
    if fast_decode is None:
        fast_decode = 0
    if codec == "libsvtav1":
        options["svtav1-params"] = f"fast-decode={max(0, min(2, fast_decode))}"
    elif codec in ("libx264", "libx265") and fast_decode:
        options["tune"] = "fastdecode"
    for key, value in (info.get("video.extra_options") or {}).items():
        if key not in options and value is not None:
            options[key] = str(value)
    return options


def repack_video(source: Path, output: Path, ranges: list[tuple[int, int]], fps: float, info: dict):
    """Retain half-open frame ranges in source order and reset their video timestamps."""
    if not ranges or any(start < 0 or end <= start for start, end in ranges):
        raise ValueError("Invalid video frame ranges")
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:], strict=False)):
        raise ValueError("Overlapping or unordered video frame ranges")
    if info.get("is_depth_map") or info.get("video.is_depth_map"):
        raise ValueError("Depth video repacking requires its depth encoder")
    with av.open(str(source)) as container:
        if len(container.streams.video) != 1 or container.streams.audio:
            raise ValueError("Repacking requires a single video stream without audio")
        stream = container.streams.video[0]
        codec = {"h264": "libx264", "hevc": "libx265", "av1": "libsvtav1"}.get(
            stream.codec_context.name, stream.codec_context.name
        )
        options = _repack_encoder_options(codec, info)

        def selected_frames():
            range_index = 0
            for index, frame in enumerate(container.decode(stream)):
                while range_index < len(ranges) and index >= ranges[range_index][1]:
                    range_index += 1
                if range_index == len(ranges):
                    break
                if index >= ranges[range_index][0]:
                    yield frame

        count = _encode_frames(
            selected_frames(),
            output,
            fps,
            [stream.height, stream.width, 3],
            codec=codec,
            pix_fmt=stream.pix_fmt,
            options=options,
        )
    if count != sum(end - start for start, end in ranges):
        raise ValueError(f"Source video is missing required frames: {source}")
    inspected = inspect_video(output, decode=True)
    if inspected["frames"] != count or not math.isclose(inspected["fps"], fps, abs_tol=1e-3):
        raise ValueError(f"Repacked video validation failed: {output}")


def _source_signature(episode: Path) -> dict:
    signature = {
        name: file_sha256(episode / name)
        for name in ("frames.parquet", "episode.json", "safety.jsonl")
    }
    # JPEG shard references already carry content hashes in Parquet. Historical
    # PNG references contain paths only, so bind their actual contents as well.
    parquet = pq.ParquetFile(episode / "frames.parquet")
    columns = [
        field.name
        for field in parquet.schema_arrow
        if field.name.startswith("observation.images.") and pa.types.is_string(field.type)
    ]
    if columns:
        digest = hashlib.sha256()
        for batch in parquet.iter_batches(batch_size=128, columns=columns):
            for row in batch.to_pylist():
                for column in columns:
                    camera = column.removeprefix("observation.images.")
                    path = image_path(episode, camera, row[column])
                    digest.update(bytes.fromhex(file_sha256(path)))
        signature["png_images"] = digest.hexdigest()
    return signature


def check_preview(episode: Path, directory: Path, fps: int, *, decode=False) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((episode / "episode.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid preview manifest: {directory}")
    if (
        manifest.get("format") != "alohamini-preview-v1"
        or manifest.get("source_sha256") != _source_signature(episode)
        or manifest.get("frames") != summary["length"]
        or manifest.get("fps") != fps
        or set(manifest.get("videos", {})) != set(summary["image_shapes"])
    ):
        raise ValueError(f"Preview metadata does not match its source episode: {directory}")
    for camera, shape in summary["image_shapes"].items():
        path = directory / f"{camera}.mp4"
        if path.is_symlink() or file_sha256(path) != manifest["videos"][camera]["sha256"]:
            raise ValueError(f"Preview checksum mismatch: {path}")
        actual = inspect_video(path, decode=decode)
        expected_shape = [shape[0] + shape[0] % 2, shape[1] + shape[1] % 2, 3]
        if (
            actual["frames"] != summary["length"]
            or actual["fps"] != fps
            or actual["shape"] != expected_shape
        ):
            raise ValueError(f"Preview frame count, fps or dimensions mismatch: {path}")
    return manifest


def generate_previews(root: Path, output: Path | None = None) -> dict:
    """Generate one MP4 per episode/camera, preserving all authoritative source files."""
    from alohamini.datasets.tools import IntegrityChecker, _read_lock

    root = Path(root).expanduser().resolve()
    directory = Path(output).expanduser().absolute() if output is not None else root / "previews"
    if output is not None and (directory.exists() or directory.is_symlink()):
        raise FileExistsError(directory)
    if output is not None and (
        directory.resolve().is_relative_to(root) or root.is_relative_to(directory.resolve())
    ):
        raise ValueError("Custom preview output must be separate from the source dataset")
    if directory.is_symlink():
        raise ValueError("Preview directory must not be a symbolic link")
    with _read_lock(root):
        checker = IntegrityChecker(root)
        checker._run_unlocked()
        report = checker.report()
        if not report["valid"]:
            raise ValueError(f"Source requires repair before preview: {report}")
        if not checker.cameras or not checker.num_episodes:
            return {"output": str(directory), "generated": 0, "reused": 0}
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            generated = reused = 0
            for index in range(checker.num_episodes):
                episode = root / "episodes" / f"episode_{index:06d}"
                destination = directory / episode.name
                if destination.is_symlink():
                    raise ValueError(f"Preview must not be a symbolic link: {destination}")
                if destination.exists():
                    check_preview(episode, destination, checker.info["fps"], decode=True)
                    reused += 1
                    continue
                stage = Path(tempfile.mkdtemp(prefix=f"{episode.name}.pending-", dir=directory))
                summary = json.loads((episode / "episode.json").read_text())
                manifest = {
                    "format": "alohamini-preview-v1",
                    "fps": checker.info["fps"],
                    "frames": summary["length"],
                    "timeline": "frame_index / fps",
                    "source_sha256": _source_signature(episode),
                    "videos": {},
                }
                try:
                    for camera, shape in summary["image_shapes"].items():
                        path = stage / f"{camera}.mp4"

                        def frames(camera=camera, episode=episode):
                            for batch in pq.ParquetFile(episode / "frames.parquet").iter_batches(
                                batch_size=8, columns=[f"observation.images.{camera}"]
                            ):
                                for reference in batch.column(0).to_pylist():
                                    rgb = image_rgb(episode, camera, reference)
                                    # H.264 4:2:0 requires even dimensions; pad, never resize.
                                    rgb = np.pad(
                                        rgb,
                                        ((0, rgb.shape[0] % 2), (0, rgb.shape[1] % 2), (0, 0)),
                                        mode="edge",
                                    )
                                    yield av.VideoFrame.from_ndarray(rgb, format="rgb24")

                        video_shape = [shape[0] + shape[0] % 2, shape[1] + shape[1] % 2, 3]
                        _encode_frames(
                            frames(),
                            path,
                            checker.info["fps"],
                            video_shape,
                            options={"crf": "18", "preset": "fast"},
                        )
                        manifest["videos"][camera] = {"sha256": file_sha256(path)}
                    _write_json(stage / "manifest.json", manifest)
                    check_preview(episode, stage, checker.info["fps"], decode=True)
                    if destination.exists() or destination.is_symlink():
                        raise FileExistsError(destination)
                    stage.rename(destination)
                    generated += 1
                    print(f"PREVIEW {episode.name}: {destination}", flush=True)
                except BaseException as exc:
                    raise RuntimeError(
                        f"Preview unfinished; source unchanged; partial files at {stage}: {exc}"
                    ) from exc
    return {"output": str(directory), "generated": generated, "reused": reused}
