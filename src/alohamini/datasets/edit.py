# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local dataset editing, adapted from lerobot_edit_dataset and dataset_tools.

All nine operations write to a new directory (except read-only info). Physical
timestamps, command semantics and safety records are never regenerated.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import tempfile
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alohamini.datasets.images import (
    VIDEO_FORMAT,
    ImageShards,
    image_bytes,
    image_path,
    image_rgb,
)
from alohamini.datasets.native import _write_json, dataset_schema
from alohamini.datasets.statistics import (
    ExactQuantileStats,
    RunningQuantileStats,
    diagnose_statistics,
)
from alohamini.datasets.tools import IntegrityChecker, _read_lock
from alohamini.paths import WorkspacePaths

OPERATIONS = (
    "delete_episodes",
    "split",
    "merge",
    "remove_feature",
    "modify_tasks",
    "convert_image_to_video",
    "recompute_stats",
    "reencode_videos",
    "info",
)


def _indices(values, count):
    if (
        not isinstance(values, list)
        or not values
        or any(type(i) is not int or not 0 <= i < count for i in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            f"Episode indices must be unique integers in 0..{count - 1} "
            f"({count} episodes); received {values!r}"
        )
    return sorted(values)


def _fractions_to_episode_indices(total_episodes, splits):
    """Keep source order; assign rounding remainder only when fractions sum to one."""
    if (
        any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1
            for v in splits.values()
        )
        or sum(splits.values()) > 1 + 1e-10
    ):
        raise ValueError("Split fractions must be positive and sum to <= 1")
    indices = list(range(total_episodes))
    result = {}
    start_idx = 0
    for split_name, fraction in splits.items():
        num_episodes = int(total_episodes * fraction)
        end_idx = start_idx + num_episodes
        if split_name == list(splits)[-1] and math.isclose(sum(splits.values()), 1):
            end_idx = total_episodes
        if end_idx == start_idx:
            raise ValueError(f"Split '{split_name}' has no episodes; use explicit episode indices")
        result[split_name] = indices[start_idx:end_idx]
        start_idx = end_idx
    return result


@contextmanager
def _sources(roots):
    with ExitStack() as stack:
        datasets = []
        for root in roots:
            root = Path(root).expanduser().resolve()
            if ".pending-" in root.name or root.name.endswith(".pending"):
                raise ValueError("Cannot edit an unfinished dataset; use the completed source")
            stack.enter_context(_read_lock(root))
            checker = IntegrityChecker(root, decode_images=True, decode_videos=True)
            checker._run_unlocked()
            if not checker.report()["valid"] or not checker.num_episodes:
                raise ValueError(f"Source requires review before editing: {checker.report()}")
            datasets.append(checker)
        yield datasets


def _output_path(value, roots):
    output = Path(value).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must not already exist: {output}")
    resolved = output.resolve()
    if any(resolved.is_relative_to(root) or root.is_relative_to(resolved) for root in roots):
        raise ValueError("Output must be separate from all source datasets")
    return output


@contextmanager
def _destination(output, sources):
    output = _output_path(output, [s.root for s in sources])
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"{output.name}.pending-", dir=output.parent))
    try:
        yield stage
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        stage.rename(output)
    except BaseException as exc:
        raise RuntimeError(
            f"Edit unfinished; source unchanged; partial output retained at {stage}: {exc}"
        ) from exc


def _task(source, episode):
    summary = json.loads(
        (source.root / "episodes" / f"episode_{episode:06d}" / "episode.json").read_text()
    )
    return summary.get("task", source.info["task"])


def _compatible(sources):
    first = sources[0]
    for source in sources[1:]:
        for key in (
            "fps",
            "features",
            "robot_metadata",
            "image_format",
            "image_color",
            "video_encoder",
        ):
            if source.info.get(key) != first.info.get(key):
                raise ValueError(f"Cannot merge datasets with different {key}")
        if source.cameras != first.cameras or source.shapes != first.shapes:
            raise ValueError("Cannot merge different cameras or image dimensions")


def _copy_episode(
    source,
    old,
    target,
    new,
    offset,
    info,
    task,
    task_index,
    removed,
    encoder,
    *,
    compact_images=False,
):
    episode = source.root / "episodes" / f"episode_{old:06d}"
    target.mkdir(parents=True)
    summary = json.loads((episode / "episode.json").read_text())
    cameras = info["cameras"]
    schema = dataset_schema(info["features"], cameras, info["image_format"])
    video_refs = {}
    if encoder is not None:
        import av

        from alohamini.datasets.video import _encode_frames, file_sha256, inspect_video

        (target / "videos").mkdir()
        for camera in cameras:
            path = target / "videos" / f"{camera}.mp4"

            def frames(camera=camera):
                for batch in pq.ParquetFile(episode / "frames.parquet").iter_batches(
                    batch_size=8, columns=[f"observation.images.{camera}"]
                ):
                    for ref in batch.column(0).to_pylist():
                        yield av.VideoFrame.from_ndarray(
                            image_rgb(episode, camera, ref), format="rgb24"
                        )

            shape = summary["image_shapes"][camera]
            count = _encode_frames(frames(), path, info["fps"], shape, **encoder)
            actual = inspect_video(path, decode=True)
            if count != summary["length"] or actual != {
                "frames": count,
                "fps": info["fps"],
                "shape": shape,
            }:
                raise ValueError(f"Encoded video does not match source: {path}")
            video_refs[camera] = {"path": f"videos/{camera}.mp4", "sha256": file_sha256(path)}
    # TAR shards may interleave cameras: removing a camera rebuilds retained JPEG
    # entries byte-for-byte so the removed images do not survive as orphan payloads.
    repack = (compact_images or cameras != source.cameras) and info[
        "image_format"
    ] == "host-jpeg-tar"
    with (
        ImageShards(target) as shards,
        pq.ParquetWriter(target / "frames.parquet", schema, compression="zstd") as writer,
    ):
        for batch in pq.ParquetFile(episode / "frames.parquet").iter_batches(batch_size=128):
            rows = batch.to_pylist()
            for row in rows:
                row["index"] = offset + row["frame_index"]
                row["episode_index"], row["task_index"], row["task"] = new, task_index, task
                for camera in cameras:
                    key = f"observation.images.{camera}"
                    reference = row[key]
                    if encoder is not None:
                        row[key] = {**video_refs[camera], "frame_index": row["frame_index"]}
                    elif repack:
                        row[key] = shards.append(
                            camera, row["frame_index"], image_bytes(episode, camera, reference)
                        )
                    else:
                        path = image_path(episode, camera, reference)
                        destination = target / path.relative_to(episode)
                        if not destination.exists():
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(path, destination)
                for key in removed:
                    row.pop(key, None)
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    with (episode / "safety.jsonl").open() as reader, (target / "safety.jsonl").open("x") as writer:
        for line in reader:
            record = json.loads(line)
            record["episode_index"] = new
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary.update(
        episode_index=new,
        task=task,
        task_index=task_index,
        image_shapes={c: summary["image_shapes"][c] for c in cameras},
    )
    _write_json(target / "episode.json", summary)
    old_preview = source.root / "previews" / episode.name
    if encoder is None and old_preview.is_dir() and not removed:
        from alohamini.datasets.video import _source_signature, check_preview

        try:
            manifest = check_preview(episode, old_preview, info["fps"], decode=True)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            logging.warning("Not copying invalid preview: %s", old_preview)
        else:
            preview = target.parent.parent / "previews" / target.name
            shutil.copytree(old_preview, preview)
            manifest["source_sha256"] = _source_signature(target)
            _write_json(preview / "manifest.json", manifest)
    return summary["length"]


def _rewrite(
    stage,
    sources,
    episodes,
    *,
    operation,
    removed=(),
    tasks=None,
    encoder=None,
    compact_images=False,
):
    info = deepcopy(sources[0].info)
    info["version"] = 3
    info["cameras"] = [c for c in sources[0].cameras if f"observation.images.{c}" not in removed]
    info["features"] = {k: v for k, v in info["features"].items() if k not in removed}
    if encoder is not None:
        info.update(image_format=VIDEO_FORMAT, image_color="rgb", video_encoder=encoder)
    episode_tasks = [
        tasks.get(old, _task(source, old)) if tasks is not None else _task(source, old)
        for source, old in episodes
    ]
    info["tasks"] = sorted(set(episode_tasks))
    info["task"] = info["tasks"][0]
    (stage / "meta").mkdir()
    (stage / "episodes").mkdir()
    (stage / "recording.lock").touch(exist_ok=False)
    _write_json(stage / "meta/info.json", info)
    offset = 0
    mapping = []
    for new, ((source, old), task) in enumerate(zip(episodes, episode_tasks, strict=True)):
        target = stage / "episodes" / f"episode_{new:06d}"
        offset += _copy_episode(
            source,
            old,
            target,
            new,
            offset,
            info,
            task,
            info["tasks"].index(task),
            removed,
            encoder,
            compact_images=compact_images,
        )
        mapping.append({"source": str(source.root), "source_episode": old, "episode_index": new})
    _write_json(
        stage / "meta/edit.json",
        {
            "operation": operation,
            "episode_mapping": mapping,
            "sources": [
                {
                    "path": str(s.root),
                    "info": s.info,
                    "training_review": s.report()["training_review"],
                }
                for s in sources
            ],
            "removed_features": list(removed),
            "pairing": "unchanged physical images, state and subsequent submitted targets",
        },
    )
    # Never retain stale statistics/previews or silently label them as current.
    report = IntegrityChecker(stage, decode_images=True, decode_videos=True).run()
    if not report["valid"]:
        raise ValueError(f"Edited dataset failed validation: {report}")
    return report


def _all_episodes(sources):
    return [(s, i) for s in sources for i in range(s.num_episodes)]


def _kept_after_delete(args, source):
    removed = set(_indices(args.episode_indices, source.num_episodes))
    kept = [(source, i) for i in range(source.num_episodes) if i not in removed]
    if not kept:
        raise ValueError("Cannot delete all episodes")
    return kept


def handle_delete_episodes(args, sources, stage):
    kept = _kept_after_delete(args, sources[0])
    return _rewrite(stage, sources, kept, operation="delete_episodes")


def handle_split(args, sources, stage):
    source = sources[0]
    splits = args.splits
    if not isinstance(splits, dict) or not splits:
        raise ValueError("splits must be a nonempty mapping")
    if all(type(v) in (int, float) for v in splits.values()):
        splits = _fractions_to_episode_indices(source.num_episodes, splits)
    seen = set()
    for name, values in splits.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Split names must be simple directory names")
        values = _indices(values, source.num_episodes)
        if seen.intersection(values):
            raise ValueError("Episodes cannot appear in multiple splits")
        seen.update(values)
    reports = {}
    for name, values in splits.items():
        directory = stage / name
        directory.mkdir()
        reports[name] = _rewrite(
            directory, sources, [(source, i) for i in sorted(values)], operation="split"
        )
    return reports


def handle_merge(args, sources, stage):
    _compatible(sources)
    return _rewrite(stage, sources, _all_episodes(sources), operation="merge")


def handle_remove_feature(args, sources, stage):
    source = sources[0]
    removed = args.feature_names
    known = set(source.info["features"]) | {f"observation.images.{c}" for c in source.cameras}
    if not isinstance(removed, list) or not removed or any(k not in known for k in removed):
        raise ValueError(
            "feature_names must contain existing feature names; indices cannot be removed"
        )
    removed = set(removed)
    for key in list(removed):
        if key.startswith("observation.motor_"):
            removed.add(f"motor_feedback.{key.removeprefix('observation.motor_')}_valid")
        elif key.startswith("motor_feedback.") and key.endswith("_valid"):
            removed.add(
                f"observation.motor_{key.removeprefix('motor_feedback.').removesuffix('_valid')}"
            )
    times = {"motor_feedback.sample_started_s", "motor_feedback.sample_finished_s"}
    if removed & times:
        if any(k.startswith("observation.motor_") and k not in removed for k in known):
            raise ValueError("Feedback sample times cannot be removed while motor feedback remains")
        removed.update(times)
    return _rewrite(
        stage, sources, _all_episodes(sources), operation="remove_feature", removed=sorted(removed)
    )


def handle_modify_tasks(args, sources, stage):
    source = sources[0]
    if args.new_task is None and args.episode_tasks is None:
        raise ValueError("Specify new_task or episode_tasks")
    tasks = {
        i: args.new_task if args.new_task is not None else _task(source, i)
        for i in range(source.num_episodes)
    }
    if args.episode_tasks is not None:
        if not isinstance(args.episode_tasks, dict):
            raise ValueError("episode_tasks must be a mapping")
        overrides = {int(k): v for k, v in args.episode_tasks.items()}
        _indices(list(overrides), source.num_episodes)
        tasks.update(overrides)
    if any(not isinstance(v, str) or not v.strip() for v in tasks.values()):
        raise ValueError("Tasks must be nonempty strings")
    return _rewrite(stage, sources, _all_episodes(sources), operation="modify_tasks", tasks=tasks)


def _encoder(args):
    codec = {"h264": "libx264", "hevc": "libx265", "av1": "libsvtav1"}.get(args.vcodec, args.vcodec)
    options = {"crf": str(args.crf), "g": str(args.g)}
    if args.preset is not None:
        options["preset"] = str(args.preset)
    if codec == "libsvtav1":
        options.setdefault("preset", "12")
        options["svtav1-params"] = "lp=2"
    return {"codec": codec, "pix_fmt": args.pix_fmt, "options": options}


def _video_edit(args, sources, stage, *, reencode):
    source = sources[0]
    if not source.cameras:
        raise ValueError("Dataset has no camera images")
    if (source.info["image_format"] == VIDEO_FORMAT) != reencode:
        raise ValueError(
            "Use reencode_videos for video datasets; convert_image_to_video for images"
        )
    indices = (
        list(range(source.num_episodes))
        if args.episode_indices is None
        else _indices(
            args.episode_indices,
            source.num_episodes,
        )
    )
    return _rewrite(
        stage,
        sources,
        [(source, i) for i in indices],
        operation="reencode_videos" if reencode else "convert_image_to_video",
        encoder=_encoder(args),
    )


def handle_convert_image_to_video(args, sources, stage):
    return _video_edit(args, sources, stage, reencode=False)


def handle_reencode_videos(args, sources, stage):
    return _video_edit(args, sources, stage, reencode=True)


def _statistics(root, info, args):
    features = info["features"]
    if args.relative_action:
        if args.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if "action" not in features or "observation.state" not in features:
            raise ValueError("Relative action statistics require action and observation.state")
        names = features["action"]["names"]
        if names != features["observation.state"]["names"]:
            raise ValueError(
                "Relative action statistics require identical state/action coordinates"
            )
        # Body velocities are already velocity commands: subtracting them changes
        # their meaning. Only position dimensions become position displacements.
        mask = np.array(
            [
                not name.endswith(".vel")
                and not any(
                    excluded.lower() in name.lower() for excluded in args.relative_exclude_joints
                )
                for name in names
            ],
            dtype=np.float32,
        )
    running = {}
    for episode in sorted((root / "episodes").iterdir()):
        for batch in pq.ParquetFile(episode / "frames.parquet").iter_batches(batch_size=128):
            rows = batch.to_pylist()
            for key, feature in features.items():
                if args.relative_action and key == "action":
                    continue
                values = np.asarray([r[key] for r in rows], dtype=np.float64)
                validity = f"motor_feedback.{key.removeprefix('observation.motor_')}_valid"
                for dimension in range(len(feature["names"])):
                    selected = values[:, dimension : dimension + 1]
                    if key.startswith("observation.motor_"):
                        valid = np.asarray([r[validity][dimension] for r in rows]) == 1
                        selected = selected[valid]
                    if len(selected):
                        if (key, dimension) not in running:
                            running[key, dimension] = ExactQuantileStats()
                        running[key, dimension].update(selected)
            if not args.skip_image_video:
                for camera in info["cameras"]:
                    key = f"observation.images.{camera}"
                    for row in rows:
                        rgb = image_rgb(episode, camera, row[key]).astype(np.float64) / 255
                        running.setdefault((key, 0), RunningQuantileStats()).update(
                            rgb.reshape(-1, 3)
                        )
        if args.relative_action:
            # Sliding windows never cross episodes and never pad terminal chunks.
            from collections import deque

            window = deque(maxlen=args.chunk_size)
            for batch in pq.ParquetFile(episode / "frames.parquet").iter_batches(
                batch_size=128, columns=["action", "observation.state"]
            ):
                for row in batch.to_pylist():
                    window.append(row)
                    if len(window) == args.chunk_size:
                        actions = np.asarray([r["action"] for r in window], dtype=np.float64)
                        actions -= (
                            np.asarray(window[0]["observation.state"], dtype=np.float64) * mask
                        )
                        for dimension in range(actions.shape[1]):
                            if ("action", dimension) not in running:
                                running["action", dimension] = ExactQuantileStats()
                            running["action", dimension].update(
                                actions[:, dimension : dimension + 1]
                            )
    if args.relative_action and ("action", 0) not in running:
        raise ValueError("No full action chunks within any episode")
    result = {}
    for key, feature in features.items():
        per_dim = []
        for dimension in range(len(feature["names"])):
            value = running.get((key, dimension))
            if value is None:
                per_dim.append(None)
            elif value._count == 1:
                stats = {
                    k: value._mean.copy()
                    for k in ("mean", "min", "max", "q01", "q10", "q50", "q90", "q99")
                }
                stats.update(std=np.zeros(1), count=np.ones(1, dtype=int))
                per_dim.append(stats)
            else:
                per_dim.append(value.get_statistics())
        result[key] = {
            stat: [float(v[stat][0]) if v is not None else None for v in per_dim]
            for stat in ("mean", "std", "min", "max", "count", "q01", "q10", "q50", "q90", "q99")
        }
        result[key]["count"] = [0 if v is None else int(v["count"][0]) for v in per_dim]
    for camera in info["cameras"]:
        key = f"observation.images.{camera}"
        if (key, 0) in running:
            stats = running[key, 0].get_statistics()
            result[key] = {
                k: (v if k == "count" else v[:, None, None]).tolist() for k, v in stats.items()
            }
    _write_json(root / "meta/stats.json", result)
    diagnostics = {}
    for key, feature in features.items():
        diagnostics[key] = []
        for dimension, name in enumerate(feature["names"]):
            if not result[key]["count"][dimension]:
                diagnostics[key].append({"name": name, "unavailable": True})
                continue
            values = {
                stat: np.array([result[key][stat][dimension]])
                for stat in ("q01", "q99", "min", "max", "std")
            }
            diagnostics[key].extend(diagnose_statistics(values, names=[name]))
    from alohamini.datasets.video import file_sha256

    _write_json(
        root / "meta/stats_info.json",
        {
            "relative_action": args.relative_action,
            "chunk_size": args.chunk_size,
            "relative_exclude_joints": args.relative_exclude_joints,
            "base_velocity": "absolute; never subtracted",
            "feedback": "invalid excluded per dimension; null with count=0 means unavailable",
            "numeric_statistics": "float64 centered moments; exact linear quantiles; version 2",
            "source_info_sha256": file_sha256(root / "meta/info.json"),
            "image_quantiles": "approximate streaming histograms",
            "source_sha256": {
                str(path.relative_to(root)): file_sha256(path)
                for path in sorted(root.glob("episodes/*/frames.parquet"))
            },
            "diagnostics": diagnostics,
            "training": "whole-dataset stats; fit training normalization on training episodes only",
        },
    )


def handle_recompute_stats(args, sources, stage):
    report = _rewrite(stage, sources, _all_episodes(sources), operation="recompute_stats")
    _statistics(stage, json.loads((stage / "meta/info.json").read_text()), args)
    return report


def handle_info(args, sources, stage=None):
    source = sources[0]
    result = {
        "root": str(source.root),
        "episodes": source.num_episodes,
        "frames": source.total_frames,
        "fps": source.info["fps"],
        "tasks": source.info.get("tasks", [source.info["task"]]),
        "cameras": source.cameras,
        "image_format": source.info["image_format"],
        "size_bytes": sum(p.stat().st_size for p in source.root.rglob("*") if p.is_file()),
        "training_review": source.report()["training_review"],
    }
    if args.show_features:
        result["features"] = source.info["features"]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def edit_dataset(args):
    if args.operation == "merge":
        roots = args.roots
        if not isinstance(roots, list) or not roots or any(not isinstance(p, str) for p in roots):
            raise ValueError("merge requires --operation.roots with local dataset directories")
    else:
        if not args.root:
            raise ValueError("--root is required")
        roots = [args.root]
    with _sources(roots) as sources:
        handler = globals()[f"handle_{args.operation}"]
        if args.operation == "info":
            return handler(args, sources)
        if not args.output:
            raise ValueError("--output is required; source datasets are never modified")
        output = _output_path(args.output, [s.root for s in sources])
        if args.operation == "delete_episodes":
            _kept_after_delete(args, sources[0])
        with _destination(output, sources) as stage:
            result = handler(args, sources, stage)
        if args.operation == "split":
            for name, report in result.items():
                report["dataset_root"] = str(output / name)
        else:
            result["dataset_root"] = str(output)
        print(f"Dataset saved at {output}")
        return result


def _boolean(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("Expected true or false")
    return value.lower() == "true"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="alohamini dataset edit", description=__doc__)
    parser.add_argument("--config_path", help="Local JSON configuration; CLI values override it")
    parser.add_argument("--root", help="Input AlohaMini dataset directory")
    parser.add_argument("--output", "--new_root", dest="output", help="New output directory")
    parser.add_argument("--dataset", help="Input dataset name in Alohamini_workspace/datasets")
    parser.add_argument("--operation.type", "--type", dest="operation", choices=OPERATIONS)
    parser.add_argument("--operation.episode_indices", dest="episode_indices", type=json.loads)
    parser.add_argument("--operation.splits", dest="splits", type=json.loads)
    parser.add_argument("--operation.roots", dest="roots", type=json.loads)
    parser.add_argument("--operation.feature_names", dest="feature_names", type=json.loads)
    parser.add_argument("--operation.new_task", dest="new_task")
    parser.add_argument("--operation.episode_tasks", dest="episode_tasks", type=json.loads)
    parser.add_argument(
        "--operation.show_features", dest="show_features", type=_boolean, default=False
    )
    parser.add_argument(
        "--operation.skip_image_video", dest="skip_image_video", type=_boolean, default=True
    )
    parser.add_argument(
        "--operation.relative_action", dest="relative_action", type=_boolean, default=False
    )
    parser.add_argument(
        "--operation.relative_exclude_joints",
        dest="relative_exclude_joints",
        type=json.loads,
        default=["gripper"],
    )
    parser.add_argument("--operation.chunk_size", dest="chunk_size", type=int, default=50)
    parser.add_argument("--operation.rgb_encoder.vcodec", dest="vcodec", default="h264")
    parser.add_argument("--operation.rgb_encoder.pix_fmt", dest="pix_fmt", default="yuv420p")
    parser.add_argument("--operation.rgb_encoder.crf", dest="crf", type=int, default=18)
    parser.add_argument("--operation.rgb_encoder.g", dest="g", type=int, default=2)
    parser.add_argument("--operation.rgb_encoder.preset", dest="preset")
    argv = list(argv) if argv is not None else __import__("sys").argv[1:]
    config_arg, _ = parser.parse_known_args(argv)
    if config_arg.config_path:
        config = json.loads(Path(config_arg.config_path).expanduser().read_text())
        if not isinstance(config, dict):
            parser.error("Configuration must be a JSON object")

        def flatten(value, prefix=""):
            result = []
            for key, item in value.items():
                name = f"{prefix}.{key}" if prefix else key
                if isinstance(item, dict) and name in ("operation", "operation.rgb_encoder"):
                    result.extend(flatten(item, name))
                else:
                    result += [
                        f"--{name}",
                        json.dumps(item) if isinstance(item, (dict, list, bool)) else str(item),
                    ]
            return result

        argv = flatten(config) + argv
    args = parser.parse_args(argv)
    if args.operation is None:
        parser.error("--operation.type is required")
    if args.dataset:
        if args.root:
            parser.error("Use either --dataset or --root")
        args.root = str(WorkspacePaths().dataset(args.dataset))
    if not isinstance(args.relative_exclude_joints, list) or any(
        not isinstance(v, str) or not v for v in args.relative_exclude_joints
    ):
        parser.error("relative_exclude_joints must be a list of joint name fragments")
    return args


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    edit_dataset(parse_args(argv))


if __name__ == "__main__":
    main()
