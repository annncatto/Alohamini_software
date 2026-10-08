#!/usr/bin/env python3
"""Migrate earlier AlohaMini episode datasets using existing MP4 previews only."""

import argparse
import json
import tempfile
import time
from copy import deepcopy
from pathlib import Path

from alohamini.datasets.images import VIDEO_FORMAT, image_path
from alohamini.datasets.lerobot_tools import IntegrityChecker as V3Checker
from alohamini.datasets.lerobotv3 import _validate_export, _write_dataset
from alohamini.datasets.record import StateSelection, _write_json
from alohamini.datasets.tools import IntegrityChecker, _read_lock
from alohamini.datasets.video import check_preview, inspect_video


class Progress:
    def __init__(self, phase, total):
        self.phase, self.total = phase, total
        self.started = self.last = time.monotonic()
        print(f"[MIGRATE] {phase}: 0/{total}", flush=True)

    def update(self, count):
        now = time.monotonic()
        if count == self.total or now - self.last >= 10:
            print(
                f"[MIGRATE] {self.phase}: {count}/{self.total} ({now - self.started:.1f}s)",
                flush=True,
            )
            self.last = now


class PreviewSource(IntegrityChecker):
    """Validate pairing/indexes; a verified preview supplies the image payloads."""

    def _check_episode(self, episode):
        super()._check_episode(episode)
        self.progress.update(self.num_episodes + 1)

    def _check_image(self, episode, row, camera):
        if self.info["image_format"] == VIDEO_FORMAT:
            return super()._check_image(episode, row, camera)
        reference = row[f"observation.images.{camera}"]
        path = image_path(episode, camera, reference)
        if isinstance(reference, dict):
            return path, reference.get("offset", reference.get("frame_index"))
        return (path,)


def _preflight(source):
    """Fail on absent media before scanning all recorded rows."""
    info = json.loads((source / "meta/info.json").read_text())
    episodes = sorted((source / "episodes").glob("episode_[0-9]*"))
    cameras = info.get("cameras", info["robot_metadata"]["cameras"])
    missing = []
    for episode in episodes:
        if info["image_format"] == VIDEO_FORMAT:
            directory, names = episode / "videos", []
        else:
            directory, names = source / "previews" / episode.name, ["manifest.json"]
        for name in [*names, *(f"{c}.mp4" for c in cameras)]:
            if not (directory / name).is_file():
                missing.append(str((directory / name).relative_to(source)))
    if missing:
        raise ValueError(
            "Required videos/previews missing; no conversion started: " + ", ".join(missing)
        )
    return len(episodes)


def migrate(source, output):
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Output already exists: {output}; choose a new --output. Nothing overwritten."
        )
    if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        raise ValueError("Output must be separate from the source")
    with _read_lock(source):
        count = _preflight(source)
        checker = PreviewSource(source)
        checker.progress = Progress("Check source", count)
        checker._run_unlocked()
        if not checker.report()["valid"] or not checker.num_episodes:
            raise ValueError(f"Source requires repair: {checker.report()['issues']}")
        checker.video_paths = {}
        checker.video_hashes = {}
        manifest = []
        progress = Progress("Check videos", checker.num_episodes)
        for index in range(checker.num_episodes):
            episode = source / "episodes" / f"episode_{index:06d}"
            if checker.info["image_format"] == VIDEO_FORMAT:
                directory = episode / "videos"
            else:
                directory = source / "previews" / episode.name
                if not (directory / "manifest.json").is_file():
                    raise ValueError(
                        f"Verified preview missing: {directory}; no encoding was started"
                    )
                verified = check_preview(episode, directory, checker.info["fps"], decode=False)
            summary = json.loads((episode / "episode.json").read_text())
            checker.video_paths[index] = {}
            checker.video_hashes[index] = {}
            for camera in checker.cameras:
                path = directory / f"{camera}.mp4"
                actual = inspect_video(path)
                if actual != {
                    "frames": summary["length"],
                    "fps": checker.info["fps"],
                    "shape": summary["image_shapes"][camera],
                }:
                    raise ValueError(f"Video does not match recorded frames/fps/dimensions: {path}")
                checker.video_paths[index][camera] = path
                digest = (
                    checker.videos[path][0]
                    if checker.info["image_format"] == VIDEO_FORMAT
                    else verified["videos"][camera]["sha256"]
                )
                checker.video_hashes[index][camera] = digest
                manifest.append(
                    {
                        "episode": index,
                        "camera": camera,
                        "source": str(path.relative_to(source)),
                        "sha256": digest,
                    }
                )
            progress.update(index + 1)
        checker.source_info = deepcopy(checker.info)
        checker.info = {**checker.info, "image_format": VIDEO_FORMAT, "image_color": "rgb"}
        # Numeric statistics use every recorded row. Image statistics sample
        # at most 32 existing video frames per episode/camera; no re-encoding.
        checker.image_sample_limit = 32
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f"{output.name}.pending-", dir=output.parent))
        try:
            selection = StateSelection(checker.info)
            checker.export_progress = Progress("Export", checker.num_episodes).update
            _write_dataset(source, stage, checker, selection)
            print("[MIGRATE] Verifying output", flush=True)
            _validate_export(source, stage, checker, selection)
            _write_json(
                stage / "meta/migration.json",
                {
                    "source": str(source),
                    "operation": "reuse_mp4_previews",
                    "videos": manifest,
                    "media": "byte-for-byte copies; no encoding",
                    "numeric_statistics": "all rows; float64 centered moments and exact quantiles",
                    "image_statistics": "up to 32 evenly spaced decoded frames per episode/camera",
                    "pairing": (
                        "unchanged row order, actions, feedback masks and capture timestamps"
                    ),
                },
            )
            (stage / "recording.lock").touch()
            report = V3Checker(stage).run()
            if not report["valid"]:
                raise ValueError(f"Output failed validation: {report['issues']}")
            if output.exists() or output.is_symlink():
                raise FileExistsError(
                    f"Output already exists: {output}; choose a new --output. Nothing overwritten."
                )
            stage.rename(output)
            print(f"Dataset saved at {output}", flush=True)
            return report
        except BaseException as exc:
            raise RuntimeError(
                f"Migration unfinished; source unchanged; partial output: {stage}: {exc}"
            ) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        migrate(args.source, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Migration failed: {exc}\n")


if __name__ == "__main__":
    main()
