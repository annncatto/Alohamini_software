"""Read local AlohaMini v3 exports without LeRobot or a Hub connection."""

import io
import json
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from alohamini.datasets.images import video_rgb
from alohamini.datasets.lerobot_tools import _dataset_path


class LeRobotSource:
    """Storage adapter only; episode windows and normalization stay shared."""

    def __init__(self, root, info):
        self.root, self.storage_info = root, info
        path = root / "meta/alohamini.json"
        if not path.is_file():
            raise ValueError(
                "Training requires a LeRobot v3 dataset exported by AlohaMini "
                "with calibration and safety metadata"
            )
        metadata = json.loads(path.read_text())
        source_info = metadata["source_info"]
        self.original_state = source_info["features"].get("observation.state")
        if (
            source_info["fps"] != info["fps"]
            or source_info["features"]["action"] != info["features"].get("action")
            or source_info["robot_metadata"]["robot_model"] != info["robot_type"]
        ):
            raise ValueError(
                "V3 action, FPS or robot coordinates differ from the original recording"
            )
        self.cameras = [
            k.removeprefix("observation.images.")
            for k in info["features"]
            if k.startswith("observation.images.")
        ]
        if any(
            info["features"][f"observation.images.{c}"]["dtype"] not in ("image", "video")
            for c in self.cameras
        ):
            raise ValueError("LeRobot v3 training requires image or video cameras")
        self.info = deepcopy(source_info)
        self.info.update(features=deepcopy(info["features"]), cameras=self.cameras)
        self.episodes = {}
        self.metadata_paths = [root / "meta/info.json", path]
        for path in sorted((root / "meta/episodes").rglob("*.parquet")):
            self.metadata_paths.append(path)
            for row in pq.read_table(path).to_pylist():
                self.episodes[row["episode_index"]] = row
        self._image_cache_key = self._image_cache = None

    def read_episode(self, episode, columns):
        return next(self.read_episodes([episode], columns))

    def read_episodes(self, episodes, columns):
        """Scan each selected physical shard once, yielding in requested episode order.

        Only selected numeric rows are materialized. Original row-group offsets
        are retained for lazy image reads; yielded rows are owned by the caller.
        Shard buffers are local to this traversal, never reused across file edits.
        """
        episodes = list(episodes)
        if not episodes:
            return
        tasks = None
        if "task" in columns:
            tasks_path = self.root / "meta/tasks.parquet"
            tasks_table = pq.read_table(tasks_path).to_pandas()
            tasks = dict(zip(tasks_table["task_index"], tasks_table.index, strict=True))
            if tasks_path not in self.metadata_paths:
                self.metadata_paths.append(tasks_path)
            columns = ["task_index" if key == "task" else key for key in columns]
        # Keep only numeric data and image locations in memory. Compressed images
        # are read on demand rather than loading an entire dataset into RAM.
        numeric = list(
            dict.fromkeys(
                [
                    "episode_index",
                    "frame_index",
                    *[k for k in columns if not k.startswith("observation.images.")],
                ]
            )
        )
        paths, selected = {}, defaultdict(set)
        for episode in episodes:
            meta = self.episodes[episode]
            path = _dataset_path(
                self.root,
                self.storage_info["data_path"],
                chunk_index=meta["data/chunk_index"],
                file_index=meta["data/file_index"],
            )
            paths[episode] = path
            selected[path].add(episode)
        remaining = Counter(episodes)
        buffered = {}
        for episode in episodes:
            path = paths[episode]
            safety = self.root / "meta/safety" / f"episode_{episode:06d}.jsonl"
            if not safety.is_file():
                raise ValueError(f"Episode {episode}: missing original safety/timing sidecar")
            if path not in buffered:
                partitions = {index: [] for index in selected[path]}
                with pq.ParquetFile(path) as file:
                    for group in range(file.num_row_groups):
                        table = file.read_row_group(group, columns=numeric)
                        indices = [
                            i
                            for i, index in enumerate(table["episode_index"].to_pylist())
                            if index in partitions
                        ]
                        if not indices:
                            continue
                        for index, row in zip(
                            indices, table.take(indices).to_pylist(), strict=True
                        ):
                            partitions[row["episode_index"]].append((group, index, row))
                buffered[path] = partitions
            remaining[episode] -= 1
            if remaining[episode]:
                located = deepcopy(buffered[path][episode])
            else:
                located = buffered[path].pop(episode)
                if not buffered[path]:
                    del buffered[path]
            yield self._episode_rows(episode, path, safety, located, tasks)

    def _episode_rows(self, episode, path, safety, located, tasks):
        meta = self.episodes[episode]
        video_refs = {}
        for camera in self.cameras:
            key = f"observation.images.{camera}"
            if self.storage_info["features"][key]["dtype"] == "video":
                video = _dataset_path(
                    self.root,
                    self.storage_info["video_path"],
                    video_key=key,
                    chunk_index=meta[f"videos/{key}/chunk_index"],
                    file_index=meta[f"videos/{key}/file_index"],
                )
                start = meta[f"videos/{key}/from_timestamp"] * self.storage_info["fps"]
                video_refs[camera] = (video, round(start))
        rows = []
        for group, index, row in located:
            if tasks is not None:
                row["task"] = tasks[row["task_index"]]
            for camera in self.cameras:
                if camera in video_refs:
                    video, first = video_refs[camera]
                    reference = (str(video), None, first + row["frame_index"])
                else:
                    reference = (str(path), group, index)
                row[f"observation.images.{camera}"] = reference
            rows.append(row)
        if len(rows) != meta["length"] or [r["frame_index"] for r in rows] != list(
            range(len(rows))
        ):
            raise ValueError(f"Episode {episode}: v3 row order or boundary mismatch")
        return (
            rows,
            safety,
            [*self.metadata_paths, path, safety, *(v[0] for v in video_refs.values())],
        )

    def image(self, reference, camera, *, video_reader=video_rgb):
        path, group, index = reference
        if group is None:
            return video_reader(path, index)
        cache_key = (path, group)
        if cache_key != self._image_cache_key:
            columns = [
                f"observation.images.{c}"
                for c in self.cameras
                if self.storage_info["features"][f"observation.images.{c}"]["dtype"] == "image"
            ]
            self._image_cache = pq.ParquetFile(path).read_row_group(group, columns=columns)
            self._image_cache_key = cache_key
        cell = self._image_cache[f"observation.images.{camera}"][index].as_py()
        source = (
            io.BytesIO(cell["bytes"])
            if cell.get("bytes") is not None
            else _dataset_path(self.root, cell["path"])
        )
        with Image.open(source) as image:
            return np.array(image.convert("RGB"), copy=True)
