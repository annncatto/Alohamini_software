"""Read local AlohaMini v3 exports without LeRobot or a Hub connection."""

import io
import json
from copy import deepcopy

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from alohamini.datasets.lerobot_tools import _dataset_path


class LeRobotSource:
    """Storage adapter only; filtering, action chunks and normalization stay shared."""

    def __init__(self, root, info):
        self.root, self.storage_info = root, info
        path = root / "meta/alohamini.json"
        if not path.is_file():
            raise ValueError(
                "Native training requires an AlohaMini v3 export "
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
            info["features"][f"observation.images.{c}"]["dtype"] != "image" for c in self.cameras
        ):
            raise ValueError("Native v3 training currently requires embedded-image cameras")
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
        meta = self.episodes[episode]
        tasks = None
        if "task" in columns:
            tasks_path = self.root / "meta/tasks.parquet"
            tasks_table = pq.read_table(tasks_path).to_pandas()
            tasks = dict(zip(tasks_table["task_index"], tasks_table.index, strict=True))
            if tasks_path not in self.metadata_paths:
                self.metadata_paths.append(tasks_path)
            columns = ["task_index" if key == "task" else key for key in columns]
        path = _dataset_path(
            self.root,
            self.storage_info["data_path"],
            chunk_index=meta["data/chunk_index"],
            file_index=meta["data/file_index"],
        )
        safety = self.root / "meta/safety" / f"episode_{episode:06d}.jsonl"
        if not safety.is_file():
            raise ValueError(f"Episode {episode}: missing original safety/timing sidecar")
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
        rows = []
        file = pq.ParquetFile(path)
        for group in range(file.num_row_groups):
            for index, row in enumerate(file.read_row_group(group, columns=numeric).to_pylist()):
                if row["episode_index"] != episode:
                    continue
                if tasks is not None:
                    row["task"] = tasks[row["task_index"]]
                for camera in self.cameras:
                    row[f"observation.images.{camera}"] = (str(path), group, index)
                rows.append(row)
        if len(rows) != meta["length"] or [r["frame_index"] for r in rows] != list(
            range(len(rows))
        ):
            raise ValueError(f"Episode {episode}: v3 row order or boundary mismatch")
        return rows, safety, [*self.metadata_paths, path, safety]

    def image(self, reference, camera):
        path, group, index = reference
        cache_key = (path, group)
        if cache_key != self._image_cache_key:
            columns = [f"observation.images.{c}" for c in self.cameras]
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
