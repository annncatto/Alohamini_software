"""Shared-shard reads preserve physical image offsets and requested episode order."""

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from alohamini.learning.lerobot import LeRobotSource


@pytest.fixture
def source(tmp_path):
    (tmp_path / "meta/episodes").mkdir(parents=True)
    (tmp_path / "meta/safety").mkdir()
    features = {
        "action": {"dtype": "float32", "shape": [1]},
        "observation.images.forward": {"dtype": "image"},
    }
    info = {
        "features": features,
        "fps": 30,
        "robot_type": "alohamini2pro",
        "data_path": "data_{file_index}.parquet",
    }
    original = {"features": features, "fps": 30, "robot_metadata": {"robot_model": "alohamini2pro"}}
    (tmp_path / "meta/alohamini.json").write_text(json.dumps({"source_info": original}))
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame({"task_index": [0]}, index=["pick"])),
        tmp_path / "meta/tasks.parquet",
    )
    metadata = []
    for ep in range(4):
        metadata.append(
            {"episode_index": ep, "data/chunk_index": 0, "data/file_index": ep % 2, "length": 2}
        )
        (tmp_path / f"meta/safety/episode_{ep:06d}.jsonl").write_text("")
    pq.write_table(pa.Table.from_pylist(metadata), tmp_path / "meta/episodes/episodes.parquet")
    for shard in range(2):
        rows = []
        for frame in range(2):
            for ep in (shard, shard + 2):
                value = ep * 20 + frame
                image = io.BytesIO()
                Image.fromarray(np.full((2, 3, 3), value, np.uint8)).save(image, format="PNG")
                rows.append(
                    {
                        "episode_index": ep,
                        "frame_index": frame,
                        "action": [value],
                        "task_index": 0,
                        "observation.images.forward": {"bytes": image.getvalue()},
                    }
                )
        pq.write_table(
            pa.Table.from_pylist(rows), tmp_path / f"data_{shard}.parquet", row_group_size=3
        )
    return LeRobotSource(tmp_path, info)


def test_bulk_reads_each_shard_once_and_keeps_image_offsets(source):
    opened = []
    parquet = pq.ParquetFile

    def tracked(path):
        file = parquet(path)
        spy = MagicMock(wraps=file)
        spy.num_row_groups = file.num_row_groups
        spy.__enter__.return_value = spy
        spy.__exit__.side_effect = lambda *args: file.close()
        opened.append(spy)
        return spy

    # Alternate physical shards and request reverse episode order.
    with patch("alohamini.learning.lerobot.pq.ParquetFile", side_effect=tracked):
        results = list(source.read_episodes([3, 2, 1, 0], ["action", "task"]))
    assert len(opened) == 2
    assert [file.read_row_group.call_count for file in opened] == [2, 2]
    for episode, (rows, safety, paths) in zip([3, 2, 1, 0], results, strict=True):
        assert [row["action"] for row in rows] == [[episode * 20], [episode * 20 + 1]]
        assert [row["task"] for row in rows] == ["pick", "pick"]
        assert safety.name == f"episode_{episode:06d}.jsonl"
        assert source.root / "meta/tasks.parquet" in paths
        for frame, row in enumerate(rows):
            np.testing.assert_array_equal(
                source.image(row["observation.images.forward"], "forward"),
                np.full((2, 3, 3), episode * 20 + frame, np.uint8),
            )


def test_duplicate_requests_own_rows_and_new_traversal_reads_fresh_files(source):
    results = source.read_episodes([0, 0], ["action"])
    first = next(results)[0]
    first[0]["action"][0] = -1
    assert next(results)[0][0]["action"] == [0]
    path = source.root / "data_0.parquet"
    table = pq.read_table(path).to_pylist()
    table[0]["action"] = [99]
    pq.write_table(pa.Table.from_pylist(table), path)
    assert source.read_episode(0, ["action"])[0][0]["action"] == [99]


def test_bulk_rejects_missing_sidecar_and_reordered_frames(source):
    safety = source.root / "meta/safety/episode_000002.jsonl"
    safety.unlink()
    with pytest.raises(ValueError, match="sidecar"):
        list(source.read_episodes([0, 2], ["action"]))
    safety.touch()
    path = source.root / "data_0.parquet"
    table = pq.read_table(path).to_pylist()
    table[0]["frame_index"] = 1
    with pytest.raises(ValueError, match="row order or boundary"):
        pq.write_table(pa.Table.from_pylist(table), path)
        list(source.read_episodes([0, 2], ["action"]))
