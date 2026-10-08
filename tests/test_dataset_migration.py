"""Historical previews are reused without image or video re-encoding."""

import contextlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_dataset import as_png_v1, jpeg
from test_dataset_tools import hashes
from test_recording_storage import add_episode, metadata, rows

from alohamini.datasets.images import IMAGE_COLOR, IMAGE_FORMAT, ImageShards
from alohamini.datasets.record import _EpisodeWriter, dataset_schema
from alohamini.datasets.tools import check_dataset
from alohamini.datasets.video import generate_previews
from alohamini.learning.data import AlohaMiniDataset

spec = importlib.util.spec_from_file_location(
    "migrate_dataset_v3", Path(__file__).resolve().parents[1] / "scripts/migrate_dataset_v3.py"
)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def historical(root, image_format="tar", count=3):
    with contextlib.closing(
        _EpisodeWriter(root, fps=30, task="pick", robot_metadata=metadata(("forward", "wrist")))
    ) as dataset:
        for _ in range(2):
            add_episode(dataset, count)
            dataset.save_episode()
    info = as_png_v1(root)
    if image_format == "tar":
        info.update(version=2, image_format=IMAGE_FORMAT, image_color=IMAGE_COLOR)
        for episode in sorted((root / "episodes").iterdir()):
            values = pq.read_table(episode / "frames.parquet").to_pylist()
            with ImageShards(episode) as writer:
                for row in values:
                    for camera in info["robot_metadata"]["cameras"]:
                        row[f"observation.images.{camera}"] = writer.append(
                            camera, row["frame_index"], jpeg((240, 20, 10), (32, 32, 3))
                        )
            pq.write_table(
                pa.Table.from_pylist(
                    values,
                    schema=dataset_schema(
                        info["features"], info["robot_metadata"]["cameras"], IMAGE_FORMAT
                    ),
                ),
                episode / "frames.parquet",
            )
        (root / "meta/info.json").write_text(json.dumps(info))
    generate_previews(root)
    return root


@pytest.mark.parametrize("image_format", ["tar", "png"])
def test_migration_reuses_previews_and_preserves_samples(tmp_path, image_format):
    source = historical(tmp_path / "source", image_format)
    before = hashes(source)
    output = tmp_path / "v3"
    with (
        patch(
            "alohamini.datasets.video._encode_frames", side_effect=AssertionError("Video encode")
        ),
        patch("PIL.Image.Image.save", side_effect=AssertionError("Image encode")),
        patch(
            "alohamini.datasets.lerobotv3._export_image", side_effect=AssertionError("Raw images")
        ),
    ):
        report = migration.migrate(source, output)
    assert report["valid"], report
    assert hashes(source) == before
    assert check_dataset(output, decode_images=True, decode_videos=True)["valid"]
    assert not list(output.rglob("*.tar")) and not list(output.rglob("*.png"))
    converted = rows(output)
    offset = 0
    for index in range(2):
        episode = source / "episodes" / f"episode_{index:06d}"
        original = pq.read_table(episode / "frames.parquet").to_pylist()
        for row, actual in zip(original, converted[offset:], strict=False):
            for key, value in row.items():
                if key != "task" and not key.startswith("observation.images."):
                    assert actual[key] == value, key
        offset += len(original)
        assert (episode / "safety.jsonl").read_bytes() == (
            output / "meta/safety" / f"episode_{index:06d}.jsonl"
        ).read_bytes()
        for camera in ("forward", "wrist"):
            assert (source / "previews" / episode.name / f"{camera}.mp4").read_bytes() == (
                output
                / "videos"
                / f"observation.images.{camera}"
                / "chunk-000"
                / f"file-{index:03d}.mp4"
            ).read_bytes()
    samples = AlohaMiniDataset(
        output,
        episodes=[0, 1],
        state="joint_velocity,joint_current,base_velocity,lift_height",
        chunk_size=3,
        image_size=(32, 32),
    )
    assert len(samples) == 6
    assert samples[0]["observation.state"].shape == (32,)
    assert samples[0]["observation.images.forward"][0].mean() > 0.8
    assert samples[2]["action_is_pad"].tolist() == [False, True, True]
    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["action"]["count"] == [6]
    assert stats["action"]["mean"] == [1.0] * 18


@pytest.mark.parametrize("defect", ["missing", "stale", "video_hash"])
def test_migration_refuses_unverified_previews_before_creating_output(tmp_path, defect):
    source = historical(tmp_path / "source")
    manifest_path = source / "previews/episode_000000/manifest.json"
    if defect == "missing":
        manifest_path.unlink()
    else:
        manifest = json.loads(manifest_path.read_text())
        if defect == "stale":
            manifest["source_sha256"]["frames.parquet"] = "0" * 64
        else:
            manifest["videos"]["forward"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    before = hashes(source)
    output = tmp_path / "v3"
    with pytest.raises(ValueError):
        migration.migrate(source, output)
    assert not output.exists() and not list(tmp_path.glob("v3.pending-*"))
    assert hashes(source) == before


def test_image_statistics_decoding_is_bounded(tmp_path):
    source = historical(tmp_path / "source", count=40)
    from alohamini.datasets.lerobotv3 import _sample_video_frames

    sampled = []

    def collect(container, indices):
        for image in _sample_video_frames(container, indices):
            sampled.append(image.shape)
            yield image

    with patch("alohamini.datasets.lerobotv3._sample_video_frames", side_effect=collect) as decode:
        migration.migrate(source, tmp_path / "v3")
    assert decode.call_count == 2 * 2  # One stream per episode/camera, not one per sample.
    assert len(sampled) == 32 * 2 * 2
    data = next((tmp_path / "v3/data").rglob("*.parquet"))
    assert pq.ParquetFile(data).num_row_groups == 2
    stats = json.loads((tmp_path / "v3/meta/stats.json").read_text())
    np.testing.assert_allclose(stats["action"]["mean"], [19.5] * 18)
    np.testing.assert_allclose(stats["action"]["std"], [np.arange(40).std()] * 18)
    np.testing.assert_allclose(stats["action"]["q01"], [0.0] * 18)


def test_sequential_samples_match_previous_seek_reader(tmp_path):
    from alohamini.datasets.images import video_rgb
    from alohamini.datasets.lerobotv3 import _sample_video_frames

    source = historical(tmp_path / "source", "png", count=40)
    path = source / "previews/episode_000000/forward.mp4"
    indices = [0, 5, 13, 22, 39]
    with av.open(str(path)) as container:
        actual = list(_sample_video_frames(container, indices))
    assert len(actual) == len(indices)
    for index, image in zip(indices, actual, strict=True):
        np.testing.assert_array_equal(image, video_rgb(path, index))
    with av.open(str(path)) as container, pytest.raises(ValueError, match="Missing video frame 40"):
        list(_sample_video_frames(container, [0, 40]))


def test_missing_preview_fails_before_source_scan(tmp_path):
    source = historical(tmp_path / "source")
    (source / "previews/episode_000001/wrist.mp4").unlink()
    with patch.object(
        migration.PreviewSource, "_run_unlocked", side_effect=AssertionError("Scanned")
    ):
        with pytest.raises(ValueError, match="Required videos/previews missing"):
            migration.migrate(source, tmp_path / "v3")


def test_output_video_hash_is_still_checked(tmp_path):
    source = historical(tmp_path / "source")
    write = migration._write_dataset

    def corrupt(source, stage, checker, selection):
        result = write(source, stage, checker, selection)
        path = next((stage / "videos").rglob("*.mp4"))
        with path.open("ab") as stream:
            stream.write(b"corrupt")
        return result

    with patch.object(migration, "_write_dataset", side_effect=corrupt):
        with pytest.raises(RuntimeError, match="Exported video changed"):
            migration.migrate(source, tmp_path / "v3")
    assert not (tmp_path / "v3").exists()


@pytest.mark.skipif(not os.environ.get("ALOHAMINI_LEROBOT_REFERENCE"), reason="Fork not selected")
def test_fork_reads_migrated_video_and_extra_feedback_offline(tmp_path):
    source = historical(tmp_path / "source")
    output = tmp_path / "v3"
    migration.migrate(source, output)
    code = """
import os, socket, sys
from pathlib import Path
socket.socket.connect = lambda *_: (_ for _ in ()).throw(AssertionError('network access'))
import lerobot
assert Path(lerobot.__file__).is_relative_to(Path(os.environ['ALOHAMINI_LEROBOT_REFERENCE']))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('local/migrated', root=sys.argv[1], video_backend='pyav',
                        delta_timestamps={'action': [0, 1/30, 2/30]})
assert len(dataset) == 6
item = dataset[2]
assert item['observation.state'].shape == (18,)
assert item['observation.motor_velocity_raw'].shape == (18,)
assert item['motor_feedback.current_ma_valid'].shape == (18,)
assert item['action'].shape == (3,18)
assert item['action_is_pad'].tolist() == [False, True, True]
assert item['observation.images.forward'].shape == (3,32,32)
assert item['observation.images.forward'][0].mean() > .8
assert item['task'] == 'pick'
"""
    result = subprocess.run(
        [os.environ["ALOHAMINI_LEROBOT_PYTHON"], "-c", code, str(output)],
        env={
            **os.environ,
            "PYTHONPATH": os.environ["ALOHAMINI_LEROBOT_REFERENCE"],
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_DATASETS_CACHE": str(tmp_path / "cache"),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
