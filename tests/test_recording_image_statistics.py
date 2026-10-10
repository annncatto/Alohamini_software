"""Recording RGB summaries use Host JPEG samples and survive resume without decoding."""

import contextlib
import json
from unittest.mock import patch

import av
import numpy as np
import pyarrow.parquet as pq
import pytest
from test_dataset import frame, jpeg
from test_recording_storage import add_episode, create, metadata

from alohamini.datasets.image_statistics import ImageStatistics, sample_indices
from alohamini.datasets.images import decode_host_image
from alohamini.datasets.lerobotv3 import recording_statistics
from alohamini.datasets.record import LocalDataset


@pytest.mark.parametrize("shape", [(1, 1, 3), (20, 30, 3), (300, 450, 3)])
def test_histogram_statistics_match_numpy_after_merge_and_restore(shape):
    rng = np.random.default_rng(17)
    images = [rng.integers(0, 256, shape, dtype=np.uint8) for _ in range(3)]
    combined = ImageStatistics()
    for image in images:
        single = ImageStatistics()
        single.update("rgb", image, source="pre_encoding_png")
        combined.merge(ImageStatistics(json.loads(json.dumps(single.payload()))))
    stride = max(shape[:2]) // 150 if max(shape[:2]) >= 300 else 1
    pixels = np.concatenate([x[::stride, ::stride].reshape(-1, 3) for x in images]) / 255
    actual = combined.result()["rgb"]
    for name, expected in {
        "mean": pixels.mean(axis=0),
        "std": pixels.std(axis=0),
        "min": pixels.min(axis=0),
        "max": pixels.max(axis=0),
        **{
            f"q{int(q * 100):02d}": np.quantile(pixels, q, axis=0)
            for q in (0.01, 0.1, 0.5, 0.9, 0.99)
        },
    }.items():
        np.testing.assert_allclose(actual[name].reshape(3), expected, atol=1e-12)
    assert actual["count"].tolist() == [3]


class HeaderOnlyInput:
    def __init__(self, container):
        self.container = container

    def __getattr__(self, name):
        return getattr(self.container, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.container.close()

    def decode(self, *args, **kwargs):
        raise AssertionError("Saving must not decode MP4 for statistics")


def test_jpeg_sampling_preserves_all_rows_without_decoding_video(tmp_path):
    root = tmp_path / "record"
    pixels = []
    original_open = av.open

    def headers_only(*args, **kwargs):
        container = original_open(*args, **kwargs)
        return (
            HeaderOnlyInput(container)
            if isinstance(container, av.container.InputContainer)
            else container
        )

    with contextlib.closing(
        LocalDataset(root, fps=30, task="pick", robot_metadata=metadata())
    ) as dataset:
        dataset.begin_episode()
        for i in range(120):
            values = frame(dataset)
            values["action"][:] = i
            encoded = jpeg((i * 2, i, 255 - i), (32, 32, 3))
            pixels.append(decode_host_image(encoded).reshape(-1, 3))
            assert dataset.add_frame(values, {"forward": encoded}, {})
            dataset._queue.join()
        with patch("av.open", side_effect=headers_only):
            dataset.save_episode()
        assert dataset.last_save_timings["image_decode"] == 0
    selected = sample_indices(120)
    assert len(selected) == 100 and selected[0] == 0 and selected[-1] == 119
    rgb = np.concatenate([pixels[i] for i in selected]) / 255
    stats = json.loads((root / "meta/stats.json").read_text())
    np.testing.assert_allclose(
        np.array(stats["observation.images.forward"]["mean"]).reshape(3), rgb.mean(axis=0)
    )
    assert stats["observation.images.forward"]["count"] == [100]
    assert stats["action"]["count"] == [120]
    np.testing.assert_allclose(stats["action"]["q99"], np.quantile(np.arange(120), 0.99))
    assert sum(pq.read_metadata(p).num_rows for p in (root / "data").rglob("*.parquet")) == 120
    with patch("av.open", side_effect=AssertionError("Histogram restore must not open videos")):
        restored = recording_statistics(root).result()
    for key, values in stats.items():
        for name, expected in values.items():
            np.testing.assert_allclose(restored[key][name], expected, atol=1e-12)


def test_resume_matches_uninterrupted_statistics(tmp_path):
    uninterrupted = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    with contextlib.closing(
        LocalDataset(uninterrupted, fps=30, task="pick", robot_metadata=metadata())
    ) as dataset:
        for count in (3, 5):
            add_episode(dataset, count)
            dataset.save_episode()
    create(resumed, episodes=1, cameras=("forward",))
    with contextlib.closing(
        LocalDataset(resumed, fps=30, task="pick", robot_metadata=metadata(), resume=True)
    ) as dataset:
        add_episode(dataset, 5)
        dataset.save_episode()
    expected = json.loads((uninterrupted / "meta/stats.json").read_text())
    actual = json.loads((resumed / "meta/stats.json").read_text())
    for key, values in expected.items():
        for name, value in values.items():
            np.testing.assert_allclose(actual[key][name], value, atol=1e-12)


def test_legacy_recording_without_histogram_can_resume(tmp_path):
    root = create(tmp_path / "legacy", episodes=1, cameras=("forward",))
    path = next((root / "meta/episodes").rglob("*.parquet"))
    table = pq.read_table(path)
    pq.write_table(table.drop(["image_statistics"]), path)
    before = path.read_bytes()
    with contextlib.closing(
        LocalDataset(root, fps=30, task="pick", robot_metadata=metadata(), resume=True)
    ) as dataset:
        add_episode(dataset)
        dataset.save_episode()
    assert path.read_bytes() == before
    summary = json.loads((root / "meta/alohamini.json").read_text())
    assert summary["image_statistics"]["sources"] == ["host_jpeg", "legacy_decoded_video"]
    stats = json.loads((root / "meta/stats.json").read_text())
    assert stats["action"]["count"] == [6]
    assert stats["observation.images.forward"]["count"] == [6]
