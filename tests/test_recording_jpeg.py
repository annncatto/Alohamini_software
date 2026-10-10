"""Raw Host JPEG staging preserves bytes, RGB, final videos and recovery."""

import contextlib
import io
import json
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from test_dataset import frame, jpeg
from test_dataset_tools import hashes
from test_recording_storage import metadata, rows

from alohamini.datasets.images import decode_host_image, image_png, image_rgb
from alohamini.datasets.lerobotv3 import recover_recording
from alohamini.datasets.record import LocalDataset
from alohamini.datasets.tools import check_dataset


def populate(dataset, encoded):
    dataset.begin_episode()
    for i in range(3):
        values = frame(dataset)
        values["action"][:] = i
        assert dataset.add_frame(values, {c: encoded for c in dataset.cameras}, {})
    dataset._finish_writer()
    dataset.check_writer()


def convert_pending_to_legacy_png(dataset):
    """Reproduce the previous writer's bytes and journal without changing final metadata."""
    pending = dataset._pending
    journal = pending / "journal.jsonl"
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    for entry in entries:
        if "frame" not in entry:
            continue
        for camera in dataset.cameras:
            key = f"observation.images.{camera}"
            old = pending / entry["frame"][key]
            new = old.with_suffix(".png")
            Image.fromarray(decode_host_image(old.read_bytes())).save(new, compress_level=1)
            old.unlink()
            entry["frame"][key] = str(new.relative_to(pending))
    journal.write_text("".join(json.dumps(entry) + "\n" for entry in entries))


def test_capture_retains_wire_bytes_without_pixel_decode_or_encode(tmp_path):
    encoded = jpeg((240, 35, 10), (32, 32, 3))
    root = tmp_path / "record"
    with contextlib.closing(
        LocalDataset(root, fps=30, task="pick", robot_metadata=metadata())
    ) as dataset:
        with (
            patch("cv2.imdecode", side_effect=AssertionError("Capture decoded JPEG")),
            patch("PIL.Image.Image.load", side_effect=AssertionError("Capture loaded pixels")),
            patch("PIL.Image.Image.save", side_effect=AssertionError("Capture re-encoded image")),
        ):
            populate(dataset, encoded)
        assert not list(dataset._pending.rglob("*.png"))
        paths = sorted(dataset._pending.rglob("*.jpg"))
        assert len(paths) == 3 and all(path.read_bytes() == encoded for path in paths)
        reference = str(paths[0].relative_to(dataset._pending))
        expected = decode_host_image(encoded)
        np.testing.assert_array_equal(image_rgb(dataset._pending, "forward", reference), expected)
        with Image.open(io.BytesIO(image_png(dataset._pending, "forward", reference))) as png:
            assert png.format == "PNG"
            np.testing.assert_array_equal(np.asarray(png), expected)
        dataset.save_episode()
    assert check_dataset(root, decode_videos=True)["valid"]


def test_jpeg_and_legacy_png_produce_identical_videos_rows_and_statistics(tmp_path):
    encoded = jpeg((240, 35, 10), (32, 32, 3))
    saved = []
    for legacy in (False, True):
        root = tmp_path / str(legacy)
        with contextlib.closing(
            LocalDataset(root, fps=30, task="pick", robot_metadata=metadata(("left", "right")))
        ) as dataset:
            populate(dataset, encoded)
            if legacy:
                convert_pending_to_legacy_png(dataset)
            dataset.save_episode()
        saved.append(root)
    left, right = saved
    assert rows(left) == rows(right)
    a, b = hashes(left), hashes(right)
    assert {k: v for k, v in a.items() if k.startswith("videos/")} == {
        k: v for k, v in b.items() if k.startswith("videos/")
    }
    assert json.loads((left / "meta/stats.json").read_text()) == json.loads(
        (right / "meta/stats.json").read_text()
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_interrupted_jpeg_and_legacy_png_recordings_recover(tmp_path, legacy):
    root = tmp_path / "record"
    dataset = LocalDataset(root, fps=30, task="pick", robot_metadata=metadata())
    populate(dataset, jpeg())
    if legacy:
        convert_pending_to_legacy_png(dataset)
    with patch("alohamini.datasets.video._encode_frames", side_effect=OSError("interrupted")):
        with pytest.raises(OSError):
            dataset.save_episode()
    dataset.close()
    before = hashes(root)
    report = recover_recording(root, tmp_path / "recovered")
    assert report["valid"], report
    assert len(rows(tmp_path / "recovered")) == 3
    assert hashes(root) == before
