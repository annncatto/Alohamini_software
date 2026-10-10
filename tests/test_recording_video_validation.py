"""Recording checks video headers; full video validation remains an explicit operation."""

import contextlib
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from test_recording_storage import add_episode, create, metadata

from alohamini.datasets.record import LocalDataset
from alohamini.datasets.tools import check_dataset
from alohamini.datasets.video import inspect_video


@pytest.mark.parametrize(
    "cameras", [(), ("forward",), ("left", "right"), ("forward", "left", "right")]
)
def test_recording_avoids_deep_video_verification_and_explicit_check_still_works(tmp_path, cameras):
    with (
        patch("alohamini.datasets.video.inspect_video", wraps=inspect_video) as inspect,
        patch("concurrent.futures.ThreadPoolExecutor", wraps=ThreadPoolExecutor) as pool,
    ):
        root = create(tmp_path / "record", episodes=1, cameras=cameras)
    if cameras:
        pool.assert_called_once_with(max_workers=len(cameras), thread_name_prefix="AlohaMiniVideo")
    else:
        pool.assert_not_called()
    assert inspect.call_count == len(cameras)
    assert all(not call.kwargs.get("decode", False) for call in inspect.call_args_list)
    report = check_dataset(root, decode_images=True, decode_videos=True)
    assert report["valid"], report


def test_header_mismatch_still_prevents_publishing_and_retains_source_images(tmp_path):
    root = tmp_path / "record"

    def wrong_frame_count(path, **kwargs):
        header = inspect_video(path, **kwargs)
        return {**header, "frames": header["frames"] + 1}

    with contextlib.closing(
        LocalDataset(root, fps=30, task="pick", robot_metadata=metadata())
    ) as dataset:
        add_episode(dataset)
        with patch("alohamini.datasets.video.inspect_video", side_effect=wrong_frame_count):
            with pytest.raises(ValueError, match="Encoded video does not match recording"):
                dataset.save_episode()
        assert dataset.num_episodes == 0
        assert list((root / ".recording").rglob("*.jpg"))


def test_worker_counts_preserve_data_statistics_and_report_wall_time(tmp_path):
    from test_dataset_tools import hashes

    saved = []
    for workers in (None, 2, 3):
        root = tmp_path / f"workers-{workers}"
        with contextlib.closing(
            LocalDataset(
                root,
                fps=30,
                task="pick",
                robot_metadata=metadata(("forward", "left", "right")),
                video_encoding_workers=workers,
            )
        ) as dataset:
            add_episode(dataset)
            dataset.save_episode()
            timings = dataset.last_save_timings
            assert set(timings) == {
                "queue_drain",
                "journal_to_parquet",
                "video_encode_headers_hash",
                "video_index",
                "temporary_cleanup",
                "image_decode",
                "image_statistics",
                "statistics_update",
                "episode_statistics",
                "global_statistics",
                "v3_data_metadata",
                "v3_publish",
                "cleanup",
                "total",
            }
            assert all(value >= 0 for value in timings.values())
            assert (
                sum(value for key, value in timings.items() if key != "total") <= timings["total"]
            )
            saved.append((hashes(root), json.loads((root / "meta/stats.json").read_text())))
    left = saved[0]
    for right in saved[1:]:
        for name, digest in left[0].items():
            if name.startswith(("data/", "videos/", "meta/episodes/", "meta/safety/")):
                assert right[0][name] == digest
        assert left[1] == right[1]


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_count_does_not_create_recording(tmp_path, workers):
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match="positive integer"):
        LocalDataset(
            root, fps=30, task="pick", robot_metadata=metadata(), video_encoding_workers=workers
        )
    assert not root.exists()
