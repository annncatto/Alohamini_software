import multiprocessing
import os
import pickle
import shutil

import av
import numpy as np
import pytest

from alohamini.datasets.images import video_rgb
from alohamini.datasets.video import _encode_frames
from alohamini.datasets.video_reader import VideoFrameCache


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "video.mp4"
    rng = np.random.default_rng(1)
    _encode_frames(
        (
            av.VideoFrame.from_ndarray(
                rng.integers(0, 256, (48, 64, 3), dtype=np.uint8), format="rgb24"
            )
            for _ in range(70)
        ),
        path,
        30,
        (48, 64, 3),
        options={"g": "12", "bf": "2"},
    )
    return path


def test_exact_sequential_random_repeated_frames_and_bounded_eviction(video, tmp_path):
    cache = VideoFrameCache(2)
    indices = [0, 1, 1, 2, 11, 12, 35, 69, 4, 3, 34, 68, 0]
    for target in indices:
        expected = video_rgb(video, target)
        actual = cache.read(video, target)
        np.testing.assert_array_equal(actual, expected)
        actual[:] = 0  # Caller mutation must not corrupt the cached frame.
        np.testing.assert_array_equal(cache.read(video, target), expected)
    assert cache.misses == 1 and cache.hits == len(indices) * 2 - 1
    for i in range(6):
        path = tmp_path / f"copy-{i}.mp4"
        shutil.copyfile(video, path)
        cache.read(path, 0)
        assert len(cache.entries) <= 2
    cache.close()
    assert not cache.entries


def test_errors_invalidate_reader_and_zero_cache_matches_reference(video, tmp_path):
    cache = VideoFrameCache(1)
    with pytest.raises(ValueError, match="Missing video frame 70"):
        cache.read(video, 70)
    assert not cache.entries
    np.testing.assert_array_equal(cache.read(video, 2), VideoFrameCache(0).read(video, 2))
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"broken")
    with pytest.raises(av.error.InvalidDataError):
        cache.read(broken, 0)
    assert not cache.entries
    with pytest.raises(ValueError, match="nonnegative"):
        cache.read(video, -1)
    cache.close()


def _child_read(cache, path, queue):
    image = cache.read(path, 4)
    queue.put((cache.pid, cache.misses, int(image.sum())))
    cache.close()


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_worker_never_reuses_parent_decoder(video, method):
    cache = VideoFrameCache(2)
    expected = cache.read(video, 4)
    copy = pickle.loads(pickle.dumps(cache))
    assert not copy.entries
    ctx = multiprocessing.get_context(method)
    queue = ctx.Queue()
    process = ctx.Process(target=_child_read, args=(cache, video, queue))
    process.start()
    try:
        pid, misses, total = queue.get(timeout=30)
        process.join(timeout=30)
        assert process.exitcode == 0
        assert pid != os.getpid() and misses == 1 and total == int(expected.sum())
        np.testing.assert_array_equal(cache.read(video, 4), expected)
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        queue.close()
        cache.close()


def test_replaced_file_reopens_decoder(video, tmp_path):
    cache = VideoFrameCache(1)
    cache.read(video, 0)
    replacement = tmp_path / "replacement.mp4"
    shutil.copyfile(video, replacement)
    replacement.replace(video)
    np.testing.assert_array_equal(cache.read(video, 0), video_rgb(video, 0))
    assert cache.misses == 2
    cache.close()
