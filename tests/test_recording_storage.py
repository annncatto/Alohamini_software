"""Complete v3 recording: fork RGB boundary, media, extra feedback and recovery."""

import contextlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest
from test_dataset import frame, jpeg
from test_dataset import metadata as uncalibrated_metadata
from test_dataset_tools import hashes

from alohamini.datasets.images import video_rgb
from alohamini.datasets.lerobotv3 import export_lerobot, recover_recording
from alohamini.datasets.record import LocalDataset, motor_feedback_frame
from alohamini.datasets.tools import check_dataset
from alohamini.learning.data import AlohaMiniDataset


def metadata(cameras=("forward",)):
    result = uncalibrated_metadata(cameras)
    for motor in result["motors"].values():
        motor.update(range_min=1000, range_max=3000, drive_mode=0)
    return result


def add_episode(dataset, count=3):
    dataset.begin_episode()
    for i in range(count):
        values = frame(dataset)
        values["action"][:] = i
        motors = {
            name: {"velocity_raw": 100 + i, "current_ma": 650 + i,
                   "sample_started_s": 100 + i / 30, "sample_finished_s": 100.001 + i / 30}
            for name in dataset.features["observation.motor_current_ma"]["names"]
        }
        values.update(motor_feedback_frame(dataset.features, {"version": 1, "motors": motors}))
        dataset.add_frame(values, {c: jpeg((240, 20 + i, 10), (32, 32, 3)) for c in dataset.cameras}, {
            "safety": {"feedback_valid": True},
            "host_timing": {"state_sample_monotonic_s": 100 + i / 30},
        })


def rows(root):
    return [r for p in sorted((root / "data").rglob("*.parquet")) for r in pq.read_table(p).to_pylist()]


def create(root, episodes=2, cameras=("forward", "wrist")):
    with contextlib.closing(LocalDataset(root, fps=30, task="pick", robot_metadata=metadata(cameras))) as dataset:
        for _ in range(episodes):
            add_episode(dataset)
            dataset.save_episode()
    return root


def test_recording_is_complete_v3_without_intermediate_dataset(tmp_path):
    root = create(tmp_path / "record")
    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert (info["total_episodes"], info["total_frames"]) == (2, 6)
    assert not (root / "episodes").exists()
    assert not (root / ".recording").exists()
    assert not (root / "previews").exists()
    assert not list(root.rglob("*.png")) and not list(root.rglob("*.tar"))
    assert len(list(root.rglob("*.mp4"))) == 4
    report = check_dataset(root, decode_images=True, decode_videos=True)
    assert report["valid"], report
    values = rows(root)
    assert [r["index"] for r in values] == list(range(6))
    assert [r["frame_index"] for r in values] == [0, 1, 2, 0, 1, 2]
    assert len(values[0]["observation.state"]) == len(values[0]["action"]) == 18
    assert len(values[0]["observation.motor_current_ma"]) == 18
    assert values[0]["motor_feedback.current_ma_valid"] == [1] * 18
    assert values[0]["motor_feedback.temperature_raw_valid"] == [0] * 18
    image = video_rgb(root / "videos/observation.images.forward/chunk-000/file-000.mp4", 1)
    assert image[0, 0, 0] > 220 and image[0, 0, 2] < 30
    samples = AlohaMiniDataset(root, episodes=[0], state="joint_velocity,joint_current,base_velocity,lift_height", chunk_size=3, image_size=(32, 32))
    assert samples[0]["observation.state"].shape == (32,)
    assert samples[0]["action"].shape == (3, 18)
    assert samples[2]["action_is_pad"].tolist() == [False, True, True]
    np.testing.assert_array_equal((samples[1]["observation.images.forward"].numpy() * 255).round().astype(np.uint8).transpose(1, 2, 0), image)


def test_resume_and_state_projection_keep_media_and_action(tmp_path):
    root = create(tmp_path / "record", episodes=1)
    previous = hashes(root)
    with contextlib.closing(LocalDataset(root, fps=30, task="pick", robot_metadata=metadata(("forward", "wrist")), resume=True)) as dataset:
        assert dataset.num_episodes == 1 and dataset.total_frames == 3
        add_episode(dataset, 2)
        dataset.save_episode()
    assert check_dataset(root, decode_videos=True)["valid"]
    for path, digest in previous.items():
        if path.startswith(("data/", "videos/", "meta/episodes/", "meta/safety/")):
            assert hashes(root)[path] == digest
    stats = json.loads((root / "meta/stats.json").read_text())
    assert stats["action"]["count"] == [5]
    assert stats["action"]["mean"] == [0.8] * 18
    before = hashes(root)
    with patch("alohamini.datasets.video._encode_frames", side_effect=AssertionError("Re-encode")):
        visual, selected = tmp_path / "visual", tmp_path / "selected"
        export_lerobot(root, visual, vision_only=True)
        export_lerobot(root, selected, state="joint_velocity,joint_current,base_velocity,lift_height")
    assert hashes(root) == before
    assert "observation.state" not in rows(visual)[0]
    assert len(rows(selected)[0]["observation.state"]) == 32
    assert rows(selected)[0]["action"] == rows(root)[0]["action"]
    for output in (selected, visual):
        assert check_dataset(output, decode_videos=True)["valid"]
        for path, digest in before.items():
            if path.startswith("videos/"):
                assert hashes(output)[path] == digest


@pytest.mark.parametrize("failure", ["encode", "publish", "after_commit"])
@pytest.mark.parametrize("committed", [0, 1])
def test_failed_save_can_recover_without_modifying_source(tmp_path, failure, committed):
    root = tmp_path / "record"
    ds = LocalDataset(root, fps=30, task="pick", robot_metadata=metadata())
    for _ in range(committed):
        add_episode(ds)
        ds.save_episode()
    add_episode(ds)
    if failure == "encode":
        target = "alohamini.datasets.video._encode_frames"
        def replacement(*a, **kw):
            raise OSError("encoder failed")
    elif failure == "publish":
        target = "pathlib.Path.replace"
        original = Path.replace

        def replacement(path, to):
            if path.name == "info.json" and path.parent.parent.name == "v3":
                raise OSError("metadata write failed")
            return original(path, to)
    else:
        target = "alohamini.datasets.record.shutil.rmtree"
        def replacement(*a, **kw):
            raise OSError("cleanup failed")
    with patch(target, new=replacement), pytest.raises(OSError):
        ds.save_episode()
    ds.close()
    before = hashes(root)
    recovered = tmp_path / "recovered"
    report = recover_recording(root, recovered)
    assert report["valid"], report
    assert hashes(root) == before
    assert len(rows(recovered)) == 3 * (committed + 1)
    with contextlib.closing(LocalDataset(recovered, fps=30, task="pick", robot_metadata=metadata(), resume=True)) as resumed:
        assert resumed.num_episodes == committed + 1
