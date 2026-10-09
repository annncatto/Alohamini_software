import numpy as np
import pytest
import torch
from test_native_learning import recording as recording

from alohamini.datasets.lerobotv3 import export_lerobot
from alohamini.learning.data import AlohaMiniDataset, DatasetInspection
from alohamini.learning.processor import Processor, image_tensor


@pytest.mark.parametrize("size", [(32, 32), (24, 48)])
def test_uint8_delivery_preserves_preprocessed_pixels_and_numeric_fields(size):
    rgb = np.random.default_rng(3).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    key = "observation.images.forward"
    regular = image_tensor(rgb, size)
    compact = image_tensor(rgb, size, return_uint8=True)
    assert compact.dtype == (torch.uint8 if size == (32, 32) else torch.float32)
    processor = Processor({key: {"mean": [[[0.4]]] * 3, "std": [[[0.2]]] * 3}})
    extras = {"action": torch.randn(2, 3, 18), "action_is_pad": torch.zeros(2, 3).bool()}
    actual = processor({key: compact[None].repeat(2, 1, 1, 1), **extras})
    expected = processor({key: regular[None].repeat(2, 1, 1, 1), **extras})
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert torch.equal(compact, image_tensor(rgb, size, return_uint8=True))


@pytest.mark.parametrize("v3", [False, True])
def test_compact_dataset_preserves_windows_statistics_and_observations(recording, tmp_path, v3):
    pytest.importorskip("torchcodec")
    root = recording
    if v3:
        root = tmp_path / "v3"
        export_lerobot(recording, root)
    kwargs = dict(
        episodes=[0, 1],
        state="none",
        image_size=(32, 32),
        delta_indices={"action": [0, 1, 2], "observation.images.forward": [-1, 0, 1]},
    )
    regular = AlohaMiniDataset(root, **kwargs)
    compact = AlohaMiniDataset(
        root, **kwargs, return_uint8=True, camera_workers=3, video_backend="torchcodec"
    )
    processor = Processor({})
    for index in [0, 3, 4, 7]:
        expected = regular[index]
        actual = processor(compact[index])
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert compact.statistics() == regular.statistics()
    assert compact.locations == regular.locations


def test_inspection_reuses_only_unchanged_dataset_and_returns_independent_reports(
    recording, monkeypatch
):
    import alohamini.learning.data as module

    check = module.check_dataset
    calls = []

    def tracked(root, **kwargs):
        calls.append(root)
        return check(root, **kwargs)

    monkeypatch.setattr(module, "check_dataset", tracked)
    inspection = DatasetInspection()
    a = AlohaMiniDataset(recording, episodes=[0], state="none", inspection=inspection)
    b = AlohaMiniDataset(recording, episodes=[1], state="none", inspection=inspection)
    assert len(calls) == 1 and a.report == b.report and a.report is not b.report
    # A modification between views must trigger a fresh check and preserve rejection.
    path = recording / "meta/info.json"
    original = path.read_text()
    path.write_text("{}")
    with pytest.raises(ValueError, match="integrity check failed"):
        AlohaMiniDataset(recording, episodes=[1], state="none", inspection=inspection)
    assert len(calls) == 2
    path.write_text(original)


def test_inspection_rejects_mutation_during_scan(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()

    def changing_check(path, **kwargs):
        (path / "changed").write_text("modified")
        return {"valid": True}

    monkeypatch.setattr("alohamini.learning.data.check_dataset", changing_check)
    with pytest.raises(RuntimeError, match="changed during integrity"):
        DatasetInspection().check(root)


def test_training_pipeline_defaults_resolve_backend_and_preserve_overrides(monkeypatch):
    from alohamini.learning.loading import resolve_data_pipeline
    from alohamini.learning.train_config import parse_training_args

    monkeypatch.setattr(
        "alohamini.learning.loading.importlib.util.find_spec", lambda name: object()
    )
    cfg, _ = parse_training_args(["--dataset.root=/tmp/data", "--output_dir=/tmp/run"])
    assert cfg["video_backend"] == "torchcodec"
    assert cfg["camera_workers"] == 3 and cfg["video_cache_size"] == 32
    assert cfg["return_uint8"] is True
    override = dict(video_backend="pyav", camera_workers=0, video_cache_size=8, return_uint8=False)
    assert resolve_data_pipeline(override) == override
    monkeypatch.setattr("alohamini.learning.loading.importlib.util.find_spec", lambda name: None)
    assert resolve_data_pipeline({})["video_backend"] == "pyav"
    assert resolve_data_pipeline({"policy": "smolvla"})["return_uint8"] is True


def test_parallel_three_camera_windows_match_serial_reads(tmp_path):
    import io
    from contextlib import closing

    from PIL import Image
    from test_dataset import frame, metadata

    from alohamini.datasets.record import _EpisodeWriter

    pytest.importorskip("torchcodec")
    root = tmp_path / "multi"
    meta = metadata()
    meta["cameras"] = ["forward", "wrist_left", "wrist_right"]
    with closing(_EpisodeWriter(root, fps=30, task="test", robot_metadata=meta)) as writer:
        writer.begin_episode()
        for index in range(4):
            images = {}
            for camera_id, camera in enumerate(meta["cameras"]):
                rgb = np.random.default_rng(index * 3 + camera_id).integers(
                    0, 256, (32, 32, 3), dtype=np.uint8
                )
                stream = io.BytesIO()
                Image.fromarray(rgb).save(stream, format="JPEG")
                images[camera] = stream.getvalue()
            writer.add_frame(frame(writer), images, {})
        writer.save_episode()
    windows = {f"observation.images.{c}": [-1, 0, 1] for c in meta["cameras"]}
    options = dict(
        episodes=[0],
        state="none",
        image_size=(32, 32),
        delta_indices={"action": [0, 1, 2], **windows},
    )
    serial = AlohaMiniDataset(root, **options)
    parallel = AlohaMiniDataset(
        root,
        **options,
        camera_workers=3,
        video_backend="torchcodec",
        return_uint8=True,
        video_cache_size=2,
    )
    processor = Processor({})
    for index in [0, 3, 1, 2]:
        actual, expected = processor(parallel[index]), serial[index]
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
