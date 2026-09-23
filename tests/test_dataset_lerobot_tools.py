from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alohamini.datasets.lerobot_tools import (
    DatasetRepairer,
    IntegrityChecker,
    flatten_dict,
    get_feature_stats,
)
from alohamini.datasets.tools import check_dataset, repair_dataset
from alohamini.datasets.video import _encode_frames, inspect_video


def test_single_frame_statistics_keep_source_basic_stats():
    stats = get_feature_stats(np.array([[7.0, 3.0]]))
    for key in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
        np.testing.assert_array_equal(stats[key], [7, 3])
    np.testing.assert_array_equal(stats["std"], [0, 0])
    np.testing.assert_array_equal(stats["count"], [1])


def test_recovered_single_frame_can_export_check_and_repair(tmp_path):
    from test_dataset import frame, metadata

    from alohamini.datasets.lerobot import export_lerobot
    from alohamini.datasets.native import LocalDataset
    from alohamini.datasets.tools import export_dataset

    source = tmp_path / "interrupted"
    dataset = LocalDataset(source, fps=30, task="pick", robot_metadata=metadata(()))
    dataset.begin_episode()
    assert dataset.add_frame(frame(dataset), {}, {})
    with patch("alohamini.datasets.native._write_json", side_effect=OSError("interrupted")):
        with pytest.raises(OSError):
            dataset.save_episode()
    dataset.close()
    recovered, exported = tmp_path / "recovered", tmp_path / "exported"
    assert export_dataset(source, recovered, recover=True)["valid"]
    assert export_lerobot(recovered, exported)["valid"]
    assert check_dataset(exported)["valid"]
    info_path = exported / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = 2
    info_path.write_text(json.dumps(info))
    output = tmp_path / "repaired"
    repair_dataset(exported, output)
    assert check_dataset(output)["valid"]
    assert json.loads(info_path.read_text())["total_frames"] == 2


def test_unknown_recovery_drop_counts_remain_warning_not_invalid(tmp_path):
    _make_gapped_dataset(tmp_path)
    directory = tmp_path / "meta/safety"
    directory.mkdir()
    for episode in (0, 2):
        rows = [
            {"episode_index": episode, "frame_index": frame, "client_monotonic_s": 1 + frame / 25}
            for frame in range(2)
        ]
        rows.append(
            {
                "episode_index": episode,
                "event": {
                    "type": "recorder_closed",
                    "frame_count": 2,
                    "queue_overflows": None,
                    "rejected_images": None,
                },
            }
        )
        (directory / f"episode_{episode:06d}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
    report = IntegrityChecker(tmp_path, decode_videos=False, timestamp_tolerance_s=1e-4).run()
    codes = {issue["code"] for issue in report["issues"]}
    assert "SAFETY_SIDECAR_INVALID" not in codes
    assert "SAFETY_LOG_INCOMPLETE" in codes


def _add_video(root, *, starts=(1, 4), count=8):
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info["features"]["observation.images.forward"] = {
        "dtype": "video",
        "shape": [16, 24, 3],
        "info": {"video.codec": "h264", "video.pix_fmt": "yuv420p", "video.crf": 18},
    }
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    path.write_text(json.dumps(info))
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    for row, start in zip(rows, starts, strict=True):
        row.update(
            {
                "videos/observation.images.forward/chunk_index": 0,
                "videos/observation.images.forward/file_index": 0,
                "videos/observation.images.forward/from_timestamp": start / 25,
                "videos/observation.images.forward/to_timestamp": (start + 2) / 25,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)
    video = root / "videos/observation.images.forward/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    colors = [
        (0, 0, 0),
        (250, 0, 0),
        (250, 0, 0),
        (0, 0, 0),
        (0, 0, 250),
        (0, 0, 250),
        (0, 0, 0),
        (0, 0, 0),
    ]
    frames = (
        av.VideoFrame.from_ndarray(np.full((16, 24, 3), color, np.uint8), format="rgb24")
        for color in colors[:count]
    )
    _encode_frames(frames, video, 25, [16, 24, 3], options={"crf": "18"})
    return video


FEATURES = {
    "action": {"dtype": "float32", "shape": [1], "names": ["joint"]},
    "observation.state": {"dtype": "float32", "shape": [1], "names": ["joint"]},
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def _stats(values: dict[str, np.ndarray]) -> dict[str, object]:
    episode_stats = {
        key: get_feature_stats(value, axis=0, keepdims=False) for key, value in values.items()
    }
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in flatten_dict({"stats": episode_stats}).items()
    }


def _make_gapped_dataset(root: Path, *, metadata_ids: tuple[int, int] = (0, 2)) -> None:
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)

    old_episode_ids = np.array([0, 0, 2, 2], dtype=np.int64)
    frame_indices = np.array([0, 1, 0, 1], dtype=np.int64)
    global_indices = np.array([0, 1, 4, 5], dtype=np.int64)
    timestamps = np.array([0.0, 0.04, 0.0, 0.04], dtype=np.float32)
    action = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    state = action + 10
    task_indices = np.zeros(4, dtype=np.int64)

    data_table = pa.table(
        {
            "action": action.tolist(),
            "observation.state": state.tolist(),
            "timestamp": timestamps,
            "frame_index": frame_indices,
            "episode_index": old_episode_ids,
            "index": global_indices,
            "task_index": task_indices,
        }
    )
    pq.write_table(data_table, root / "data/chunk-000/file-000.parquet")

    rows = []
    for row_number, old_id in enumerate(metadata_ids):
        mask = old_episode_ids == (0 if row_number == 0 else 2)
        values = {
            "action": action[mask],
            "observation.state": state[mask],
            "timestamp": timestamps[mask].reshape(-1, 1),
            "frame_index": frame_indices[mask].reshape(-1, 1),
            "episode_index": old_episode_ids[mask].reshape(-1, 1),
            "index": global_indices[mask].reshape(-1, 1),
            "task_index": task_indices[mask].reshape(-1, 1),
        }
        row = {
            "episode_index": old_id,
            "tasks": ["test"],
            "length": 2,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 0 if row_number == 0 else 2,
            "dataset_to_index": 2 if row_number == 0 else 4,
        }
        row.update(_stats(values))
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"task_index": [0], "task": ["test"]}), root / "meta/tasks.parquet")

    info = {
        "codebase_version": "v3.0",
        "fps": 25,
        "features": FEATURES,
        "total_episodes": 3,
        "total_frames": 6,
        "total_tasks": 1,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "robot_type": "test",
        "splits": {"train": "0:3"},
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/stats.json").write_text("{}", encoding="utf-8")


def test_repair_normalizes_episode_and_global_indices(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "repaired"
    _make_gapped_dataset(source)

    checker = IntegrityChecker(source, decode_videos=True, timestamp_tolerance_s=1e-4)
    source_report = checker.run()
    assert not source_report["valid"]

    manifest, repaired_report = DatasetRepairer(checker, output).run(
        source_report, timestamp_tolerance_s=1e-4
    )

    assert repaired_report["valid"]
    assert manifest["episode_mapping"] == {"0": 0, "2": 1}
    assert manifest["video_repair"]["output"] == str(output)
    repaired_info = json.loads((output / "meta/info.json").read_text())
    assert repaired_info["total_episodes"] == 2
    assert repaired_info["total_frames"] == 4
    assert repaired_info["splits"] == {"train": "0:2"}

    data = pq.read_table(output / "data/chunk-000/file-000.parquet")
    assert data["episode_index"].to_pylist() == [0, 0, 1, 1]
    assert data["index"].to_pylist() == [0, 1, 2, 3]
    stats = json.loads((output / "meta/stats.json").read_text())
    assert {"action", "observation.state", "episode_index", "index"} <= stats.keys()


def test_repair_refuses_data_metadata_episode_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "repaired"
    _make_gapped_dataset(source, metadata_ids=(0, 1))

    checker = IntegrityChecker(source, decode_videos=True, timestamp_tolerance_s=1e-4)
    report = checker.run()

    with pytest.raises(ValueError, match="Repair refused"):
        DatasetRepairer(checker, output).run(report, timestamp_tolerance_s=1e-4)
    assert not output.exists()


def test_video_gaps_are_repacked_without_changing_samples_or_source(tmp_path):
    from test_dataset_tools import hashes

    source, output = tmp_path / "source", tmp_path / "repaired"
    _make_gapped_dataset(source)
    video = _add_video(source)
    (source / "meta/alohamini.json").write_text('{"state_units": ["A"]}')
    (source / "meta/motor_feedback.json").write_text('{"version": 1}')
    before = hashes(source)
    report = check_dataset(source, decode_videos=True)
    assert {"VIDEO_LEADING_GAP", "VIDEO_FRAME_GAP", "VIDEO_TOTAL_FRAMES_MISMATCH"} <= {
        issue["code"] for issue in report["issues"]
    }
    repaired = repair_dataset(source, output)
    assert repaired["valid"], repaired
    assert repaired["repair"]["video_repair"]["removed_video_frames"] == 4
    target = output / video.relative_to(source)
    assert inspect_video(target, decode=True)["frames"] == 4
    with av.open(str(target)) as container:
        colors = [f.to_ndarray(format="rgb24").mean(axis=(0, 1)) for f in container.decode(video=0)]
    np.testing.assert_allclose(colors, [(250, 0, 0)] * 2 + [(0, 0, 250)] * 2, atol=12)
    for name in ("action", "observation.state", "timestamp", "frame_index", "task_index"):
        assert pq.read_table(source / "data/chunk-000/file-000.parquet")[name].to_pylist() == (
            pq.read_table(output / "data/chunk-000/file-000.parquet")[name].to_pylist()
        )
    assert (output / "meta/alohamini.json").read_bytes() == (
        source / "meta/alohamini.json"
    ).read_bytes()
    assert (output / "meta/motor_feedback.json").is_file()
    assert hashes(source) == before


def test_video_repair_preserves_source_encoder_options(tmp_path):
    source = tmp_path / "source"
    _make_gapped_dataset(source)
    _add_video(source)
    path = source / "meta/info.json"
    info = json.loads(path.read_text())
    info["features"]["observation.images.forward"]["info"].update(
        {
            "video.g": None,
            "video.fast_decode": 1,
            "video.extra_options": {"g": 99, "crf": 40, "preset": "fast", "unused": None},
        }
    )
    path.write_text(json.dumps(info))
    with patch("alohamini.datasets.video._encode_frames", wraps=_encode_frames) as encode:
        repaired = repair_dataset(source, tmp_path / "repaired")
    assert repaired["valid"], repaired
    assert encode.call_args.kwargs["options"] == {
        "g": "2",
        "crf": "18",
        "tune": "fastdecode",
        "preset": "fast",
    }


@pytest.mark.parametrize("starts,count", [((1, 2), 8), ((1, 7), 8), ((1, 4), 5)])
def test_video_overlap_and_missing_required_frames_cannot_be_guessed(tmp_path, starts, count):
    from test_dataset_tools import hashes

    source, output = tmp_path / "source", tmp_path / "repaired"
    _make_gapped_dataset(source)
    _add_video(source, starts=starts, count=count)
    before = hashes(source)
    with pytest.raises((ValueError, RuntimeError)):
        repair_dataset(source, output)
    assert not output.exists()
    assert hashes(source) == before


def test_path_escape_invalid_features_and_malformed_info_return_invalid(tmp_path):
    source = tmp_path / "source"
    _make_gapped_dataset(source)
    path = source / "meta/info.json"
    info = json.loads(path.read_text())
    info["data_path"] = "../outside.parquet"
    path.write_text(json.dumps(info))
    report = check_dataset(source)
    assert "PATH_TEMPLATE_INVALID" in {issue["code"] for issue in report["issues"]}
    with pytest.raises(ValueError):
        repair_dataset(source, tmp_path / "output")
    for value in ([], {"codebase_version": "v3.0"}, {**info, "fps": float("nan")}):
        path.write_text(json.dumps(value))
        assert not check_dataset(source)["valid"]


def test_out_of_order_rows_are_not_silently_repaired(tmp_path):
    source = tmp_path / "source"
    _make_gapped_dataset(source)
    path = source / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    pq.write_table(table.take([1, 0, 2, 3]), path)
    report = check_dataset(source)
    assert "DATA_ROW_ORDER_INVALID" in {issue["code"] for issue in report["issues"]}
    with pytest.raises(ValueError):
        repair_dataset(source, tmp_path / "repaired")


def test_fractional_video_ranges_are_not_rounded_into_a_repair(tmp_path):
    source = tmp_path / "source"
    _make_gapped_dataset(source)
    _add_video(source)
    path = source / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["videos/observation.images.forward/from_timestamp"] += 0.01
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    report = check_dataset(source)
    assert "VIDEO_TIMESTAMP_MISMATCH" in {issue["code"] for issue in report["issues"]}
    with pytest.raises(ValueError):
        repair_dataset(source, tmp_path / "repaired")


def test_repair_refuses_nested_output_before_writing(tmp_path):
    source = tmp_path / "source"
    _make_gapped_dataset(source)
    with pytest.raises(ValueError, match="separate"):
        repair_dataset(source, source / "repaired")
    assert not (source / "repaired").exists()


def test_cli_checks_and_repairs_the_old_video_format(tmp_path, capsys):
    from alohamini.cli import main

    source, output = tmp_path / "source", tmp_path / "repaired"
    _make_gapped_dataset(source)
    _add_video(source)
    assert main(["dataset", "check", str(source), "--decode-videos"]) == 1
    assert main(["dataset", "repair", str(source), "--output", str(output)]) == 0
    assert main(["dataset", "check", str(output), "--decode-videos"]) == 0
    assert "Result: VALID" in capsys.readouterr().out


def test_native_export_embedded_images_are_checked_and_repair_keeps_metadata(tmp_path):
    from test_dataset import frame, jpeg, metadata

    from alohamini.datasets.lerobot import export_lerobot
    from alohamini.datasets.native import LocalDataset

    native, exported = tmp_path / "native", tmp_path / "exported"
    with closing(LocalDataset(native, fps=30, task="pick", robot_metadata=metadata())) as dataset:
        dataset.begin_episode()
        for _ in range(2):
            dataset.add_frame(frame(dataset), {"forward": jpeg()}, {})
        dataset.save_episode()
    export_lerobot(native, exported)
    report = check_dataset(exported, decode_images=True, decode_videos=True)
    assert report["valid"], report
    output = tmp_path / "repaired"
    assert repair_dataset(exported, output)["valid"]
    assert (output / "meta/alohamini.json").read_bytes() == (
        exported / "meta/alohamini.json"
    ).read_bytes()
    path = next((exported / "data").rglob("*.parquet"))
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["observation.images.forward"] = {"bytes": b"bad", "path": None}
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    report = check_dataset(exported, decode_images=True)
    assert "IMAGE_INVALID" in {issue["code"] for issue in report["issues"]}
    with pytest.raises(ValueError):
        repair_dataset(exported, tmp_path / "unsafe")


def test_repair_refuses_in_place_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_gapped_dataset(source)
    checker = IntegrityChecker(source, decode_videos=True, timestamp_tolerance_s=1e-4)

    with pytest.raises(ValueError, match="in-place repair is forbidden"):
        DatasetRepairer(checker, source).run(checker.run(), timestamp_tolerance_s=1e-4)


def test_repair_preserves_and_remaps_safety_without_certifying_training(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "repaired"
    _make_gapped_dataset(source)
    sidecar = source / "meta/safety/episode_000002.jsonl"
    sidecar.parent.mkdir()
    rows = [
        {
            "frame_index": 0,
            "client_monotonic_s": 1.0,
            "safety": {"joint_holds": {"joint": 2}},
            "requested_action": {"joint": 8},
        },
        {"frame_index": None, "event": {"type": "feedback_wait"}},
        {"frame_index": 1, "client_monotonic_s": 1.04, "safety": {}},
        {
            "frame_index": None,
            "event": {"type": "recorder_closed", "frame_count": 2, "dropped_records": 0},
        },
    ]
    sidecar.write_text("\n".join(json.dumps({"episode_index": 2, **row}) for row in rows) + "\n")
    checker = IntegrityChecker(source, decode_videos=True, timestamp_tolerance_s=1e-4)
    report = checker.run()
    _, repaired = DatasetRepairer(checker, output).run(report, timestamp_tolerance_s=1e-4)
    assert repaired["valid"]
    assert repaired["training_review"] == "required"
    original = [
        json.loads(line)
        for line in (source / "meta/safety/episode_000002.jsonl").read_text().splitlines()
    ]
    restored = [
        json.loads(line)
        for line in (output / "meta/safety/episode_000001.jsonl").read_text().splitlines()
    ]
    assert restored == [{**row, "episode_index": 1} for row in original]


def test_checker_marks_truncated_log_and_capture_gap(tmp_path):
    source = tmp_path / "source"
    _make_gapped_dataset(source)
    path = source / "meta/safety/episode_000002.jsonl"
    path.parent.mkdir()
    rows = [{"episode_index": 2, "frame_index": i, "client_monotonic_s": 2 * i} for i in range(2)]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    report = IntegrityChecker(source, decode_videos=True, timestamp_tolerance_s=1e-4).run()
    codes = {issue["code"] for issue in report["issues"]}
    assert {"SAFETY_LOG_INCOMPLETE", "SAFETY_TIME_GAP", "SAFETY_SIDECAR_MISSING"} <= codes


@pytest.mark.parametrize("actual_fps", [30, 23.72])
def test_capture_clock_detects_gradual_drift_without_a_large_gap(tmp_path, actual_fps):
    checker = IntegrityChecker(tmp_path, decode_videos=False, timestamp_tolerance_s=1e-4)
    checker.info = {"fps": 30}
    rows = [
        {
            "frame_index": index,
            "client_monotonic_s": 100 + index / 30,
            "host_timing": {"camera_capture_monotonic_s": {"forward": 5000 + index / actual_fps}},
        }
        for index in range(240)
    ]
    checker._check_capture_timeline(0, rows)
    codes = {issue.code for issue in checker.issues}
    assert ("SAFETY_CAPTURE_TIMEBASE" in codes) == (actual_fps != 30)
    assert "SAFETY_CAPTURE_GAP" not in codes
    if actual_fps != 30:
        assert checker.report()["training_review"] == "required"


@pytest.mark.parametrize("stamps", [(1.0, 1.2), (1.0, 0.9), (1.0, 1.0)])
def test_capture_clock_detects_skips_and_rollback(tmp_path, stamps):
    checker = IntegrityChecker(tmp_path, decode_videos=False, timestamp_tolerance_s=1e-4)
    checker.info = {"fps": 30}
    checker._check_capture_timeline(
        0,
        [
            {
                "frame_index": index,
                "host_timing": {"camera_capture_monotonic_s": {"forward": stamp}},
            }
            for index, stamp in enumerate(stamps)
        ],
    )
    assert "SAFETY_CAPTURE_GAP" in {issue.code for issue in checker.issues}


def test_checker_rejects_invalid_action_chronology(tmp_path):
    source = tmp_path / "source"
    _make_gapped_dataset(source)
    path = source / "meta/safety/episode_000002.jsonl"
    path.parent.mkdir()
    rows = [
        {
            "episode_index": 2,
            "frame_index": index,
            "client_monotonic_s": 1 + index * 0.04,
            "client_timing": {
                "observation_received_monotonic_s": 1,
                "action_sample_started_monotonic_s": 0,
                "action_sample_finished_monotonic_s": 2,
                "command_sent_monotonic_s": 3,
            },
        }
        for index in range(2)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    report = IntegrityChecker(source, decode_videos=False, timestamp_tolerance_s=1e-4).run()
    assert "SAFETY_SIDECAR_INVALID" in {issue["code"] for issue in report["issues"]}
