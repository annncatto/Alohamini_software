"""Offline native editing: no source mutation, robot access or framework dependency."""

import io
import json
import tarfile
from contextlib import closing

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_dataset import frame, jpeg, metadata
from test_dataset_tools import hashes

from alohamini.cli import main
from alohamini.datasets.edit import edit_dataset, parse_args
from alohamini.datasets.images import image_bytes, image_rgb
from alohamini.datasets.lerobot import export_lerobot
from alohamini.datasets.native import LocalDataset, motor_feedback_frame
from alohamini.datasets.tools import check_dataset, export_dataset, repair_dataset


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    with closing(
        LocalDataset(root, fps=30, task="pick", robot_metadata=metadata(("forward", "wrist")))
    ) as ds:
        for episode in range(3):
            ds.begin_episode()
            for i in range(4):
                values = frame(ds)
                values["action"][:] = episode * 10 + i
                values["observation.state"][:] = episode * 10
                values.update(
                    motor_feedback_frame(
                        ds.features,
                        {
                            "version": 1,
                            "motors": {
                                name: {
                                    "current_ma": 100 + i,
                                    "sample_started_s": 100 + i / 30,
                                    "sample_finished_s": 100.001 + i / 30,
                                }
                                for name in ds.features["observation.motor_current_ma"]["names"]
                            },
                        },
                    )
                )
                record = {
                    "requested_action": dict(zip(ds.names, values["action"].tolist(), strict=True)),
                    "robot_metadata": ds.robot_metadata,
                    "safety": {"feedback_valid": True, "control_epoch": 1},
                    "host_timing": {
                        "state_sample_monotonic_s": 100 + i / 30,
                        "camera_capture_monotonic_s": {
                            "forward": 100 + i / 30,
                            "wrist": 100 + i / 30,
                        },
                    },
                }
                assert ds.add_frame(
                    values,
                    {
                        "forward": jpeg((200, i * 30, 0), (32, 32, 3)),
                        "wrist": jpeg((0, 100, 0), (32, 32, 3)),
                    },
                    record,
                )
            ds.save_episode()
    assert check_dataset(root, decode_images=True)["valid"]
    return root


def run(source, output, operation, **options):
    args = ["--root", str(source), "--output", str(output), "--operation.type", operation]
    for key, value in options.items():
        args += [f"--operation.{key}", json.dumps(value) if not isinstance(value, str) else value]
    return edit_dataset(parse_args(args))


def rows(root, episode=0):
    return pq.read_table(
        root / "episodes" / f"episode_{episode:06d}" / "frames.parquet"
    ).to_pylist()


def records(root, episode=0):
    return [
        json.loads(line)
        for line in (root / "episodes" / f"episode_{episode:06d}" / "safety.jsonl")
        .read_text()
        .splitlines()
    ]


def test_delete_preserves_physical_pairing_and_source(source, tmp_path):
    before = hashes(source)
    output = tmp_path / "filtered"
    result = run(source, output, "delete_episodes", episode_indices=[1])
    assert result["dataset_root"] == str(output)
    assert hashes(source) == before
    assert check_dataset(output, decode_images=True)["valid"]
    assert [r["index"] for r in rows(output, 1)] == [4, 5, 6, 7]
    for original, copied in zip(rows(source, 2), rows(output, 1), strict=True):
        for key in original.keys() - {"index", "episode_index"}:
            assert original[key] == copied[key]
    for original, copied in zip(records(source, 2), records(output, 1), strict=True):
        assert copied == {**original, "episode_index": 1}


def test_split_source_order_and_unused_fraction(source, tmp_path):
    output = tmp_path / "splits"
    run(source, output, "split", splits={"train": [2, 0], "val": [1]})
    assert rows(output / "train", 0)[0]["action"][0] == 0
    assert rows(output / "train", 1)[0]["action"][0] == 20
    assert check_dataset(output / "val")["valid"]
    from alohamini.datasets.edit import _fractions_to_episode_indices

    assert _fractions_to_episode_indices(10, {"train": 0.5, "val": 0.2}) == {
        "train": [0, 1, 2, 3, 4],
        "val": [5, 6],
    }


def test_tasks_merge_export_and_info(source, tmp_path, capsys):
    changed = tmp_path / "tasks"
    run(source, changed, "modify_tasks", new_task="default", episode_tasks={"1": "place"})
    assert [rows(changed, i)[0]["task"] for i in range(3)] == ["default", "place", "default"]
    merged = tmp_path / "merged"
    run(source, merged, "merge", roots=[str(source), str(changed)])
    report = check_dataset(merged, decode_images=True)
    assert report["valid"] and report["summary"]["episodes"] == 6
    assert rows(merged, 5)[0]["index"] == 20
    exported = tmp_path / "lerobot"
    export_lerobot(changed, exported)
    assert check_dataset(exported, decode_images=True)["valid"]
    assert json.loads((exported / "meta/info.json").read_text())["total_tasks"] == 2
    result = edit_dataset(
        parse_args(["--root", str(changed), "--type", "info", "--operation.show_features", "true"])
    )
    assert result["tasks"] == ["default", "place"]
    assert "features" in capsys.readouterr().out


def test_remove_camera_and_feedback_pairs(source, tmp_path):
    output = tmp_path / "reduced"
    run(
        source,
        output,
        "remove_feature",
        feature_names=["observation.images.wrist", "observation.motor_current_ma"],
    )
    report = check_dataset(output, decode_images=True)
    assert report["valid"], report
    assert "observation.images.wrist" not in rows(output)[0]
    assert "motor_feedback.current_ma_valid" not in rows(output)[0]
    for path in output.rglob("*.tar"):
        with tarfile.open(path) as stream:
            assert all(member.name.startswith("forward/") for member in stream)
    old, new = rows(source)[0], rows(output)[0]
    assert image_bytes(
        source / "episodes/episode_000000", "forward", old["observation.images.forward"]
    ) == image_bytes(
        output / "episodes/episode_000000", "forward", new["observation.images.forward"]
    )
    assert records(source) == records(output)


def test_stats_masks_relative_chunks_and_source(source, tmp_path):
    before = hashes(source)
    output = tmp_path / "stats"
    run(
        source,
        output,
        "recompute_stats",
        relative_action=True,
        chunk_size=2,
        skip_image_video=False,
    )
    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["action"]["mean"][0] == pytest.approx(1.5)
    assert stats["action"]["count"][0] == 18
    # Base velocity is never changed into a delta velocity.
    assert stats["action"]["mean"][14] == pytest.approx(11.5)
    assert stats["observation.motor_current_ma"]["mean"][0] == pytest.approx(101.5)
    assert stats["observation.motor_velocity_raw"]["count"][0] == 0
    assert stats["observation.motor_velocity_raw"]["mean"][0] is None
    assert "observation.images.forward" in stats
    info = json.loads((output / "meta/stats_info.json").read_text())
    assert "exact linear" in info["numeric_statistics"]
    assert info["source_sha256"] and info["source_info_sha256"]
    assert info["diagnostics"]["observation.motor_velocity_raw"][0]["unavailable"]
    assert hashes(source) == before


def test_video_convert_reencode_delete_and_native_reader(source, tmp_path):
    video = tmp_path / "video"
    run(source, video, "convert_image_to_video", episode_indices=[0, 2])
    report = check_dataset(video, decode_images=True, decode_videos=True)
    assert report["valid"], report
    original = image_rgb(
        source / "episodes/episode_000000", "forward", rows(source)[0]["observation.images.forward"]
    )
    for i in (3, 0, 2, 1):
        rgb = image_rgb(
            video / "episodes/episode_000000",
            "forward",
            rows(video)[i]["observation.images.forward"],
        )
        assert rgb.shape == original.shape
        assert rgb[:, :, 0].mean() > 190 and rgb[:, :, 2].mean() < 10
    transcoded = tmp_path / "transcoded"
    run(video, transcoded, "reencode_videos", **{"rgb_encoder.crf": 20})
    assert check_dataset(transcoded, decode_videos=True)["valid"]
    filtered = tmp_path / "video_filtered"
    run(video, filtered, "delete_episodes", episode_indices=[0])
    assert rows(filtered)[0]["action"][0] == 20
    assert check_dataset(filtered, decode_videos=True)["valid"]
    copy = tmp_path / "copy"
    assert export_dataset(filtered, copy)["valid"]
    from alohamini.learning.data import AlohaMiniDataset

    samples = AlohaMiniDataset(
        video, episodes=[0], chunk_size=2, state="none", review_note="fixture"
    )
    assert samples[0]["observation.images.forward"].shape == (3, 480, 640)
    assert samples[0]["action"].shape == (2, 18)


@pytest.mark.parametrize(
    "operation,options",
    [
        ("delete_episodes", {"episode_indices": [0, 1, 2]}),
        ("delete_episodes", {"episode_indices": [-1]}),
        ("delete_episodes", {"episode_indices": [True]}),
        ("delete_episodes", {"episode_indices": [1, 1]}),
        ("split", {"splits": {"../escape": [0]}}),
        ("split", {"splits": {"a": [0], "b": [0]}}),
        ("remove_feature", {"feature_names": ["index"]}),
        ("modify_tasks", {"episode_tasks": {"9": "bad"}}),
        ("recompute_stats", {"relative_action": True, "chunk_size": 100}),
        ("reencode_videos", {}),
    ],
)
def test_invalid_operations_preserve_source(source, tmp_path, operation, options):
    before = hashes(source)
    output = tmp_path / "output"
    with pytest.raises((ValueError, RuntimeError)):
        run(source, output, operation, **options)
    assert hashes(source) == before
    assert not output.exists()


def test_path_guard_config_and_cli(source, tmp_path):
    before = hashes(source)
    for output in (source, source / "child", source.parent):
        with pytest.raises((ValueError, FileExistsError)):
            run(source, output, "delete_episodes", episode_indices=[1])
    cfg = tmp_path / "edit.json"
    cfg.write_text(
        json.dumps(
            {
                "root": str(source),
                "output": str(tmp_path / "configured"),
                "operation": {"type": "delete_episodes", "episode_indices": [1]},
            }
        )
    )
    assert main(["dataset", "edit", "--config_path", str(cfg)]) == 0
    assert hashes(source) == before
    assert main(["dataset", "edit", "--root", str(source), "--type", "info"]) == 0


def test_preview_reindex_and_current_only_training(source, tmp_path):
    from alohamini.datasets.video import generate_previews
    from alohamini.learning.data import AlohaMiniDataset

    generate_previews(source)
    output = tmp_path / "preview_kept"
    run(source, output, "delete_episodes", episode_indices=[1])
    assert (output / "previews/episode_000001/forward.mp4").read_bytes() == (
        source / "previews/episode_000002/forward.mp4"
    ).read_bytes()
    report = check_dataset(output, decode_images=True, decode_videos=True)
    assert report["valid"]
    assert "PREVIEW_INVALID" not in {issue["code"] for issue in report["issues"]}
    reduced = tmp_path / "current_only"
    run(
        source,
        reduced,
        "remove_feature",
        feature_names=["observation.state", "observation.images.wrist"],
    )
    samples = AlohaMiniDataset(reduced, episodes=[0], state="joint_current", review_note="fixture")
    assert samples.cameras == ["forward"]
    assert samples[0]["observation.state"].shape == (14,)
    assert samples[0]["observation.state"][0].item() == pytest.approx(0.1)
    with pytest.raises(ValueError, match="original state and action"):
        export_lerobot(reduced, tmp_path / "no_state_export")


def test_reject_busy_source_and_camera_calibration_mismatch(source, tmp_path):
    import fcntl

    with (source / "recording.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="in use"):
            run(source, tmp_path / "busy", "delete_episodes", episode_indices=[1])
    second = tmp_path / "second"
    export_dataset(source, second)
    path = second / "meta/info.json"
    info = json.loads(path.read_text())
    info["fps"] = 20
    path.write_text(json.dumps(info))
    with pytest.raises((RuntimeError, ValueError)):
        run(source, tmp_path / "bad_merge", "merge", roots=[str(source), str(second)])


def test_historical_png_and_video_corruption(source, tmp_path):
    from test_dataset import as_png_v1

    as_png_v1(source)
    output = tmp_path / "png"
    run(source, output, "delete_episodes", episode_indices=[1])
    assert check_dataset(output, decode_images=True)["valid"]
    video = tmp_path / "video"
    run(output, video, "convert_image_to_video")
    path = video / "episodes/episode_000000/videos/forward.mp4"
    with path.open("ab") as stream:
        stream.write(b"changed")
    assert not check_dataset(video)["valid"]


def test_delete_invalid_index_does_not_create_pending_directory(source, tmp_path):
    with pytest.raises(ValueError, match=r"0\.\.2.*received \[3, 8\]"):
        run(source, tmp_path / "bad_indices", "delete_episodes", episode_indices=[3, 8])
    assert not list(tmp_path.glob("bad_indices*"))


def test_native_repair_indices_and_orphan_jpegs(source, tmp_path):
    for old, new in reversed([(0, 2), (1, 5), (2, 8)]):
        directory = source / "episodes" / f"episode_{old:06d}"
        table = pq.read_table(directory / "frames.parquet")
        values = table.to_pylist()
        for row in values:
            row["episode_index"] = new
            row["index"] += 100
        pq.write_table(
            pa.Table.from_pylist(values, schema=table.schema), directory / "frames.parquet"
        )
        path = directory / "episode.json"
        summary = json.loads(path.read_text())
        summary["episode_index"] = new
        path.write_text(json.dumps(summary))
        log = records(source, old)
        for record in log:
            record["episode_index"] = new
        (directory / "safety.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
        directory.rename(source / "episodes" / f"episode_{new:06d}")
    shard = next((source / "episodes/episode_000002/images").glob("*.tar"))
    data = jpeg()
    with tarfile.open(shard, "a") as archive:
        member = tarfile.TarInfo("forward/orphan.jpg")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    before = hashes(source)
    assert not check_dataset(source)["valid"]
    output = tmp_path / "repaired"
    report = repair_dataset(source, output)
    assert report["valid"] and report["repair"]["episode_mapping"] == {"2": 0, "5": 1, "8": 2}
    assert hashes(source) == before
    assert [r["index"] for r in rows(output, 2)] == [8, 9, 10, 11]
    assert rows(output, 2)[0]["action"] == rows(source, 8)[0]["action"]
    for path in output.rglob("*.tar"):
        with tarfile.open(path) as archive:
            assert all("orphan" not in member.name for member in archive)
    assert check_dataset(output, decode_images=True)["valid"]


@pytest.mark.parametrize("field", ["frame_index", "timestamp", "episode_index", "action"])
def test_native_repair_refuses_ambiguous_pairing(source, tmp_path, field):
    path = source / "episodes/episode_000000/frames.parquet"
    table = pq.read_table(path)
    values = table.to_pylist()
    values[1][field] = [999.0] * 18 if field == "action" else 999
    pq.write_table(pa.Table.from_pylist(values, schema=table.schema), path)
    before = hashes(source)
    output = tmp_path / "bad_repair"
    with pytest.raises(ValueError):
        repair_dataset(source, output)
    assert not output.exists()
    assert hashes(source) == before


@pytest.mark.parametrize("overlap", [False, True])
def test_native_repair_compacts_only_ordered_video_references(source, tmp_path, overlap):
    import av

    from alohamini.datasets.video import _encode_frames, file_sha256

    video = tmp_path / "video"
    run(source, video, "convert_image_to_video")
    directory = video / "episodes/episode_000000"
    path = directory / "frames.parquet"
    table = pq.read_table(path)
    values = table.to_pylist()
    originals = [image_rgb(directory, "forward", r["observation.images.forward"]) for r in values]
    media = directory / "videos/forward.mp4"
    _encode_frames(
        (
            av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for rgb in [originals[0], *originals, originals[-1]]
        ),
        media,
        30,
        originals[0].shape,
    )
    digest = file_sha256(media)
    for i, row in enumerate(values):
        row["observation.images.forward"].update(sha256=digest, frame_index=1 if overlap else i + 1)
    pq.write_table(pa.Table.from_pylist(values, schema=table.schema), path)
    before = hashes(video)
    assert not check_dataset(video)["valid"]
    output = tmp_path / "repaired_video"
    if overlap:
        with pytest.raises(ValueError, match="refused"):
            repair_dataset(video, output)
    else:
        report = repair_dataset(video, output)
        assert report["repair"]["video_reencoded"]
        assert check_dataset(output, decode_images=True, decode_videos=True)["valid"]
        assert [r["observation.images.forward"]["frame_index"] for r in rows(output)] == [
            0,
            1,
            2,
            3,
        ]
    assert hashes(video) == before
