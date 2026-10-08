"""Episode selection defines training data; quality flags never silently remove rows."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_native_learning import model_options
from test_native_learning import recording as recording

from alohamini.datasets.lerobotv3 import export_lerobot
from alohamini.learning.checkpoint import read_checkpoint, write_checkpoint_metadata
from alohamini.learning.data import AlohaMiniDataset, inspect_feedback
from alohamini.learning.policy import NativePolicy, make_policy, save_checkpoint
from alohamini.learning.statistics import training_statistics, write_statistics
from alohamini.learning.train import launch_training
from alohamini.learning.train_config import parse_training_args
from alohamini.learning.training_state import load_training_checkpoint
from alohamini.policies.registry import algorithm


def mark_frame(recording, **safety):
    path = recording / "episodes/episode_000000/safety.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        if record.get("frame_index") == 1:
            record["safety"].update(safety)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


@pytest.mark.parametrize("storage", ["native", "v3"])
def test_reviewed_episode_keeps_flagged_actions_windows_and_statistics(
    recording, tmp_path, storage
):
    mark_frame(recording, fault="reviewed", watchdog_active=True, joint_holds={"arm": {}})
    root = recording
    if storage == "v3":
        root = tmp_path / "v3"
        export_lerobot(recording, root)
    data = AlohaMiniDataset(root, episodes=[0, 1], state="none", chunk_size=3, cameras=[])
    assert data.sample_indices == list(range(8))
    assert data.excluded == 0
    assert data[0]["action"][:, 0].tolist() == [0, 1, 2]
    assert data[3]["action"][:, 0].tolist() == [3, 3, 3]
    assert data[3]["action_is_pad"].tolist() == [False, True, True]
    assert data[4]["action"][:, 0].tolist() == [10, 11, 12]
    assert data.statistics()["action"]["mean"] == [6.5] * 18
    assert data.records[1]["safety"]["fault"] == "reviewed"


@pytest.mark.parametrize("storage", ["native", "v3"])
def test_feedback_inspection_preserves_invalid_rows_before_training(recording, tmp_path, storage):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = recording / "episodes/episode_000000/frames.parquet"
    table = pq.read_table(path)
    key = "motor_feedback.current_ma_valid"
    masks = table[key].to_pylist()
    masks[1][0] = 0.0
    table = table.set_column(
        table.column_names.index(key),
        table.schema.field(key),
        pa.array(masks, type=table.schema.field(key).type),
    )
    pq.write_table(table, path)
    root = recording
    if storage == "v3":
        root = tmp_path / "v3"
        export_lerobot(recording, root)
    report = inspect_feedback(root, [0, 1], state="joint_current")
    assert report["values"].shape == (8, 14)
    assert np.isnan(report["values"][1]).all()
    assert np.isfinite(report["values"][[0, 2, 3, 4, 5, 6, 7]]).all()
    assert report["locations"][1] == (0, 1)
    assert report["episode_starts"] == [0, 4]
    assert [(issue["episode"], issue["frame"]) for issue in report["issues"]] == [(0, 1)]
    with pytest.raises(ValueError, match="Episode 0, frame 1.*current"):
        AlohaMiniDataset(root, episodes=[0], state="joint_current")
    assert len(AlohaMiniDataset(root, episodes=[1], state="joint_current")) == 4


def test_feedback_age_is_review_information_not_implicit_window_filter(recording):
    path = recording / "episodes/episode_000000/safety.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        if record.get("frame_index") is not None:
            record["host_timing"]["state_sample_monotonic_s"] += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    data = AlohaMiniDataset(recording, episodes=[0], state="joint_current", cameras=[])
    assert data.sample_indices == [0, 1, 2, 3]
    assert data[0]["observation.state"].tolist() == [1.0] * 14


def test_legacy_statistics_require_refit_and_legacy_checkpoint_is_inference_only(
    recording, tmp_path
):
    data = AlohaMiniDataset(
        recording, episodes=[0], state="none", chunk_size=3, image_size=(32, 32)
    )
    options = model_options(state=False)
    component = algorithm("act")
    artifact = training_statistics(component, data, options, "act")
    assert artifact["contract"]["sample_filter"] == "episode_windows_v2"
    artifact["contract"]["sample_filter"] = "required_fields_and_windows_v1"
    source = write_statistics(tmp_path / "old-statistics.json", artifact)
    with pytest.raises(ValueError, match="sample_filter"):
        training_statistics(component, data, options, "act", source)
    checkpoint = tmp_path / "000001"
    model_dir = checkpoint / "pretrained_model"
    stats = data.statistics()
    stats["action"]["mean"] = [999.0] * 18
    save_checkpoint(model_dir, make_policy("act", options), stats, data, training={})
    _, manifest = read_checkpoint(model_dir)
    assert manifest["sample_filter"] == data.sample_filter
    manifest["sample_filter"] = "required_fields_and_windows_v1"
    write_checkpoint_metadata(model_dir, manifest)
    assert NativePolicy(model_dir).manifest["stats"]["action"]["mean"] == [999.0] * 18
    (model_dir / "train_config.json").write_text("{}")
    (checkpoint / "training_state").mkdir()
    (checkpoint / "training_state/state.pt").touch()
    with pytest.raises(ValueError, match="different sample-selection policy"):
        load_training_checkpoint(checkpoint, {}, data, None)


@pytest.mark.parametrize("policy", [None, "required_fields_and_windows_v1"])
def test_resume_without_current_selection_version_is_explicitly_rejected(tmp_path, policy):
    (tmp_path / "training_state").mkdir()
    (tmp_path / "training_state/state.pt").touch()
    (tmp_path / "pretrained_model").mkdir()
    (tmp_path / "pretrained_model/train_config.json").write_text("{}")
    (tmp_path / "pretrained_model/policy.json").write_text(json.dumps({"sample_filter": policy}))
    with pytest.raises(ValueError, match="original trainer.*--policy.path"):
        load_training_checkpoint(
            tmp_path, {}, SimpleNamespace(sample_filter=AlohaMiniDataset.sample_filter), None
        )


def test_detached_training_uses_reviewed_episode_and_resumes_same_population(
    recording, tmp_path, monkeypatch
):
    mark_frame(recording, watchdog_active=True)
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    processes = []
    popen = subprocess.Popen

    def capture(*args, **kwargs):
        assert kwargs["start_new_session"]
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", capture)

    def run(cfg):
        job = launch_training(cfg)
        process = processes[-1]
        try:
            assert process.wait(timeout=120) == 0, Path(job["log"]).read_text()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        assert Path(job["pid_file"]).read_text().strip() == str(process.pid)
        return Path(job["checkpoint"]).resolve(), Path(job["log"])

    cfg = dict(
        dataset=str(recording),
        output_dir=str(tmp_path / "run"),
        policy="act",
        device="cpu",
        state="none",
        steps=1,
        batch_size=2,
        train_episodes=[0],
        val_episodes=[1],
        num_workers=0,
        eval_num_workers=0,
        image_size=[32, 32],
        model=model_options(state=False),
    )
    checkpoint, log = run(cfg)
    _, manifest = read_checkpoint(checkpoint)
    assert manifest["training"]["samples"] == 4
    assert manifest["training"]["excluded_samples"] == 0
    assert manifest["stats"]["action"]["mean"] == [1.5] * 18
    assert manifest["sample_filter"] == "episode_windows_v2"
    assert "window_excluded=0 quality_filtered=0" in log.read_text()
    resume, _ = parse_training_args(
        [f"--config_path={checkpoint / 'train_config.json'}", "--resume=true", "--steps=2"]
    )
    final, _ = run(resume)
    assert read_checkpoint(final)[1]["stats"] == manifest["stats"]
    assert final.parent.name == "000002"
