"""Submitted JSON is an owned snapshot reused by the journal writer."""

import json
from copy import deepcopy
from unittest.mock import patch

import pytest
from test_dataset import frame, jpeg, metadata

from alohamini.datasets.record import _EpisodeWriter


def test_writer_reuses_serialized_snapshot_and_preserves_metadata(tmp_path):
    ds = _EpisodeWriter(tmp_path / "record", fps=30, task="pick", robot_metadata=metadata())
    ds.begin_episode()
    motor_metadata = deepcopy(ds.robot_metadata)
    record = {
        "robot_metadata": motor_metadata,
        "safety": {"label": '中文\n"quoted"', "values": [1, 2]},
        "episode_index": -99,
        "frame_index": -99,
        "feedback_phase": "overridden",
        "event": {"type": "overridden"},
    }
    expected = deepcopy(record)
    expected.update(
        episode_index=0, frame_index=0, feedback_phase="before_issued_command", event=None
    )
    values = frame(ds)
    try:
        # Neither the submitter nor writer may decode a frame's JSON snapshot.
        with patch("alohamini.datasets.record.json.loads", side_effect=AssertionError("decode")):
            assert ds.add_frame(values, {"forward": jpeg()}, record)
            record["safety"]["values"][0] = 99
            motor_metadata["motors"]["arm_left_gripper"]["normalization"] = "changed"
            values["action"][:] = 99
            ds._finish_writer()
        items = [
            json.loads(line) for line in (ds._pending / "journal.jsonl").read_text().splitlines()
        ]
        assert items[0]["record"] == expected
        assert items[0]["frame"]["action"] == [1.0] * len(ds.names)
        assert ds._queued_bytes == 0
    finally:
        ds.close()


def test_dynamic_payload_encodes_once_and_rejected_images_do_not_consume_frame_index(tmp_path):
    class Diagnostic:
        calls = 0

        def tolist(self):
            self.calls += 1
            return [1, 2, 3]

    ds = _EpisodeWriter(tmp_path / "record", fps=30, task="pick", robot_metadata=metadata())
    ds.begin_episode()
    payload = Diagnostic()
    changed = deepcopy(ds.robot_metadata)
    changed["motors"]["arm_left_gripper"]["normalization"] = "changed"
    try:
        assert ds.add_frame(frame(ds), {"forward": b"broken"}, {})
        assert ds.add_frame(
            frame(ds),
            {"forward": jpeg()},
            {
                "diagnostic": payload,
                "robot_metadata": changed,
            },
        )
        ds.event({"type": "test"})
        ds._finish_writer()
        items = [
            json.loads(line) for line in (ds._pending / "journal.jsonl").read_text().splitlines()
        ]
        assert payload.calls == 1
        assert items[0]["record"]["event"]["type"] == "image_rejected"
        assert items[1]["record"]["diagnostic"] == [1, 2, 3]
        assert items[1]["record"]["robot_metadata"] == changed
        assert items[1]["record"]["frame_index"] == items[1]["frame"]["frame_index"] == 0
        assert items[2]["record"]["event"] == {"type": "test"}
        assert ds._queued_bytes == 0
    finally:
        ds.close()


@pytest.mark.parametrize("key", ["safety", "frame_index", "event"])
def test_nonfinite_input_still_fails_before_enqueue_even_in_overridden_fields(tmp_path, key):
    ds = _EpisodeWriter(tmp_path / "record", fps=30, task="pick", robot_metadata=metadata())
    ds.begin_episode()
    try:
        with pytest.raises(ValueError):
            ds.add_frame(frame(ds), {"forward": jpeg()}, {key: float("nan")})
        assert ds.submitted == 0
    finally:
        ds.close()
