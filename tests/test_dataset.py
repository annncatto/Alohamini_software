import json
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from test_teleoperation import snapshot

from alohamini.datasets.images import ImageShards, image_bytes, image_rgb
from alohamini.datasets.native import LocalDataset, dataset_schema, motor_feedback_frame


def metadata(cameras=("forward",)):
    result = snapshot().payload["_robot_metadata"]
    result["cameras"] = list(cameras)
    return result


def jpeg(color=(255, 0, 0), shape=(16, 24, 3)):
    rgb = np.zeros(shape, np.uint8)
    rgb[:] = color
    return cv2.imencode(".jpg", rgb)[1].tobytes()


def frame(dataset):
    result = motor_feedback_frame(dataset.features, {})
    result["observation.state"] = np.zeros(len(dataset.names), np.float32)
    result["action"] = np.ones(len(dataset.names), np.float32)
    return result


def as_png_v1(root):
    """Build a historical PNG fixture without retaining a second capture implementation."""
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info.update(version=1, image_format="png", image_compression=0)
    info.pop("image_color")
    for episode in (root / "episodes").iterdir():
        rows = pq.read_table(episode / "frames.parquet").to_pylist()
        for row in rows:
            for camera in info["robot_metadata"]["cameras"]:
                key = f"observation.images.{camera}"
                pixels = image_rgb(episode, camera, row[key])
                relative = f"images/{camera}/frame_{row['frame_index']:06d}.png"
                target = episode / relative
                target.parent.mkdir(exist_ok=True)
                Image.fromarray(pixels).save(target, compress_level=0)
                row[key] = relative
        schema = dataset_schema(info["features"], info["robot_metadata"]["cameras"], "png")
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), episode / "frames.parquet")
        journal = episode / "journal.jsonl"
        if journal.exists():
            items = [json.loads(line) for line in journal.read_text().splitlines()]
            for item in items:
                if "frame" in item:
                    item["frame"] = rows[item["frame"]["frame_index"]]
            journal.write_text("".join(json.dumps(item) + "\n" for item in items))
    path.write_text(json.dumps(info))
    return info


class LocalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "capture"
        self.dataset = LocalDataset(self.root, fps=30, task="pick", robot_metadata=metadata())
        self.addCleanup(self.dataset.close)

    def test_complete_frames_have_old_vectors_masks_indices_and_correct_rgb(self):
        self.dataset.begin_episode()
        value = frame(self.dataset)
        self.dataset.event({"type": "capture_wait"}, {})
        self.assertTrue(self.dataset.add_frame(value, {"forward": jpeg()}, {"safety": {"test": 1}}))
        value["action"][:] = 99  # The writer owns the submitted row.
        self.dataset.save_episode()
        episode = self.root / "episodes/episode_000000"
        table = pq.read_table(episode / "frames.parquet")
        row = table.to_pylist()[0]
        self.assertEqual(row["action"], [1.0] * 18)
        self.assertEqual(len([key for key in row if key.startswith("motor_feedback.")]), 11)
        self.assertEqual(row["motor_feedback.current_ma_valid"], [0.0] * 18)
        self.assertEqual(
            (row["index"], row["frame_index"], row["episode_index"], row["timestamp"]), (0, 0, 0, 0)
        )
        image = image_rgb(episode, "forward", row["observation.images.forward"])
        self.assertGreater(image[0, 0, 0], 240)
        self.assertLess(image[0, 0, 2], 10)
        safety = [json.loads(line) for line in (episode / "safety.jsonl").read_text().splitlines()]
        self.assertEqual(safety[0]["event"]["type"], "capture_wait")
        self.assertEqual(safety[1]["frame_index"], 0)
        self.assertEqual(safety[-1]["event"]["frame_count"], 1)
        self.assertFalse((episode / "journal.jsonl").exists())

    def test_partial_episode_is_saved_on_close_and_resume_appends_without_overwrite(self):
        self.dataset.begin_episode()
        for _ in range(3):
            self.dataset.add_frame(frame(self.dataset), {"forward": jpeg()}, {})
        self.dataset.close()
        resumed = LocalDataset(
            self.root, fps=30, task="pick", robot_metadata=metadata(), resume=True
        )
        self.addCleanup(resumed.close)
        self.assertEqual((resumed.num_episodes, resumed.total_frames), (1, 3))
        resumed.begin_episode()
        resumed.add_frame(frame(resumed), {"forward": jpeg()}, {})
        resumed.save_episode()
        row = pq.read_table(self.root / "episodes/episode_000001/frames.parquet").to_pylist()[0]
        self.assertEqual((row["index"], row["episode_index"], row["frame_index"]), (3, 1, 0))

    def test_discard_is_recoverable_and_does_not_increment_episode(self):
        self.dataset.begin_episode()
        self.dataset.add_frame(frame(self.dataset), {"forward": jpeg()}, {})
        self.dataset.discard_episode()
        self.assertEqual(self.dataset.num_episodes, 0)
        saved = list((self.root / "discarded").iterdir())
        self.assertEqual(len(saved), 1)
        self.assertTrue((saved[0] / "journal.jsonl").is_file())
        self.dataset.begin_episode()
        self.dataset.add_frame(frame(self.dataset), {"forward": jpeg()}, {})
        self.dataset.save_episode()
        self.assertEqual(self.dataset.num_episodes, 1)

    def test_bad_camera_is_not_fabricated_and_later_valid_frames_survive(self):
        self.dataset.begin_episode()
        self.dataset.add_frame(frame(self.dataset), {"forward": b"bad"}, {})
        self.dataset.add_frame(frame(self.dataset), {"forward": jpeg()}, {})
        self.dataset.save_episode()
        self.assertEqual((self.dataset.saved, self.dataset.rejected_images), (1, 1))
        row = pq.read_table(self.root / "episodes/episode_000000/frames.parquet").to_pylist()[0]
        self.assertEqual(row["frame_index"], 0)
        self.assertIn("frame_000001", row["observation.images.forward"]["member"])

    def test_queue_byte_limit_is_nonblocking_and_counted(self):
        self.dataset.begin_episode()
        with patch.object(self.dataset, "QUEUE_BYTES", 1):
            self.assertFalse(self.dataset.add_frame(frame(self.dataset), {"forward": jpeg()}, {}))
        self.assertEqual(self.dataset.queue_overflows, 1)
        self.assertEqual(self.dataset.submitted, 0)

    def test_frame_queue_budget_counts_utf8_bytes_not_characters(self):
        self.dataset.begin_episode()
        values = frame(self.dataset)
        image = jpeg()
        record = {"note": "采集" * 1000}
        clean = {key: value.tolist() for key, value in values.items()}
        character_budget = (
            len(image)
            + len(json.dumps(clean, ensure_ascii=False))
            + len(json.dumps(record, ensure_ascii=False))
            + 2  # One trailing newline per JSON object.
        )
        with patch.object(self.dataset, "QUEUE_BYTES", character_budget):
            self.assertFalse(self.dataset.add_frame(values, {"forward": image}, record))
        self.assertEqual(self.dataset.queue_overflows, 1)
        self.assertEqual(self.dataset.submitted, 0)
        self.assertEqual(self.dataset._queued_bytes, 0)

    def test_event_queue_respects_byte_limit_and_later_frames_remain_usable(self):
        self.dataset.begin_episode()
        with patch.object(self.dataset, "QUEUE_BYTES", 1):
            self.dataset.event({"type": "capture_wait", "reason": "缺少图像"})
        self.assertEqual(self.dataset.queue_overflows, 1)
        self.dataset.event({"type": "capture_recovered"})
        self.assertTrue(self.dataset.add_frame(frame(self.dataset), {"forward": jpeg()}, {}))
        self.dataset.save_episode()
        self.assertEqual(self.dataset._queued_bytes, 0)
        episode = self.root / "episodes/episode_000000"
        records = [json.loads(line) for line in (episode / "safety.jsonl").read_text().splitlines()]
        events = [row["event"] for row in records if row.get("event")]
        self.assertEqual(
            [event["type"] for event in events], ["capture_recovered", "recorder_closed"]
        )
        self.assertEqual(events[-1]["queue_overflows"], 1)
        self.assertEqual(pq.ParquetFile(episode / "frames.parquet").metadata.num_rows, 1)

    def test_slow_image_worker_does_not_block_submission(self):
        entered, release = threading.Event(), threading.Event()

        def save(*_args):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test stalled")
            raise ValueError("test image")

        with patch("alohamini.datasets.native.validate_wire_image", side_effect=save):
            self.dataset.begin_episode()
            self.dataset.add_frame(frame(self.dataset), {"forward": jpeg()}, {})
            self.assertTrue(entered.wait(1))
            for _ in range(5):
                self.assertTrue(
                    self.dataset.add_frame(frame(self.dataset), {"forward": jpeg()}, {})
                )
            release.set()
            self.dataset.save_episode()

    def test_disk_error_retains_journal_and_refuses_fake_success_or_retry(self):
        self.dataset.begin_episode()
        with patch.object(ImageShards, "append", side_effect=OSError("disk full")):
            self.dataset.add_frame(frame(self.dataset), {"forward": jpeg()}, {})
            with self.assertRaises(OSError):
                self.dataset.save_episode()
        self.assertTrue(self.dataset.save_failed)
        self.assertTrue((self.root / "episodes/episode_000000.pending/journal.jsonl").is_file())
        with self.assertRaises(RuntimeError):
            self.dataset.save_episode()
        self.dataset.close()
        with self.assertRaisesRegex(RuntimeError, "Uncommitted"):
            LocalDataset(self.root, fps=30, task="pick", robot_metadata=metadata(), resume=True)

    def test_commit_error_retains_complete_recoverable_frames(self):
        self.dataset.begin_episode()
        self.dataset.add_frame(frame(self.dataset), {"forward": jpeg()}, {})
        with patch("alohamini.datasets.native._write_json", side_effect=OSError("commit failed")):
            with self.assertRaises(OSError):
                self.dataset.save_episode()
        pending = self.root / "episodes/episode_000000.pending"
        self.assertTrue((pending / "journal.jsonl").is_file())
        self.assertEqual(pq.read_table(pending / "frames.parquet").num_rows, 1)
        self.assertEqual(self.dataset.num_episodes, 0)

    def test_existing_dataset_lock_and_metadata_mismatch_are_rejected(self):
        with self.assertRaises(OSError):
            LocalDataset(self.root, fps=30, task="pick", robot_metadata=metadata(), resume=True)
        self.dataset.close()
        for options in ({"fps": 25}, {"task": "other"}, {"robot_metadata": metadata(())}):
            with self.assertRaises(ValueError):
                LocalDataset(
                    self.root,
                    **{"fps": 30, "task": "pick", "robot_metadata": metadata(), **options},
                    resume=True,
                )

    def test_invalid_vector_or_missing_camera_is_rejected_before_enqueue(self):
        self.dataset.begin_episode()
        with self.assertRaises(ValueError):
            self.dataset.add_frame(frame(self.dataset), {}, {})
        value = frame(self.dataset)
        value["action"][0] = np.nan
        with self.assertRaises(ValueError):
            self.dataset.add_frame(value, {"forward": jpeg()}, {})
        self.assertEqual(self.dataset.submitted, 0)

    def test_camera_shape_is_stable_across_episodes_and_resume(self):
        self.dataset.begin_episode()
        self.dataset.add_frame(frame(self.dataset), {"forward": jpeg()}, {})
        self.dataset.save_episode()
        self.dataset.begin_episode()
        self.dataset.add_frame(frame(self.dataset), {"forward": jpeg(shape=(32, 24, 3))}, {})
        self.dataset.save_episode()
        self.assertEqual(self.dataset.num_episodes, 1)
        self.dataset.close()
        resumed = LocalDataset(
            self.root, fps=30, task="pick", robot_metadata=metadata(), resume=True
        )
        self.addCleanup(resumed.close)
        resumed.begin_episode()
        resumed.add_frame(frame(resumed), {"forward": jpeg(shape=(32, 24, 3))}, {})
        resumed.save_episode()
        self.assertEqual(resumed.num_episodes, 1)
        self.assertEqual(resumed.rejected_images, 1)

    def test_shards_retain_wire_bytes_without_capture_encoding_and_rotate_with_size_bound(self):
        encoded = jpeg()
        self.dataset.begin_episode()
        with patch.object(ImageShards, "MAX_SHARD_BYTES", 10240), patch("cv2.imencode") as encode:
            for _ in range(20):
                self.dataset.add_frame(frame(self.dataset), {"forward": encoded}, {})
            self.dataset.save_episode()
        encode.assert_not_called()
        episode = self.root / "episodes/episode_000000"
        shards = list((episode / "images").glob("*.tar"))
        self.assertGreater(len(shards), 1)
        self.assertFalse(list(episode.rglob("*.png")))
        members = 0
        for path in shards:
            self.assertLessEqual(path.stat().st_size, 10240)
            with tarfile.open(path) as archive:
                for member in archive:
                    self.assertEqual(archive.extractfile(member).read(), encoded)
                    members += 1
        rows = pq.read_table(episode / "frames.parquet").to_pylist()
        self.assertEqual(members, 20)
        for row in rows:
            self.assertEqual(
                image_bytes(episode, "forward", row["observation.images.forward"]), encoded
            )

    def test_rejected_camera_group_does_not_fix_shapes_or_append_orphan_images(self):
        root = self.root.with_name("two-cameras")
        dataset = LocalDataset(
            root, fps=30, task="pick", robot_metadata=metadata(("forward", "wrist"))
        )
        self.addCleanup(dataset.close)
        dataset.begin_episode()
        dataset.add_frame(frame(dataset), {"forward": jpeg(shape=(32, 24, 3)), "wrist": b"bad"}, {})
        dataset.add_frame(frame(dataset), {"forward": jpeg(), "wrist": jpeg()}, {})
        dataset.save_episode()
        self.assertEqual((dataset.saved, dataset.rejected_images), (1, 1))
        with tarfile.open(root / "episodes/episode_000000/images/chunk-000000.tar") as archive:
            self.assertEqual(
                archive.getnames(), ["forward/frame_000001.jpg", "wrist/frame_000001.jpg"]
            )
