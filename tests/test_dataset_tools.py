import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from test_dataset import as_png_v1, frame, jpeg, metadata

from alohamini.cli import main
from alohamini.datasets.images import image_bytes, image_path, image_rgb
from alohamini.datasets.native import LocalDataset, motor_feedback_frame
from alohamini.datasets.tools import IntegrityChecker, export_dataset
from alohamini.datasets.video import generate_previews, inspect_video


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


class DatasetToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.output = Path(self.temp.name) / "output"
        self.dataset = LocalDataset(self.root, fps=30, task="pick", robot_metadata=metadata())
        self.addCleanup(self.dataset.close)

    def add_frames(self, count=3, *, start=0):
        for i in range(start, start + count):
            value = frame(self.dataset)
            motors = {
                name: {
                    "sample_started_s": 100 + i / 30,
                    "sample_finished_s": 100.001 + i / 30,
                    **{
                        field: 0
                        for field in (
                            "position_raw",
                            "velocity_raw",
                            "current_raw",
                            "current_ma",
                            "load_raw",
                            "voltage_raw",
                            "temperature_raw",
                            "status_raw",
                            "moving",
                        )
                    },
                }
                for name in self.dataset.features["observation.motor_current_ma"]["names"]
            }
            value.update(
                motor_feedback_frame(self.dataset.features, {"version": 1, "motors": motors})
            )
            record = {
                "safety": {},
                "robot_metadata": self.dataset.robot_metadata,
                "requested_action": dict.fromkeys(self.dataset.names, 1.0),
                "client_timing": {
                    "observation_received_monotonic_s": 9000 + i / 30,
                    "action_sample_started_monotonic_s": 9000.001 + i / 30,
                    "action_sample_finished_monotonic_s": 9000.002 + i / 30,
                    "command_sent_monotonic_s": 9000.003 + i / 30,
                },
                "host_timing": {
                    "state_sample_monotonic_s": 100 + i / 30,
                    "camera_capture_monotonic_s": {"forward": 100.01 + i / 30},
                },
            }
            self.assertTrue(self.dataset.add_frame(value, {"forward": jpeg()}, record))

    def finish(self):
        self.dataset.begin_episode()
        self.add_frames()
        self.dataset.close()
        return self.root / "episodes/episode_000000"

    def pending(self):
        self.dataset.begin_episode()
        self.add_frames()
        with patch(
            "alohamini.datasets.native._write_json", side_effect=OSError("interrupted commit")
        ):
            with self.assertRaises(OSError):
                self.dataset.save_episode()
        self.dataset.close()
        return self.root / f"episodes/episode_{self.dataset.num_episodes:06d}.pending"

    def check(self, **kwargs):
        return IntegrityChecker(self.root, **kwargs).run()

    def codes(self, report):
        return {item["code"] for item in report["issues"]}

    def change_rows(self, episode, key, value):
        path = episode / "frames.parquet"
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows[0][key] = value
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)

    def change_safety(self, episode, mutate):
        path = episode / "safety.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(rows)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_read_only_check_passes_with_separate_host_and_client_clock_origins(self):
        self.finish()
        before = hashes(self.root)
        report = self.check(decode_images=True)
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["training_review"], "not_assessed")
        self.assertEqual(report["summary"], {"episodes": 1, "frames": 3, "pending_episodes": 0})
        self.assertEqual(hashes(self.root), before)

    def test_busy_dataset_is_not_examined_or_exported(self):
        report = self.check()
        self.assertFalse(report["valid"])
        self.assertIn("in use", report["issues"][0]["message"])
        with self.assertRaisesRegex(RuntimeError, "in use"):
            export_dataset(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_mp4_previews_preserve_rgb_frame_order_and_source_files(self):
        import av

        self.dataset.begin_episode()
        colors = [(250, 0, 0), (0, 250, 0), (0, 0, 250)]
        for color in colors:
            self.dataset.add_frame(frame(self.dataset), {"forward": jpeg(color)}, {})
        self.dataset.close()
        before = hashes(self.root)
        result = generate_previews(self.root)
        self.assertEqual(result["generated"], 1)
        path = self.root / "previews/episode_000000/forward.mp4"
        self.assertEqual(
            inspect_video(path, decode=True), {"frames": 3, "fps": 30.0, "shape": [16, 24, 3]}
        )
        with av.open(str(path)) as container:
            pixels = [
                f.to_ndarray(format="rgb24").mean(axis=(0, 1)) for f in container.decode(video=0)
            ]
        for actual, expected in zip(pixels, colors, strict=True):
            np.testing.assert_allclose(actual, expected, atol=8)
        after = hashes(self.root)
        self.assertEqual({k: after[k] for k in before}, before)
        reused = generate_previews(self.root)
        self.assertEqual((reused["generated"], reused["reused"]), (0, 1))
        self.assertTrue(self.check(decode_videos=True)["valid"])

    def test_preview_failure_retains_originals_and_can_retry(self):
        self.finish()
        before = hashes(self.root)
        with patch("alohamini.datasets.video._encode_frames", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "source unchanged"):
                generate_previews(self.root)
        self.assertFalse((self.root / "previews/episode_000000").exists())
        after = hashes(self.root)
        self.assertEqual({k: after[k] for k in before}, before)
        result = generate_previews(self.root)
        self.assertEqual(result["generated"], 1)
        report = self.check(decode_videos=True)
        self.assertTrue(report["valid"])
        self.assertIn("PREVIEW_INCOMPLETE", self.codes(report))

    def test_png_preview_reuse_requires_unchanged_image_contents(self):
        from PIL import Image

        episode = self.finish()
        as_png_v1(self.root)
        generate_previews(self.root)
        self.assertEqual(generate_previews(self.root)["reused"], 1)
        before = hashes(self.root / "previews")
        path = episode / "images/forward/frame_000000.png"
        Image.fromarray(np.full((16, 24, 3), (0, 250, 0), np.uint8)).save(path)
        report = self.check(decode_images=True, decode_videos=True)
        self.assertTrue(report["valid"], report)
        self.assertIn("PREVIEW_INVALID", self.codes(report))
        with self.assertRaisesRegex(ValueError, "does not match"):
            generate_previews(self.root)
        self.assertEqual(hashes(self.root / "previews"), before)
        self.assertEqual(generate_previews(self.root, self.output)["generated"], 1)

    def test_preview_handles_multiple_cameras_and_pads_odd_dimensions(self):
        from contextlib import closing

        self.dataset.close()
        source = self.root.parent / "two_cameras"
        with closing(
            LocalDataset(
                source, fps=30, task="pick", robot_metadata=metadata(("forward", "wrist_left"))
            )
        ) as dataset:
            dataset.begin_episode()
            for _ in range(2):
                dataset.add_frame(
                    frame(dataset),
                    {
                        "forward": jpeg(shape=(15, 23, 3)),
                        "wrist_left": jpeg((0, 250, 0)),
                    },
                    {},
                )
        generate_previews(source)
        directory = source / "previews/episode_000000"
        for camera in ("forward", "wrist_left"):
            self.assertEqual(
                inspect_video(directory / f"{camera}.mp4", decode=True),
                {"frames": 2, "fps": 30.0, "shape": [16, 24, 3]},
            )

    def test_preview_resume_preserves_existing_videos(self):
        from contextlib import closing

        self.finish()
        generate_previews(self.root)
        original = hashes(self.root / "previews/episode_000000")
        with closing(
            LocalDataset(self.root, fps=30, task="pick", robot_metadata=metadata(), resume=True)
        ) as dataset:
            dataset.begin_episode()
            dataset.add_frame(frame(dataset), {"forward": jpeg()}, {})
        result = generate_previews(self.root)
        self.assertEqual((result["generated"], result["reused"]), (1, 1))
        self.assertEqual(hashes(self.root / "previews/episode_000000"), original)

    def test_preview_corruption_warns_but_does_not_invalidate_raw_capture(self):
        self.finish()
        generate_previews(self.root)
        path = self.root / "previews/episode_000000/forward.mp4"
        path.write_bytes(b"broken")
        report = self.check(decode_videos=True)
        self.assertTrue(report["valid"])
        self.assertIn("PREVIEW_INVALID", self.codes(report))
        with self.assertRaises(ValueError):
            generate_previews(self.root)
        self.assertEqual(generate_previews(self.root, self.output)["generated"], 1)
        manifest = self.root / "previews/episode_000000/manifest.json"
        manifest.write_text("[]")
        self.assertIn("PREVIEW_INVALID", self.codes(self.check()))

    def test_preview_refuses_active_capture_and_ambiguous_source(self):
        with self.assertRaisesRegex(RuntimeError, "in use"):
            generate_previews(self.root)
        episode = self.finish()
        self.change_rows(episode, "action", [float("nan")] * 18)
        with self.assertRaises(ValueError):
            generate_previews(self.root)
        self.assertFalse((self.root / "previews").exists())

    def test_all_bad_images_are_reported_without_stopping_at_first(self):
        episode = self.finish()
        path = episode / "images/chunk-000000.tar"
        path.write_bytes(b"broken")
        report = self.check()
        self.assertEqual(sum(item["code"] == "IMAGE_INVALID" for item in report["issues"]), 3)
        self.assertEqual(report["summary"]["frames"], 3)

    def test_one_stalled_camera_is_not_hidden_by_median_timestamps(self):
        episode = self.finish()

        def mutate(rows):
            for row in rows[:3]:
                cameras = row["host_timing"]["camera_capture_monotonic_s"]
                cameras["wrist_left"] = 100.01
                cameras["wrist_right"] = cameras["forward"]

        self.change_safety(episode, mutate)
        report = self.check()
        self.assertIn("SAFETY_CAMERA_GAP", self.codes(report))
        self.assertIn("SAFETY_CAMERA_SKEW", self.codes(report))

    def test_cli_can_save_report_and_generate_preview(self):
        self.finish()
        report = self.root.parent / "report.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(["dataset", "check", str(self.root), "--output-json", str(report)]), 0
            )
            self.assertEqual(main(["dataset", "preview", str(self.root)]), 0)
            self.assertEqual(main(["dataset", "check", str(self.root), "--decode-videos"]), 0)
        self.assertTrue(json.loads(report.read_text())["valid"])

    def test_numeric_nan_and_nonbinary_masks_are_errors(self):
        episode = self.finish()
        self.change_rows(episode, "action", [float("nan")] * 18)
        self.change_rows(episode, "motor_feedback.current_ma_valid", [2.0] * 18)
        report = self.check()
        self.assertFalse(report["valid"])
        self.assertIn("FEATURE_NONFINITE", self.codes(report))
        self.assertIn("MOTOR_FEEDBACK_MASK_INVALID", self.codes(report))

    def test_missing_feedback_is_valid_structure_but_requires_training_review(self):
        self.dataset.begin_episode()
        self.dataset.add_frame(frame(self.dataset), {"forward": jpeg()}, {})
        self.dataset.close()
        report = self.check()
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["training_review"], "required")
        self.assertIn("MOTOR_FEEDBACK_INCOMPLETE", self.codes(report))

    def test_indices_and_fixed_fps_timestamps_are_checked(self):
        episode = self.finish()
        self.change_rows(episode, "index", 20)
        self.change_rows(episode, "timestamp", 5.0)
        self.assertIn("FRAME_INDEX_INVALID", self.codes(self.check()))
        self.assertIn("FRAME_TIMESTAMP_INVALID", self.codes(self.check()))

    def test_changed_column_schema_and_feature_names_are_rejected(self):
        episode = self.finish()
        path = episode / "frames.parquet"
        table = pq.read_table(path).drop(["action"])
        pq.write_table(table, path)
        self.assertFalse(self.check()["valid"])
        path = self.root / "meta/info.json"
        info = json.loads(path.read_text())
        info["features"]["action"]["names"].reverse()
        path.write_text(json.dumps(info))
        self.assertIn("DATASET_UNREADABLE", self.codes(self.check()))

    def test_repeated_missing_and_external_image_references_are_rejected(self):
        episode = self.finish()
        refs = pq.read_table(episode / "frames.parquet")["observation.images.forward"].to_pylist()
        for value in (
            refs[1],
            {**refs[0], "path": "images/chunk-999999.tar"},
            {**refs[0], "path": "../../outside.tar"},
            {**refs[0], "path": "/tmp/outside.tar"},
            {**refs[0], "offset": 1},
            {**refs[0], "size": 2**40},
            {**refs[0], "member": "wrist/frame_000000.jpg"},
        ):
            self.change_rows(episode, "observation.images.forward", value)
            self.assertFalse(self.check()["valid"], value)

    def test_corrupt_image_body_is_detected_even_without_decode(self):
        episode = self.finish()
        image = episode / "images/chunk-000000.tar"
        image.write_bytes(image.read_bytes()[:600])
        self.assertFalse(self.check()["valid"])
        self.assertFalse(self.check(decode_images=True)["valid"])

    def test_image_header_and_color_convention_are_validated(self):
        episode = self.finish()
        image = episode / "images/chunk-000000.tar"
        with image.open("r+b") as stream:
            stream.write(b"bad")
        self.assertFalse(self.check()["valid"])
        path = self.root / "meta/info.json"
        info = json.loads(path.read_text())
        info["image_color"] = "standard_jpeg_rgb"
        path.write_text(json.dumps(info))
        self.assertIn("DATASET_UNREADABLE", self.codes(self.check()))

    def test_safety_frame_coverage_is_required(self):
        episode = self.finish()
        self.change_safety(episode, lambda rows: rows.pop(1))
        self.assertIn("SAFETY_FRAME_COVERAGE", self.codes(self.check()))

    def test_native_fault_is_preserved_and_requires_training_review(self):
        episode = self.finish()
        self.change_safety(episode, lambda rows: rows[0]["safety"].update({"fault": "overcurrent"}))
        report = export_dataset(self.root, self.output)
        self.assertTrue(report["valid"])
        self.assertEqual(report["training_review"], "required")
        self.assertIn("SAFETY_REVIEW_REQUIRED", self.codes(report))
        self.assertEqual(
            (episode / "safety.jsonl").read_bytes(),
            (self.output / "episodes/episode_000000/safety.jsonl").read_bytes(),
        )

    def test_action_pairing_is_not_silently_changed(self):
        episode = self.finish()
        self.change_safety(
            episode, lambda rows: rows[0]["requested_action"].update({self.dataset.names[0]: 42})
        )
        self.assertIn("SAFETY_ACTION_MISMATCH", self.codes(self.check()))

    def test_client_chronology_errors_and_capture_gaps_are_distinct(self):
        episode = self.finish()
        self.change_safety(
            episode,
            lambda rows: rows[1]["host_timing"]["camera_capture_monotonic_s"].update(
                {"forward": 200}
            ),
        )
        report = self.check()
        self.assertTrue(report["valid"], report)
        self.assertIn("SAFETY_CAPTURE_GAP", self.codes(report))
        self.change_safety(
            episode,
            lambda rows: rows[0]["client_timing"].update({"command_sent_monotonic_s": 8999}),
        )
        self.assertFalse(self.check()["valid"])

    def test_lossless_export_keeps_pixels_numeric_bytes_and_sidecar_bytes(self):
        episode = self.finish()
        before = hashes(self.root)
        report = export_dataset(self.root, self.output)
        self.assertTrue(report["valid"], report)
        self.assertEqual(hashes(self.root), before)
        exported = self.output / "episodes/episode_000000"
        for name in ("frames.parquet", "safety.jsonl", "episode.json"):
            self.assertEqual((episode / name).read_bytes(), (exported / name).read_bytes())
        for source in (episode / "images").glob("*.tar"):
            target = exported / source.relative_to(episode)
            self.assertEqual(target.read_bytes(), source.read_bytes())
        self.assertEqual(
            (self.output / "meta/info.json").read_bytes(),
            (self.root / "meta/info.json").read_bytes(),
        )

    def test_old_png_dataset_is_readable_and_exported_without_source_changes(self):
        episode = self.finish()
        as_png_v1(self.root)
        before = hashes(self.root)
        self.assertTrue(self.check(decode_images=True)["valid"])
        self.assertTrue(export_dataset(self.root, self.output)["valid"])
        exported = self.output / "episodes/episode_000000"
        for row in pq.read_table(episode / "frames.parquet").to_pylist():
            reference = row["observation.images.forward"]
            np.testing.assert_array_equal(
                image_rgb(episode, "forward", reference), image_rgb(exported, "forward", reference)
            )
            self.assertLess(
                (exported / reference).stat().st_size, (episode / reference).stat().st_size
            )
        self.assertEqual(hashes(self.root), before)
        self.assertEqual(
            json.loads((self.output / "meta/info.json").read_text())["image_compression"], 6
        )

    def test_output_must_be_new_and_separate(self):
        self.finish()
        for path in (self.root, self.root / "nested", self.root.parent):
            with self.assertRaises((ValueError, FileExistsError)):
                export_dataset(self.root, path)
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            export_dataset(self.root, self.output)

    def test_symlink_source_is_rejected_before_copy(self):
        self.finish()
        (self.root / "extra_link").symlink_to(self.root / "meta/info.json")
        self.assertFalse(self.check()["valid"])
        with self.assertRaises(ValueError):
            export_dataset(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_recover_interrupted_commit_keeps_indices_and_is_resumable(self):
        self.dataset.begin_episode()
        self.add_frames(count=2)
        self.dataset.save_episode()
        pending = self.pending()
        before = hashes(self.root)
        self.assertIn("EPISODE_RECOVERY_PENDING", self.codes(self.check()))
        report = export_dataset(self.root, self.output, recover=True)
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["summary"]["frames"], 5)
        self.assertEqual(hashes(self.root), before)
        recovered = self.output / "episodes/episode_000001"
        self.assertEqual(
            pq.read_table(recovered / "frames.parquet")["index"].to_pylist(), [2, 3, 4]
        )
        for original, saved in zip(
            pq.read_table(pending / "frames.parquet").to_pylist(),
            pq.read_table(recovered / "frames.parquet").to_pylist(),
            strict=True,
        ):
            self.assertEqual(
                image_bytes(pending, "forward", original["observation.images.forward"]),
                image_bytes(recovered, "forward", saved["observation.images.forward"]),
            )
        resumed = LocalDataset(
            self.output, fps=30, task="pick", robot_metadata=metadata(), resume=True
        )
        self.addCleanup(resumed.close)
        self.assertEqual((resumed.num_episodes, resumed.total_frames), (2, 5))
        self.assertEqual(report["training_review"], "required")

    def test_recovery_ignores_only_unfinished_tail_not_bad_complete_records(self):
        pending = self.pending()
        journal = pending / "journal.jsonl"
        original = journal.read_bytes()
        journal.write_bytes(original + b'{"frame":')
        report = export_dataset(self.root, self.output, recover=True)
        self.assertEqual(report["summary"]["frames"], 3)
        journal.write_bytes(original + b'{"frame":\n')
        second = self.output.with_name("bad-record")
        with self.assertRaisesRegex(RuntimeError, "source unchanged"):
            export_dataset(self.root, second, recover=True)
        self.assertFalse(second.exists())

    def test_recovery_stops_before_incomplete_image_group_without_stitching(self):
        pending = self.pending()
        refs = pq.read_table(pending / "frames.parquet")["observation.images.forward"].to_pylist()
        with image_path(pending, "forward", refs[1]).open("r+b") as stream:
            stream.seek(refs[1]["offset"])
            stream.write(b"bad")
        report = export_dataset(self.root, self.output, recover=True)
        self.assertEqual(report["summary"]["frames"], 1)
        self.assertEqual(report["training_review"], "required")
        self.assertEqual(image_bytes(pending, "forward", refs[2]), jpeg())

    def test_recovery_reads_complete_prefix_from_unclosed_truncated_tar(self):
        pending = self.pending()
        refs = pq.read_table(pending / "frames.parquet")["observation.images.forward"].to_pylist()
        with image_path(pending, "forward", refs[1]).open("r+b") as stream:
            stream.truncate(refs[1]["offset"] + refs[1]["size"] // 2)
        before = hashes(self.root)
        report = export_dataset(self.root, self.output, recover=True)
        self.assertEqual(report["summary"]["frames"], 1)
        self.assertTrue(report["valid"], report)
        self.assertEqual(hashes(self.root), before)

    def test_old_png_pending_journal_can_still_be_recovered(self):
        self.pending()
        as_png_v1(self.root)
        before = hashes(self.root)
        report = export_dataset(self.root, self.output, recover=True)
        self.assertEqual(report["summary"]["frames"], 3)
        self.assertTrue(report["valid"], report)
        self.assertEqual(hashes(self.root), before)

    def test_recovery_rejects_ambiguous_numeric_data_and_retains_source(self):
        pending = self.pending()
        journal = pending / "journal.jsonl"
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        rows[1]["frame"]["index"] = 99
        journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
        before = hashes(self.root)
        with self.assertRaises(RuntimeError):
            export_dataset(self.root, self.output, recover=True)
        self.assertFalse(self.output.exists())
        self.assertEqual(hashes(self.root), before)

    def test_recovery_does_not_coerce_fractional_indices_to_integers(self):
        pending = self.pending()
        journal = pending / "journal.jsonl"
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        rows[0]["frame"]["index"] = 0.5
        journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaises(RuntimeError):
            export_dataset(self.root, self.output, recover=True)
        self.assertFalse(self.output.exists())

    def test_malformed_info_is_reported_without_traceback(self):
        self.finish()
        path = self.root / "meta/info.json"
        original = path.read_text()
        for invalid in ([], None, {}, {**json.loads(original), "robot_metadata": []}):
            path.write_text(json.dumps(invalid))
            self.assertFalse(self.check()["valid"])

    def test_check_cli_never_imports_frameworks(self):
        import os
        import subprocess
        import sys

        self.finish()
        script = (
            "import sys; from alohamini.cli import main; "
            "assert main(['dataset', 'check', sys.argv[1]]) == 0; "
            "assert not {'torch', 'lerobot', 'huggingface_hub', 'rclpy', 'serial'} "
            "& sys.modules.keys()"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.root)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_copy_failure_retains_stage_and_never_publishes_half_dataset(self):
        self.finish()
        before = hashes(self.root)
        with patch("alohamini.datasets.tools.shutil.copyfile", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "files retained"):
                export_dataset(self.root, self.output)
        self.assertEqual(hashes(self.root), before)
        self.assertFalse(self.output.exists())
        self.assertEqual(len(list(self.output.parent.glob("output.pending-*"))), 1)

    def test_cli_is_offline_and_warning_exit_status_is_explicit(self):
        self.dataset.begin_episode()
        self.dataset.add_frame(frame(self.dataset), {"forward": jpeg()}, {})
        self.dataset.close()
        with (
            patch("alohamini.cli.HostClient") as client,
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            self.assertEqual(main(["dataset", "check", str(self.root), "--decode-images"]), 0)
            self.assertEqual(main(["dataset", "check", str(self.root), "--fail-on-warnings"]), 1)
        client.assert_not_called()
        self.assertIn("Training review: required", out.getvalue())


if __name__ == "__main__":
    unittest.main()
