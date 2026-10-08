import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from test_dataset import as_png_v1, frame, jpeg
from test_dataset_tools import hashes
from test_teleoperation import snapshot

from alohamini.cli import main
from alohamini.datasets.images import decode_host_image, image_rgb
from alohamini.datasets.lerobotv3 import export_lerobot
from alohamini.datasets.record import (
    FEEDBACK_FIELDS,
    StateSelection,
    motor_feedback_frame,
)
from alohamini.datasets.record import (
    _EpisodeWriter as LocalDataset,
)


class LeRobotExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.output = Path(self.temp.name) / "export"

    def make_source(self, model="alohamini2pro", *, episodes=2, frames=3, invalid=False):
        metadata = snapshot(model).payload["_robot_metadata"]
        metadata["cameras"] = ["forward"]
        for index, motor in enumerate(metadata["motors"].values()):
            motor.update(range_min=1000, range_max=3000, drive_mode=index % 2)
        dataset = LocalDataset(self.root, fps=30, task="pick", robot_metadata=metadata)
        try:
            for _episode in range(episodes):
                dataset.begin_episode()
                for index in range(frames):
                    sample = {key: 0.0 for key in FEEDBACK_FIELDS}
                    sample.update(
                        velocity_raw=100.0 + index,
                        current_ma=650.0 + index,
                        sample_started_s=100.0 + index / 30,
                        sample_finished_s=100.001 + index / 30,
                    )
                    motors = {
                        name: dict(sample)
                        for name in dataset.features["observation.motor_current_ma"]["names"]
                    }
                    if invalid:
                        motors[dataset.names[0].removesuffix(".pos")].pop("current_ma")
                    value = frame(dataset)
                    value.update(
                        motor_feedback_frame(dataset.features, {"version": 1, "motors": motors})
                    )
                    value["observation.state"] = np.arange(len(dataset.names), dtype=np.float32)
                    value["action"] = np.arange(len(dataset.names), dtype=np.float32) + 0.25
                    dataset.add_frame(
                        value,
                        {"forward": jpeg()},
                        {},
                    )
                dataset.save_episode()
        finally:
            dataset.close()
        return json.loads((self.root / "meta/info.json").read_text())

    def read_rows(self):
        result = []
        for path in sorted((self.output / "data").rglob("*.parquet")):
            result.extend(pq.read_table(path).to_pylist())
        return result

    def test_fork_wire_is_saved_as_rgb_video_and_exported_without_encoding(self):
        import cv2

        from alohamini.datasets.tools import check_dataset
        from alohamini.protocol import HostSnapshot, decode_reply, encode_reply

        rgb = np.zeros((32, 32, 3), np.uint8)
        rgb[:, :16] = (240, 20, 10)
        rgb[:, 16:] = (10, 20, 240)
        encoded = cv2.imencode(".jpg", rgb)[1].tobytes()
        metadata = snapshot().payload["_robot_metadata"]
        metadata["cameras"] = ["forward"]
        payload = {**snapshot().payload, "_robot_metadata": metadata}
        payload, images = decode_reply(
            [b"rgb:full", *encode_reply(payload, {"forward": encoded})],
            token=b"rgb:full",
            include_images=True,
        )
        observation = HostSnapshot(payload, images, 1.0, 1.1)
        with contextlib.closing(
            LocalDataset(
                self.root,
                fps=30,
                task="pick",
                robot_metadata=metadata,
            )
        ) as dataset:
            dataset.begin_episode()
            for _ in range(2):
                dataset.add_frame(frame(dataset), observation.images, {})
            dataset._finish_writer()
            # Exactly the fork PC decoded pixels enter video encoding.
            with Image.open(dataset._pending / "images/forward/frame_000000.png") as image:
                np.testing.assert_array_equal(np.asarray(image), decode_host_image(encoded))
            dataset.save_episode()
        before = hashes(self.root)
        self.assertTrue(check_dataset(self.root, decode_images=True, decode_videos=True)["valid"])
        with (
            patch("PIL.Image.Image.save", side_effect=AssertionError("Unexpected image encoding")),
            patch(
                "alohamini.datasets.video._encode_frames",
                side_effect=AssertionError("Unexpected video encoding"),
            ),
        ):
            export_lerobot(self.root, self.output)
            visual = self.output.with_name("visual")
            export_lerobot(self.output, visual, vision_only=True)
        self.assertEqual(before, hashes(self.root))
        episode = self.root / "episodes/episode_000000"
        original = pq.read_table(episode / "frames.parquet").to_pylist()[0]
        ref = original["observation.images.forward"]
        stored = (episode / "videos/forward.mp4").read_bytes()
        self.assertFalse((episode / "images").exists())
        self.assertFalse((self.root / "previews").exists())
        info = json.loads((self.root / "meta/info.json").read_text())
        self.assertEqual((info["image_format"], info["image_color"]), ("rgb-mp4", "rgb"))
        for root in (self.output, visual):
            self.assertEqual(
                (root / "videos/observation.images.forward/chunk-000/file-000.mp4").read_bytes(),
                stored,
            )
        pixels = image_rgb(episode, "forward", ref)
        self.assertGreater(int(pixels[8, 4, 0]), 220)
        self.assertLess(int(pixels[8, 4, 2]), 30)
        self.assertGreater(int(pixels[8, 24, 2]), 220)
        for row in self.read_rows():
            self.assertNotIn("observation.images.forward", row)
            self.assertEqual(row["action"], original["action"])

    def test_velocity_current_state_preserves_every_other_field_and_source(self):
        info = self.make_source()
        before = hashes(self.root)
        report = export_lerobot(
            self.root, self.output, state="joint_velocity,joint_current,base_velocity,lift_height"
        )
        self.assertTrue(report["valid"])
        self.assertEqual(hashes(self.root), before)
        rows = self.read_rows()
        self.assertEqual(len(rows), 6)
        row = rows[0]
        self.assertEqual(len(row["observation.state"]), 32)
        self.assertEqual(len(row["action"]), 18)
        self.assertEqual(row["action"], [i + 0.25 for i in range(18)])
        self.assertEqual(row["observation.source_state"], list(range(18)))
        self.assertAlmostEqual(row["observation.state"][0], 10)
        self.assertAlmostEqual(row["observation.state"][1], -10)
        self.assertAlmostEqual(row["observation.state"][6], 5)  # gripper percent/s
        self.assertAlmostEqual(row["observation.state"][14], 0.65)
        self.assertEqual(row["observation.state"][-4:], list(range(14, 18)))
        for key in info["features"]:
            if key != "observation.state":
                original = pq.read_table(self.root / "episodes/episode_000000/frames.parquet")[key][
                    0
                ].as_py()
                self.assertEqual(row[key], original)
        for index in range(2):
            self.assertEqual(
                (self.root / f"episodes/episode_{index:06d}/safety.jsonl").read_bytes(),
                (self.output / f"meta/safety/episode_{index:06d}.jsonl").read_bytes(),
            )
        archive_info = json.loads((self.output / "meta/info.json").read_text())
        self.assertEqual(archive_info["codebase_version"], "v3.0")
        self.assertEqual(archive_info["total_frames"], 6)
        self.assertEqual(archive_info["features"]["observation.state"]["shape"], [32])
        stats = json.loads((self.output / "meta/stats.json").read_text())
        self.assertEqual(len(stats["observation.state"]["mean"]), 32)
        self.assertEqual(stats["observation.state"]["count"], [6])
        self.assertIn("q01", stats["observation.state"])
        self.assertEqual(len(stats["observation.images.forward"]["mean"]), 3)

    def test_default_state_is_unchanged_and_all_model_dimensions_are_supported(self):
        for model, dimension in (("alohamini1", 16), ("alohamini2", 18), ("alohamini2pro", 18)):
            with self.subTest(model=model):
                self.root = self.root.with_name(model)
                self.output = self.output.with_name(f"{model}-export")
                self.make_source(model)
                export_lerobot(self.root, self.output)
                row = self.read_rows()[0]
                self.assertEqual(row["observation.state"], list(range(dimension)))
                self.assertNotIn("observation.source_state", row)

    def test_visual_v3_projection_preserves_retained_values_and_sidecars(self):
        self.make_source()
        export_lerobot(self.root, self.output)
        before = hashes(self.output)
        visual = self.output.with_name("visual")
        self.assertEqual(
            main(
                [
                    "dataset",
                    "export",
                    str(self.output),
                    "--output",
                    str(visual),
                    "--format",
                    "lerobot-v3",
                    "--vision-only",
                ]
            ),
            0,
        )
        self.assertEqual(before, hashes(self.output))
        info = json.loads((visual / "meta/info.json").read_text())
        self.assertNotIn("observation.state", info["features"])
        self.assertEqual(
            set(info["features"]),
            {
                "action",
                "observation.images.forward",
                "timestamp",
                "index",
                "frame_index",
                "episode_index",
                "task_index",
            },
        )
        original = self.read_rows()
        rows = pq.read_table(visual / "data/chunk-000/file-000.parquet").to_pylist()
        self.assertEqual(
            rows,
            [
                {k: r[k] for k, ft in info["features"].items() if ft["dtype"] != "video"}
                for r in original
            ],
        )
        stats = json.loads((visual / "meta/stats.json").read_text())
        self.assertEqual(set(stats), set(info["features"]))
        for episode in range(2):
            relative = f"meta/safety/episode_{episode:06d}.jsonl"
            self.assertEqual(
                (self.output / relative).read_bytes(), (visual / relative).read_bytes()
            )
        with self.assertRaises(FileExistsError):
            export_lerobot(self.output, visual, vision_only=True)

    def test_visual_export_rejects_conflicting_state_and_native_format(self):
        self.make_source()
        with self.assertRaisesRegex(ValueError, "custom --state"):
            export_lerobot(self.root, self.output, vision_only=True, state="joint_current")
        self.assertEqual(
            main(
                [
                    "dataset",
                    "export",
                    str(self.root),
                    "--output",
                    str(self.output),
                    "--format",
                    "native",
                    "--vision-only",
                ]
            ),
            1,
        )
        self.assertFalse(self.output.exists())

    def test_images_are_embedded_losslessly_with_no_absolute_path_dependency(self):
        self.make_source()
        as_png_v1(self.root)
        export_lerobot(self.root, self.output)
        row = self.read_rows()[0]
        encoded = row["observation.images.forward"]
        self.assertIsNone(encoded["path"])
        episode = self.root / "episodes/episode_000000"
        original = pq.read_table(episode / "frames.parquet").to_pylist()[0]
        with Image.open(io.BytesIO(encoded["bytes"])) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertGreater(image.getpixel((0, 0))[0], 240)
            np.testing.assert_array_equal(
                np.asarray(image),
                image_rgb(episode, "forward", original["observation.images.forward"]),
            )

    def test_old_png_dataset_exports_with_identical_embedded_image_bytes(self):
        self.make_source()
        as_png_v1(self.root)
        before = hashes(self.root)
        export_lerobot(self.root, self.output)
        original = self.root / "episodes/episode_000000/images/forward/frame_000000.png"
        self.assertEqual(
            self.read_rows()[0]["observation.images.forward"]["bytes"], original.read_bytes()
        )
        self.assertEqual(hashes(self.root), before)

    def test_missing_selected_current_rejects_before_creating_output(self):
        self.make_source(invalid=True)
        with self.assertRaisesRegex(ValueError, "frame 0: Unavailable"):
            export_lerobot(self.root, self.output, state="joint_velocity,joint_current")
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.output.parent.glob("export.pending-*")))
        # The same missing feedback can remain stored without becoming training state.
        report = export_lerobot(self.root, self.output)
        self.assertEqual(report["training_review"], "required")
        self.assertEqual(self.read_rows()[0]["motor_feedback.current_ma_valid"][0], 0)

    def test_selection_rejects_unknown_duplicate_or_uncalibrated_fields(self):
        info = self.make_source()
        for selection in ("", "joint_position,joint_position", "joint_torque", "current"):
            with self.assertRaises(ValueError):
                StateSelection(info, selection)
        first = next(iter(info["robot_metadata"]["motors"].values()))
        first.pop("drive_mode")
        with self.assertRaisesRegex(ValueError, "ranges/direction"):
            StateSelection(info, "joint_velocity")

    def test_selection_order_and_degree_direction_follow_host_units(self):
        info = self.make_source()
        first = next(iter(info["robot_metadata"]["motors"].values()))
        first.update(normalization="degrees", drive_mode=1)
        selection = StateSelection(info, "joint_current,joint_velocity")
        row = pq.read_table(self.root / "episodes/episode_000000/frames.parquet").to_pylist()[0]
        values = selection.frame(row)
        self.assertAlmostEqual(values[0], 0.65)
        self.assertAlmostEqual(values[14], 100 * 360 / 4095, places=5)
        self.assertEqual(selection.units[14], "deg/s")

    def test_one_frame_episode_has_finite_statistics(self):
        self.make_source(episodes=1, frames=1)
        export_lerobot(self.root, self.output, state="joint_current")
        stats = json.loads((self.output / "meta/stats.json").read_text())
        self.assertEqual(stats["observation.state"]["count"], [1])
        self.assertEqual(stats["observation.state"]["std"], [0] * 14)

    def test_shards_rotate_only_between_episodes_with_continuous_global_indices(self):
        self.make_source(episodes=3)
        with patch("alohamini.datasets.lerobotv3.DATA_FILE_BYTES", 1):
            export_lerobot(self.root, self.output)
        paths = sorted((self.output / "data").rglob("*.parquet"))
        self.assertEqual(len(paths), 3)
        self.assertEqual([row["index"] for row in self.read_rows()], list(range(9)))
        metadata = pq.read_table(
            self.output / "meta/episodes/chunk-000/file-000.parquet"
        ).to_pylist()
        self.assertEqual([row["data/file_index"] for row in metadata], [0, 1, 2])

    def test_output_is_new_and_failure_does_not_publish_partial_export(self):
        self.make_source()
        before = hashes(self.root)
        with patch(
            "alohamini.datasets.lerobotv3._validate_export", side_effect=OSError("read failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "source unchanged"):
                export_lerobot(self.root, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(hashes(self.root), before)
        self.assertEqual(len(list(self.output.parent.glob("export.pending-*"))), 1)
        for destination in (self.root, self.root / "nested"):
            with self.assertRaises((ValueError, FileExistsError)):
                export_lerobot(self.root, destination)

    def test_existing_output_has_actionable_cli_error_without_overwriting(self):
        self.make_source(episodes=1)
        export_lerobot(self.root, self.output)
        before = hashes(self.output)
        for source in (self.root, self.output):
            out = io.StringIO()
            with contextlib.redirect_stderr(out):
                result = main(
                    [
                        "dataset",
                        "export",
                        str(source),
                        "--output",
                        str(self.output),
                        "--format",
                        "lerobot-v3",
                    ]
                )
            self.assertEqual(result, 1)
            self.assertIn("Output already exists", out.getvalue())
            self.assertIn("choose a new --output", out.getvalue())
            self.assertIn("Nothing overwritten", out.getvalue())
            self.assertEqual(hashes(self.output), before)

    def test_cli_routes_state_selection_only_to_lerobot_export(self):
        self.make_source()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                main(
                    [
                        "dataset",
                        "export",
                        str(self.root),
                        "--output",
                        str(self.output),
                        "--format",
                        "lerobot-v3",
                        "--state",
                        "joint_velocity,joint_current",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "dataset",
                        "export",
                        str(self.root),
                        "--output",
                        str(self.output),
                        "--state",
                        "joint_current",
                    ]
                ),
                1,
            )
        self.assertEqual(len(self.read_rows()[0]["observation.state"]), 28)

    @unittest.skipUnless(
        os.environ.get("ALOHAMINI_LEROBOT_REFERENCE"), "Official reader checkout not selected"
    )
    def test_official_lerobot_reader_loads_images_fields_and_action_chunks_offline(self):
        self.make_source()
        export_lerobot(
            self.root, self.output, state="joint_velocity,joint_current,base_velocity,lift_height"
        )
        code = """
import os, socket, sys
socket.socket.connect = lambda *_: (_ for _ in ()).throw(AssertionError('network access'))
import lerobot
from pathlib import Path
assert Path(lerobot.__file__).is_relative_to(Path(os.environ['ALOHAMINI_LEROBOT_REFERENCE']))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('local/test', root=sys.argv[1], video_backend='pyav',
                        delta_timestamps={'action': [0, 1/30, 2/30]})
assert len(dataset) == 6
sample = dataset[2]
assert sample['observation.state'].shape == (32,)
assert sample['observation.source_state'].shape == (18,)
assert sample['observation.motor_velocity_raw'].shape == (18,)
assert sample['motor_feedback.current_ma_valid'].shape == (18,)
assert sample['action'].shape == (3,18)
assert sample['action_is_pad'].tolist() == [False, True, True]
assert sample['observation.images.forward'].shape == (3,16,24)
assert sample['observation.images.forward'][0,0,0] > .9
assert sample['task'] == 'pick'
print('Official LeRobot reader: OK')
"""
        env = {
            **os.environ,
            "PYTHONPATH": os.environ["ALOHAMINI_LEROBOT_REFERENCE"],
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_DATASETS_CACHE": str(Path(self.temp.name) / "reader_cache"),
        }
        python = os.environ["ALOHAMINI_LEROBOT_PYTHON"]
        result = subprocess.run(
            [python, "-c", code, str(self.output)],
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
