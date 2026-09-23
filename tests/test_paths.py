import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alohamini.paths import WorkspacePaths


class WorkspacePathsTests(unittest.TestCase):
    def test_default_is_visible_and_independent_of_huggingface_paths(self):
        with (
            patch.dict(
                os.environ, {"HF_HOME": "/ignored", "HF_LEROBOT_HOME": "/ignored"}, clear=True
            ),
            patch("pathlib.Path.home", return_value=Path("/home/operator")),
        ):
            paths = WorkspacePaths()
        self.assertEqual(paths.root, Path("/home/operator/Alohamini_workspace"))
        self.assertEqual(paths.datasets, paths.root / "datasets")
        self.assertEqual(paths.pretrained, paths.root / "pretrained")

    def test_explicit_root_overrides_environment_and_stays_fixed(self):
        with patch.dict(os.environ, {"ALOHAMINI_WORKSPACE": "/mnt/robot-workspace"}):
            default = WorkspacePaths()
            explicit = WorkspacePaths(Path("/mnt/another-disk"))
            os.environ["ALOHAMINI_WORKSPACE"] = "/changed"
        self.assertEqual(default.root, Path("/mnt/robot-workspace"))
        self.assertEqual(explicit.root, Path("/mnt/another-disk"))

    def test_empty_or_relative_environment_root_is_not_silently_replaced(self):
        for value in ("", " ", "relative/data"):
            with patch.dict(os.environ, {"ALOHAMINI_WORKSPACE": value}):
                with self.assertRaises(ValueError):
                    WorkspacePaths()

    def test_explicit_relative_root_is_rejected(self):
        with self.assertRaises(ValueError):
            WorkspacePaths(Path("workspace"))

    def test_calibration_preserves_robot_and_bimanual_device_identifiers(self):
        paths = WorkspacePaths(Path("/data"))
        self.assertEqual(
            paths.calibration_file("robots", "AlohaMiniRobot"),
            Path("/data/calibration/robots/AlohaMiniRobot.json"),
        )
        self.assertEqual(
            paths.calibration_file("teleoperators", "am_leader_bi_left"),
            Path("/data/calibration/teleoperators/am_leader_bi_left.json"),
        )
        self.assertNotEqual(
            paths.calibration_file("teleoperators", "am_leader_bi_left"),
            paths.calibration_file("teleoperators", "am_leader_bi_right"),
        )

    def test_named_artifacts_stay_in_their_assigned_directory(self):
        paths = WorkspacePaths(Path("/data"))
        self.assertEqual(paths.dataset("0913_test_3"), Path("/data/datasets/0913_test_3"))
        self.assertEqual(paths.run("act_001"), Path("/data/runs/act_001"))
        self.assertEqual(paths.incoming_batch("robot01_001"), Path("/data/incoming/robot01_001"))
        self.assertEqual(paths.dataset("抓取测试"), Path("/data/datasets/抓取测试"))

    def test_names_cannot_be_paths_or_traversal(self):
        paths = WorkspacePaths(Path("/data"))
        for name in ("", ".", "..", "../other", "/tmp/other", "a/b", "a\\b", "\x00", " task "):
            for resolver in (
                paths.dataset,
                paths.run,
                paths.incoming_batch,
                lambda value: paths.calibration_file("robots", value),
            ):
                with self.assertRaises(ValueError):
                    resolver(name)
        with self.assertRaises(ValueError):
            paths.calibration_file("../other", "robot")

    def test_resolving_paths_does_not_create_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = WorkspacePaths(Path(directory) / "not-created")
            paths.calibration_file("robots", "robot")
            paths.dataset("episode")
            paths.run("train")
            paths.incoming_batch("batch")
            self.assertFalse(paths.root.exists())


if __name__ == "__main__":
    unittest.main()
