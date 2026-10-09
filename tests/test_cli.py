import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch

from support import state_payload

from alohamini.cli import main
from alohamini.errors import ResponseTimeoutError
from alohamini.protocol import HostSnapshot


class CliTests(unittest.TestCase):
    def test_dataset_format_name_preserves_existing_export_commands(self):
        for options in ([], ["--format", "alohamini"], ["--format", "native"]):
            with (
                patch("alohamini.datasets.tools.export_dataset") as export,
                patch("alohamini.datasets.tools.print_report"),
            ):
                export.return_value = {"valid": True, "warnings": 0}
                self.assertEqual(
                    main(["dataset", "export", "/input", "--output", "/output", *options]), 0
                )
                export.assert_called_once_with("/input", "/output", recover=False)

    def test_record_accepts_25_fps_and_preview_with_both_fps_names(self):
        for option in ("--fps", "--dataset.fps"):
            with patch("alohamini.apps.recording.record") as record:
                self.assertEqual(
                    main(
                        [
                            "record",
                            "--dataset",
                            "test",
                            "--task",
                            "pick",
                            "--host",
                            "192.168.8.55",
                            "--robot_model",
                            "alohamini2pro",
                            option,
                            "25",
                            "--display_data",
                        ]
                    ),
                    0,
                )
            self.assertEqual(record.call_args.kwargs["fps"], 25)
            self.assertTrue(record.call_args.kwargs["display_data"])

    def test_record_entry_keeps_familiar_names_without_hub_upload(self):
        with patch("alohamini.apps.recording.record") as record:
            self.assertEqual(
                main(
                    [
                        "record",
                        "--dataset.repo_id",
                        "test",
                        "--dataset.single_task",
                        "pick",
                        "--robot.remote_ip",
                        "192.168.8.161",
                        "--robot.robot_model",
                        "alohamini2pro",
                        "--teleop.id",
                        "am_leader_bi",
                        "--dataset.episode_time_s",
                        "8",
                        "--dataset.reset_time_s",
                        "3",
                        "--profile_timing",
                    ]
                ),
                0,
            )
        args = record.call_args.kwargs
        self.assertEqual((args["dataset_name"], args["task"], args["fps"]), ("test", "pick", 30))
        self.assertEqual((args["episode_time_s"], args["reset_time_s"]), (8, 3))
        self.assertTrue(args["profile_timing"])
        self.assertIsNone(args["video_encoding_workers"])
        self.assertNotIn("push_to_hub", args)

    def test_local_calibration_entry_selects_explicit_device_and_model(self):
        for target in ("leader", "robot", "arms"):
            with patch("alohamini.calibration.procedure.calibrate") as calibrate:
                self.assertEqual(
                    main(["calibrate", target, "--robot_model", "alohamini2pro", "--id", "test"]),
                    0,
                )
                self.assertEqual(calibrate.call_args.args, (target, "alohamini2pro"))
                self.assertEqual(calibrate.call_args.kwargs["device_id"], "test")
                self.assertFalse(calibrate.call_args.kwargs["rehome"])

    def test_arms_rehome_requires_explicit_flag(self):
        with patch("alohamini.calibration.procedure.calibrate") as calibrate:
            self.assertEqual(
                main(["calibrate", "arms", "--robot_model", "alohamini2pro", "--rehome"]), 0
            )
            self.assertEqual(calibrate.call_args.args[0], "arms")
            self.assertTrue(calibrate.call_args.kwargs["rehome"])

    def test_teleoperation_accepts_existing_cli_names_with_optional_profile(self):
        for profile in ([], ["--teleop.arm_profile", "am-leader-6dof"]):
            with patch("alohamini.apps.teleoperation.teleoperate") as teleoperate:
                self.assertEqual(
                    main(
                        [
                            "teleoperate",
                            "--robot.remote_ip",
                            "192.168.8.161",
                            "--robot.robot_model",
                            "alohamini2pro",
                            "--teleop.id",
                            "am_leader_bi",
                            *profile,
                        ]
                    ),
                    0,
                )
                self.assertEqual(teleoperate.call_args.args, ("192.168.8.161", "alohamini2pro"))
                options = teleoperate.call_args.kwargs
                self.assertEqual(options["leader_id"], "am_leader_bi")
                self.assertEqual(options["arm_profile"], "am-leader-6dof" if profile else None)
                self.assertEqual((options["fps"], options["camera_fps"]), (50, 30))
                self.assertFalse(options["no_robot"] or options["no_preview"])

    def test_no_robot_debug_entry_does_not_require_a_host_address(self):
        with patch("alohamini.apps.teleoperation.teleoperate") as teleoperate:
            self.assertEqual(
                main(
                    [
                        "teleoperate",
                        "--robot_model",
                        "alohamini2pro",
                        "--no_robot",
                        "--no_leader",
                        "--no_preview",
                    ]
                ),
                0,
            )
            for option in ("no_robot", "no_leader", "no_preview"):
                self.assertTrue(teleoperate.call_args.kwargs[option])

    def test_host_profile_flag_retains_optional_boolean_and_alias(self):
        for option, enabled in (
            ([], False),
            (["--profile_timing"], True),
            (["--profile_timing", "true"], True),
            (["--profile-timing", "false"], False),
        ):
            with (
                patch("alohamini.runtime.startup.open_host") as open_host,
                patch("alohamini.cli.logging.info"),
            ):
                self.assertEqual(main(["host", "--robot_model", "alohamini2pro", *option]), 0)
                open_host.return_value.run.assert_called_once_with(profile_timing=enabled)

    def test_paths_is_local_only_and_reports_all_storage_directories(self):
        with (
            patch.dict(os.environ, {"ALOHAMINI_WORKSPACE": "/mnt/robot-workspace"}),
            patch("alohamini.cli.HostClient") as client,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(main(["paths"]), 0)
        client.assert_not_called()
        paths = json.loads(output.getvalue())
        self.assertEqual(paths.pop("root"), "/mnt/robot-workspace")
        self.assertEqual(set(paths), {"calibration", "datasets", "runs", "incoming", "logs"})
        for name, location in paths.items():
            self.assertEqual(location, f"/mnt/robot-workspace/{name}")

    def test_prints_state_json_and_closes_client(self):
        snapshot = HostSnapshot(state_payload(), {}, 1, 1.02)
        with patch("alohamini.cli.HostClient") as client:
            client.return_value.__enter__.return_value.read.return_value = snapshot
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["inspect", "--host", "127.0.0.1", "--model", "alohamini2pro"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue()), state_payload())
            client.return_value.__exit__.assert_called_once()

    def test_timeout_is_an_error_not_an_empty_success(self):
        with patch("alohamini.cli.HostClient") as client:
            client.return_value.__enter__.return_value.read.side_effect = ResponseTimeoutError(
                "timeout"
            )
            output, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                code = main(["inspect", "--host", "127.0.0.1"])
            self.assertEqual(code, 1)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("timeout", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
