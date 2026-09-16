import contextlib
import io
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_dataset import metadata
from test_teleoperation import snapshot

from alohamini.apps.recording import RecordingKeyboard, record, record_loop
from alohamini.datasets.native import motor_feedback_features, state_names
from alohamini.schema import CommandIdentity


class RecordingLoopTests(unittest.TestCase):
    def setUp(self):
        self.clock = SimpleNamespace(now=0.0, count=0)
        self.client = Mock(client_id="client")
        self.keyboard = Mock(
            events=dict(exit_early=False, rerecord_episode=False, stop_recording=False)
        )
        self.keyboard.read.return_value = {"w"}
        self.leader = Mock()
        names = state_names("alohamini2pro")
        feature = {"names": names, "dtype": "float32"}
        self.dataset = Mock(
            cameras=("forward",),
            queue_overflows=0,
            features={
                "observation.state": feature,
                "action": feature,
                **motor_feedback_features(names),
            },
        )
        self.frames, self.calls, self.requests = [], [], []
        self.dataset.add_frame.side_effect = lambda *args: (
            self.frames.append(deepcopy(args)) or True
        )
        self.meta = metadata()
        self.client.read.side_effect = self.read
        self.client.send_command.side_effect = self.send
        self.leader.read.side_effect = self.input

    def read(self, include_images=False):
        self.requests.append(include_images)
        self.calls.append("read")
        result = snapshot()
        result.payload["_robot_metadata"] = deepcopy(self.meta)
        result.payload["arm_left_shoulder_pan.pos"] = float(self.clock.count)
        result.payload["_host_timing"] = {
            "state_sample_started_monotonic_s": 100 + self.clock.now,
            "state_sample_finished_monotonic_s": 100 + self.clock.now,
            "camera_capture_monotonic_s": {"forward": 100 + self.clock.now / 10}
            if include_images
            else {},
        }
        result.payload["_motor_feedback"] = {
            "version": 1,
            "motors": {
                "arm_left_shoulder_pan": {
                    "current_raw": self.clock.count + 10,
                    "sample_started_s": 100 + self.clock.now,
                    "sample_finished_s": 100 + self.clock.now,
                }
            },
        }
        result.images = {"forward": b"jpeg"} if include_images else {}
        self.clock.count += 1
        return result

    def input(self, units):
        self.calls.append("input")
        return {key: self.clock.count + 20.0 for key in units}

    def send(self, *_args, **_kwargs):
        self.calls.append("send")
        return CommandIdentity("client", self.clock.count, "session", 0)

    def run_loop(self, duration=0.1):
        def sleep(seconds):
            self.clock.now += seconds

        with (
            patch("time.perf_counter", side_effect=lambda: self.clock.now),
            patch("time.monotonic", side_effect=lambda: self.clock.now),
            patch("time.sleep", side_effect=sleep),
            patch("alohamini.apps.recording.stop_owned_robot") as stop,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                record_loop(
                    self.client,
                    "alohamini2pro",
                    self.leader,
                    self.keyboard,
                    fps=30,
                    duration_s=duration,
                    metadata=self.meta,
                    dataset=self.dataset,
                )
            finally:
                self.stop_call = stop.call_args

    def test_image_aligned_old_state_uses_current_subsequent_action_and_matching_feedback(self):
        self.run_loop()
        self.assertEqual(self.calls, ["read", "input", "send"] * 5)
        self.assertEqual(len(self.frames), 3)
        value, images, log = self.frames[1]
        self.assertEqual(value["observation.state"][0], 0.0)
        self.assertEqual(value["action"][0], 23.0)
        self.assertEqual(value["observation.motor_current_raw"][0], 10.0)
        self.assertEqual(value["motor_feedback.current_raw_valid"][0], 1.0)
        self.assertEqual(log["host_timing"]["state_sample_monotonic_s"], 100)
        self.assertEqual(images, {"forward": b"jpeg"})
        timing = log["client_timing"]
        self.assertLessEqual(
            timing["observation_received_monotonic_s"], timing["action_sample_started_monotonic_s"]
        )
        self.assertEqual(log["issued_command"]["sequence"], 3)
        self.assertEqual(self.requests, [True, False, True, False, True])
        self.client.set_recording_cameras.assert_any_call(True)
        self.assertFalse(self.client.set_recording_cameras.call_args.args[0])
        self.assertEqual(self.stop_call.args[-1].sequence, 5)

    def test_deadline_during_observation_does_not_read_or_send_post_task_target(self):
        def late(**kwargs):
            result = self.read(**kwargs)
            self.clock.now = 0.2
            return result

        self.client.read.side_effect = late
        self.run_loop()
        self.assertEqual(self.calls, ["read"])
        self.assertFalse(self.frames)

    def test_missing_camera_stalls_only_capture_then_recovers_without_cached_images(self):
        def delayed(**kwargs):
            result = self.read(**kwargs)
            if self.clock.now < 1.1:
                result.images = {}
                result.payload["_host_timing"]["camera_capture_monotonic_s"] = {}
            else:
                result.payload["_host_timing"]["camera_capture_monotonic_s"] = (
                    {"forward": 100 + self.clock.now} if kwargs["include_images"] else {}
                )
            return result

        self.client.read.side_effect = delayed
        self.run_loop(1.4)
        self.assertGreater(self.client.send_command.call_count, 60)
        self.assertGreater(len(self.frames), 0)
        events = [call.args[0]["type"] for call in self.dataset.event.call_args_list]
        self.assertEqual(events, ["capture_wait", "capture_recovered"])

    def test_epoch_change_stops_and_does_not_pair_across_host_sessions(self):
        def restart(**kwargs):
            result = self.read(**kwargs)
            if self.clock.count >= 3:
                result.payload["_safety"]["control_epoch"] = 1
            return result

        self.client.read.side_effect = restart
        with self.assertRaisesRegex(RuntimeError, "lease changed"):
            self.run_loop()
        self.assertEqual(self.client.send_command.call_count, 2)
        self.assertEqual(self.stop_call.args[-1].sequence, 2)

    def test_contact_hold_is_recorded_but_does_not_block_retreat_control(self):
        original = self.read

        def contact(**kwargs):
            result = original(**kwargs)
            result.payload["_safety"]["joint_holds"] = {"arm_left_shoulder_pan": 0}
            return result

        self.client.read.side_effect = contact
        self.run_loop()
        self.assertEqual(self.client.send_command.call_count, 5)
        self.assertTrue(self.frames[0][2]["safety"]["joint_holds"])

    def test_state_only_sampling_keeps_nominal_fps_and_has_no_camera_requests(self):
        self.dataset.cameras = ()
        self.meta["cameras"] = []
        self.run_loop()
        self.assertEqual(len(self.frames), 3)
        self.assertFalse(any(self.requests))
        self.assertTrue(all(not images for _, images, _ in self.frames))


class RecordingEntryTests(unittest.TestCase):
    def test_previews_start_after_cleanup_and_failure_does_not_fail_saved_recording(self):
        client = Mock(client_id="client")
        initial = snapshot()
        initial.payload["_robot_metadata"] = metadata()
        client.read.return_value = initial
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        keyboard = RecordingKeyboard()
        dataset = Mock(num_episodes=0, submitted=3, saved=3, cameras=("forward",))
        dataset.save_episode.side_effect = lambda: setattr(dataset, "num_episodes", 1)
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("alohamini.apps.recording.BimanualLeader") as leader,
                patch("alohamini.apps.recording.RecordingKeyboard") as keyboard_class,
                patch("alohamini.apps.recording.HostClient", return_value=client),
                patch("alohamini.datasets.native.LocalDataset", return_value=dataset),
                patch("alohamini.apps.recording.record_loop"),
                patch("alohamini.datasets.video.generate_previews") as preview,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                keyboard_class.return_value.__enter__.return_value = keyboard

                def fail_preview(path):
                    dataset.close.assert_called_once()
                    leader.return_value.__exit__.assert_called_once()
                    client.__exit__.assert_called_once()
                    raise OSError("encoder failed")

                preview.side_effect = fail_preview
                with self.assertLogs(level="WARNING") as logs:
                    record(
                        "127.0.0.1",
                        "alohamini2pro",
                        dataset_name="test",
                        task="pick",
                        root=Path(directory) / "capture",
                        reset_time_s=0,
                    )
                preview.assert_called_once()
                self.assertIn("Dataset is saved", logs.output[0])

    def test_interrupt_preserves_partial_data_after_device_cleanup(self):
        client = Mock(client_id="client")
        initial = snapshot()
        initial.payload["_robot_metadata"] = metadata()
        client.read.return_value = initial
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        keyboard = RecordingKeyboard()
        dataset = Mock(num_episodes=0)
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("alohamini.apps.recording.BimanualLeader") as leader,
                patch("alohamini.apps.recording.RecordingKeyboard") as keyboard_class,
                patch("alohamini.apps.recording.HostClient", return_value=client),
                patch("alohamini.datasets.native.LocalDataset", return_value=dataset),
                patch("alohamini.apps.recording.record_loop", side_effect=KeyboardInterrupt),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                keyboard_class.return_value.__enter__.return_value = keyboard
                with self.assertRaises(KeyboardInterrupt):
                    record(
                        "127.0.0.1",
                        "alohamini2pro",
                        dataset_name="test",
                        task="pick",
                        root=Path(directory) / "capture",
                    )
                dataset.begin_episode.assert_called_once()
                dataset.close.assert_called_once()
                leader.return_value.__exit__.assert_called_once()
                client.__exit__.assert_called_once()

    def test_recording_keys_preserve_movement_keys_and_episode_controls(self):
        keyboard = RecordingKeyboard()
        keyboard.update("w", True)
        with contextlib.redirect_stdout(io.StringIO()):
            keyboard.update("r", True)
            self.assertTrue(keyboard.events["exit_early"] and keyboard.events["rerecord_episode"])
            self.assertIn("w", keyboard._pressed)
            keyboard.update("esc", True)
        self.assertTrue(keyboard.events["stop_recording"])
        self.assertFalse(keyboard._pressed)

    def test_invalid_settings_fail_before_opening_leaders(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("alohamini.apps.recording.BimanualLeader") as leader:
                for options in (
                    {"fps": 60},
                    {"episode_time_s": 0},
                    {"task": ""},
                    {"num_episodes": 0},
                ):
                    with self.assertRaises(ValueError):
                        record(
                            "127.0.0.1",
                            "alohamini2pro",
                            **{
                                "dataset_name": "test",
                                "task": "pick",
                                "root": Path(directory) / "new",
                                **options,
                            },
                        )
                leader.assert_not_called()
