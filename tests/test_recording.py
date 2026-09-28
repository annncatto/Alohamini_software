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
from alohamini.errors import ResponseTimeoutError
from alohamini.schema import CommandIdentity


class RecordingLoopTests(unittest.TestCase):
    def test_skipped_send_does_not_record_an_unsent_action(self):
        self.client.send_command.side_effect = None
        self.client.send_command.return_value = None
        self.run_loop()
        self.assertGreater(self.client.send_command.call_count, 1)
        self.assertEqual(self.frames, [])
        self.assertIsNone(self.stop_call.args[2])

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
        self.client.read_recording.side_effect = self.read
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

    def run_loop(self, duration=0.1, fps=30):
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
                    fps=fps,
                    duration_s=duration,
                    metadata=self.meta,
                    dataset=self.dataset,
                )
            finally:
                self.stop_call = stop.call_args

    def test_queued_images_are_consumed_even_when_next_request_is_state_only(self):
        def read(*, include_images=False):
            # The ordered client returns an earlier camera request, while the
            # recorder schedules a state-only request for a later cycle.
            result = self.read(include_images=True)
            return result

        self.client.read_recording.side_effect = read
        self.run_loop()
        self.assertEqual(len(self.frames), 5)
        self.assertTrue(
            any(
                not call.kwargs["include_images"]
                for call in self.client.read_recording.call_args_list
            )
        )
        self.client.prefetch.assert_not_called()

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

    def test_brief_timeout_reads_new_input_without_recording_cached_frames(self):
        attempts = []

        def read(**kwargs):
            attempts.append(kwargs)
            if len(attempts) == 2:
                raise ResponseTimeoutError("temporary")
            return self.read(**kwargs)

        self.client.read_recording.side_effect = read
        self.run_loop()
        self.assertEqual(len(attempts), 5)
        self.assertEqual(
            self.calls, ["read", "input", "send", "input", "send"] + ["read", "input", "send"] * 3
        )
        self.assertEqual(self.client.send_command.call_count, 5)
        self.assertTrue(attempts[2]["include_images"])
        events = [call.args[0]["type"] for call in self.dataset.event.call_args_list]
        self.assertEqual(events, ["response_timeout", "response_recovered"])

    def test_timeout_never_resumes_into_another_host_session(self):
        attempts = 0

        def read(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise ResponseTimeoutError("temporary")
            result = self.read(**kwargs)
            if attempts >= 3:
                result.payload["_safety"]["host_session_id"] = "restarted"
            return result

        self.client.read_recording.side_effect = read
        with self.assertRaisesRegex(RuntimeError, "session or control lease changed"):
            self.run_loop()
        self.assertEqual(self.client.send_command.call_count, 2)

    def test_own_watchdog_stop_resumes_recording_with_new_input_and_state_history(self):
        attempts = 0

        def read(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise ResponseTimeoutError("temporary")
            result = self.read(**kwargs)
            result.payload["_safety"].update(
                control_owner="client" if attempts == 1 or attempts > 3 else None,
                phase="ready" if attempts == 3 else "active",
                control_epoch=0 if attempts == 1 else 1,
                watchdog_events=0 if attempts == 1 else 1,
                watchdog_active=attempts == 3,
                joint_hold_events=0,
            )
            return result

        def send(action, *, based_on):
            self.calls.append("send")
            return CommandIdentity(
                "client", self.clock.count, "session", based_on.payload["_safety"]["control_epoch"]
            )

        self.client.read_recording.side_effect = read
        self.client.send_command.side_effect = send
        self.run_loop()
        self.assertEqual(
            self.calls, ["read", "input", "send", "input", "send"] + ["read", "input", "send"] * 3
        )
        events = [call.args[0]["type"] for call in self.dataset.event.call_args_list]
        self.assertEqual(events, ["response_timeout", "watchdog_recovered", "response_recovered"])
        for _frame, _images, log in self.frames:
            self.assertEqual(log["safety"]["control_epoch"], log["issued_command"]["control_epoch"])

    def test_deadline_during_observation_does_not_read_or_send_post_task_target(self):
        def late(**kwargs):
            result = self.read(**kwargs)
            self.clock.now = 0.2
            return result

        self.client.read_recording.side_effect = late
        self.run_loop()
        self.assertEqual(self.calls, ["read"])
        self.assertFalse(self.frames)

    def test_leader_delay_over_250ms_preserves_real_sample_and_timestamps(self):
        self.dataset.cameras = ()
        self.meta["cameras"] = []

        def delayed(units):
            action = self.input(units)
            if self.clock.count == 1:
                self.clock.now += 0.26
            return action

        self.leader.read.side_effect = delayed
        self.run_loop(0.32)
        self.assertEqual(self.calls[:3], ["read", "input", "send"])
        self.assertTrue(self.frames)
        frame, _, log = self.frames[0]
        self.assertEqual(frame["action"][0], 21)
        self.assertEqual(log["client_timing"]["action_sample_finished_monotonic_s"], 0.26)
        self.client.read_recording.assert_any_call(include_images=False)

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

        self.client.read_recording.side_effect = delayed
        self.run_loop(1.4)
        self.assertGreater(self.client.send_command.call_count, 60)
        self.assertGreater(len(self.frames), 0)
        events = [call.args[0]["type"] for call in self.dataset.event.call_args_list]
        self.assertEqual(events, ["capture_wait", "capture_recovered"])

    def test_host_restart_stops_and_does_not_pair_across_sessions(self):
        def restart(**kwargs):
            result = self.read(**kwargs)
            if self.clock.count >= 3:
                result.payload["_safety"]["host_session_id"] = "restarted"
            return result

        self.client.read_recording.side_effect = restart
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

        self.client.read_recording.side_effect = contact
        self.run_loop()
        self.assertEqual(self.client.send_command.call_count, 5)
        self.assertTrue(self.frames[0][2]["safety"]["joint_holds"])

    def test_invalid_feedback_pauses_capture_and_control_then_recovers(self):
        def intermittent(**kwargs):
            result = self.read(**kwargs)
            result.payload["_safety"]["feedback_valid"] = self.clock.count != 2
            return result

        self.client.read_recording.side_effect = intermittent
        self.run_loop()
        self.assertEqual(self.client.send_command.call_count, 4)
        self.assertTrue(self.frames)
        self.assertTrue(all(log["safety"]["feedback_valid"] for _, _, log in self.frames))

    def test_state_only_sampling_keeps_nominal_fps_and_has_no_camera_requests(self):
        self.dataset.cameras = ()
        self.meta["cameras"] = []
        self.run_loop()
        self.assertEqual(len(self.frames), 3)
        self.assertFalse(any(self.requests))
        self.assertTrue(all(not images for _, images, _ in self.frames))

    def test_25_fps_requests_do_not_reduce_control_rate(self):
        self.run_loop(duration=0.12, fps=25)
        self.assertEqual(self.client.send_command.call_count, 6)
        self.assertEqual(self.requests, [True, False, True, False, True, False])
        self.assertEqual(len(self.frames), 3)


class RecordingEntryTests(unittest.TestCase):
    def test_leader_calibration_cancel_does_not_create_dataset_or_start_keyboard(self):
        client = Mock(client_id="client")
        client.connect_control.return_value = snapshot()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("alohamini.apps.recording.BimanualLeader") as leader,
                patch("alohamini.apps.recording.RecordingKeyboard") as keyboard,
                patch("alohamini.apps.recording.HostClient", return_value=client),
                patch("alohamini.datasets.native.LocalDataset") as dataset,
                patch("alohamini.apps.recording.record_loop") as loop,
            ):
                leader.return_value.__enter__.side_effect = InterruptedError(
                    "Calibration cancelled"
                )
                with self.assertRaises(InterruptedError):
                    record(
                        "pi",
                        "alohamini2pro",
                        dataset_name="test",
                        task="pick",
                        root=Path(directory) / "capture",
                    )
                dataset.assert_not_called()
                keyboard.assert_not_called()
                loop.assert_not_called()
                client.__exit__.assert_called_once()

    def test_dataset_metadata_is_refreshed_after_leader_calibration(self):
        initial, fresh = snapshot(), snapshot()
        initial.payload["_robot_metadata"] = metadata()
        fresh.payload["_robot_metadata"] = deepcopy(metadata())
        fresh.payload["_robot_metadata"]["motors"]["arm_left_elbow_flex"]["homing_offset"] = 123
        client = Mock(client_id="client")
        client.connect_control.side_effect = [initial, fresh]
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        keyboard = RecordingKeyboard()
        keyboard.events["stop_recording"] = True
        dataset = Mock(num_episodes=0)
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("alohamini.apps.recording.BimanualLeader") as leader,
                patch("alohamini.apps.recording.RecordingKeyboard") as keys,
                patch("alohamini.apps.recording.HostClient", return_value=client),
                patch("alohamini.datasets.native.LocalDataset", return_value=dataset) as create,
            ):
                keys.return_value.__enter__.return_value = keyboard

                def after_setup(*args, **kwargs):
                    leader.return_value.__enter__.assert_called_once()
                    self.assertEqual(client.connect_control.call_count, 2)
                    return dataset

                create.side_effect = after_setup
                record(
                    "pi",
                    "alohamini2pro",
                    dataset_name="test",
                    task="pick",
                    root=Path(directory) / "capture",
                )
                self.assertEqual(
                    create.call_args.kwargs["robot_metadata"], fresh.payload["_robot_metadata"]
                )

    def test_previews_start_after_cleanup_and_failure_does_not_fail_saved_recording(self):
        client = Mock(client_id="client")
        initial = snapshot()
        initial.payload["_robot_metadata"] = metadata()
        client.connect_control.return_value = initial
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
        client.connect_control.return_value = initial
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        keyboard = RecordingKeyboard()
        dataset = Mock(num_episodes=0, close=Mock(side_effect=OSError("disk failed")))
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
                with self.assertLogs(level="ERROR"), self.assertRaises(KeyboardInterrupt):
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
