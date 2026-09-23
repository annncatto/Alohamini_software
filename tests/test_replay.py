import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from test_dataset import frame, jpeg
from test_teleoperation import snapshot

from alohamini.apps.replay import (
    ReplayEpisode,
    check_coordinates,
    load_episode,
    replay,
    run_replay,
)
from alohamini.cli import main
from alohamini.datasets.lerobot import export_lerobot
from alohamini.datasets.native import LocalDataset, state_names
from alohamini.errors import ResponseTimeoutError
from alohamini.model import get_robot_model
from alohamini.schema import CommandIdentity


def replay_snapshot(model="alohamini2pro"):
    result = snapshot(model)
    metadata = result.payload["_robot_metadata"]
    metadata["cameras"] = ["forward"]
    for motor in get_robot_model(model).actuators:
        metadata["motors"].setdefault(motor.name, {"normalization": "degrees"})
        metadata["motors"][motor.name].update(
            id=motor.motor_id,
            model=motor.motor_model,
            range_min=1000,
            range_max=3000,
            drive_mode=0,
            homing_offset=0,
        )
    result.payload["_safety"].update(
        joint_hold_events=0,
        watchdog_events=0,
        watchdog_active=False,
        command_watchdog_timeout_s=1.0,
        command={},
    )
    return result


def episode_fixture(count=3, model="alohamini2pro"):
    source = replay_snapshot(model)
    names = state_names(model)
    actions = np.array([[source.payload[key] for key in names] for _ in range(count)])
    actions[:, 0] = np.arange(count)
    actions[:, names.index("x.vel")] = 0.15
    return ReplayEpisode(
        Path("/test-only"), 0, 30, source.payload["_robot_metadata"], tuple(names), actions
    )


class Clock:
    now = 0.0

    def sleep(self, duration):
        self.now += duration


class Client:
    """In-memory Host with explicit delayed command acceptance; never opens sockets."""

    client_id = "replay-test"

    def __init__(self, clock):
        self.clock = clock
        self.state = replay_snapshot()
        self.sent = []
        self.on_read = None
        self.acknowledge = True
        self.read_delay = 0.002

    def read(self):
        self.clock.sleep(self.read_delay)
        if self.on_read:
            self.on_read(self)
        if self.sent and self.acknowledge:
            status = self.state.payload["_safety"]
            status.update(
                command=asdict(self.sent[-1][2]),
                control_owner=self.client_id,
                watchdog_active=False,
            )
        result = copy.deepcopy(self.state)
        result.request_started_s = result.received_s = self.clock.now
        return result

    def send_command(self, action, *, based_on):
        status = based_on.payload["_safety"]
        identity = CommandIdentity(
            self.client_id, len(self.sent) + 1, status["host_session_id"], status["control_epoch"]
        )
        self.sent.append((self.clock.now, dict(action), identity))
        return identity


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.client = Client(self.clock)
        self.enterContext(
            patch("alohamini.apps.replay.time.monotonic", side_effect=lambda: self.clock.now)
        )
        self.enterContext(patch("alohamini.apps.replay.time.sleep", side_effect=self.clock.sleep))
        self.stop = self.enterContext(patch("alohamini.apps.replay.stop_owned_robot"))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_named_actions_not_state_are_replayed_in_order(self):
        episode = episode_fixture(10)
        run_replay(self.client, episode)
        self.assertEqual(len(self.client.sent), 10)
        for row, (_, command, _) in zip(episode.actions, self.client.sent, strict=True):
            self.assertEqual(command, dict(zip(episode.names, row, strict=True)))
        self.assertAlmostEqual(self.clock.now, 10 / 30, delta=0.04)
        self.assertTrue(
            all(
                b[0] - a[0] >= 0.019
                for a, b in zip(self.client.sent, self.client.sent[1:], strict=False)
            )
        )
        self.stop.assert_called_once_with(self.client, "alohamini2pro", self.client.sent[-1][2])

    def test_reordered_feature_names_map_by_name_without_reordering_rows(self):
        original = episode_fixture()
        episode = replace(original, names=original.names[::-1], actions=original.actions[:, ::-1])
        run_replay(self.client, episode)
        self.assertEqual(self.client.sent[1][1][original.names[0]], 1.0)
        self.assertEqual(self.client.sent[1][1]["x.vel"], 0.15)

    def test_slow_replay_renews_accepted_target_without_scaling_base_velocity(self):
        run_replay(self.client, episode_fixture(1), fps=0.5)
        self.assertGreater(len(self.client.sent), 5)
        self.assertTrue(all(row[1] == self.client.sent[0][1] for row in self.client.sent))
        self.assertTrue(
            all(
                b[0] - a[0] < 0.3
                for a, b in zip(self.client.sent, self.client.sent[1:], strict=False)
            )
        )
        self.assertAlmostEqual(self.clock.now, 2.0, delta=0.04)

    def test_slow_reads_skip_expired_rows_without_replaying_them_for_extra_time(self):
        self.client.read_delay = 0.09
        run_replay(self.client, episode_fixture(5))
        self.assertEqual([row[1]["arm_left_shoulder_pan.pos"] for row in self.client.sent], [0, 3])
        self.assertTrue(
            all(
                b[0] - a[0] >= 0.09 - 1e-9
                for a, b in zip(self.client.sent, self.client.sent[1:], strict=False)
            )
        )

        elapsed = self.clock.now - self.client.sent[0][0]
        self.assertLess(elapsed, 5 / 30 + 0.1)

    def test_short_replay_without_ack_reports_missing_confirmation_without_extending_time(self):
        self.client.acknowledge = False
        with self.assertLogs(level="WARNING") as messages:
            run_replay(self.client, episode_fixture())
        self.assertIn("未观察到", messages.output[0])
        self.assertEqual(len(self.client.sent), 3)
        self.assertLess(self.clock.now, 1.1)
        self.stop.assert_called_once()

    def test_sustained_six_hz_delivery_skips_rows_instead_of_stretching_episode_fivefold(self):
        self.client.read_delay = 1 / 6
        run_replay(self.client, episode_fixture(30))
        self.assertGreater(len(self.client.sent), 1)
        self.assertLess(len(self.client.sent), 10)
        self.assertLess(self.clock.now - self.client.sent[0][0], 1.2)
        self.stop.assert_called_once()

    def test_long_replay_requires_ack_progress_but_not_every_row_ack(self):
        self.client.acknowledge = False
        with self.assertRaisesRegex(RuntimeError, "命令确认未推进"):
            run_replay(self.client, episode_fixture(90))
        self.assertGreater(len(self.client.sent), 1)
        self.assertLess(len(self.client.sent), 90)
        self.assertGreaterEqual(self.clock.now, 1.0)
        self.assertLess(self.clock.now, 1.1)

    def test_lagging_ack_does_not_block_next_timed_action(self):
        self.client.acknowledge = False

        def ack_earlier_row(client):
            if len(client.sent) >= 2:
                client.state.payload["_safety"].update(
                    command=asdict(client.sent[-2][2]), control_owner=client.client_id
                )

        self.client.on_read = ack_earlier_row
        run_replay(self.client, episode_fixture(30))
        self.assertEqual(len(self.client.sent), 30)
        self.assertAlmostEqual(self.clock.now - self.client.sent[0][0], 1.0, delta=0.01)

    def test_temporary_stall_does_not_accumulate_base_velocity_duration(self):
        delayed = False

        def stall_once(client):
            nonlocal delayed
            if len(client.sent) == 8 and not delayed:
                delayed = True
                self.clock.sleep(0.06)

        self.client.on_read = stall_once
        episode = episode_fixture(30)
        run_replay(self.client, episode)
        self.assertTrue(delayed)
        self.assertLess(len(self.client.sent), 30)
        duration = self.clock.now - self.client.sent[0][0]
        self.assertAlmostEqual(duration * 0.15, 0.15, delta=0.002)

    def test_replay_cli_uses_bounded_runtime_reads_and_prefetch(self):
        with (
            patch("alohamini.apps.replay.load_episode", return_value=episode_fixture()),
            patch("alohamini.apps.replay.HostClient") as client,
            patch("alohamini.apps.replay.run_replay"),
        ):
            replay("test-only", "127.0.0.1", "alohamini2pro")
        self.assertEqual(client.call_args.kwargs["timeout_s"], 0.2)
        self.assertTrue(client.call_args.kwargs["prefetch_before_decode"])

    def test_transient_response_timeout_retries_and_skips_without_replaying_cached_action(self):
        interrupted = False
        original = self.client.read

        def read():
            nonlocal interrupted
            if len(self.client.sent) == 3 and not interrupted:
                interrupted = True
                self.clock.sleep(0.2)
                raise ResponseTimeoutError("temporary")
            return original()

        self.client.read = read
        run_replay(self.client, episode_fixture(30))
        self.assertTrue(interrupted)
        indices = [command["arm_left_shoulder_pan.pos"] for _, command, _ in self.client.sent]
        self.assertEqual(indices[:3], [0, 1, 2])
        self.assertGreater(indices[3], 7)
        self.assertLess(self.clock.now - self.client.sent[0][0], 1.01)
        self.stop.assert_called_once()

    def test_persistent_response_loss_stops_at_host_budget_not_first_poll(self):
        original = self.client.read

        def read():
            if self.client.sent:
                self.clock.sleep(0.2)
                raise ResponseTimeoutError("offline")
            return original()

        self.client.read = read
        with self.assertRaisesRegex(RuntimeError, "看门狗时限"):
            run_replay(self.client, episode_fixture(90))
        self.assertEqual(len(self.client.sent), 1)
        self.assertGreaterEqual(self.clock.now - self.client.sent[0][0], 1.0)
        self.assertLess(self.clock.now - self.client.sent[0][0], 1.3)
        self.stop.assert_called_once()

    def test_protection_and_ownership_changes_stop_before_the_next_action(self):
        for change in (
            {"joint_holds": {"arm_left_shoulder_pan": {"current_ma": 2500}}},
            {"joint_hold_events": 1},
            {"watchdog_events": 1},
            {"watchdog_active": True},
            {"feedback_valid": False},
            {"lift_reference_valid": False},
            {"host_session_id": "restarted"},
            {"control_epoch": 1},
            {"control_owner": "other"},
            {"phase": "fault", "fault": "overcurrent"},
        ):
            with self.subTest(change=change):
                client = Client(self.clock)
                client.acknowledge = False

                def changed(current, change=change):
                    if current.sent:
                        current.state.payload["_safety"].update(change)

                client.on_read = changed
                with self.assertRaises(RuntimeError):
                    run_replay(client, episode_fixture())
                self.assertEqual(len(client.sent), 1)

    def test_gripper_contact_does_not_abort_a_grasp(self):
        self.client.state.payload["_safety"]["gripper_holds"] = {"arm_left_gripper": {}}
        run_replay(self.client, episode_fixture())
        self.assertEqual(len(self.client.sent), 3)

    def test_idle_watchdog_from_previous_client_does_not_require_host_restart(self):
        self.client.state.payload["_safety"].update(
            watchdog_active=True, watchdog_events=3, control_epoch=3, control_owner=None
        )
        run_replay(self.client, episode_fixture())
        self.assertEqual(len(self.client.sent), 3)
        self.assertTrue(all(row[2].control_epoch == 3 for row in self.client.sent))

    def test_new_watchdog_event_during_initial_idle_claim_is_not_ignored(self):
        self.client.state.payload["_safety"].update(watchdog_active=True, watchdog_events=3)
        self.client.acknowledge = False

        def timeout(current):
            if current.sent:
                current.state.payload["_safety"]["watchdog_events"] += 1

        self.client.on_read = timeout
        with self.assertRaisesRegex(RuntimeError, "保护事件"):
            run_replay(self.client, episode_fixture())
        self.assertEqual(len(self.client.sent), 1)

    def test_failure_or_interrupt_still_attempts_owned_stop(self):
        for error in (ConnectionError("transport failed"), KeyboardInterrupt()):
            client = Client(self.clock)

            def fail(current, error=error):
                if current.sent:
                    raise error

            client.on_read = fail
            with self.assertRaises(type(error)):
                run_replay(client, episode_fixture())
            self.stop.assert_called_with(client, "alohamini2pro", client.sent[0][2])

    def test_invalid_rate_or_calibration_sends_nothing(self):
        for options in ({"speed": 0}, {"fps": 100}, {"speed": float("nan")}, {"fps": -2}):
            with self.assertRaises(ValueError):
                run_replay(self.client, episode_fixture(), **options)
        self.client.state.payload["_robot_metadata"]["motors"]["arm_left_shoulder_pan"][
            "drive_mode"
        ] = 1
        with self.assertRaisesRegex(ValueError, "calibration or units"):
            run_replay(self.client, episode_fixture())
        self.assertFalse(self.client.sent)

    def test_all_models_match_calibration_and_reject_missing_or_out_of_range_values(self):
        for model in ("alohamini1", "alohamini2", "alohamini2pro"):
            data, live = episode_fixture(model=model), replay_snapshot(model)
            check_coordinates(data, live)
            broken = copy.deepcopy(data)
            broken.metadata["motors"]["lift_axis"].pop("homing_offset")
            with self.assertRaisesRegex(ValueError, "lift_axis"):
                check_coordinates(broken, live)
            changed_lift = copy.deepcopy(data)
            changed_lift.metadata["lift_axis"]["soft_max_mm"] += 10
            with self.assertRaisesRegex(ValueError, "lift geometry"):
                check_coordinates(changed_lift, live)
            for key, value in (("arm_left_shoulder_pan.pos", 150), ("lift_axis.height_mm", 700)):
                actions = data.actions.copy()
                actions[-1, data.names.index(key)] = value
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    check_coordinates(replace(data, actions=actions), live)


class ReplayDatasetTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory) / "dataset"
        with contextlib.closing(
            LocalDataset(
                self.root,
                fps=30,
                task="pick",
                robot_metadata=replay_snapshot().payload["_robot_metadata"],
            )
        ) as dataset:
            for episode in range(2):
                dataset.begin_episode()
                for index in range(3):
                    value = frame(dataset)
                    value["action"][:] = index + episode * 10
                    dataset.add_frame(value, {"forward": jpeg()}, {})
                dataset.save_episode()

    def test_loads_selected_native_episode_without_reading_images(self):
        with patch("alohamini.datasets.images.image_rgb", side_effect=AssertionError("image read")):
            episode = load_episode(self.root, 1)
        np.testing.assert_array_equal(episode.actions[:, 0], [10, 11, 12])
        self.assertEqual(episode.names, tuple(state_names("alohamini2pro")))
        self.assertFalse(episode.actions.flags.writeable)

    def test_native_and_state_selected_lerobot_export_replay_identical_actions(self):
        output = self.root.parent / "export"
        export_lerobot(self.root, output, state="lift_height")
        native, exported = load_episode(self.root, 1), load_episode(output, 1)
        np.testing.assert_array_equal(native.actions, exported.actions)
        self.assertEqual(native.metadata, exported.metadata)
        self.assertEqual(native.names, exported.names)

    def test_in_use_dataset_is_rejected(self):
        with contextlib.closing(
            LocalDataset(
                self.root,
                fps=30,
                task="pick",
                robot_metadata=replay_snapshot().payload["_robot_metadata"],
                resume=True,
            )
        ):
            with self.assertRaisesRegex(RuntimeError, "in use"):
                load_episode(self.root)

    def test_corruption_is_rejected_before_creating_client(self):
        path = self.root / "episodes/episode_000000/frames.parquet"
        original = pq.read_table(path)
        for change in (
            {"action": [float("nan")] * 18},
            {"frame_index": 0},
            {"timestamp": 0},
            {"episode_index": 1},
        ):
            rows = original.to_pylist()
            rows[-1].update(change)
            pq.write_table(pa.Table.from_pylist(rows, schema=original.schema), path)
            with patch("alohamini.apps.replay.HostClient") as client:
                with self.assertRaises(ValueError):
                    replay("dataset", "127.0.0.1", "alohamini2pro", root=self.root)
                client.assert_not_called()

    def test_missing_calibration_sidecar_in_generic_lerobot_is_not_guessed(self):
        path = self.root / "meta/info.json"
        path.write_text(json.dumps({"codebase_version": "v3.0"}))
        with self.assertRaisesRegex(ValueError, "alohamini.json"):
            load_episode(self.root)

    def test_cli_maps_existing_parameter_aliases_without_motion(self):
        with patch("alohamini.apps.replay.replay") as run:
            self.assertEqual(
                main(
                    [
                        "replay",
                        "--dataset.repo_id",
                        "demo",
                        "--robot.remote_ip",
                        "127.0.0.1",
                        "--robot.robot_model",
                        "alohamini2pro",
                        "--dataset.episode",
                        "2",
                        "--replay.fps",
                        "15",
                    ]
                ),
                0,
            )
        self.assertEqual(run.call_args.kwargs["episode"], 2)
        self.assertEqual(run.call_args.kwargs["fps"], 15)
        with patch("alohamini.apps.replay.HostClient") as client:
            with contextlib.redirect_stderr(io.StringIO()):
                result = main(
                    [
                        "replay",
                        "--dataset",
                        "demo",
                        "--root",
                        str(self.root / "absent"),
                        "--host",
                        "127.0.0.1",
                        "--robot_model",
                        "alohamini2pro",
                    ]
                )
            self.assertEqual(result, 1)
            client.assert_not_called()
