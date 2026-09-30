"""Optional differential checks against explicitly selected source repositories.

Set ALOHAMINI_SOURCE_REPO and ALOHAMINI_ROS_SOURCE to trusted local checkouts.
Only named pure computations are loaded from their actual files; no LeRobot,
ROS, camera or serial initialization runs. Normal regressions need neither repo.
"""

import ast
import collections
import collections.abc
import contextlib
import io
import json
import logging
import math
import os
import random
import threading
import time
import unittest
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from alohamini.apps.teleop_monitor import TeleopMonitor, tracking_rows
from alohamini.apps.teleoperation import KeyboardTargets
from alohamini.calibration.encoder import EncoderCalibration, HostPositionUnits
from alohamini.calibration.lift import LiftCalibration
from alohamini.calibration.servo import MotorCalibration
from alohamini.hardware.feetech import _FIELDS, decode_feedback
from alohamini.kinematics import OmniBaseKinematics
from alohamini.model import ActuatorSpec, get_robot_model
from alohamini.runtime.arm_contact import (
    ArmContactGuard,
    ArmJointSpec,
    GripperContactCalibration,
    JointContactCalibration,
)
from alohamini.runtime.base_control import BaseDrive
from alohamini.runtime.command_owner import CommandOwner
from alohamini.runtime.current_protection import CurrentProtection
from alohamini.runtime.host import RecordingCameraBuffer
from alohamini.runtime.lift_control import lift_height_target
from alohamini.schema import BodyVelocity, CommandIdentity

SOURCE = os.environ.get("ALOHAMINI_SOURCE_REPO")
ROS_SOURCE = os.environ.get("ALOHAMINI_ROS_SOURCE")


class NormMode(Enum):
    RANGE_M100_100 = "range_m100_100"
    RANGE_0_100 = "range_0_100"
    DEGREES = "degrees"


def computations(path, *, functions=(), classes=None, constants=(), namespace=None):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in constants for target in targets):
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in (classes or {}):
            selected = classes[node.name]
            if selected is not None:
                node.body = [
                    method
                    for method in node.body
                    if isinstance(method, ast.FunctionDef) and method.name in selected
                ]
                node.bases = []
            body.append(node)
    scope = dict(namespace or {})
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"
        ),
        scope,
    )
    return scope


@unittest.skipUnless(SOURCE, "Set ALOHAMINI_SOURCE_REPO for source differential checks")
class SourceDifferentialTests(unittest.TestCase):
    def test_camera_stream_preserves_original_jpeg_and_clock_metadata(self):
        import cv2
        import numpy as np
        from test_camera_stream import capture_frame

        from alohamini.runtime.camera_stream import encode_camera_stream_message

        source = computations(
            Path(SOURCE) / "src/lerobot/robots/alohamini/camera_stream.py",
            constants=("CAMERA_STREAM_SCHEMA_VERSION",),
            classes={"CameraSnapshot": None},
            functions=("encode_camera_stream_message",),
            namespace={"dataclass": dataclass, "cv2": cv2, "json": json, "time": time},
        )
        rgb = np.zeros((8, 16, 3), np.uint8)
        rgb[:, :, 0] = 255
        rgb[:, :, 1] = np.arange(8, dtype=np.uint8)[:, None] * 12
        rgb[:, :, 2] = np.arange(16, dtype=np.uint8)[None, :] * 6
        for rotation in (0, 90, 180, 270):
            (_, _, _, snapshot), _ = capture_frame(rotation=rotation)
            frame = (
                rgb
                if not rotation
                else cv2.rotate(
                    rgb,
                    {
                        90: cv2.ROTATE_90_CLOCKWISE,
                        180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                    }[rotation],
                )
            )
            stamp = snapshot.capture_monotonic_s
            old, _ = source["encode_camera_stream_message"](
                "forward",
                source["CameraSnapshot"](frame, stamp),
                7,
                70,
                host_monotonic_s=stamp + 0.01,
                host_unix_ns=1_000_000_000,
            )
            new = encode_camera_stream_message(
                "forward",
                snapshot,
                7,
                host_session_id="native-session",
                host_monotonic_s=stamp + 0.01,
                host_unix_ns=1_000_000_000,
            )
            self.assertEqual((old[0], old[2]), (new[0], new[2]))
            metadata = json.loads(new[1])
            self.assertEqual(metadata.pop("host_session_id"), "native-session")
            self.assertEqual(metadata, json.loads(old[1]))

    def test_export_statistics_match_existing_lerobot_running_statistics(self):
        import numpy as np

        from alohamini.datasets.lerobot import RunningQuantileStats

        source = computations(
            Path(SOURCE) / "src/lerobot/datasets/compute_stats.py",
            constants=("DEFAULT_QUANTILES",),
            classes={"RunningQuantileStats": None},
            namespace={"np": np},
        )
        original, migrated = source["RunningQuantileStats"](), RunningQuantileStats()
        for batch in (
            np.array([[0, 10, -2], [1, 30, -1]], dtype=np.float64),
            np.array([[2, 100, 5], [3, 40, 7]], dtype=np.float64),
        ):
            original.update(batch)
            migrated.update(batch)
        for key, value in original.get_statistics().items():
            np.testing.assert_array_equal(value, migrated.get_statistics()[key])

    def test_dataset_checks_keep_source_feedback_and_physical_timing_rules(self):
        import numpy as np
        import pyarrow as pa

        from alohamini.datasets.tools import IntegrityChecker

        methods = (
            "error",
            "warning",
            "_check_motor_feedback",
            "_check_finite_feature",
            "_check_capture_timeline",
        )
        source = computations(
            Path(SOURCE) / "scripts/check_lerobot_dataset_integrity.py",
            classes={"Issue": None, "IntegrityChecker": methods},
            namespace={"np": np, "pa": pa, "Path": Path, "math": math, "dataclass": dataclass},
        )
        old = source["IntegrityChecker"]()
        old.root = Path("/dataset")
        old.info = {"fps": 30}
        old.timestamp_tolerance_s = 1e-4
        new = IntegrityChecker(Path("/dataset"))
        new.info = old.info
        for mask in ([1.0, 1.0], [1.0, 0.0], [2.0, 0.0]):
            table = pa.table(
                {
                    "observation.motor_current_ma": [[1.0, 0.0]],
                    "motor_feedback.current_ma_valid": [mask],
                    "motor_feedback.sample_started_s": [[1.0, 2.0]],
                    "motor_feedback.sample_finished_s": [[1.001, 2.001]],
                }
            )
            old.issues, new.issues = [], []
            for checker in (old, new):
                checker._check_motor_feedback(Path("/dataset/episode"), table)
            self.assertEqual(
                [asdict(issue) for issue in old.issues], [asdict(issue) for issue in new.issues]
            )
        for stamps, expected_gaps in (
            ([100, 100 + 1 / 30, 100 + 2 / 30], 0),
            ([100, 100.1, 100.2], 1),
            ([100, 100, 99], 1),
        ):
            frames = [
                {"host_timing": {"camera_capture_monotonic_s": {"forward": stamp}}}
                for stamp in stamps
            ]
            old.issues, new.issues = [], []
            for checker in (old, new):
                checker._check_capture_timeline(0, frames)
            self.assertEqual(
                [asdict(issue) for issue in old.issues],
                [asdict(issue) for issue in new.issues if issue.code != "SAFETY_CAMERA_GAP"],
            )
            # Per-camera gap detection supplements, rather than replaces, the
            # source check of the aggregate acquisition timeline.
            camera_gaps = [issue for issue in new.issues if issue.code == "SAFETY_CAMERA_GAP"]
            self.assertEqual(len(camera_gaps), expected_gaps)

    def test_motor_feedback_columns_and_validity_masks_match_original(self):
        from alohamini.datasets import native as dataset

        np = self.ns["np"]
        original = computations(
            self.robot / "motor_feedback.py",
            constants=("FEEDBACK_FIELDS", "FEEDBACK_TIMES"),
            functions=(
                "motor_feedback_features",
                "is_motor_feedback_feature",
                "motor_feedback_frame",
            ),
            namespace={"np": np, "math": math, "Mapping": collections.abc.Mapping},
        )
        for model_name in ("alohamini1", "alohamini2", "alohamini2pro"):
            names = dataset.state_names(model_name)
            features = dataset.motor_feedback_features(names)
            self.assertEqual(features, original["motor_feedback_features"](names))
            for current in (123, None, float("nan"), True, 1e40):
                feedback = {
                    "version": 1,
                    "motors": {
                        "arm_left_shoulder_pan": {
                            "current_raw": current,
                            "sample_started_s": 100,
                            "sample_finished_s": 100.01,
                        },
                        "base_left_wheel": {
                            "velocity_raw": -10,
                            "sample_started_s": 100,
                            "sample_finished_s": 100.01,
                        },
                    },
                }
                old = original["motor_feedback_frame"](features, feedback)
                new = dataset.motor_feedback_frame(features, feedback)
                self.assertEqual(old.keys(), new.keys())
                for key in old:
                    np.testing.assert_array_equal(old[key], new[key])

    def test_recording_alignment_and_camera_gate_match_original(self):
        from alohamini.apps.recording import FreshCameraGate, StateSample, StateSampleBuffer

        source = computations(
            self.source / "examples/alohamini/record_utils_multirate.py",
            classes={"StateSample": None, "StateSampleBuffer": None, "FreshCameraGate": None},
            namespace={
                "math": math,
                "statistics": __import__("statistics"),
                "deque": collections.deque,
                "dataclass": dataclass,
                "field": dataclass_field,
            },
        )
        old, new = source["StateSampleBuffer"](3), StateSampleBuffer(3)
        for stamp in (1.0, 1.02, 1.04, 1.06):
            old.append(source["StateSample"]({"state": stamp}, stamp))
            new.append(StateSample({"state": stamp}, stamp))
        for stamp in (1.018, 1.033, 1.05, 1.11):
            a, ae = old.nearest(stamp, max_error_s=0.1)
            b, be = new.nearest(stamp, max_error_s=0.1)
            self.assertEqual((a.observation, ae), (b.observation, be))
        gates = [
            cls(("a", "b"), started_at=0, stall_timeout_s=1, max_skew_s=0.05)
            for cls in (source["FreshCameraGate"], FreshCameraGate)
        ]
        for stamps, now in (
            ({}, 0.1),
            ({"a": 1.0}, 0.2),
            ({"a": 1.0, "b": 1.02}, 0.3),
            ({"a": 1.0, "b": 1.02}, 0.4),
            ({"a": 1.1, "b": 1.11}, 0.5),
        ):
            self.assertEqual(gates[0].observe(stamps, now=now), gates[1].observe(stamps, now=now))

    def test_partial_save_errors_preserve_source_exception_priority(self):
        from alohamini.datasets.native import preserve_dataset

        source = computations(
            self.source / "examples/alohamini/safety_utils.py",
            functions=("preserve_dataset",),
            namespace={"contextmanager": contextlib.contextmanager, "logger": Mock()},
        )["preserve_dataset"]
        for primary in (None, RuntimeError("policy/leader failed"), KeyboardInterrupt()):
            for save_error in (None, OSError("disk failed")):
                with self.subTest(primary=primary, save_error=save_error):
                    old = Mock()
                    old.writer.save_failed = False
                    old.has_pending_frames.return_value = True
                    old.save_episode.side_effect = save_error
                    new = Mock()
                    new.close.side_effect = save_error
                    results = []
                    with patch("alohamini.datasets.native.logging.exception"):
                        for manager, dataset in ((source, old), (preserve_dataset, new)):
                            try:
                                with manager(dataset):
                                    if primary is not None:
                                        raise primary
                            except BaseException as error:
                                results.append(error)
                            else:
                                results.append(None)
                    self.assertIs(results[0], primary if primary is not None else save_error)
                    self.assertIs(results[1], results[0])
                    old.finalize.assert_called_once()
                    new.close.assert_called_once()

    def test_evaluation_guard_rejects_source_protection_events(self):
        from test_replay import replay_snapshot

        from alohamini.apps.replay import ReplayGuard

        old_type = computations(
            self.source / "examples/alohamini/evaluation_safety.py",
            classes={"EvaluationSafetyGuard": ("__init__", "acknowledge", "reason")},
            namespace={"time": SimpleNamespace(monotonic=lambda: 1.0)},
        )["EvaluationSafetyGuard"]
        for change in (
            {"joint_holds": {"arm_left_elbow_flex": {}}},
            {"joint_hold_events": 1},
            {"watchdog_events": 1},
            {"watchdog_active": True},
            {"host_session_id": "restarted"},
        ):
            with self.subTest(change=change):
                snapshot = replay_snapshot()
                status = snapshot.payload["_safety"]
                old = old_type()
                robot = SimpleNamespace(
                    latest_safety_status=status,
                    _last_safety_received_at=1.0,
                    command_permitted=True,
                    feedback_fresh=True,
                )
                new = ReplayGuard(SimpleNamespace(client_id="replay-test"), "alohamini2pro")
                self.assertIsNone(old.reason(robot))
                new.check(snapshot)
                status.update(change)
                self.assertIsNotNone(old.reason(robot))
                with self.assertRaises(RuntimeError):
                    new.check(snapshot)

    @classmethod
    def setUpClass(cls):
        import numpy as np

        cls.source = Path(SOURCE)
        cls.robot = cls.source / "src/lerobot/robots/alohamini"
        cls.ns = {
            "np": np,
            "dataclass": dataclass,
            "time": time,
            "MotorNormMode": NormMode,
            "logger": logging.getLogger("source-differential"),
        }

    def test_new_calibration_values_and_hand_movement_prompts_match_source(self):
        from alohamini.calibration.procedure import _calibrate_leader, _calibrate_robot

        def ranges(names):
            return ({name: 1000 for name in names}, {name: 3000 for name in names})

        def bus(motors):
            return SimpleNamespace(
                actuators=motors,
                motors={m.name: SimpleNamespace(id=m.motor_id) for m in motors},
                disable_torque=Mock(),
                write=Mock(),
                write_calibration=Mock(),
                configure_calibration=Mock(),
                calibration_session=contextlib.nullcontext,
                set_half_turn_homings=lambda names=None: {
                    name: 123 for name in (names if names is not None else [m.name for m in motors])
                },
                record_ranges_of_motion=ranges,
            )

        for model_name in ("alohamini1", "alohamini2", "alohamini2pro"):
            for leader in (True, False):
                model = get_robot_model(model_name)
                buses = {}
                for side in ("left", "right"):
                    motors = tuple(m for m in model.actuators if m.bus == side)
                    if leader:
                        motors = tuple(
                            ActuatorSpec(
                                m.name.removeprefix(f"arm_{side}_"), side, m.motor_id, "sts3215"
                            )
                            for m in motors
                            if m.name.startswith("arm_")
                        )
                    buses[side] = bus(motors)
                class_name = "SOLeader" if leader else "AlohaMini"
                path = (
                    self.source / "src/lerobot/teleoperators/so_leader/so_leader.py"
                    if leader
                    else self.robot / "alohamini.py"
                )
                cls = computations(
                    path,
                    classes={class_name: {"calibrate"}},
                    namespace={
                        "MotorCalibration": MotorCalibration,
                        "logger": logging.getLogger("calibration-source"),
                        "OperatingMode": SimpleNamespace(POSITION=SimpleNamespace(value=0)),
                    },
                )[class_name]
                cls.__str__ = lambda self: self.id
                old = cls()
                old.id, old.calibration, old.calibration_fpath = (
                    "test",
                    {},
                    Path("calibration.json"),
                )
                old._save_calibration = Mock()
                if leader:
                    old.bus = buses["left"]
                else:
                    old.config = SimpleNamespace(no_follower=False)
                    old.left_bus, old.right_bus = buses["left"], buses["right"]
                    old.left_arm_motors = [m for m in old.left_bus.motors if m.startswith("arm_")]
                    old.right_arm_motors = list(old.right_bus.motors)
                    old.base_motors = [m for m in old.left_bus.motors if m.startswith("base_")]

                with patch("builtins.input", return_value="") as prompts:
                    with contextlib.redirect_stdout(io.StringIO()) as original_output:
                        old.calibrate()
                    original_prompts = list(prompts.call_args_list)
                    prompts.reset_mock()
                    with (
                        patch(
                            "alohamini.calibration.procedure.record_ranges_of_motion",
                            side_effect=lambda _bus, names: ranges(names),
                        ),
                        patch("alohamini.calibration.procedure.save_motor_calibration") as save,
                        contextlib.redirect_stdout(io.StringIO()) as migrated_output,
                    ):
                        if leader:
                            _calibrate_leader(buses["left"], old.id, old.calibration_fpath, {})
                        else:
                            _calibrate_robot(buses, old.id, old.calibration_fpath, {})
                    self.assertEqual(prompts.call_args_list, original_prompts)
                    self.assertEqual(migrated_output.getvalue(), original_output.getvalue())
                    self.assertEqual(save.call_args.args[1], old.calibration)

    def test_teleop_monitor_output_matches_original_file_verbatim(self):
        source = computations(
            self.source / "examples/alohamini/teleop_monitor.py",
            functions=("tracking_rows",),
            classes={"TeleopMonitor": None},
            namespace={"time": time, "math": math},
        )
        payload = {
            "arm_left_shoulder_pan.pos": 1.0,
            "arm_left_gripper.pos": 2.0,
            "arm_right_wrist_roll.pos": 3.0,
            "arm_right_gripper.pos": 4.0,
            "_robot_metadata": {
                "motors": {
                    "arm_left_shoulder_pan": {"normalization": "degrees"},
                    "arm_left_gripper": {"normalization": "range_0_100"},
                    "arm_right_wrist_roll": {"normalization": "degrees"},
                    "arm_right_gripper": {"normalization": "range_0_100"},
                }
            },
            "_safety": {
                "accepted_targets": {
                    "arm_right_gripper.pos": 10.0,
                    "arm_right_wrist_roll.pos": 9.0,
                    "arm_left_gripper.pos": 8.0,
                    "arm_left_shoulder_pan.pos": 7.0,
                },
                "requested_targets": {"arm_left_gripper.pos": 9.0},
                "currents_ma": {"arm_left_gripper": 58.5},
                "joint_holds": {"arm_right_wrist_roll": 9.0},
            },
        }
        for fresh, permitted, sent in (
            (True, True, True),
            (False, False, False),
            (True, False, False),
            (True, True, False),
        ):
            robot = SimpleNamespace(
                latest_safety_status=payload["_safety"],
                latest_robot_metadata=payload["_robot_metadata"],
                last_remote_state=payload,
                feedback_fresh=fresh,
                command_permitted=permitted,
            )
            self.assertEqual(tracking_rows(payload), source["tracking_rows"](robot))
            with patch("time.perf_counter", return_value=0):
                original, migrated = source["TeleopMonitor"](robot), TeleopMonitor()
            with (
                patch("time.perf_counter", return_value=0.5),
                contextlib.redirect_stdout(io.StringIO()) as early,
            ):
                original.update(sent=sent)
                migrated.update(payload, sent=sent, fresh=fresh, permitted=permitted)
            self.assertEqual(early.getvalue(), "")
            with patch("time.perf_counter", return_value=1.25):
                with contextlib.redirect_stdout(io.StringIO()) as expected:
                    original.update(sent=sent)
                with contextlib.redirect_stdout(io.StringIO()) as actual:
                    migrated.update(payload, sent=sent, fresh=fresh, permitted=permitted)
            self.assertEqual(actual.getvalue(), expected.getvalue())

    def test_visualization_dispatch_retains_original_implementation(self):
        from alohamini.apps import visualization

        def dispatch(path):
            tree = ast.parse(Path(path).read_text(encoding="utf-8"))
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "log_rerun_data"
            )
            # The framework dependency guard is replaced by the native PC environment.
            body = [
                node
                for node in function.body[1:]  # Exclude the module-specific docstring.
                if not (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "require_package"
                )
            ]
            module = ast.Module(body=body, type_ignores=[])
            for node in ast.walk(module):
                if isinstance(node, ast.Call) and ast.unparse(node.func) == "rr.log":
                    # Intentional defect fix: images belong on the time series.
                    node.keywords = [kw for kw in node.keywords if kw.arg != "static"]
            return ast.dump(module)

        self.assertEqual(
            dispatch(self.source / "src/lerobot/utils/rerun_visualization.py"),
            dispatch(visualization.__file__),
        )

    def test_keyboard_trajectory_and_axis_signs_match_source(self):
        source = computations(
            self.robot / "alohamini_client.py",
            classes={
                "AlohaMiniClient": {
                    "_from_keyboard_to_base_action",
                    "_from_keyboard_to_lift_action",
                }
            },
            namespace={
                **self.ns,
                "LiftAxisConfig": lambda: SimpleNamespace(soft_min_mm=0.0, soft_max_mm=600.0),
            },
        )["AlohaMiniClient"]
        old = source()
        old.teleop_keys = dict(
            zip(
                (
                    "forward",
                    "backward",
                    "left",
                    "right",
                    "rotate_left",
                    "rotate_right",
                    "speed_up",
                    "speed_down",
                    "lift_up",
                    "lift_down",
                ),
                "wszxadtguj",
                strict=True,
            )
        )
        old.speed_levels = [
            {"xy": 0.15, "theta": 45},
            {"xy": 0.2, "theta": 60},
            {"xy": 0.25, "theta": 75},
        ]
        old.speed_index = old._lift_direction = 0
        old._lift_target_mm = old._lift_last_update_t = None
        old.config = SimpleNamespace(lift_target_speed_mm_s=150.0, lift_target_max_lead_mm=50.0)
        old.latest_robot_metadata = {"lift_axis": {"soft_min_mm": 0.0, "soft_max_mm": 600.0}}
        mapper = KeyboardTargets()
        for i, keys in enumerate(
            ("wu", "zu", "tau", "u", "u", "u", "uj", "", "j", "sxgd", "j", "u")
        ):
            height, now = 100 + i * 0.2, i * 0.07
            old.last_remote_state = {"lift_axis.height_mm": height}
            payload = {**old.last_remote_state, "_robot_metadata": old.latest_robot_metadata}
            with patch("time.monotonic", return_value=now):
                expected = {
                    **old._from_keyboard_to_base_action(keys),
                    **old._from_keyboard_to_lift_action(keys),
                }
            self.assertEqual(mapper.targets(set(keys), payload, now=now), expected)

    def test_model_geometry_actuator_order_and_types_match_source(self):
        namespace = {}
        exec(
            compile((self.robot / "model_specs.py").read_text(), "model_specs.py", "exec"),
            namespace,
        )
        profiles = computations(
            self.robot / "alohamini.py", constants=("_ARM_PROFILES",), namespace=self.ns
        )["_ARM_PROFILES"]
        for name, spec in namespace["ROBOT_SPECS"].items():
            model = get_robot_model(name)
            self.assertEqual(model.wheel_radius_m, spec["wheel_radius"])
            self.assertEqual(model.base_radius_m, spec["base_radius"])
            self.assertEqual(model.lift_lead_m_per_rev * 1000, spec["lead_mm_per_rev"])
            expected = [
                ActuatorSpec(f"arm_{side}_{joint}", side, motor_id, motor_model)
                for side in ("left", "right")
                for joint, motor_id, motor_model, _ in profiles[spec["arm_profile"]]
            ]
            expected += [
                ActuatorSpec(f"base_{side}_wheel", "left", i, spec["base_motor"])
                for side, i in zip(("left", "back", "right"), (8, 9, 10), strict=True)
            ]
            expected += [ActuatorSpec("lift_axis", "left", 11, spec["lift_motor"])]
            self.assertEqual(model.actuators, tuple(expected))

    def test_encoder_normalization_matches_actual_motor_bus(self):
        old = computations(
            self.source / "src/lerobot/motors/motors_bus.py",
            classes={"SerialMotorsBus": ("_normalize", "_unnormalize")},
            namespace=self.ns,
        )["SerialMotorsBus"]()
        old._id_to_name = lambda _: "joint"
        old._id_to_model = lambda _: "sts3250"
        old.model_resolution_table = {"sts3250": 4096}
        old.apply_drive_mode = True
        for drive in (0, 1):
            for mode in NormMode:
                for lower, upper in ((0, 4095), (1000, 3000), (1897, 3471)):
                    units = HostPositionUnits(mode.value, lower, upper, drive)
                    old.motors = {"joint": SimpleNamespace(norm_mode=mode)}
                    old.calibration = {
                        "joint": SimpleNamespace(range_min=lower, range_max=upper, drive_mode=drive)
                    }
                    for tick in (0, lower, 2048, upper, 4095):
                        self.assertEqual(units.from_tick(tick), old._normalize({1: tick})[1])
                    for target in (-150, -100, -1, 0, 1, 45, 100, 150):
                        expected = old._unnormalize({1: target})[1]
                        if mode is NormMode.DEGREES and not lower <= expected <= upper:
                            with self.assertRaises(ValueError):
                                units.to_tick(target)  # Explicit native target safety restriction.
                        else:
                            self.assertEqual(units.to_tick(target), expected)

    def test_passive_leader_settings_match_source_configure_without_motion_writes(self):
        from test_feetech_device import RegisterSerial

        from alohamini.hardware.feetech_device import FeetechBusDevice

        tables = computations(
            self.source / "src/lerobot/motors/feetech/tables.py",
            constants=(
                "FIRMWARE_MAJOR_VERSION",
                "FIRMWARE_MINOR_VERSION",
                "MODEL_NUMBER",
                "STS_SMS_SERIES_CONTROL_TABLE",
            ),
        )["STS_SMS_SERIES_CONTROL_TABLE"]
        original_bus = computations(
            self.source / "src/lerobot/motors/feetech/feetech.py",
            classes={"FeetechMotorsBus": ("configure_motors",)},
        )["FeetechMotorsBus"]
        original_leader = computations(
            self.source / "src/lerobot/teleoperators/so_leader/so_leader.py",
            classes={"SOLeader": ("configure",)},
            namespace={"OperatingMode": SimpleNamespace(POSITION=SimpleNamespace(value=0))},
        )["SOLeader"]()
        old, new = RegisterSerial(), RegisterSerial()
        motors = (ActuatorSpec("joint", "left", 1, "sts3215"),)
        calibration = {"joint": MotorCalibration(1, 0, -123, 1000, 3000)}
        for serial in (old, new):
            for address, width, value in (
                (3, 2, 777),
                (7, 1, 250),
                (18, 1, 0x1C),
                (33, 1, 1),
                (31, 2, calibration["joint"].offset_register),
                (9, 2, 1000),
                (11, 2, 3000),
            ):
                serial.set(1, address, width, value)
        bus = original_bus()
        bus.protocol_version = 0
        bus.motors = {"joint": SimpleNamespace(model="sts3215")}
        bus.read = lambda register, motor, **kwargs: old.get(1, *tables[register])
        bus.write = lambda register, motor, value: old.set(1, *tables[register], value)
        bus.disable_torque = lambda: old.set(1, *tables["Torque_Enable"], 0)
        original_leader.bus = bus
        original_leader.configure()
        with patch("serial.Serial", return_value=new):
            migrated = FeetechBusDevice(
                "/dev/differential-only",
                motors,
                position_calibrations={"joint": calibration["joint"].encoder_calibration()},
                velocity_limits={},
            )
            migrated.connect_passive("test")
            try:
                migrated.disable_torque()
                migrated.prepare_passive(calibration)
                for register in (
                    "Return_Delay_Time",
                    "Phase",
                    "Operating_Mode",
                    "Maximum_Acceleration",
                    "Acceleration",
                    "Torque_Enable",
                    "Homing_Offset",
                    "Min_Position_Limit",
                    "Max_Position_Limit",
                    "Goal_Position",
                    "Goal_Velocity",
                    "Protection_Current",
                ):
                    with self.subTest(register=register):
                        self.assertEqual(
                            new.get(1, *tables[register]), old.get(1, *tables[register])
                        )
                written = {p[5] for p in new.requests if p[4] == 3}
                self.assertLessEqual(written, {7, 18, 33, 40, 41, 55, 85})
                self.assertFalse(migrated._ready or migrated._prepared)
            finally:
                migrated.close()

    def test_passive_sync_read_matches_source_sdk_packets_and_retry_count(self):
        import inspect

        from scservo_sdk import GroupSyncRead, PortHandler
        from scservo_sdk.protocol_packet_handler import protocol_packet_handler
        from test_feetech_device import RegisterSerial

        from alohamini.hardware.feetech_device import FeetechBusDevice

        scope = computations(
            self.source / "src/lerobot/motors/motors_bus.py",
            classes={"SerialMotorsBus": ("sync_read", "_sync_read", "_setup_sync_reader")},
            namespace={**self.ns, "check_if_not_connected": lambda method: method},
        )
        timeout = computations(
            self.source / "src/lerobot/motors/feetech/feetech.py",
            functions=("patch_setPacketTimeout",),
        )["patch_setPacketTimeout"]
        source_type = scope["SerialMotorsBus"]
        retries = inspect.signature(source_type.sync_read).parameters["num_retry"].default

        class ScriptedSerial(RegisterSerial):
            def __init__(self, failures):
                super().__init__()
                self.failures = failures
                self.queries = 0

            def flush(self):
                pass  # In-memory writes complete synchronously.

            def write(self, packet):
                if packet[4] == 0x82:
                    self.queries += 1
                    self.drop = {(1, 0x82, 56)} if self.queries <= self.failures else set()
                return super().write(bytes(packet))

        for failures in (0, 1, 3, 4):
            old_serial, new_serial = ScriptedSerial(failures), ScriptedSerial(failures)
            port = PortHandler("/dev/never-opened")
            port.ser = old_serial
            port.tx_time_per_byte = 0.01
            port.setPacketTimeout = timeout.__get__(port, PortHandler)
            old = source_type()
            old.packet_handler = protocol_packet_handler()
            old.sync_reader = GroupSyncRead(port, old.packet_handler, 56, 2)
            old._is_comm_success = lambda result: result == 0
            entry = MotorCalibration(1, 0, 0, 0, 4095)
            new_serial.set(1, 3, 2, 777)
            new_serial.set(1, 11, 2, 4095)
            with patch("serial.Serial", return_value=new_serial):
                new = FeetechBusDevice(
                    "/dev/differential-only",
                    (ActuatorSpec("joint", "left", 1, "sts3215"),),
                    position_calibrations={"joint": entry.encoder_calibration()},
                    velocity_limits={},
                )
                new.connect_passive("test")
                try:
                    new.disable_torque()
                    new.prepare_passive({"joint": entry})
                    new_serial.requests.clear()
                    for original in (True, False):
                        with self.subTest(failures=failures, original=original):
                            try:
                                values = (
                                    old._sync_read(56, 2, [1], num_retry=retries)[0]
                                    if original
                                    else new.read_positions()
                                )
                            except ConnectionError:
                                self.assertGreater(failures, retries)
                            else:
                                self.assertLessEqual(failures, retries)
                                self.assertEqual(list(values.values()), [1234])
                    self.assertEqual(old_serial.queries, new_serial.queries)
                    self.assertEqual(old_serial.requests, new_serial.requests)
                finally:
                    new.close()

    def test_body_commands_and_feedback_match_source_for_all_models(self):
        old_class = computations(
            self.robot / "alohamini.py",
            namespace=self.ns,
            classes={
                "AlohaMini": (
                    "_degps_to_raw",
                    "_raw_to_degps",
                    "_body_to_wheel_raw",
                    "_wheel_raw_to_body",
                )
            },
        )["AlohaMini"]
        rng = random.Random(1409)
        inputs = [(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 90)]
        inputs += [
            (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-180, 180)) for _ in range(100)
        ]
        for name in ("alohamini1", "alohamini2", "alohamini2pro"):
            model = get_robot_model(name)
            old = old_class()
            old.wheel_radius, old.base_radius = model.wheel_radius_m, model.base_radius_m
            new = BaseDrive(OmniBaseKinematics(model.wheel_radius_m, model.base_radius_m))
            for x, y, yaw in inputs:
                expected = tuple(old._body_to_wheel_raw(x, y, yaw).values())
                self.assertEqual(
                    new.target(BodyVelocity(x, y, math.radians(yaw))).wheel_ticks_s, expected
                )
                measured = new.measured_velocity(expected)
                old_measured = old._wheel_raw_to_body(*expected)
                for actual, expected in zip(
                    (measured.x_m_s, measured.y_m_s, math.degrees(measured.yaw_rad_s)),
                    old_measured.values(),
                    strict=True,
                ):
                    self.assertAlmostEqual(actual, expected, places=10)

    def test_lift_height_accumulation_matches_source_across_pauses_and_wraps(self):
        from test_base_lift_control import layout, lift_batch

        from alohamini.runtime.lift_control import LiftHeightTracker

        old = computations(
            self.robot / "lift_axis.py", classes={"LiftAxis": ("_update_extended_ticks",)}
        )["LiftAxis"]()
        old.enabled = True
        old.cfg = SimpleNamespace(name="lift")
        old._ticks_per_rev = 4096.0
        old._last_tick = 4000.0
        old._extended_ticks = 0.0
        old._bus = Mock()
        _, spec = layout()
        tracker = LiftHeightTracker(spec)
        tracker.bind_session("session")
        tracker.observe(lift_batch(0, 4000))
        tracker.establish_reference(0.1)
        # Includes >62.5 ms stalls and wrap-around in both directions.
        for sequence, (stamp, tick) in enumerate(
            ((0.02, 4050), (0.12, 20), (0.2, 4090), (0.4, 4000), (0.9, 4000)), 1
        ):
            old._bus.read.return_value = tick
            old._update_extended_ticks()
            tracker.observe(lift_batch(sequence, tick, time_s=stamp))
            self.assertAlmostEqual(
                tracker.height_m,
                0.1 + spec.direction * old._extended_ticks * spec.lead_m_per_revolution / 4096,
            )

    def test_lift_homing_keeps_source_unloading_before_set_zero(self):
        from test_base_lift_control import BaseLiftIntegrationTests

        events = []
        old = computations(
            self.robot / "lift_axis.py",
            classes={"LiftAxis": ("home",)},
            namespace={
                "time": SimpleNamespace(sleep=lambda duration: events.append(("wait", duration)))
            },
        )["LiftAxis"]()
        old.enabled = True
        old.cfg = SimpleNamespace(name="lift", home_down_speed=1300, home_stall_current_ma=300)
        old._bus = SimpleNamespace(
            read=lambda register, *_args, **_kwargs: 50 if register == "Present_Current" else 1000,
            write=lambda register, name, value: events.append((register, name, value)),
        )
        old.configure = lambda: None
        old._last_tick, old._extended_ticks = 1000, 0
        old._update_extended_ticks = lambda: None
        old._extended_deg = lambda: 0
        old.get_height_mm = lambda: events.append(("zero",)) or 0
        with contextlib.redirect_stdout(io.StringIO()):
            old.home()
        release = events.index(("Torque_Enable", "lift", 0))
        self.assertEqual(events[release + 1], ("wait", 1))
        self.assertGreater(events.index(("zero",)), release + 1)

        fixture = BaseLiftIntegrationTests()
        fixture.setUp()
        try:
            fixture.control.begin_lift_homing()
            fixture.serial.set(4, 69, 2, 50)
            released_at = None
            for _ in range(80):
                fixture.time_s += 0.02
                fixture.host.cycle()
                if released_at is None and fixture.serial.get(4, 40, 1) == 0:
                    released_at = fixture.time_s
                if fixture.control.lift_homing_phase == "complete":
                    break
            self.assertIsNotNone(released_at)
            self.assertEqual(fixture.control.lift_height_m, 0)
            self.assertGreaterEqual(fixture.time_s - released_at, 1)
            self.assertEqual(fixture.serial.get(4, 46, 2), 0)
            self.assertEqual(fixture.serial.get(4, 40, 1), 0)
        finally:
            fixture.host.close()
            fixture.doCleanups()

    def test_lift_height_target_matches_source_limits_direction_and_gain(self):
        old = computations(
            self.robot / "lift_axis.py", classes={"LiftAxis": ("apply_action",)}, namespace=self.ns
        )["LiftAxis"]()
        old.enabled = True
        old.cfg = SimpleNamespace(
            name="lift_axis",
            soft_min_mm=0,
            soft_max_mm=600,
            on_target_mm=1,
            kp_vel=300,
            v_max=1300,
            descent_floor_mm=5,
            dir_sign=-1,
        )
        writes = []
        old._bus = SimpleNamespace(write=lambda *args: writes.append(args[-1]))
        for height in (-1, 0, 5, 6, 100, 599, 600, 601):
            for target in (-10, 0, 5, 100, 101, 103, 599, 600, 700):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = old.apply_action(
                        {"lift_axis.height_mm": target}, current_height_mm=height
                    )
                new = lift_height_target(target / 1000, height / 1000, direction=-1)
                self.assertEqual(new.velocity_raw, writes[-1])
                self.assertEqual(new.target_height_m * 1000, result["lift_axis.height_mm"])

    def test_gripper_holds_match_source_and_joint_stalls_are_intentionally_advisory(self):
        names = (
            "_CURRENT_MA_PER_RAW_UNIT",
            "_JOINT_COLLISION_DURATION_S",
            "_JOINT_STALL_MIN_COMMAND_ERROR_DEG",
            "_JOINT_STALL_MIN_PROGRESS_DEG",
        )
        scope = computations(
            self.robot / "alohamini.py",
            namespace=self.ns,
            constants=names,
            functions=("_position_delta_degrees",),
            classes={
                "_JointStallCandidate": None,
                "AlohaMini": ("_limit_gripper_goal_by_current", "_limit_joint_goal_by_current"),
            },
        )
        for gripper in (False, True):
            for drive in (0, 1):
                name = "arm_left_gripper" if gripper else "arm_left_shoulder_pan"
                installed = MotorCalibration(1, drive, 0, 1000, 3000)
                units = installed.position_units("range_0_100" if gripper else "range_m100_100")
                encoder = installed.encoder_calibration()

                def position(value, encoder=encoder, units=units):
                    return encoder.position_from_tick(units.to_tick(value))

                contact = (
                    GripperContactCalibration(position(0), position(100))
                    if gripper
                    else JointContactCalibration(10 * math.tau / 4096)
                )
                new = ArmContactGuard(
                    {name: ArmJointSpec(ActuatorSpec(name, "left", 1, "sts3250"), encoder, contact)}
                )
                old = scope["AlohaMini"]()
                for field in (
                    "_gripper_hold_goal",
                    "_gripper_hold_direction",
                    "_joint_hold_goal",
                    "_joint_hold_direction",
                    "_joint_stall_candidates",
                ):
                    setattr(old, field, {})
                old._gripper_open_direction = {name: 1}
                old._gripper_current_limit_ma, old._gripper_hold_close_step = 500, 3
                old._gripper_release_margin = old._joint_release_margin = 1
                old._joint_hold_events = 0
                old._current_limits = {name: SimpleNamespace(collision_ma=2100)}
                bus = SimpleNamespace(
                    motors={
                        name: SimpleNamespace(
                            model="sts3250", norm_mode=NormMode(units.normalization)
                        )
                    },
                    calibration={name: installed},
                    model_resolution_table={"sts3250": 4096},
                )
                sequence = (
                    [(0, 0, 50, 520), (0.2, 0, 50, 0), (0.3, 60, 50, 0)]
                    if gripper
                    else [
                        (0, 50, 0, 2200),
                        (0.1, 50, 0, 2200),
                        (0.16, 50, 0, 2200),
                        (0.2, -2, 0, 0),
                    ]
                )
                for stamp, goal, present, current in sequence:
                    old._read_force_feedback = (
                        lambda *_, sample=({name: current / 6.5}, {name: present}): sample
                    )
                    method = (
                        old._limit_gripper_goal_by_current
                        if gripper
                        else old._limit_joint_goal_by_current
                    )
                    with (
                        patch("time.monotonic", return_value=stamp),
                        patch.object(scope["logger"], "warning"),
                    ):
                        expected = method(bus, {name + ".pos": goal})[name + ".pos"]
                    actual = new.limit(
                        {name: position(goal)},
                        {name: position(present)},
                        {name: current / 1000},
                        now=stamp,
                    )[name]
                    if gripper:
                        self.assertLessEqual(
                            abs(encoder.position_to_tick(actual) - units.to_tick(expected)), 1
                        )
                    else:
                        self.assertEqual(actual, position(goal))
                        self.assertFalse(new.holds)
                        self.assertEqual(new.joint_hold_events, 0)
                        if stamp == 0.16:
                            self.assertEqual(expected, present)  # Source latched the joint.
                            self.assertIn(name, new.joint_stall_currents_a)

    def test_camera_group_selection_matches_original_episode_cursors(self):
        scope = {}
        exec(
            compile((self.robot / "camera_buffer.py").read_text(), "camera_buffer.py", "exec"),
            scope,
        )
        old, new = scope["RecordingCameraBuffer"](), RecordingCameraBuffer()
        for episode in (b"first", b"second"):
            for cycle in range(12):
                now = 1 + cycle * 0.02
                history = {
                    "forward": tuple(
                        (1 + n / 30, bytes([n + 1])) for n in range(6) if 1 + n / 30 <= now
                    ),
                    "wrist_right": tuple(
                        (1.003 + n / 30, bytes([n + 1])) for n in range(6) if 1.003 + n / 30 <= now
                    ),
                }
                cameras = {
                    name: SimpleNamespace(read_frame_history=lambda h=h: h)
                    for name, h in history.items()
                }
                token = b"request:" + episode + b":record"
                expected = old.apply({}, cameras, b"pc", token, now=now)
                actual = new.select(history, b"pc", token, now=now)
                self.assertEqual(
                    {name: jpeg for name, (_, jpeg) in actual.items()},
                    {name: expected[name] for name in history if name in expected},
                )

    def test_overcurrent_limits_and_elapsed_timers_match_source(self):
        scope = computations(
            self.robot / "alohamini.py",
            namespace=self.ns,
            constants=(
                "_MOTOR_CURRENT_RATINGS_MA",
                "_COLLISION_RATED_CURRENT_MULTIPLIER",
                "_SUSTAINED_RATED_CURRENT_MULTIPLIER",
                "_NEAR_STALL_CURRENT_FRACTION",
            ),
            functions=("_current_limits_for_motor", "_has_sustained_overcurrent"),
            classes={"_MotorCurrentLimits": None},
        )
        for model in ("sts3215", "sts3250", "sts3095"):
            limits = scope["_current_limits_for_motor"](SimpleNamespace(model=model))
            for limit, duration in ((limits.near_stall_ma, 0.080), (limits.sustained_ma, 0.650)):
                for current in (limit - 1, limit, limit + 1):
                    new = CurrentProtection({"motor": model})
                    timers = ({}, {})
                    for now in (0, 0.02, duration - 0.001, duration, duration + 0.01):
                        expected = any(
                            scope["_has_sustained_overcurrent"](
                                timer, "motor", current, threshold, now, dwell
                            )
                            for timer, threshold, dwell in zip(
                                timers,
                                (limits.near_stall_ma, limits.sustained_ma),
                                (0.080, 0.650),
                                strict=True,
                            )
                        )
                        self.assertEqual(
                            new.update({"motor": current / 1000}, now=now) is not None, expected
                        )

    def test_fault_stop_preserves_source_velocity_only_motion_writes(self):
        from test_base_lift_control import BaseLiftIntegrationTests

        events = []
        old = computations(
            self.robot / "alohamini.py",
            classes={"AlohaMini": ("stop_base", "stop_lift", "stop_motion", "disconnect")},
            namespace={**self.ns, "check_if_not_connected": lambda method: method},
        )["AlohaMini"]()
        old.lift = computations(self.robot / "lift_axis.py", classes={"LiftAxis": ("stop",)})[
            "LiftAxis"
        ]()
        old.lift.enabled = True
        old.lift.cfg = SimpleNamespace(name="lift")
        old.left_bus = old.lift._bus = SimpleNamespace(
            sync_write=lambda register, values, **_: events.extend(
                (register, name, value) for name, value in values.items()
            ),
            write=lambda register, name, value: events.append((register, name, value)),
            disconnect=lambda disable: events.append(("disconnect", disable)),
        )
        old.right_bus = None
        old.base_motors = ("wheel_1", "wheel_2", "wheel_3")
        old.config = SimpleNamespace(disable_torque_on_disconnect=True)
        old.cameras = {}
        old.disconnect()
        self.assertEqual(events[-1], ("disconnect", True))
        expected = {name: value for register, name, value in events[:-1]}
        self.assertEqual({event[0] for event in events[:-1]}, {"Goal_Velocity"})

        fixture = BaseLiftIntegrationTests()
        fixture.setUp()
        try:
            for motor in fixture.actuators:
                fixture.serial.set(motor.motor_id, 46, 2, 100)
            fixture.serial.requests.clear()
            fixture.device.stop_velocity()
            writes = [p for p in fixture.serial.requests if p[4] in (3, 0x83)]
            self.assertEqual({p[5] for p in writes}, {46})
            actual = {
                motor.name: fixture.serial.get(motor.motor_id, 46, 2) for motor in fixture.actuators
            }
            self.assertEqual(actual, expected)
        finally:
            fixture.host.close()
            fixture.doCleanups()

    def test_register_layout_and_sign_decoding_match_original_tables(self):
        tables = {}
        path = self.source / "src/lerobot/motors/feetech/tables.py"
        exec(compile(path.read_text(), str(path), "exec"), tables)
        decoding = computations(
            self.source / "src/lerobot/motors/encoding_utils.py",
            functions=("decode_sign_magnitude",),
        )["decode_sign_magnitude"]
        names = (
            "Present_Position",
            "Present_Velocity",
            "Present_Load",
            "Present_Voltage",
            "Present_Temperature",
            "Status",
            "Moving",
            "Present_Current",
        )
        for model in ("sts3215", "sts3250", "sts3095"):
            for byte in (0, 1, 127, 128, 255):
                block = bytes([byte]) * 15
                decoded = decode_feedback(block)
                for register, (field, offset, width, sign) in zip(names, _FIELDS, strict=True):
                    self.assertEqual(
                        tables["MODEL_CONTROL_TABLE"][model][register], (offset + 56, width)
                    )
                    self.assertEqual(tables["MODEL_ENCODING_TABLE"][model].get(register), sign)
                    raw = int.from_bytes(block[offset : offset + width], "little")
                    expected = decoding(raw, sign) if sign is not None else raw
                    self.assertEqual(decoded.registers[field], expected)
                self.assertAlmostEqual(
                    decoded.current_a * 1000, decoded.registers["current_raw"] * 6.5
                )

    def test_identified_owner_epoch_and_sequence_match_original_gate(self):
        scope = {}
        path = self.robot / "command_owner.py"
        exec(compile(path.read_text(), str(path), "exec"), scope)
        old, new = scope["CommandOwner"](), CommandOwner("host")
        for metadata in (
            CommandIdentity("pc", 0, "host", 0),
            CommandIdentity("pc", 0, "host", 0),
            CommandIdentity("other", 1, "host", 0),
            CommandIdentity("pc", 2, "stale", 0),
            CommandIdentity("pc", 2, "host", 0),
        ):
            self.assertEqual(new.accept(metadata), old.accept(asdict(metadata), "host"))
        old.release()
        new.release()
        for identity in (
            CommandIdentity("pc", 3, "host", 0),
            CommandIdentity("other", 1, "host", 1),
        ):
            self.assertEqual(new.accept(identity), old.accept(asdict(identity), "host"))

    @unittest.skipUnless(ROS_SOURCE, "Set ALOHAMINI_ROS_SOURCE for full Host/ROS boundary check")
    def test_existing_ros_validator_consumes_native_host_state(self):
        from test_startup import StartupTests

        path = (
            Path(ROS_SOURCE) / "src/alohamini_lerobot_bridge/alohamini_lerobot_bridge/protocol.py"
        )
        scope = computations(
            path,
            functions=("finite_number", "validate_state_observation"),
            classes={"BodyVelocity": None},
            namespace={"math": math, "dataclass": dataclass},
        )
        fixture = StartupTests()
        fixture.setUp()
        try:
            host = fixture.open()
            try:
                fixture.serials["/dev/am_arm_follower_left"].set(11, 69, 2, 50)
                clock = time.monotonic()
                for cycle in range(70):
                    with patch("time.monotonic", return_value=clock + cycle * 0.02):
                        result = host.step()
                    if host.control.base_lift.lift_homing_phase == "complete":
                        break
                self.assertEqual(host.control.base_lift.lift_homing_phase, "complete")
                payload = host._payload(result)
                metadata, _, velocity = scope["validate_state_observation"](
                    payload, SimpleNamespace(observation_to_joint_positions=lambda *_: {})
                )
                self.assertEqual(len(metadata["motors"]), 18)
                self.assertEqual((velocity.x, velocity.y, velocity.yaw), (0, 0, 0))
            finally:
                host.close()
        finally:
            fixture.doCleanups()


class RosConnectionLifecycleTests(unittest.TestCase):
    def test_read_only_worker_connects_commands_only_after_explicit_enable(self):
        from alohamini.errors import AlohaMiniError

        client = Mock()
        client.read.return_value = "state"
        client.connect_control.return_value = "handshake"
        clock = SimpleNamespace(cycle=0)
        source = Path(__file__).resolve().parents[1] / (
            "ros2/src/alohamini_bridge/alohamini_bridge/bridge_node.py"
        )
        receiver_type = computations(
            source,
            classes={"StateReceiver": None},
            namespace={
                "HostClient": lambda *args, **kwargs: contextlib.nullcontext(client),
                "AlohaMiniError": AlohaMiniError,
                "time": time,
            },
        )["StateReceiver"]
        receiver = object.__new__(receiver_type)
        receiver._lock = threading.Lock()
        receiver.commands = Mock()
        receiver.commands.status.side_effect = lambda: (clock.cycle > 0, False, "")
        receiver._stop = Mock()
        receiver._stop.is_set.side_effect = lambda: clock.cycle >= 3
        receiver._stop.wait.side_effect = lambda _: setattr(clock, "cycle", clock.cycle + 1)
        receiver._run("localhost", 5556, "alohamini2pro", 0.25, 50, 5555)
        self.assertEqual(client.read.call_count, 2)
        client.connect_control.assert_called_once()
        self.assertEqual(
            [call.args[1] for call in receiver.commands.step.call_args_list],
            ["state", "handshake", "state"],
        )
        receiver.commands.fail.assert_not_called()


@unittest.skipUnless(ROS_SOURCE, "Set ALOHAMINI_ROS_SOURCE for ROS differential checks")
class RosDifferentialTests(unittest.TestCase):
    def test_original_ros_camera_parser_accepts_native_stream(self):
        from test_camera_stream import capture_frame

        from alohamini.runtime.camera_stream import encode_camera_stream_message

        source = computations(
            Path(ROS_SOURCE) / "src/alohamini_camera/alohamini_camera/protocol.py",
            constants=("CAMERA_STREAM_SCHEMA_VERSION",),
            classes={"CameraFrame": None},
            functions=("parse_camera_message",),
            namespace={"dataclass": dataclass, "json": json, "math": math},
        )
        (_, _, _, snapshot), _ = capture_frame()
        parts = encode_camera_stream_message("forward", snapshot, 5, host_session_id="test-session")
        frame = source["parse_camera_message"](parts)
        self.assertEqual(
            (frame.camera_name, frame.sequence, frame.width, frame.height), ("forward", 5, 16, 8)
        )
        self.assertEqual(frame.jpeg, snapshot.jpeg)
        self.assertEqual(frame.capture_monotonic_s, snapshot.capture_monotonic_s)

    def test_joint_and_lift_mappings_match_explicit_ros_calibration(self):
        path = (
            Path(ROS_SOURCE) / "src/alohamini_lerobot_bridge/alohamini_lerobot_bridge/protocol.py"
        )
        scope = computations(
            path,
            constants=("ARM_JOINTS",),
            functions=("signed_tick_delta", "finite_number"),
            classes={"JointMapper": None},
            namespace={"math": math},
        )
        for direction in (-1, 1):
            for ratio in (0.5, 1):
                entry = {
                    "reference_tick": 1000,
                    "reference_q_rad": 0.1,
                    "sign": direction,
                    "joint_per_encoder_ratio": ratio,
                }
                old = scope["JointMapper"](
                    {"ticks_per_revolution": 4096, "joints": {"shoulder_pan": entry}}
                )
                new = EncoderCalibration(4096, 1000, 0.1, direction, ratio)
                previous = None
                for tick in (1000, 1800, 2600, 3400, 4095, 3, 1000):
                    value = new.position_from_tick(tick, previous_position_rad=previous)
                    self.assertEqual(
                        value, old.tick_to_urdf("shoulder_pan", tick, "left_shoulder_pan")
                    )
                    previous = value
        old.lift_calibration = {
            "mechanism": {"physical_min_mm": 0, "physical_max_mm": 600},
            "urdf": {
                "q_at_physical_min_m": 0.1,
                "q_at_physical_max_m": 0.7,
                "clamp_to_physical_range": False,
            },
        }
        new_lift = LiftCalibration(0, 0.6, 0.1, 0.7)
        for height in (-10, 0, 5, 300, 600, 610):
            self.assertAlmostEqual(
                new_lift.height_to_position(height / 1000), old.lift_height_to_urdf(height)
            )
        for position in (0.1, 0.3, 0.7):
            self.assertAlmostEqual(
                new_lift.position_to_height(position) * 1000, old.lift_urdf_to_height(position)
            )

    def test_command_envelope_matches_actual_ros_sender(self):
        from alohamini.protocol import command_target_keys, encode_command
        from alohamini.schema import CommandIdentity

        path = (
            Path(ROS_SOURCE) / "src/alohamini_lerobot_bridge/alohamini_lerobot_bridge/protocol.py"
        )
        old = computations(
            path,
            classes={"ZmqHostTransport": ("send_action",)},
            namespace={"json": json, "zmq": SimpleNamespace(NOBLOCK=1)},
        )["ZmqHostTransport"]()
        sent = []
        old.client_id, old.command_sequence = "pc", 0
        old.command = SimpleNamespace(send_string=lambda value, **_: sent.append(value))
        for model in ("alohamini1", "alohamini2", "alohamini2pro"):
            keys = command_target_keys(model)
            for epoch in (0, 1, 10):
                old.safety_status = {
                    "version": 1,
                    "control_owner": None,
                    "host_session_id": "session",
                    "control_epoch": epoch,
                }
                for key in sorted(keys - {"lift_axis.stop"}):
                    action = {key: 0.25}
                    self.assertTrue(old.send_action(action))
                    encoded = encode_command(
                        action,
                        CommandIdentity("pc", old.command_sequence, "session", epoch),
                        allowed_targets=keys,
                    )
                    self.assertEqual(json.loads(encoded), json.loads(sent[-1]))


if __name__ == "__main__":
    unittest.main()
