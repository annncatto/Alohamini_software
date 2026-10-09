# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from examples/alohamini/evaluate_bi.py and evaluation_safety.py.
"""Synchronous policy episodes using Host snapshots, commands and local recording."""

import importlib
import json
import logging
import math
import statistics
import time
from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from alohamini._validation import finite_number
from alohamini.apps.replay import check_calibration, check_target_ranges
from alohamini.apps.teleoperation import ready_units, stop_owned_robot
from alohamini.calibration.encoder import HostPositionUnits
from alohamini.client import HostClient, control_feedback_valid
from alohamini.datasets.record import (
    LocalDataset,
    motor_feedback_frame,
    preserve_dataset,
    state_names,
)
from alohamini.errors import ResponseTimeoutError
from alohamini.paths import WorkspacePaths


def _options(fps, duration_s):
    finite_number(duration_s, "evaluation duration")
    if type(fps) is not int or not 1 <= fps <= 30 or duration_s <= 0:
        raise ValueError("Evaluation requires fps in [1, 30] and a positive duration")


def _action(value, names, snapshot):
    """Bound normalized targets as MotorsBus._unnormalize and LiftAxis.apply_action do."""
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError("Policy must return every named absolute Host target, without extra keys")
    for name in names:
        finite_number(value[name], name)
    action = {name: float(value[name]) for name in names}
    metadata = snapshot.payload["_robot_metadata"]
    for name, motor in metadata["motors"].items():
        key = f"{name}.pos"
        if key not in action:
            continue
        units = HostPositionUnits(
            **{k: motor[k] for k in ("normalization", "range_min", "range_max", "drive_mode")}
        )
        if units.normalization in ("range_0_100", "range_m100_100"):
            lower, upper = sorted(
                (units.from_tick(units.range_min), units.from_tick(units.range_max))
            )
            action[key] = min(upper, max(lower, action[key]))
    limits = metadata["lift_axis"]
    action["lift_axis.height_mm"] = min(
        limits["soft_max_mm"], max(limits["soft_min_mm"], action["lift_axis.height_mm"])
    )
    # Degree-based joints retain the Host's strict calibrated encoder bounds.
    check_target_ranges(np.asarray([[action[name] for name in names]]), names, snapshot)
    return action


def _recordable(snapshot, cameras, previous):
    """RecordingGate.frame_ready semantics; never gate policy execution on capture cadence."""
    if previous is not None and snapshot.request_started_s <= previous.request_started_s:
        return False
    if not cameras:
        return True
    timing = snapshot.payload.get("_host_timing", {})
    stamps = timing.get("camera_capture_monotonic_s", {})
    if not all(name in snapshot.images and name in stamps for name in cameras):
        return False
    values = [stamps[name] for name in cameras]
    if not all(math.isfinite(v) for v in values) or max(values) - min(values) > 0.05:
        return False
    if previous is not None:
        old = previous.payload["_host_timing"]["camera_capture_monotonic_s"]
        if any(stamps[name] <= old[name] for name in cameras):
            return False
    state_time = timing.get("state_sample_finished_monotonic_s")
    return state_time is not None and abs(statistics.median(values) - state_time) <= 0.1


class EvaluationGuard:
    """Check feedback/session health without pausing on contact reports."""

    def __init__(self, client, robot_model):
        self.client, self.robot_model = client, robot_model
        self.context = None
        self.ready = False
        self.reason = None
        self._hold_events = 0
        self._held_joints = ()
        self._last_hold_warning = -math.inf

    def check(self, snapshot):
        safety = snapshot.payload["_safety"]
        if self.context is not None and safety["host_session_id"] != self.context[0]:
            raise RuntimeError("Host 已重启；评估停止。")
        if safety.get("control_owner") not in (None, self.client.client_id):
            raise RuntimeError("控制权由其他客户端持有；评估停止。")
        events = safety.get("joint_hold_events")
        if type(events) is not int or events < 0:
            raise ValueError("Host is missing a valid joint protection counter")
        self.context = (safety["host_session_id"],)
        holds = tuple(sorted(safety.get("joint_holds", {})))
        now = time.monotonic()
        if holds and (
            holds != self._held_joints
            or events != self._hold_events
            or now - self._last_hold_warning >= 5.0
        ):
            logging.warning(
                "Host 报告关节保持：%s；评估继续提交目标。"
                "当前 Host 不应对普通关节锁位，请检查 Host 版本。",
                ", ".join(holds),
            )
            self._last_hold_warning = now
        elif not holds and (self._held_joints or events != self._hold_events):
            logging.warning("Host 关节保持报告已清除（累计 %d 次）；评估继续。", events)
        self._held_joints, self._hold_events = holds, events
        self.ready = ready_units(
            snapshot, self.robot_model, self.client.client_id
        ) is not None and control_feedback_valid(snapshot)
        self.reason = None
        if not self.ready:
            if not control_feedback_valid(snapshot):
                self.reason = "Host 反馈中断或过期"
            else:
                self.reason = "Host 尚未提供可控制的整机反馈"


def run_evaluation(client, policy, robot_model, *, fps=30, duration_s=60, dataset=None):
    """Run one episode. The caller owns the client, policy and optional open dataset.

    policy.robot_metadata must describe its trained absolute command coordinates;
    reset() clears temporal/chunk state and select_action(snapshot) returns named
    Host targets. No framework, vector order or end-effector transform is inferred.
    Gripper holds are enforced by Host; joint stall reports do not pause the policy.
    Prolonged feedback loss pauses the episode without input.
    Host restart, calibration changes or another controller end it.
    A synchronous policy cannot be interrupted here;
    the independent Host watchdog remains active while it computes.
    """
    _options(fps, duration_s)
    from alohamini.fixed import FixedGuard, restore, validate

    fixed_config = getattr(policy, "fixed_dimensions", None)
    if not isinstance(fixed_config, Mapping):
        fixed_config = None
    fixed_targets = validate(fixed_config, robot_model)
    metadata = deepcopy(policy.robot_metadata)
    fixed_guard = FixedGuard(fixed_config, metadata)
    guard = EvaluationGuard(client, robot_model)
    names = state_names(robot_model)
    identity = None
    commands = 0
    live_metadata = reference = None
    last_snapshot = last_recorded = None
    last_clip_warning = -math.inf
    last_camera_warning = -math.inf
    last_timeout_warning = -math.inf
    wait_reason = None

    def report_wait(reason):
        nonlocal wait_reason
        if reason != wait_reason:
            if reason is None:
                logging.warning("评估恢复发送动作。")
            else:
                logging.warning("%s；暂停发送动作，等待恢复。", reason)
            wait_reason = reason

    def checked_read(*, images=False):
        nonlocal last_snapshot, last_timeout_warning
        try:
            snapshot = client.read(include_images=images)
        except ResponseTimeoutError:
            if control_feedback_valid(last_snapshot):
                if images and not last_snapshot.images:
                    return None  # A state-only handshake cannot supply the first policy image.
                if time.monotonic() - last_timeout_warning >= 5.0:
                    logging.warning("Host 响应超时；使用上次有效反馈继续评估。")
                    last_timeout_warning = time.monotonic()
                # Keep the original timestamps; cached reads never renew blind control.
                return last_snapshot
            return None
        guard.check(snapshot)
        if live_metadata is not None and snapshot.payload["_robot_metadata"] != live_metadata:
            raise RuntimeError(
                "Host model/calibration/camera configuration changed; evaluation stopped"
            )
        if (
            reference is not None
            and snapshot.payload.get("lift_axis.reference_sequence") != reference
        ):
            raise RuntimeError("Host lift reference changed; evaluation stopped")
        last_snapshot = snapshot
        return snapshot

    try:
        # A successful connection handshake does not guarantee the next reply
        # arrives within one 200 ms poll. Use the same bounded startup allowance.
        startup_deadline = time.monotonic() + 5.0
        initial = checked_read()
        while initial is None and time.monotonic() < startup_deadline:
            time.sleep(min(0.02, max(0, startup_deadline - time.monotonic())))
            initial = checked_read()
        if initial is None:
            raise ResponseTimeoutError("No initial Host feedback for evaluation")
        check_calibration(metadata, initial)
        live_metadata = deepcopy(initial.payload["_robot_metadata"])
        reference = initial.payload.get("lift_axis.reference_sequence")
        if type(reference) is not int or reference < 0:
            raise ValueError("Evaluation requires the Host lift reference sequence")
        cameras = tuple(live_metadata.get("cameras", ()))
        if dataset is not None and (dataset.robot_metadata != live_metadata or dataset.fps != fps):
            raise ValueError("Evaluation dataset must match the live Host metadata and FPS")
        if fixed_targets:
            identity = restore(
                client,
                robot_model,
                fixed_config,
                expected_metadata=live_metadata,
                expected_snapshot=initial,
            )
        policy.reset()
        started = time.monotonic()
        deadline = started + duration_s
        manifest = getattr(policy, "manifest", None)
        policy_cameras = tuple(manifest["cameras"]) if isinstance(manifest, Mapping) else cameras
        print("Starting evaluation", flush=True)
        restart_pending = False
        report_started = started
        selected = submitted_count = changed = 0
        last_action = None
        while time.monotonic() < deadline:
            loop_started = time.monotonic()
            if loop_started - report_started >= 2.0:
                elapsed = loop_started - report_started
                print(
                    f"[EVAL] select_hz={selected / elapsed:.1f} "
                    f"sent_hz={submitted_count / elapsed:.1f} target_changes={changed}",
                    flush=True,
                )
                report_started = loop_started
                selected = submitted_count = changed = 0
            previous_context = guard.context
            observation = checked_read(images=bool(cameras))
            if time.monotonic() >= deadline:
                break
            if observation is None or not guard.ready:
                report_wait(guard.reason or "Host 反馈中断或过期")
                restart_pending = True
                time.sleep(min(1 / fps, max(0, deadline - time.monotonic())))
                continue
            if fixed_targets:
                fixed_guard.check(observation)
            if restart_pending or guard.context != previous_context:
                policy.reset()
                restart_pending = False
            if policy_cameras:
                missing = [name for name in policy_cameras if name not in observation.images]
                if missing:
                    raise ValueError("Missing policy camera: " + ", ".join(missing))
                timing = observation.payload.get("_host_timing", {})
                stamps = timing.get("camera_capture_monotonic_s", {})
                # Camera/state skew is a diagnostic, not a policy queue reset.
                # The source evaluator consumes the available image on every tick.
                state_end = timing.get("state_sample_finished_monotonic_s")
                delayed = {
                    name: round(abs(state_end - stamps[name]), 3)
                    for name in policy_cameras
                    if state_end is not None
                    and name in stamps
                    and abs(state_end - stamps[name]) > 0.25
                }
                if delayed and time.monotonic() - last_camera_warning >= 5.0:
                    logging.warning(
                        "Policy camera/state time difference (s): %s; "
                        "continuing with available images. Check camera capture if persistent.",
                        delayed,
                    )
                    last_camera_warning = time.monotonic()
            if dataset is not None:
                dataset.check_writer()
            inference_started = time.monotonic()
            # Preserve the exact input for recording even if a policy mutates its argument.
            value = policy.select_action(deepcopy(observation))
            selected += 1
            inference_finished = time.monotonic()
            if time.monotonic() >= deadline:
                break
            # One observation per tick, including slow inference. Do not replace
            # its image/state or invalidate queued requests with a second read.
            if not control_feedback_valid(observation):
                report_wait("Host 反馈中断或过期")
                restart_pending = True
                time.sleep(min(1 / fps, max(0, deadline - time.monotonic())))
                continue
            requested = {**value, **fixed_targets} if fixed_targets else value
            action = _action(requested, names, observation)
            clipped = {
                name: (float(requested[name]), action[name])
                for name in names
                if requested[name] != action[name]
            }
            if clipped and time.monotonic() - last_clip_warning >= 1.0:
                logging.warning(
                    "Policy targets clipped to calibrated limits (predicted, submitted): %s",
                    clipped,
                )
                last_clip_warning = time.monotonic()
            submitted = client.send_command(action, based_on=observation)
            if submitted is None:
                # Like the source send_action() path, a dropped send consumes this
                # tick but does not rewind the policy to the start of its chunk.
                report_wait("动作暂未入队（发送拥塞或反馈过期）")
                time.sleep(max(0, min(1 / fps, deadline - time.monotonic())))
                continue
            report_wait(None)
            identity = submitted
            sent_at = time.monotonic()
            commands += 1
            submitted_count += 1
            if last_action is not None and any(
                abs(action[name] - last_action[name]) > 1e-6 for name in names
            ):
                changed += 1
            last_action = action
            # Submission is not execution acknowledgement. Later feedback carries
            # accepted targets; a slow/missing ACK must not stretch every action.
            if dataset is not None and _recordable(observation, cameras, last_recorded):
                payload = observation.payload
                feedback = deepcopy(payload.get("_motor_feedback", {}))
                frame = {
                    "observation.state": [payload[name] for name in names],
                    "action": [action[name] for name in names],
                    **motor_feedback_frame(dataset.features, feedback),
                }
                timing = deepcopy(payload.get("_host_timing", {}))
                start = timing.get("state_sample_started_monotonic_s")
                end = timing.get("state_sample_finished_monotonic_s")
                if start is not None and end is not None:
                    timing["state_sample_monotonic_s"] = (start + end) / 2
                saved = dataset.add_frame(
                    frame,
                    {name: observation.images[name] for name in cameras},
                    {
                        "safety": payload["_safety"],
                        "motor_feedback": feedback,
                        "robot_metadata": live_metadata,
                        "requested_action": action,
                        **({"fixed_dimensions": fixed_config} if fixed_targets else {}),
                        "policy_action": {name: float(value[name]) for name in names},
                        "issued_command": asdict(identity),
                        "host_timing": timing,
                        "client_timing": {
                            "observation_received_monotonic_s": observation.received_s,
                            "action_sample_started_monotonic_s": inference_started,
                            "action_sample_finished_monotonic_s": inference_finished,
                            "command_sent_monotonic_s": sent_at,
                        },
                    },
                )
                if saved:
                    last_recorded = observation
                if not saved and dataset.queue_overflows == 1:
                    logging.warning(
                        "Dataset writer queue full; evaluation continues, capture gaps are counted"
                    )
            # No catch-up burst and no repeated velocity target while inference blocks.
            time.sleep(max(0, min(loop_started + 1 / fps, deadline) - time.monotonic()))
        return commands
    finally:
        stop_owned_robot(client, robot_model, identity)


def evaluate(
    host,
    robot_model,
    *,
    policy_factory,
    fps=30,
    episode_time_s=60,
    num_episodes=1,
    reset_time_s=10,
    dataset_name=None,
    task="robot task",
):
    """Load a developer-supplied policy factory before opening any robot connection."""
    _options(fps, episode_time_s)
    finite_number(reset_time_s, "reset duration")
    if type(num_episodes) is not int or num_episodes <= 0 or reset_time_s < 0:
        raise ValueError("Invalid episode count or reset duration")
    factory = policy_factory
    if not callable(factory):
        module, separator, name = policy_factory.partition(":")
        if not separator or not module or not name.isidentifier():
            raise ValueError("--policy must be an importable module:factory")
        factory = getattr(importlib.import_module(module), name, None)
    if not callable(factory):
        raise ValueError("Policy factory is not callable")
    policy = factory()
    if not all(callable(getattr(policy, name, None)) for name in ("reset", "select_action")):
        raise ValueError("Policy must provide reset() and select_action(snapshot)")
    if not isinstance(getattr(policy, "robot_metadata", None), Mapping):
        raise ValueError("Policy must declare its trained robot_metadata")
    path = WorkspacePaths().dataset(dataset_name) if dataset_name is not None else None
    if path is not None and Path(path).exists():
        raise FileExistsError(f"Evaluation dataset already exists: {path}")
    fixed_config = getattr(policy, "fixed_dimensions", None)
    if not isinstance(fixed_config, Mapping):
        fixed_config = None
    if fixed_config:
        log_dir = WorkspacePaths().logs / "evaluation"
        log_dir.mkdir(parents=True, exist_ok=True)
        report = log_dir / f"fixed-{time.time_ns()}.json"
        details = getattr(policy, "fixed_deployment", None)
        report.write_text(
            json.dumps(
                details if isinstance(details, Mapping) else {"fixed_dimensions": fixed_config},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        print(f"Fixed deployment configuration: {report}", flush=True)
    with ExitStack() as cleanup:
        client = cleanup.enter_context(
            HostClient(
                host,
                expected_model=robot_model,
                timeout_s=0.2,
                request_window=3,
                prefetch_before_decode=True,
            )
        )
        initial = client.connect_control()
        EvaluationGuard(client, robot_model).check(initial)
        check_calibration(policy.robot_metadata, initial)
        dataset = None
        if path is not None:
            dataset = LocalDataset(
                path,
                fps=fps,
                task=task,
                robot_metadata=initial.payload["_robot_metadata"],
                **({"fixed_dimensions": fixed_config} if fixed_config else {}),
            )
            cleanup.enter_context(preserve_dataset(dataset))
        for episode in range(num_episodes):
            if dataset is not None:
                dataset.begin_episode()
            print(f"Eval episode {episode + 1} of {num_episodes}", flush=True)
            count = run_evaluation(
                client, policy, robot_model, fps=fps, duration_s=episode_time_s, dataset=dataset
            )
            if dataset is not None:
                dataset.save_episode()
            print(f"Evaluation episode ended: {count} submitted actions", flush=True)
            if episode + 1 < num_episodes:
                deadline = time.monotonic() + reset_time_s
                while time.monotonic() < deadline:
                    print(
                        f"\r[RESET] {math.ceil(deadline - time.monotonic())}s", end="", flush=True
                    )
                    try:
                        client.read()
                    except ResponseTimeoutError:
                        pass
                    time.sleep(min(0.1, max(0, deadline - time.monotonic())))
                print(flush=True)
