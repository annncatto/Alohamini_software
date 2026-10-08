# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Migrated from examples/alohamini/record_utils_multirate.py and record_bi.py.
"""Multirate collection with subsequent targets and local episode storage."""

import logging
import math
import statistics
import sys
import time
from collections import deque
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from alohamini._validation import finite_number
from alohamini.apps.teleoperation import (
    KeyboardInput,
    KeyboardTargets,
    _same_control_session,
    ready_units,
    stop_owned_robot,
)
from alohamini.client import HostClient, control_feedback_valid
from alohamini.errors import ResponseTimeoutError
from alohamini.hardware.leader import BimanualLeader
from alohamini.paths import WorkspacePaths


@dataclass(frozen=True)
class StateSample:
    """Host feedback retained for image/state alignment, never for action selection."""

    observation: dict[str, Any]
    state_timestamp: float | None
    safety: dict[str, Any] = field(default_factory=dict)
    motor_feedback: dict[str, Any] = field(default_factory=dict)


class StateSampleBuffer:
    """Keep bounded Host feedback history to align delayed camera frames."""

    def __init__(self, max_samples: int) -> None:
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2")
        self._samples: deque[StateSample] = deque(maxlen=max_samples)

    def append(self, sample: StateSample) -> None:
        self._samples.append(sample)

    def nearest(self, timestamp: float, *, max_error_s: float) -> tuple[StateSample, float]:
        timestamped = [
            sample
            for sample in self._samples
            if sample.state_timestamp is not None and math.isfinite(sample.state_timestamp)
        ]
        if not timestamped:
            raise RuntimeError("no timestamped control sample is available for dataset alignment")
        if not math.isfinite(timestamp):
            raise RuntimeError("non-finite camera capture timestamp")

        sample = min(
            timestamped,
            key=lambda item: abs(float(item.state_timestamp) - timestamp),
        )
        error_s = abs(float(sample.state_timestamp) - timestamp)
        if error_s > max_error_s:
            raise RuntimeError(
                "camera/state alignment exceeded the allowed error: "
                f"{error_s * 1000:.1f} ms > {max_error_s * 1000:.1f} ms"
            )
        return sample, error_s


class FreshCameraGate:
    """Accept complete, fresh multi-camera snapshots and diagnose stalled cameras."""

    def __init__(
        self,
        camera_names: tuple[str, ...],
        *,
        started_at: float,
        stall_timeout_s: float,
        max_skew_s: float,
    ) -> None:
        if not camera_names:
            raise ValueError("FreshCameraGate requires at least one camera")
        if stall_timeout_s <= 0.0 or max_skew_s < 0.0:
            raise ValueError("camera timeout must be positive and skew must be non-negative")
        self.camera_names = camera_names
        self.stall_timeout_s = float(stall_timeout_s)
        self.max_skew_s = float(max_skew_s)
        self._last_seen: dict[str, float] = {}
        self._last_recorded: dict[str, float] = {}
        self._last_advanced_at = dict.fromkeys(camera_names, float(started_at))
        self.last_skew_s = 0.0

    def observe(self, timestamps: dict[str, float], *, now: float) -> float | None:
        for name in self.camera_names:
            if name not in timestamps:
                continue
            timestamp = float(timestamps[name])
            if not math.isfinite(timestamp):
                raise RuntimeError("non-finite camera capture timestamp")
            if timestamp > self._last_seen.get(name, -math.inf):
                self._last_seen[name] = timestamp
                self._last_advanced_at[name] = now

        stalled = [
            name
            for name in self.camera_names
            if now - self._last_advanced_at[name] > self.stall_timeout_s
        ]
        if stalled:
            raise RuntimeError(
                f"camera capture timestamp stalled for > {self.stall_timeout_s:.1f} s: "
                f"{', '.join(stalled)}"
            )

        if not all(name in timestamps for name in self.camera_names):
            return None
        if not all(
            float(timestamps[name]) > self._last_recorded.get(name, -math.inf)
            for name in self.camera_names
        ):
            return None

        values = [float(timestamps[name]) for name in self.camera_names]
        self.last_skew_s = max(values) - min(values)
        if self.last_skew_s > self.max_skew_s:
            raise RuntimeError(
                "multi-camera capture skew exceeded the allowed limit: "
                f"{self.last_skew_s * 1000:.1f} ms > {self.max_skew_s * 1000:.1f} ms"
            )

        self._last_recorded = {name: float(timestamps[name]) for name in self.camera_names}
        return float(statistics.median(values))


def _advance_deadline(deadline: float, interval: float, now: float) -> float:
    while deadline <= now:
        deadline += interval
    return deadline


class RecordingKeyboard(KeyboardInput):
    """Existing held-key teleoperation plus original episode controls."""

    def __init__(self):
        super().__init__()
        self.events = dict(exit_early=False, rerecord_episode=False, stop_recording=False)

    def update(self, name, pressed):
        if pressed:
            with self._lock:
                if name == "right":
                    print("Right arrow key pressed. Exiting loop...")
                    self.events["exit_early"] = True
                elif name in ("left", "r", "R"):
                    print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                    self.events.update(rerecord_episode=True, exit_early=True)
                elif name in ("esc", "q"):
                    print("Escape key pressed. Stopping data recording...")
                    self.events.update(stop_recording=True, exit_early=True)
        super().update(name, pressed)


def print_countdown(phase, episode_number, remaining_s, *, fps, is_recording, actual_fps=None):
    """Retain record_bi.py's single-line countdown layout."""
    recording_state = "RECORDING" if is_recording else "NOT RECORDING"
    message = f"[{phase:<7}] Ep {episode_number:<3} | {remaining_s:>4}s | {recording_state}"
    if actual_fps is not None:
        message += f" | FPS {actual_fps:>5.1f}/{fps}"
    sys.stdout.write(f"\r\033[2K{message}")
    sys.stdout.flush()


def record_loop(
    client,
    robot_model,
    leader,
    keyboard,
    *,
    fps,
    duration_s,
    metadata,
    dataset=None,
    episode_number=1,
    profile_timing=False,
    on_frame=None,
):
    """Source observe -> fresh input -> send -> image/state pair -> save sequence.

    Image-aligned state history never supplies action labels. No frames are added
    after the wall-time deadline to reach an artificial fixed frame count.
    """
    from alohamini.datasets.record import motor_feedback_frame, state_names

    finite_number(duration_s, "episode duration")
    if duration_s <= 0 or type(fps) is not int or not 1 <= fps <= 30:
        raise ValueError("Positive duration and dataset fps in [1, 30] required")
    start_episode_t = time.perf_counter()
    deadline = start_episode_t + duration_s
    control_interval, dataset_interval = 1 / 50, 1 / fps
    next_camera_request_t = next_state_only_sample_t = start_episode_t
    expected_cameras = (
        dataset.cameras if dataset is not None else tuple(metadata["cameras"]) if on_frame else ()
    )
    sample_buffer = StateSampleBuffer(50)
    camera_gate = (
        FreshCameraGate(
            expected_cameras, started_at=start_episode_t, stall_timeout_s=1.0, max_skew_s=0.05
        )
        if dataset is not None and expected_cameras
        else None
    )
    names = state_names(robot_model)
    mapper = KeyboardTargets()
    identity = context = previous = last_recovery_reason = None
    feedback_warning = False
    waiting_response = False
    events = keyboard.events
    report_started = start_episode_t
    control_count = capture_count = 0
    totals = dict(
        observation=0.0, teleop=0.0, send_action=0.0, dataset_write=0.0, display=0.0, loop=0.0
    )
    last_remaining = actual_fps = None
    client.set_recording_cameras(dataset is not None and bool(expected_cameras))
    try:
        while True:
            start_loop_t = time.perf_counter()
            if start_loop_t >= deadline - 1e-9 or events["exit_early"]:
                events["exit_early"] = False
                break
            if dataset is not None:
                dataset.check_writer()
            remaining = max(0, math.ceil(deadline - start_loop_t))
            if remaining != last_remaining:
                print_countdown(
                    "RECORD" if dataset is not None else "RESET",
                    episode_number,
                    remaining,
                    fps=fps,
                    is_recording=dataset is not None,
                    actual_fps=actual_fps,
                )
                last_remaining = remaining
            request_cameras = bool(expected_cameras) and start_loop_t >= next_camera_request_t
            if request_cameras:
                next_camera_request_t = _advance_deadline(
                    next_camera_request_t, dataset_interval, start_loop_t
                )
            state_only_due = not expected_cameras and start_loop_t >= next_state_only_sample_t
            if state_only_due:
                next_state_only_sample_t = _advance_deadline(
                    next_state_only_sample_t, dataset_interval, start_loop_t
                )
            new_response = True
            try:
                snapshot = client.read_recording(include_images=request_cameras)
            except ResponseTimeoutError:
                new_response = False
                if not waiting_response:
                    if dataset is not None:
                        dataset.event({"type": "response_timeout"})
                waiting_response = True
                if not control_feedback_valid(previous):
                    mapper.reset()
                    if keyboard.read() is None:
                        events["stop_recording"] = True
                        break
                    time.sleep(max(0, control_interval - (time.perf_counter() - start_loop_t)))
                    continue
                snapshot = previous
            observation_received_t = time.monotonic()
            if events["exit_early"] or time.perf_counter() >= deadline:
                events["exit_early"] = False
                break
            payload = snapshot.payload
            if payload["_robot_metadata"] != metadata:
                raise RuntimeError(
                    "Host model/calibration/camera configuration changed during recording"
                )
            units = ready_units(snapshot, robot_model, client.client_id)
            safety = payload["_safety"]
            current_context = (safety["host_session_id"], safety["control_epoch"])
            if context is not None and context != current_context:
                if units is None or not _same_control_session(previous, snapshot, client.client_id):
                    raise RuntimeError(
                        "Host session or control lease changed; collected frames will be preserved"
                    )
                identity = None
                mapper.reset()
                # Never align post-stop images with pre-stop state history.
                sample_buffer = StateSampleBuffer(50)
                context = current_context
                if dataset is not None:
                    dataset.event({"type": "watchdog_recovered"}, safety)
                logging.info("Host 响应恢复，重新采样主臂和按键，继续采集。")
            previous = snapshot
            if units is None:
                mapper.reset()
                if safety.get("control_owner") not in (None, client.client_id):
                    raise RuntimeError(
                        "Another client owns Host control; collected frames will be preserved"
                    )
                if keyboard.read() is None:
                    events["stop_recording"] = True
                    break
                time.sleep(max(0, control_interval - (time.perf_counter() - start_loop_t)))
                continue
            if waiting_response and new_response:
                waiting_response = False
                logging.info("Host 响应恢复，继续采集。")
                if dataset is not None:
                    dataset.event({"type": "response_recovered"}, safety)
            sampled_safety = deepcopy(safety)
            sampled_motor_feedback = deepcopy(payload.get("_motor_feedback", {}))
            observation_done_t = time.perf_counter()
            action_started_t = time.monotonic()
            action = leader.read(units)
            keys = keyboard.read()
            action_finished_t = time.monotonic()
            if keys is None:
                events["stop_recording"] = True
                break
            if not control_feedback_valid(snapshot):
                mapper.reset()
                time.sleep(max(0, control_interval - (time.perf_counter() - start_loop_t)))
                continue
            action.update(mapper.targets(keys, payload, now=action_finished_t))
            teleop_done_t = time.perf_counter()
            if events["exit_early"] or teleop_done_t >= deadline:
                events["exit_early"] = False
                break
            submitted = client.send_command(action, based_on=snapshot)
            if submitted is None:
                time.sleep(max(0, control_interval - (time.perf_counter() - start_loop_t)))
                continue
            identity = submitted
            context = current_context
            command_sent_t = time.monotonic()
            send_action_done_t = time.perf_counter()
            if send_action_done_t >= deadline:
                break
            if not new_response:
                # Continue new operator input during brief gaps, but do not record cached samples.
                time.sleep(max(0, control_interval - (time.perf_counter() - start_loop_t)))
                continue
            timing = payload["_host_timing"]
            state_started_t = timing.get("state_sample_started_monotonic_s")
            state_finished_t = timing.get("state_sample_finished_monotonic_s")
            state_timestamp = (
                (float(state_started_t) + float(state_finished_t)) / 2
                if state_started_t is not None and state_finished_t is not None
                else None
            )
            current_sample = StateSample(
                {name: payload[name] for name in names},
                state_timestamp,
                sampled_safety,
                sampled_motor_feedback,
            )
            sample_buffer.append(current_sample)
            selected_sample = current_sample if dataset is not None and state_only_due else None
            alignment_error_s = None
            if camera_gate is not None:
                try:
                    camera_timestamp = camera_gate.observe(
                        timing.get("camera_capture_monotonic_s", {}), now=start_loop_t
                    )
                    if camera_timestamp is not None:
                        selected_sample, alignment_error_s = sample_buffer.nearest(
                            camera_timestamp, max_error_s=0.10
                        )
                        if not all(name in snapshot.images for name in expected_cameras):
                            raise RuntimeError("fresh camera timestamp has no corresponding image")
                        if last_recovery_reason is not None:
                            logging.info(
                                "Camera/state alignment recovered; dataset recording continues"
                            )
                            dataset.event({"type": "capture_recovered"}, sampled_safety)
                        last_recovery_reason = None
                except RuntimeError as exc:
                    selected_sample = None
                    reason = str(exc).partition(":")[0]
                    if reason != last_recovery_reason:
                        logging.warning(
                            "Waiting for aligned camera data: %s; teleoperation continues", exc
                        )
                        dataset.event({"type": "capture_wait", "reason": str(exc)}, sampled_safety)
                    last_recovery_reason = reason
            if dataset is not None and selected_sample is not None:
                feedback_frame = motor_feedback_frame(
                    dataset.features, selected_sample.motor_feedback
                )
                if not feedback_warning and any(
                    not value.all()
                    for key, value in feedback_frame.items()
                    if key.endswith("_valid")
                ):
                    logging.warning(
                        "Motor feedback is incomplete; recording continues with validity masks."
                    )
                    feedback_warning = True
                frame = {
                    "observation.state": [selected_sample.observation[name] for name in names],
                    "action": [action[name] for name in names],
                    **feedback_frame,
                }
                accepted = dataset.add_frame(
                    frame,
                    {name: snapshot.images[name] for name in expected_cameras},
                    {
                        "safety": selected_sample.safety,
                        "motor_feedback": selected_sample.motor_feedback,
                        "robot_metadata": metadata,
                        "requested_action": action,
                        "issued_command": asdict(identity),
                        "alignment_error_s": alignment_error_s,
                        "client_timing": {
                            "observation_received_monotonic_s": observation_received_t,
                            "action_sample_started_monotonic_s": action_started_t,
                            "action_sample_finished_monotonic_s": action_finished_t,
                            "command_sent_monotonic_s": command_sent_t,
                        },
                        "host_timing": {
                            "state_sample_monotonic_s": selected_sample.state_timestamp,
                            "camera_capture_monotonic_s": timing.get(
                                "camera_capture_monotonic_s", {}
                            ),
                        },
                    },
                )
                capture_count += int(accepted)
                if not accepted and dataset.queue_overflows == 1:
                    logging.warning(
                        "Dataset writer queue full; teleoperation continues, "
                        "capture gaps are counted"
                    )
            dataset_write_done_t = time.perf_counter()
            if on_frame is not None:
                on_frame(snapshot, action)
            work_done_t = time.perf_counter()
            time.sleep(
                max(0, min(control_interval - (work_done_t - start_loop_t), deadline - work_done_t))
            )
            loop_done_t = time.perf_counter()
            for name, value in {
                "observation": observation_done_t - start_loop_t,
                "teleop": teleop_done_t - observation_done_t,
                "send_action": send_action_done_t - teleop_done_t,
                "dataset_write": dataset_write_done_t - send_action_done_t,
                "display": work_done_t - dataset_write_done_t,
                "loop": loop_done_t - start_loop_t,
            }.items():
                totals[name] += value
            control_count += 1
            elapsed = loop_done_t - report_started
            if elapsed >= 1:
                actual_fps = capture_count / elapsed
                if profile_timing:
                    sys.stdout.write("\r\033[2K")
                    fields = " ".join(
                        f"{name}={value * 1000 / control_count:.1f}"
                        for name, value in totals.items()
                    )
                    print(
                        "[PC TIMING avg ms/control-cycle] "
                        f"control={control_count / elapsed:.1f}/50 "
                        f"capture={actual_fps:.1f}/{fps} {fields}",
                        flush=True,
                    )
                control_count = capture_count = 0
                totals = dict.fromkeys(totals, 0.0)
                report_started = loop_done_t
    finally:
        client.set_recording_cameras(False)
        stop_owned_robot(client, robot_model, identity)
        print(flush=True)


def record(
    host,
    robot_model,
    *,
    dataset_name,
    task,
    root=None,
    fps=30,
    num_episodes=1,
    episode_time_s=60,
    reset_time_s=10,
    resume=False,
    leader_id=None,
    calibration_dir=None,
    left_port="/dev/am_arm_leader_left",
    right_port="/dev/am_arm_leader_right",
    arm_profile=None,
    display_data=False,
    profile_timing=False,
):
    from alohamini.datasets.record import LocalDataset, preserve_dataset

    for value in (episode_time_s, reset_time_s):
        finite_number(value, "recording duration")
    if (
        type(fps) is not int
        or not 1 <= fps <= 30
        or type(num_episodes) is not int
        or num_episodes <= 0
        or episode_time_s <= 0
        or reset_time_s < 0
        or not isinstance(task, str)
        or not task.strip()
    ):
        raise ValueError("Invalid recording fps, episode count, duration or task")
    expected_profile = "so-arm-5dof" if robot_model == "alohamini1" else "am-leader-6dof"
    if arm_profile is not None and arm_profile != expected_profile:
        raise ValueError("Leader profile must match robot_model")
    path = Path(root).expanduser() if root is not None else WorkspacePaths().dataset(dataset_name)
    if not path.is_absolute():
        raise ValueError("Dataset root must be absolute")
    if path.exists() and not resume:
        raise FileExistsError(f"Dataset already exists: {path}; use --resume or another name")
    leader = BimanualLeader(
        robot_model,
        leader_id=leader_id,
        calibration_dir=calibration_dir,
        left_port=left_port,
        right_port=right_port,
    )
    with ExitStack() as dataset_cleanup, ExitStack() as cleanup:
        client = cleanup.enter_context(
            HostClient(host, expected_model=robot_model, timeout_s=0.2, request_window=3)
        )
        initial = client.connect_control()
        if ready_units(initial, robot_model, client.client_id) is None:
            raise RuntimeError(
                "Host not ready; finish homing and check control ownership before recording"
            )
        cleanup.enter_context(leader)
        # Interactive calibration can take minutes. Refresh the Host before
        # fixing dataset coordinates; start keyboard capture only afterwards.
        initial = client.connect_control()
        if ready_units(initial, robot_model, client.client_id) is None:
            raise RuntimeError("Host not ready after Leader setup; recording has not started")
        keyboard = cleanup.enter_context(RecordingKeyboard())
        metadata = initial.payload["_robot_metadata"]
        dataset = LocalDataset(path, fps=fps, task=task, robot_metadata=metadata, resume=resume)
        dataset_cleanup.enter_context(preserve_dataset(dataset))
        on_frame = None
        if display_data:
            from alohamini.apps.visualization import init_rerun, log_snapshot, shutdown_rerun

            cleanup.callback(shutdown_rerun)
            init_rerun("alohamini_record")
            on_frame = log_snapshot
        recorded_episodes = 0
        events = keyboard.events
        while recorded_episodes < num_episodes and not events["stop_recording"]:
            episode_number = dataset.num_episodes + 1
            dataset.begin_episode()
            print(
                f"Episode {episode_number} recording started. "
                f"{num_episodes - recorded_episodes} episode(s) remaining. "
                "Press -> to end recording; press R to discard and re-record.",
                flush=True,
            )
            started = time.monotonic()
            record_loop(
                client,
                robot_model,
                leader,
                keyboard,
                fps=fps,
                duration_s=episode_time_s,
                metadata=metadata,
                dataset=dataset,
                episode_number=episode_number,
                profile_timing=profile_timing,
                on_frame=on_frame,
            )
            elapsed = time.monotonic() - started
            print(
                f"Episode {episode_number} capture rate: {dataset.submitted / elapsed:.2f} FPS "
                f"({dataset.submitted} submitted frames in {elapsed:.2f}s; target {fps} FPS)",
                flush=True,
            )
            print(f"Episode {episode_number} recording ended. Resetting before save.", flush=True)
            if not events["stop_recording"] and reset_time_s:
                events["exit_early"] = False
                record_loop(
                    client,
                    robot_model,
                    leader,
                    keyboard,
                    fps=fps,
                    duration_s=reset_time_s,
                    metadata=metadata,
                    episode_number=episode_number,
                    on_frame=on_frame,
                )
            if events["rerecord_episode"]:
                dataset.discard_episode()
                print(
                    f"Discarding episode {episode_number}; retained under {path / 'discarded'}.",
                    flush=True,
                )
            else:
                print(f"Saving episode {episode_number}; please wait...", flush=True)
                save_started = time.perf_counter()
                previous = dataset.num_episodes
                dataset.save_episode()
                if dataset.num_episodes > previous:
                    recorded_episodes += 1
                    print(
                        f"Episode {episode_number} saved in "
                        f"{time.perf_counter() - save_started:.1f} second(s). "
                        f"{dataset.saved} valid frames.",
                        flush=True,
                    )
                else:
                    print("No frames collected; ready to record again.", flush=True)
            events["exit_early"] = events["rerecord_episode"] = False
    print(f"Dataset saved at {path.resolve()}", flush=True)
