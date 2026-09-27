# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from examples/alohamini/evaluate_bi.py and evaluation_safety.py.
"""Synchronous policy episodes using Host snapshots, commands and local recording."""

import importlib
import logging
import math
import time
from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from alohamini._validation import finite_number
from alohamini.apps.recording import FreshCameraGate
from alohamini.apps.replay import ReplayGuard, check_calibration, check_target_ranges
from alohamini.apps.teleoperation import stop_owned_robot
from alohamini.client import HostClient
from alohamini.datasets.native import (
    LocalDataset,
    motor_feedback_frame,
    preserve_dataset,
    state_names,
)
from alohamini.paths import WorkspacePaths


def _options(fps, duration_s):
    finite_number(duration_s, "evaluation duration")
    if type(fps) is not int or not 1 <= fps <= 30 or duration_s <= 0:
        raise ValueError("Evaluation requires fps in [1, 30] and a positive duration")


def _action(value, names, snapshot):
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError("Policy must return every named absolute Host target, without extra keys")
    for name in names:
        finite_number(value[name], name)
    action = {name: float(value[name]) for name in names}
    check_target_ranges(np.asarray([[action[name] for name in names]]), names, snapshot)
    return action


def run_evaluation(client, policy, robot_model, *, fps=30, duration_s=60, dataset=None):
    """Run one episode. The caller owns the client, policy and optional open dataset.

    policy.robot_metadata must describe its trained absolute command coordinates;
    reset() clears temporal/chunk state and select_action(snapshot) returns named
    Host targets. No framework, vector order or end-effector transform is inferred.
    Safety changes stop the episode; calling this function again is an explicit
    restart and resets the policy. A synchronous policy cannot be interrupted here;
    the independent Host watchdog remains active while it computes.
    """
    _options(fps, duration_s)
    metadata = deepcopy(policy.robot_metadata)
    guard = ReplayGuard(client, robot_model, operation="评估")
    names = state_names(robot_model)
    identity = None
    commands = 0
    live_metadata = reference = None

    def checked_read(*, images=False):
        snapshot = client.read(include_images=images)
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
        return snapshot

    try:
        initial = checked_read()
        check_calibration(metadata, initial)
        live_metadata = deepcopy(initial.payload["_robot_metadata"])
        reference = initial.payload.get("lift_axis.reference_sequence")
        if type(reference) is not int or reference < 0:
            raise ValueError("Evaluation requires the Host lift reference sequence")
        cameras = tuple(live_metadata.get("cameras", ()))
        if dataset is not None and (dataset.robot_metadata != live_metadata or dataset.fps != fps):
            raise ValueError("Evaluation dataset must match the live Host metadata and FPS")
        policy.reset()
        started = time.monotonic()
        deadline = started + duration_s
        gate = (
            FreshCameraGate(cameras, started_at=started, stall_timeout_s=1.0, max_skew_s=0.05)
            if cameras
            else None
        )
        print("Starting evaluation", flush=True)
        next_observation = None
        while time.monotonic() < deadline:
            loop_started = time.monotonic()
            observation = next_observation
            next_observation = None
            if observation is None or loop_started - observation.request_started_s >= 0.25:
                observation = checked_read(images=bool(cameras))
            else:
                guard.check(observation)
            if time.monotonic() >= deadline:
                break
            if gate is not None:
                timing = observation.payload["_host_timing"]
                stamps = timing.get("camera_capture_monotonic_s", {})
                timestamp = gate.observe(stamps, now=time.monotonic())
                if timestamp is None or any(name not in observation.images for name in cameras):
                    time.sleep(min(1 / fps, max(0, deadline - time.monotonic())))
                    continue
                # Compare only Host-clock timestamps, never PC/Host monotonic times.
                state_end = timing.get("state_sample_finished_monotonic_s")
                finite_number(state_end, "Host state timestamp")
                if any(abs(state_end - stamps[name]) > 0.25 for name in cameras):
                    raise RuntimeError("Policy camera/state snapshot is stale; evaluation stopped")
            if dataset is not None:
                dataset.check_writer()
            inference_started = time.monotonic()
            # Preserve the exact input for recording even if a policy mutates its argument.
            value = policy.select_action(deepcopy(observation))
            inference_finished = time.monotonic()
            # Always refresh safety after inference. Never bind old work to a new lease.
            latest = checked_read()
            if time.monotonic() >= deadline:
                break
            action = _action(value, names, latest)
            identity = client.send_command(action, based_on=observation)
            sent_at = time.monotonic()
            while True:
                accepted = checked_read(images=bool(cameras))
                status = accepted.payload["_safety"]
                if status.get("command") == asdict(identity):
                    break
                if time.monotonic() - sent_at >= status["command_watchdog_timeout_s"]:
                    raise RuntimeError("Host 未确认策略目标；评估停止。")
                time.sleep(1 / 50)
            commands += 1
            # The acknowledgement is also a checked, fresh policy observation.
            # Keep the post-inference safety refresh and avoid reading it twice.
            next_observation = accepted
            if dataset is not None:
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
                        "issued_command": asdict(identity),
                        "accepted_safety": status,
                        "host_timing": timing,
                        "client_timing": {
                            "observation_received_monotonic_s": observation.received_s,
                            "action_sample_started_monotonic_s": inference_started,
                            "action_sample_finished_monotonic_s": inference_finished,
                            "command_sent_monotonic_s": sent_at,
                        },
                    },
                )
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
    with ExitStack() as cleanup:
        client = cleanup.enter_context(
            HostClient(host, expected_model=robot_model, timeout_s=0.2, request_window=1)
        )
        initial = client.connect_control()
        ReplayGuard(client, robot_model, operation="评估").check(initial)
        check_calibration(policy.robot_metadata, initial)
        dataset = None
        if path is not None:
            dataset = LocalDataset(
                path, fps=fps, task=task, robot_metadata=initial.payload["_robot_metadata"]
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
            print(f"Evaluation episode ended: {count} acknowledged actions", flush=True)
            if episode + 1 < num_episodes:
                deadline = time.monotonic() + reset_time_s
                while time.monotonic() < deadline:
                    print(
                        f"\r[RESET] {math.ceil(deadline - time.monotonic())}s", end="", flush=True
                    )
                    client.read()
                    time.sleep(min(0.1, max(0, deadline - time.monotonic())))
                print(flush=True)
