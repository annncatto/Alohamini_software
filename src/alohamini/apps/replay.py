# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from examples/alohamini/replay_bi.py.
"""Local episode replay through the same feedback and command lease as teleoperation."""

import json
import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from alohamini._validation import finite_number
from alohamini.apps.teleoperation import ready_units, stop_owned_robot
from alohamini.calibration.encoder import HostPositionUnits
from alohamini.client import HostClient
from alohamini.datasets.record import state_names
from alohamini.datasets.tools import _read_lock
from alohamini.errors import ResponseTimeoutError
from alohamini.model import get_robot_model
from alohamini.paths import WorkspacePaths


@dataclass(frozen=True)
class ReplayEpisode:
    root: Path
    index: int
    fps: float
    metadata: dict
    names: tuple[str, ...]
    actions: np.ndarray

    @property
    def robot_model(self):
        return self.metadata["robot_model"]


def _local_file(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or external dataset file: {path}")
    return path


def _json(root, relative):
    value = json.loads(_local_file(root, relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {relative}")
    return value


def load_episode(root, episode=0):
    """Read only action/index columns; never load images, a policy, or Hub credentials.

    LeRobot exports need the AlohaMini source metadata: vector length alone cannot
    distinguish a normalized position target from radians or an end-effector delta.
    """
    if type(episode) is not int or episode < 0:
        raise ValueError("Episode index must be a nonnegative integer")
    root = Path(root).expanduser().resolve()
    if root.name.startswith(".pending-"):
        raise ValueError("Cannot replay an unfinished dataset export")
    info = _json(root, "meta/info.json")
    native = info.get("format") == "alohamini-episodes"
    columns = ["action", "frame_index", "episode_index", "timestamp"]
    with _read_lock(root) if native else nullcontext():
        if native:
            if info.get("version") not in (1, 2, 3):
                raise ValueError("Unsupported AlohaMini dataset version")
            if "action" not in info.get("features", {}):
                raise ValueError("Dataset has no action feature for replay")
            source = info
            directory = f"episodes/episode_{episode:06d}"
            summary = _json(root, f"{directory}/episode.json")
            if summary.get("episode_index") != episode:
                raise ValueError("Episode metadata index mismatch")
            length = summary.get("length")
            table = pq.read_table(_local_file(root, f"{directory}/frames.parquet"), columns=columns)
        elif info.get("codebase_version") == "v3.0":
            origin = _json(root, "meta/alohamini.json")
            source = origin.get("source_info", {})
            if origin.get("version") != 1 or source.get("format") != "alohamini-episodes":
                raise ValueError(
                    "Replay requires the recorded AlohaMini action/calibration metadata"
                )
            meta_files = sorted((root / "meta/episodes").rglob("*.parquet"))
            data_files = sorted((root / "data").rglob("*.parquet"))
            for path in (*meta_files, *data_files):
                _local_file(root, path.relative_to(root))
            if not meta_files or not data_files:
                raise ValueError("Missing LeRobot episode/data files")
            selected = ds.field("episode_index") == episode
            summaries = (
                ds.dataset(meta_files, format="parquet")
                .to_table(columns=["length"], filter=selected)
                .to_pylist()
            )
            if len(summaries) != 1:
                raise ValueError("Episode missing or duplicated in LeRobot metadata")
            length = summaries[0]["length"]
            table = ds.dataset(data_files, format="parquet").to_table(
                columns=columns, filter=selected
            )
        else:
            raise ValueError("Replay supports AlohaMini datasets and their LeRobot v3 exports")
    metadata = source.get("robot_metadata", {})
    model = metadata.get("robot_model")
    expected = state_names(model)
    feature = info.get("features", {}).get("action", {})
    names = feature.get("names", [])
    if (
        len(names) != len(expected)
        or set(names) != set(expected)
        or feature.get("shape") != [len(names)]
        or feature.get("dtype") != "float32"
        or feature != source.get("features", {}).get("action")
    ):
        raise ValueError("Dataset action must retain the recorded named Host targets")
    fps = info.get("fps")
    finite_number(fps, "dataset fps")
    if not 0 < fps <= 50 or fps != source.get("fps"):
        raise ValueError("Invalid or changed dataset FPS")
    if type(length) is not int or length <= 0 or len(table) != length:
        raise ValueError("Episode is empty or its frame count is inconsistent")
    for key in ("frame_index", "episode_index"):
        if not pa.types.is_integer(table.schema.field(key).type) or table[key].null_count:
            raise ValueError(f"Invalid {key}")
    if (
        table["frame_index"].to_pylist() != list(range(length))
        or table["episode_index"].to_pylist() != [episode] * length
        or not np.allclose(
            table["timestamp"].to_numpy(), np.arange(length) / fps, rtol=0, atol=1e-4
        )
    ):
        raise ValueError("Episode action order or fixed-FPS timestamps are inconsistent")
    dtype = table.schema.field("action").type
    if not (
        pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype)
    ) or not pa.types.is_floating(dtype.value_type):
        raise ValueError("Action must be a floating-point vector")
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    if actions.shape != (length, len(names)) or not np.isfinite(actions).all():
        raise ValueError("Action vectors have missing, nonfinite or incorrectly sized values")
    actions.setflags(write=False)
    return ReplayEpisode(root, episode, float(fps), metadata, tuple(names), actions)


def check_coordinates(episode, snapshot):
    """A matching model name is insufficient when calibration or units changed."""
    check_calibration(episode.metadata, snapshot)
    check_target_ranges(episode.actions, episode.names, snapshot)


def check_calibration(metadata, snapshot):
    """Match the recorded motor identities, offsets and position coordinates."""
    live = snapshot.payload["_robot_metadata"]
    if metadata.get("robot_model") != snapshot.robot_model or metadata.get("schema_version") != 1:
        raise ValueError("Dataset and Host robot metadata do not match")
    unit_fields = ("normalization", "range_min", "range_max", "drive_mode")
    fields = ("id", "model", *unit_fields, "homing_offset")
    for actuator in get_robot_model(snapshot.robot_model).actuators:
        name = actuator.name
        recorded = metadata.get("motors", {}).get(name, {})
        installed = live.get("motors", {}).get(name, {})
        differences = [
            f"{key}: recorded={recorded.get(key)!r}, Host={installed.get(key)!r}"
            for key in fields
            if key not in recorded or recorded[key] != installed.get(key)
        ]
        if differences:
            raise ValueError(
                f"Dataset/Host calibration or units differ: {name} ({'; '.join(differences)})"
            )
        if recorded["id"] != actuator.motor_id or recorded["model"] != actuator.motor_model:
            raise ValueError(f"Dataset motor identity mismatch: {name}")
    if metadata.get("lift_axis") != live.get("lift_axis"):
        raise ValueError("Dataset/Host lift geometry or limits differ")


def check_target_ranges(actions, names, snapshot):
    """Validate absolute Host targets without clipping or converting their units."""
    TargetRanges(names, snapshot).check(actions)


class TargetRanges:
    """Precomputed coordinates for a session whose live metadata stays unchanged.

    The evaluator checks session, metadata and lift reference before each action.
    Other callers use check_target_ranges to construct fresh bounds per snapshot.
    """

    def __init__(self, names, snapshot):
        live = snapshot.payload["_robot_metadata"]
        fields = ("normalization", "range_min", "range_max", "drive_mode")
        joints = []
        for actuator in get_robot_model(snapshot.robot_model).actuators:
            name = actuator.name
            if not name.startswith("arm_"):
                continue
            units = HostPositionUnits(**{key: live["motors"][name][key] for key in fields})
            lower, upper = sorted(
                (units.from_tick(units.range_min), units.from_tick(units.range_max))
            )
            joints.append((name, names.index(f"{name}.pos"), lower, upper, units.normalization))
        self.joints = tuple(joints)
        self.lift_index = names.index("lift_axis.height_mm")
        self.lift_min = live["lift_axis"]["soft_min_mm"]
        self.lift_max = live["lift_axis"]["soft_max_mm"]

    def clip(self, action):
        for name, _, lower, upper, normalization in self.joints:
            if normalization in ("range_0_100", "range_m100_100"):
                key = f"{name}.pos"
                action[key] = min(upper, max(lower, action[key]))
        action["lift_axis.height_mm"] = min(
            self.lift_max, max(self.lift_min, action["lift_axis.height_mm"])
        )

    def check(self, actions):
        for name, index, lower, upper, _ in self.joints:
            values = actions[:, index]
            if values.min() < lower - 1e-4 or values.max() > upper + 1e-4:
                raise ValueError(f"Recorded action exceeds the installed joint range: {name}")
        heights = actions[:, self.lift_index]
        if heights.min() < self.lift_min or heights.max() > self.lift_max:
            raise ValueError("Recorded action exceeds the installed lift range")


class ReplayGuard:
    """Stop on contact/watchdog events, including events between two reads.

    Gripper contacts are deliberately not joint holds, as in evaluation_safety.py.
    Restarting the application is an explicit operator decision; no automatic retry.
    """

    def __init__(self, client, robot_model, *, operation="回放"):
        self.client, self.robot_model = client, robot_model
        self.operation = operation
        self.context = None
        self._initial_idle = False

    def check(self, snapshot):
        if ready_units(snapshot, self.robot_model, self.client.client_id) is None:
            raise RuntimeError(f"Host 未就绪、反馈无效或由其他客户端控制；{self.operation}停止。")
        safety = snapshot.payload["_safety"]
        for key in ("joint_hold_events", "watchdog_events"):
            if type(safety.get(key)) is not int or safety[key] < 0:
                raise ValueError(f"Host is missing a valid safety counter: {key}")
        watchdog = safety.get("command_watchdog_timeout_s")
        finite_number(watchdog, "Host watchdog timeout")
        if watchdog <= 0:
            raise ValueError("Invalid Host watchdog timeout")
        if self.context is None:
            # Starting replay explicitly claims an idle Host, including one
            # released by a previous client's watchdog. Later events still stop.
            self._initial_idle = (
                safety.get("watchdog_active") is True and safety["control_owner"] is None
            )
        idle = self._initial_idle and safety["control_owner"] is None
        if safety.get("joint_holds") or (safety.get("watchdog_active") and not idle):
            raise RuntimeError(f"Host 触发关节保护或命令超时；{self.operation}停止，请检查机械臂。")
        context = tuple(
            safety[key]
            for key in ("host_session_id", "control_epoch", "joint_hold_events", "watchdog_events")
        )
        if self.context is not None and context != self.context:
            raise RuntimeError(f"Host 会话、控制权或保护事件已改变；{self.operation}停止。")
        self.context = context
        if safety["control_owner"] == self.client.client_id:
            self._initial_idle = False


def run_replay(client, episode, *, fps=None, speed=1.0, verbose_actions=False):
    """Absolute-time playback with feedback supervision, not per-row ACK blocking.

    Overdue rows are skipped rather than burst or replayed for extra time.
    No smoothing, action scaling or physical-timestamp resampling is performed.
    """
    base_fps = episode.fps if fps is None else fps
    finite_number(base_fps, "replay fps")
    finite_number(speed, "replay speed")
    rate = base_fps * speed
    if base_fps <= 0 or speed <= 0 or not math.isfinite(rate) or not 0 < rate <= 50:
        raise ValueError("Effective replay rate must be in (0, 50] Hz")
    interval = 1 / rate
    guard = ReplayGuard(client, episode.robot_model)
    identity = None
    try:
        snapshot = client.read()
        guard.check(snapshot)
        check_coordinates(episode, snapshot)
        snapshot = client.read()
        guard.check(snapshot)
        print(f"Replaying episode {episode.index} from {episode.root}", flush=True)
        print(
            f"Dataset FPS: {episode.fps:g}; replay rate: {rate:.3f} Hz; "
            f"expected duration: {len(episode.actions) / rate:.2f}s",
            flush=True,
        )
        started = time.monotonic()
        end = started + len(episode.actions) * interval
        next_frame_t = started
        index = -1
        skipped = 0
        first_sequence = last_ack_sequence = None
        last_progress = sent_at = started
        last_response = started
        watchdog = snapshot.payload["_safety"]["command_watchdog_timeout_s"]
        timeout_count = 0
        while True:
            now = time.monotonic()
            if snapshot is None:
                if now >= end:
                    break
                if now - last_response >= watchdog:
                    raise RuntimeError("Host 持续无响应已达看门狗时限；回放停止。")
                try:
                    snapshot = client.read()
                except ResponseTimeoutError:
                    timeout_count += 1
                    time.sleep(1 / 50)
                    continue
                guard.check(snapshot)
                last_response = time.monotonic()
                # Recompute the row after receiving, never reuse the loop's old time.
                continue
            status = snapshot.payload["_safety"]
            accepted = status.get("command", {})
            acknowledged = False
            if identity is not None:
                sequence = accepted.get("sequence")
                if (
                    accepted.get("client_id") == identity.client_id
                    and accepted.get("host_session_id") == identity.host_session_id
                    and accepted.get("control_epoch") == identity.control_epoch
                    and type(sequence) is int
                    and first_sequence <= sequence <= identity.sequence
                ):
                    acknowledged = sequence == identity.sequence
                    if last_ack_sequence is None or sequence > last_ack_sequence:
                        last_ack_sequence, last_progress = sequence, now
                if now - last_progress >= status["command_watchdog_timeout_s"]:
                    raise RuntimeError("Host 命令确认未推进；回放停止。")
            if now >= end:
                break
            current = min(int((now - started) / interval + 1e-9), len(episode.actions) - 1)
            renew = acknowledged and now - sent_at >= min(
                0.2, status["command_watchdog_timeout_s"] / 3
            )
            if current > index or renew:
                if current > index:
                    skipped += current - index - 1
                    index = current
                action = dict(zip(episode.names, map(float, episode.actions[index]), strict=True))
                if verbose_actions:
                    print(f"replay_bi.action:{action}", flush=True)
                # Printing may stall too: reselect by time instead of emitting
                # the old row or imposing another short network-fault threshold.
                current = int((time.monotonic() - started) / interval + 1e-9)
                if current >= len(episode.actions):
                    break
                if current > index:
                    skipped += current - index
                    index = current
                    action = dict(
                        zip(episode.names, map(float, episode.actions[index]), strict=True)
                    )
                submitted = client.send_command(action, based_on=snapshot)
                if submitted is not None:
                    identity = submitted
                    if first_sequence is None:
                        first_sequence = identity.sequence
                    sent_at = time.monotonic()
            next_frame_t = started + (index + 1) * interval
            delay = max(0.0, min(next_frame_t, end) - time.monotonic())
            time.sleep(min(1 / 50, delay))
            try:
                snapshot = client.read()
            except ResponseTimeoutError:
                snapshot = None
                timeout_count += 1
                continue
            guard.check(snapshot)
            last_response = time.monotonic()
        if identity is not None and last_ack_sequence is None:
            logging.warning("回放时间轴已结束，但未观察到 Host 命令确认。")
        skipped += len(episode.actions) - index - 1
        print(
            f"Replay timeline finished; skipped overdue rows: {skipped}; "
            f"response timeouts: {timeout_count}",
            flush=True,
        )
    finally:
        stop_owned_robot(client, episode.robot_model, identity)


def replay(
    dataset_name,
    host,
    robot_model,
    *,
    root=None,
    episode=0,
    fps=None,
    speed=1.0,
    verbose_actions=False,
):
    path = Path(root).expanduser() if root is not None else WorkspacePaths().dataset(dataset_name)
    data = load_episode(path, episode)
    if data.robot_model != robot_model:
        raise ValueError("Dataset robot_model does not match --robot_model")
    with HostClient(
        host, expected_model=robot_model, timeout_s=0.2, prefetch_before_decode=True
    ) as client:
        client.connect_control()
        run_replay(client, data, fps=fps, speed=speed, verbose_actions=verbose_actions)
