# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from examples/alohamini/replay_bi.py.
"""Local episode replay through the same feedback and command lease as teleoperation."""

import json
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
from alohamini.datasets.native import state_names
from alohamini.datasets.tools import _read_lock
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

    LeRobot exports need the native source metadata: vector length alone cannot
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
            if info.get("version") not in (1, 2):
                raise ValueError("Unsupported native dataset version")
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
            raise ValueError("Replay supports native datasets and their LeRobot v3 exports")
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
    live = snapshot.payload["_robot_metadata"]
    if episode.robot_model != snapshot.robot_model or episode.metadata.get("schema_version") != 1:
        raise ValueError("Dataset and Host robot metadata do not match")
    unit_fields = ("normalization", "range_min", "range_max", "drive_mode")
    fields = ("id", "model", *unit_fields, "homing_offset")
    for actuator in get_robot_model(episode.robot_model).actuators:
        name = actuator.name
        recorded = episode.metadata.get("motors", {}).get(name, {})
        installed = live.get("motors", {}).get(name, {})
        if any(key not in recorded or recorded[key] != installed.get(key) for key in fields):
            raise ValueError(f"Dataset/Host calibration or units differ: {name}")
        if recorded["id"] != actuator.motor_id or recorded["model"] != actuator.motor_model:
            raise ValueError(f"Dataset motor identity mismatch: {name}")
        if not name.startswith("arm_"):
            continue
        units = HostPositionUnits(**{key: recorded[key] for key in unit_fields})
        values = episode.actions[:, episode.names.index(f"{name}.pos")]
        lower, upper = units.from_tick(units.range_min), units.from_tick(units.range_max)
        if values.min() < min(lower, upper) - 1e-4 or values.max() > max(lower, upper) + 1e-4:
            raise ValueError(f"Recorded action exceeds the installed joint range: {name}")
    heights = episode.actions[:, episode.names.index("lift_axis.height_mm")]
    limits = live["lift_axis"]
    if heights.min() < limits["soft_min_mm"] or heights.max() > limits["soft_max_mm"]:
        raise ValueError("Recorded action exceeds the installed lift range")


class ReplayGuard:
    """Stop on contact/watchdog events, including events between two reads.

    Gripper contacts are deliberately not joint holds, as in evaluation_safety.py.
    Restarting the application is an explicit operator decision; no automatic retry.
    """

    def __init__(self, client, robot_model):
        self.client, self.robot_model = client, robot_model
        self.context = None
        self._initial_idle = False

    def check(self, snapshot):
        if ready_units(snapshot, self.robot_model, self.client.client_id) is None:
            raise RuntimeError("Host 未就绪、反馈无效或由其他客户端控制；回放停止。")
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
            raise RuntimeError("Host 触发关节保护或命令超时；回放停止，请检查机械臂。")
        context = tuple(
            safety[key]
            for key in ("host_session_id", "control_epoch", "joint_hold_events", "watchdog_events")
        )
        if self.context is not None and context != self.context:
            raise RuntimeError("Host 会话、控制权或保护事件已改变；回放停止。")
        self.context = context
        if safety["control_owner"] == self.client.client_id:
            self._initial_idle = False


def run_replay(client, episode, *, fps=None, speed=1.0, verbose_actions=False):
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
        next_frame_t = time.monotonic()
        for values in episode.actions:
            action = dict(zip(episode.names, map(float, values), strict=True))
            if verbose_actions:
                print(f"replay_bi.action:{action}", flush=True)
            identity = client.send_command(action, based_on=snapshot)
            sent_at = time.monotonic()
            # Preserve the original absolute schedule, but never burst through
            # queued goals after a stall. Every dataset row needs a Host ack.
            next_frame_t = max(next_frame_t + interval, sent_at + 1 / 50)
            while True:
                snapshot = client.read()
                guard.check(snapshot)
                status = snapshot.payload["_safety"]
                accepted = status.get("command", {})
                acknowledged = (
                    accepted.get("client_id") == identity.client_id
                    and accepted.get("sequence") == identity.sequence
                    and accepted.get("host_session_id") == identity.host_session_id
                    and accepted.get("control_epoch") == identity.control_epoch
                )
                now = time.monotonic()
                watchdog = status["command_watchdog_timeout_s"]
                if not acknowledged and now - sent_at >= watchdog:
                    raise RuntimeError("Host 未确认回放目标；回放停止。")
                if acknowledged and now >= next_frame_t:
                    break
                # At low playback rates, renew only this accepted target. Keep
                # reading feedback throughout the hold; never mask a lost lease.
                if acknowledged and now - sent_at >= min(0.2, watchdog / 3):
                    identity = client.send_command(action, based_on=snapshot)
                    sent_at = time.monotonic()
                delay = max(0.0, next_frame_t - time.monotonic())
                time.sleep(min(1 / 50, delay) if delay else 1 / 50)
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
    with HostClient(host, expected_model=robot_model) as client:
        run_replay(client, data, fps=fps, speed=speed, verbose_actions=verbose_actions)
