#!/usr/bin/env python3
# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline 0922demo_5 demonstration; uses the installed local Host, never serial I/O.

prepare runs on the PC. run needs only the existing Pi Host dependencies.
Calibration/range/guard functions below are copied from apps/replay.py so the
standalone deployment does not import its Parquet dependencies on the Pi.
"""

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from alohamini._validation import finite_number
from alohamini.apps.teleoperation import ready_units, stop_owned_robot
from alohamini.calibration.encoder import HostPositionUnits
from alohamini.client import HostClient
from alohamini.model import get_robot_model


def smooth_lift(values, fps, lower, upper):
    """Offline symmetric 0.5 s smoothing and a 40 mm/s target slew limit."""
    values = np.clip(np.asarray(values, dtype=float), lower, upper)
    width = int(round(0.5 * fps)) | 1
    padded = np.pad(values, (width // 2, width // 2), mode="edge")
    result = np.convolve(padded, np.ones(width) / width, mode="valid")
    result[0], result[-1] = values[0], values[-1]
    step = 40.0 / fps
    if abs(values[-1] - values[0]) > step * (len(values) - 1):
        raise ValueError("Lift endpoints cannot be reached within the recorded duration")
    for i in range(1, len(result)):
        result[i] = np.clip(result[i], result[i - 1] - step, result[i - 1] + step)
    result[-1] = values[-1]
    for i in range(len(result) - 2, -1, -1):
        result[i] = np.clip(result[i], result[i + 1] - step, result[i + 1] + step)
    return result


def transition(first, last, names, fps, minimum_s=2.0):
    """Stationary-base smoothstep between poses; no homing or calibration writes."""
    first, last = np.asarray(first), np.asarray(last)
    base = [names.index(n) for n in ("x.vel", "y.vel", "theta.vel")]
    lift = names.index("lift_axis.height_mm")
    arm = [i for i, name in enumerate(names) if name.endswith(".pos")]
    duration = max(
        minimum_s,
        1.5 * np.max(np.abs(last[arm] - first[arm])) / 10.0,
        1.5 * abs(last[lift] - first[lift]) / 40.0,
    )
    steps = max(2, math.ceil(duration * fps) + 1)
    u = np.linspace(0, 1, steps)
    result = first + (3 * u**2 - 2 * u**3)[:, None] * (last - first)
    result[:, base] = 0
    return result


def prepare(dataset, output):
    # These dependencies are used only for offline preparation on the PC.
    import pyarrow.parquet as pq

    from alohamini.apps.replay import load_episode
    from alohamini.datasets.tools import _read_lock, check_dataset

    root, output = Path(dataset).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("Output already exists; original data will not be overwritten")
    if output.is_relative_to(root):
        raise ValueError("Demo output must be outside the source dataset")
    with _read_lock(root):
        report = check_dataset(root)
        if not report["valid"]:
            raise ValueError(f"Dataset structural errors: {report['issues']}")
        episode = load_episode(root, 0)
        path = root / "episodes/episode_000000"
        logs = [json.loads(line) for line in (path / "safety.jsonl").read_text().splitlines()]
        rows = [row for row in logs if row.get("event") is None]
        if [row.get("frame_index") for row in rows] != list(range(len(episode.actions))):
            raise ValueError("Physical timestamp rows do not match episode frames")
        counters = {
            tuple(
                row["safety"].get(k)
                for k in (
                    "host_session_id",
                    "control_epoch",
                    "joint_hold_events",
                    "watchdog_events",
                )
            )
            for row in rows
        }
        if len(counters) != 1 or any(
            row["safety"].get("fault") or row["safety"].get("joint_holds") for row in rows
        ):
            raise ValueError("Protection/session changes require a separate reviewed demo")
        t = np.array([row["client_timing"]["command_sent_monotonic_s"] for row in rows])
        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("Physical command timestamps must be finite and increasing")
        t -= t[0]
        names = list(episode.names)
        original = episode.actions
        state = np.array(
            pq.read_table(path / "frames.parquet", columns=["observation.state"])[
                "observation.state"
            ].to_pylist()
        )
        # This demonstration is straight-line. Do not claim independent component
        # integrals close an arbitrary turning SE(2) trajectory.
        if np.any(original[:, [names.index("y.vel"), names.index("theta.vel")]] != 0):
            raise ValueError("This demo preparer only supports the recorded straight-line path")
        fps = 30
        grid = np.arange(math.ceil((t[-1] + 1 / fps) * fps)) / fps
        actions = np.column_stack([np.interp(grid, t, original[:, j]) for j in range(len(names))])
        indices = np.minimum(np.searchsorted(t, grid, side="right") - 1, len(t) - 1)
        base = [names.index(n) for n in ("x.vel", "y.vel", "theta.vel")]
        actions[:, base] = original[indices][:, base]
        vx = actions[:, names.index("x.vel")]
        forward = vx[vx > 0].sum() / fps
        backward = -vx[vx < 0].sum() / fps
        if min(forward, backward) <= 0:
            raise ValueError("Both forward and backward motion are required")
        distance = min(forward, backward)
        vx[vx > 0] *= distance / forward
        vx[vx < 0] *= distance / backward
        lift = names.index("lift_axis.height_mm")
        limits = episode.metadata["lift_axis"]
        actions[:, lift] = smooth_lift(
            np.interp(grid, t, state[:, lift]),
            fps,
            limits["soft_min_mm"],
            limits["soft_max_mm"],
        )
        if np.any(actions[[0, -1]][:, base] != 0):
            raise ValueError("Demo must start and end with a stopped base")
        return_pose = transition(actions[-1], actions[0], names, fps)
        actions = np.vstack([actions, return_pose, np.tile(actions[0], (fps, 1))])
        source_hashes = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "recording.lock"
        }
        bundle = {
            "format": "alohamini-demo-trajectory",
            "version": 1,
            "purpose": "Edited demonstration targets; not a training dataset or measured path",
            "fps": fps,
            "names": names,
            "robot_metadata": episode.metadata,
            "actions": actions.tolist(),
            "report": {
                "source_dataset": str(root),
                "source_sha256": source_hashes,
                "source_check": report,
                "source_frames": len(t),
                "source_span_s": float(t[-1]),
                "nominal_source_duration_s": len(t) / episode.fps,
                "forward_m": float(distance),
                "backward_m": float(distance),
                "base_velocity_integral": (actions[:, base].sum(axis=0) / fps).tolist(),
                "duration_s": len(actions) / fps,
                "lift_target_max_rate_mm_s": float(np.abs(np.diff(actions[:, lift])).max() * fps),
                "lift_target_max_step_mm": float(np.abs(np.diff(actions[:, lift])).max()),
                "source_lift_target_max_step_mm": float(np.abs(np.diff(original[:, lift])).max()),
                "lift_method": (
                    "measured height, 0.5s symmetric average, <=40mm/s; soft limits retained"
                ),
                "closure": "command integral only; wheel slip and execution jitter may cause drift",
            },
        }
        validate_bundle(bundle)
        output.mkdir(parents=True)
        (output / "trajectory.json").write_text(json.dumps(bundle, ensure_ascii=False) + "\n")
        shutil.copy2(__file__, output / "replay_demo.py")
        # Retain actual feedback and original labels separately, never relabel them
        # as measurements corresponding to the edited demonstration targets.
        shutil.copytree(root, output / "source_dataset")
    print(json.dumps(bundle["report"], ensure_ascii=False, indent=2))
    print(f"Demo saved: {output}")


def validate_bundle(bundle):
    if bundle.get("format") != "alohamini-demo-trajectory" or bundle.get("version") != 1:
        raise ValueError("Unsupported demo trajectory")
    names, fps = bundle["names"], bundle["fps"]
    expected = [
        f"{m.name}.pos"
        for m in get_robot_model(bundle["robot_metadata"]["robot_model"]).actuators
        if m.name.startswith("arm_")
    ]
    expected += ["x.vel", "y.vel", "theta.vel", "lift_axis.height_mm"]
    if len(names) != len(expected) or set(names) != set(expected) or fps != 30:
        raise ValueError("Unexpected demonstration action layout or FPS")
    actions = np.asarray(bundle["actions"], dtype=float)
    if actions.ndim != 2 or actions.shape[1] != len(names) or len(actions) < 2:
        raise ValueError("Invalid trajectory shape")
    if not np.isfinite(actions).all():
        raise ValueError("Nonfinite trajectory")
    base = actions[:, [names.index(k) for k in ("x.vel", "y.vel", "theta.vel")]]
    if np.any(base[[0, -1]] != 0) or not np.allclose(base.sum(axis=0) / fps, 0, atol=1e-8):
        raise ValueError("Base target integral does not close")
    if np.any(base[:, 1:] != 0) or np.max(np.abs(base[:, 0])) > 0.150001:
        raise ValueError("Demo base must stay straight and within 0.15 m/s")
    if not np.allclose(actions[0], actions[-1], atol=1e-8):
        raise ValueError("Loop seam is not continuous")
    lift = actions[:, names.index("lift_axis.height_mm")]
    if np.max(np.abs(np.diff(lift))) * fps > 40.00001:
        raise ValueError("Lift target slew exceeds 40 mm/s")
    return actions


def play_timeline(client, actions, names, fps, guard):
    """Absolute timeline, bounded lateness, current feedback and ACK progress."""
    identity = None
    skipped = 0
    try:
        snapshot = client.read()
        guard.check(snapshot)
        start = time.monotonic()
        index, last_ack = 0, -1
        last_progress = start
        while index < len(actions):
            due = start + index / fps
            time.sleep(max(0.0, due - time.monotonic()))
            snapshot = client.read()
            guard.check(snapshot)
            now = time.monotonic()
            if now - due > 0.10:
                raise RuntimeError(f"回放延迟 {now - due:.3f}s，停止展示，不延长底盘速度目标。")
            ack = snapshot.payload["_safety"].get("command", {})
            if identity is not None:
                if (
                    ack.get("client_id") == identity.client_id
                    and ack.get("host_session_id") == identity.host_session_id
                    and ack.get("control_epoch") == identity.control_epoch
                    and ack.get("sequence", -1) > last_ack
                ):
                    last_ack, last_progress = ack["sequence"], now
                if now - last_progress > 0.25:
                    raise RuntimeError("Host 命令确认超过 250 ms 未推进，停止展示。")
            current = max(index, int((now - start) * fps))
            if current >= len(actions):
                break
            skipped += current - index
            target = dict(zip(names, map(float, actions[current]), strict=True))
            submitted = client.send_command(target, based_on=snapshot)
            if submitted is not None:
                identity = submitted
            index = current + 1
        time.sleep(max(0.0, start + len(actions) / fps - time.monotonic()))
        return {"elapsed_s": round(time.monotonic() - start, 3), "skipped_rows": skipped}
    finally:
        stop_owned_robot(client, guard.robot_model, identity)


def run(bundle_path, loops, execute):
    root = Path(bundle_path).expanduser().resolve()
    bundle = json.loads((root / "trajectory.json").read_text())
    actions = validate_bundle(bundle)
    if loops < 0:
        raise ValueError("loops must be >= 0 (0 means repeat until Ctrl+C)")
    report = bundle["report"]
    print(
        f"DEMO {root.name}: {len(actions)} targets, {bundle['fps']}Hz, "
        f"{len(actions) / bundle['fps']:.2f}s/cycle; "
        f"forward/backward={report['forward_m']:.3f}/{report['backward_m']:.3f}m; "
        f"lift <= {report['lift_target_max_rate_mm_s']:.1f}mm/s",
        flush=True,
    )
    if not execute:
        print("只检查文件，未连接 Host。加 --execute 才会运动。")
        return
    print("底盘计划闭合不保证实际回到原点。清空运动区域并全程看护；Ctrl+C 停止。")
    input("确认物品摆放与循环动作兼容、机器人有支撑，按 Enter 开始：")
    model, names, fps = bundle["robot_metadata"]["robot_model"], bundle["names"], bundle["fps"]
    # Local loopback only. No remote host option, downloads, cameras or serial ownership.
    with HostClient(
        "127.0.0.1",
        expected_model=model,
        timeout_s=0.08,
        request_window=1,
        prefetch_before_decode=True,
    ) as client:
        snapshot = client.connect_control()
        guard = ReplayGuard(client, model, operation="展示")
        guard.check(snapshot)
        check_calibration(bundle["robot_metadata"], snapshot)
        check_target_ranges(actions, names, snapshot)
        initial = np.array([snapshot.payload[n] for n in names])
        lift = names.index("lift_axis.height_mm")
        limits = bundle["robot_metadata"]["lift_axis"]
        initial[lift] = np.clip(initial[lift], limits["soft_min_mm"], limits["soft_max_mm"])
        approach = transition(initial, actions[0], names, fps, minimum_s=3)
        check_target_ranges(approach, names, snapshot)
        print("平滑进入起始姿态……", flush=True)
        play_timeline(client, approach, names, fps, guard)
        cycle = 0
        while loops == 0 or cycle < loops:
            cycle += 1
            print(f"DEMO cycle={cycle} planned_s={len(actions) / fps:.2f}", flush=True)
            result = play_timeline(client, actions, names, fps, guard)
            print(f"DEMO cycle={cycle} {result}", flush=True)


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


def main():
    parser = argparse.ArgumentParser(description="树莓派本机离线循环展示")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="PC 离线准备，不连接机器人")
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--output", required=True)
    runner = commands.add_parser("run", help="本机 Host 回放；默认只检查文件")
    runner.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    runner.add_argument("--loops", type=int, default=3, help="重复次数；0 表示持续循环")
    runner.add_argument("--execute", action="store_true", help="显式允许真机运动")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.dataset, args.output)
        else:
            run(args.bundle, args.loops, args.execute)
    except KeyboardInterrupt:
        print("展示已中断，已尝试停止；断联时 Host 看门狗负责停止。")
    except Exception as exc:
        parser.exit(1, f"demo: {exc}\n")


if __name__ == "__main__":
    main()
