# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from teleoperate_bi.py, AlohaMiniClient keyboard mapping and teleop_monitor.py.
"""Native leader/keyboard teleoperation in the deployed Host command units."""

import logging
import math
import os
import sys
import threading
import time
from contextlib import ExitStack

from alohamini._validation import finite_number
from alohamini.apps.teleop_monitor import TeleopMonitor
from alohamini.client import HostClient
from alohamini.hardware.leader import BimanualLeader
from alohamini.model import get_robot_model
from alohamini.protocol import HostSnapshot, decode_command_context

logger = logging.getLogger(__name__)


class KeyboardInput:
    """Held-key input; terminal key-down bytes cannot substitute for releases."""

    def __init__(self):
        self._listener = None
        self._pressed = set()
        self._quit = False
        self._lock = threading.Lock()

    def __enter__(self):
        if sys.platform.startswith("linux") and (
            not os.environ.get("DISPLAY")
            or os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        ):
            raise RuntimeError("键盘遥操需要 X11 桌面；仅用主臂时加 --no_keyboard。")
        from pynput import keyboard

        def update(key, pressed):
            name = getattr(key, "char", None) or getattr(key, "name", None)
            self.update(name, pressed)

        self._listener = keyboard.Listener(
            on_press=lambda key: update(key, True), on_release=lambda key: update(key, False)
        )
        self._listener.start()
        if getattr(self._listener, "IS_TRUSTED", True) is False:
            self.close()
            raise RuntimeError("键盘监听缺少系统输入权限。")
        return self

    def __exit__(self, *_exc):
        self.close()

    def update(self, name, pressed):
        with self._lock:
            if name in ("esc", "q") and pressed:
                self._quit = True
                self._pressed.clear()
            elif isinstance(name, str) and len(name) == 1 and name in "wszxadtguj":
                if pressed:
                    self._pressed.add(name)
                else:
                    self._pressed.discard(name)

    def read(self) -> set[str] | None:
        with self._lock:
            if self._quit:
                return None
            if self._listener is None or not self._listener.is_alive():
                self._pressed.clear()
                raise RuntimeError("键盘监听已停止，遥操退出。")
            return set(self._pressed)

    def close(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener.join(timeout=0.5)
            self._listener = None
        with self._lock:
            self._pressed.clear()


class KeyboardTargets:
    """Original key map, speed levels and feedback-latched absolute lift target."""

    def __init__(self):
        self.speed_index = 0
        self._height = self._last_time = None
        self._direction = 0

    def reset(self):
        self._height = self._last_time = None
        self._direction = 0

    def base_targets(self, keys) -> dict[str, float]:
        if "t" in keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if "g" in keys:
            self.speed_index = max(self.speed_index - 1, 0)
        xy, theta = ((0.15, 45), (0.2, 60), (0.25, 75))[self.speed_index]
        return {
            "x.vel": (int("w" in keys) - int("s" in keys)) * xy,
            "y.vel": (int("z" in keys) - int("x" in keys)) * xy,
            "theta.vel": (int("a" in keys) - int("d" in keys)) * theta,
        }

    def targets(self, keys, payload, *, now: float) -> dict[str, float]:
        finite_number(now, "keyboard time")
        h_now = payload["lift_axis.height_mm"]
        lift = payload["_robot_metadata"]["lift_axis"]
        lower, upper = lift["soft_min_mm"], lift["soft_max_mm"]
        for value in (h_now, lower, upper):
            finite_number(value, "lift height/limit")
        if lower >= upper:
            raise ValueError("Invalid Host lift limits")
        direction = int("u" in keys) - int("j" in keys)
        if direction == 0:
            self._height = h_now
        else:
            relatch = self._height is None or direction != self._direction
            if relatch:
                self._height = h_now
            dt = (
                1 / 50
                if self._last_time is None or relatch
                else min(max(now - self._last_time, 0), 0.1)
            )
            self._height += direction * 150.0 * dt
            self._height = min(max(self._height, h_now - 50.0), h_now + 50.0)
        self._height = min(max(self._height, lower), upper)
        self._last_time, self._direction = now, direction
        return {
            **self.base_targets(keys),
            "lift_axis.height_mm": self._height,
        }


def ready_units(snapshot: HostSnapshot, robot_model: str, client_id: str) -> dict[str, str] | None:
    """Require valid full-robot state; joint contact holds still permit retreat."""
    payload = snapshot.payload
    if snapshot.robot_model != robot_model:
        raise ValueError("Host robot_model does not match teleoperation")
    safety = payload["_safety"]
    if safety.get("fault") or safety.get("phase") in ("fault", "closed"):
        raise RuntimeError(f"Host unavailable: {safety.get('fault') or safety.get('phase')}")
    if (
        decode_command_context(payload, client_id=client_id) is None
        or safety.get("feedback_valid") is not True
        or safety.get("lift_reference_valid") is not True
        or safety.get("phase") not in ("ready", "active")
    ):
        return None
    motors = payload["_robot_metadata"]["motors"]
    units = {}
    for motor in get_robot_model(robot_model).actuators:
        if motor.name.startswith("arm_"):
            key = f"{motor.name}.pos"
            finite_number(payload[key], key)
            unit = motors[motor.name]["normalization"]
            allowed = (
                ("range_0_100",)
                if motor.name.endswith("gripper")
                else ("range_m100_100", "degrees")
            )
            if unit not in allowed:
                raise ValueError(f"Incompatible Host position units: {key}")
            units[key] = unit
    for key in ("x.vel", "y.vel", "theta.vel", "lift_axis.height_mm"):
        finite_number(payload[key], key)
    return units


def stop_owned_robot(client, robot_model, identity):
    """Best-effort measured hold/zero, only for this unchanged control lease."""
    if identity is None:
        return
    try:
        snapshot = client.read()
        safety = snapshot.payload["_safety"]
        if (
            safety.get("control_owner") != client.client_id
            or safety.get("host_session_id") != identity.host_session_id
            or safety.get("control_epoch") != identity.control_epoch
            or ready_units(snapshot, robot_model, client.client_id) is None
        ):
            return
        units = ready_units(snapshot, robot_model, client.client_id)
        targets = {key: snapshot.payload[key] for key in units}
        targets.update(
            {
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0,
                "lift_axis.height_mm": snapshot.payload["lift_axis.height_mm"],
            }
        )
        stopped = client.send_command(targets, based_on=snapshot)
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            reply = client.read().payload["_safety"]
            command = reply.get("command", {})
            if (
                command.get("client_id") == client.client_id
                and command.get("sequence") == stopped.sequence
            ):
                return
        logger.warning("未确认停止目标；Host watchdog 将在命令超时后停止运动。")
    except Exception as exc:
        logger.warning("停止请求未完成：%s；Host watchdog 负责断联停止。", exc)


def run_loop(
    client, robot_model, leader, keyboard, *, fps=50, camera_fps=30, on_frame=None, stop_event=None
):
    """Source teleoperate_bi cadence: observe -> input -> send -> monitor -> preview.

    A missing client is explicit no_robot mode, never a connection-error fallback.
    Native lease/feedback checks and measured-stop cleanup remain authoritative.
    """
    finite_number(fps, "teleoperation fps")
    if not 0 < fps <= 50:
        raise ValueError("Teleoperation fps must be in (0, 50]")
    finite_number(camera_fps, "camera fps")
    if not 0 < camera_fps <= fps:
        raise ValueError("--camera-fps must be positive and must not exceed --fps")
    if leader is None and keyboard is None:
        raise ValueError("Teleoperation needs a leader or keyboard")
    stop = stop_event if stop_event is not None else threading.Event()
    mapper = KeyboardTargets()
    monitor = TeleopMonitor()
    identity = context = None
    # Retained from teleoperate_bi.py; request at most once per control cycle.
    next_camera_request_t = time.perf_counter()
    camera_interval_s = 1.0 / camera_fps
    debug_report_t = -math.inf
    default_units = {
        f"{m.name}.pos": "range_0_100" if m.name.endswith("gripper") else "range_m100_100"
        for m in get_robot_model(robot_model).actuators
        if m.name.startswith("arm_")
    }
    try:
        while not stop.is_set():
            t0 = time.perf_counter()
            request_cameras = t0 >= next_camera_request_t
            if request_cameras:
                while next_camera_request_t <= t0:
                    next_camera_request_t += camera_interval_s
            if client is None:
                keys = set() if keyboard is None else keyboard.read()
                if keys is None:
                    break
                action = {} if leader is None else leader.read(default_units)
                action.update(mapper.base_targets(keys))
                if t0 - debug_report_t >= 1.0:
                    print(f"[TELEOP ACTION] {action} keys={','.join(sorted(keys))}", flush=True)
                    debug_report_t = t0
                if on_frame is not None:
                    on_frame(None, action)
                # There is no physical height in no_robot mode: do not invent one.
                stop.wait(max(1.0 / fps - (time.perf_counter() - t0), 0.0))
                continue
            snapshot = (
                client.read(include_images=True)
                if request_cameras and on_frame is not None
                else client.read()
            )
            units = ready_units(snapshot, robot_model, client.client_id)
            keys = set() if keyboard is None else keyboard.read()
            if keys is None:
                break
            safety = snapshot.payload["_safety"]
            current_context = (safety.get("host_session_id"), safety.get("control_epoch"))
            if context is not None and current_context != context:
                raise RuntimeError("Host 会话或控制权已改变，请检查后重新启动遥操。")
            sent = False
            if units is None:
                mapper.reset()
                if identity is not None:
                    raise RuntimeError("Host 反馈或控制权失效，遥操停止。")
            else:
                action = {} if leader is None else leader.read(units)
                keys = set() if keyboard is None else keyboard.read()
                if keys is None:
                    break
                action.update(mapper.targets(keys, snapshot.payload, now=time.monotonic()))
                if stop.is_set():
                    break
                identity = client.send_command(action, based_on=snapshot)
                context = current_context
                sent = True
            monitor.update(
                snapshot.payload,
                sent=sent,
                fresh=(
                    safety.get("feedback_valid") is True
                    and safety.get("lift_reference_valid") is True
                ),
                permitted=units is not None,
            )
            if sent and on_frame is not None:
                on_frame(snapshot, action)
            # Same per-cycle sleep as source precise_sleep on Linux, cancellable.
            stop.wait(max(1.0 / fps - (time.perf_counter() - t0), 0.0))
    finally:
        if client is not None:
            stop_owned_robot(client, robot_model, identity)


def teleoperate(
    host,
    robot_model,
    *,
    leader_id=None,
    calibration_dir=None,
    left_port="/dev/am_arm_leader_left",
    right_port="/dev/am_arm_leader_right",
    no_leader=False,
    no_keyboard=False,
    no_robot=False,
    no_preview=False,
    fps=50,
    camera_fps=30,
    arm_profile=None,
):
    if no_leader and no_keyboard:
        raise ValueError("不能同时关闭主臂与键盘。")
    finite_number(fps, "teleoperation fps")
    if not 0 < fps <= 50:
        raise ValueError("Teleoperation fps must be in (0, 50]")
    finite_number(camera_fps, "camera fps")
    if not 0 < camera_fps <= fps:
        raise ValueError("--camera-fps must be positive and must not exceed --fps")
    get_robot_model(robot_model)
    expected_profile = "so-arm-5dof" if robot_model == "alohamini1" else "am-leader-6dof"
    if arm_profile is not None and arm_profile != expected_profile:
        raise ValueError(f"{robot_model} requires leader profile {expected_profile}")
    with ExitStack() as cleanup:
        # Validate both calibration files and keyboard availability before serial I/O.
        leader = (
            None
            if no_leader
            else BimanualLeader(
                robot_model,
                leader_id=leader_id,
                calibration_dir=calibration_dir,
                left_port=left_port,
                right_port=right_port,
            )
        )
        keyboard = None if no_keyboard else cleanup.enter_context(KeyboardInput())
        client = None
        if no_robot:
            print("🧪 NO_ROBOT mode enabled: robot will not connect, only print actions.")
            print("No lift height feedback: u/j keys are shown without generating a height target.")
        else:
            client = cleanup.enter_context(
                HostClient(host, expected_model=robot_model, timeout_s=0.2, request_window=1)
            )
            client.read()  # Fail a mismatched/unreachable Host before touching leaders.
        on_frame = None
        if not no_preview:
            from alohamini.apps.visualization import init_rerun, log_snapshot, shutdown_rerun

            cleanup.callback(shutdown_rerun)
            init_rerun(session_name="alohamini_teleop")
            on_frame = log_snapshot
        if leader is not None:
            cleanup.enter_context(leader)
        if no_leader:
            print(
                "🧪 NO_LEADER mode enabled: leader arms will not connect; "
                "keyboard controls base and lift."
            )
        run_loop(
            client, robot_model, leader, keyboard, fps=fps, camera_fps=camera_fps, on_frame=on_frame
        )
