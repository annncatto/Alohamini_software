# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from teleoperate_bi.py, AlohaMiniClient keyboard mapping and teleop_monitor.py.
"""Leader/keyboard teleoperation in Host command units."""

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
from alohamini.errors import ResponseTimeoutError
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
            elif name == "enter" or (
                isinstance(name, str) and len(name) == 1 and name in "wszxadtguj"
            ):
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
    if identity is None or identity.client_id != client.client_id:
        return
    try:
        snapshot = client.read()
        safety = snapshot.payload["_safety"]
        if (
            safety.get("control_owner") not in (None, client.client_id)
            or safety.get("host_session_id") != identity.host_session_id
            or safety.get("control_epoch") != identity.control_epoch
            or ready_units(snapshot, robot_model, client.client_id) is None
        ):
            return
        # A prefetched reply can precede our first command's ownership claim.
        # A newer sequence on the same lease also supersedes that in-flight command.
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


def _own_watchdog_release(previous, snapshot, client_id):
    """An available lease after one watchdog stop, not a restart or active takeover.

    A prefetched state may precede the first command's ownership claim. The
    recovery input is sampled anew, so observing that claim is not a prerequisite.
    """
    if previous is None:
        return False
    before, after = previous.payload["_safety"], snapshot.payload["_safety"]
    old_events, new_events = before.get("watchdog_events"), after.get("watchdog_events")
    return (
        before.get("control_owner") in (None, client_id)
        and before.get("host_session_id") == after.get("host_session_id")
        and type(old_events) is int
        and type(new_events) is int
        and new_events == old_events + 1
        and after.get("control_epoch") == before.get("control_epoch", -2) + 1
        and after.get("control_owner") is None
        and after.get("phase") == "ready"
        and after.get("watchdog_active") is True
        and not after.get("joint_holds")
        and after.get("joint_hold_events") == before.get("joint_hold_events")
        and snapshot.payload["_robot_metadata"] == previous.payload["_robot_metadata"]
        and snapshot.payload.get("lift_axis.reference_sequence")
        == previous.payload.get("lift_axis.reference_sequence")
    )


def run_loop(
    client,
    robot_model,
    leader,
    keyboard,
    *,
    fps=50,
    camera_fps=30,
    on_frame=None,
    stop_event=None,
    tracking=None,
):
    """Observe -> input -> send -> monitor -> submit the latest preview state.

    A missing client is explicit no_robot mode, never a connection-error fallback.
    Host lease/feedback checks and measured-stop cleanup remain authoritative.
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
    identity = context = previous = None
    debug_report_t = -math.inf
    timeout_report_t = -math.inf
    waiting_since = None
    timeout_warned = False
    default_units = {
        f"{m.name}.pos": "range_0_100" if m.name.endswith("gripper") else "range_m100_100"
        for m in get_robot_model(robot_model).actuators
        if m.name.startswith("arm_")
    }
    try:
        while not stop.is_set():
            t0 = time.perf_counter()
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
            try:
                snapshot = client.read()
            except ResponseTimeoutError as exc:
                # Source teleoperation waits for valid feedback after a missed
                # response. Never send from cached state or queue old leader input.
                mapper.reset()
                now = time.perf_counter()
                if waiting_since is None:
                    waiting_since = now
                if now - waiting_since >= 0.5 and now - timeout_report_t >= 1.0:
                    logger.warning("等待 Host 新反馈，暂不发送动作：%s", exc)
                    timeout_report_t = now
                    timeout_warned = True
                if keyboard is not None and keyboard.read() is None:
                    break
                stop.wait(max(1.0 / fps - (now - t0), 0.0))
                continue
            if timeout_warned:
                logger.info("Host 反馈恢复，继续遥操。")
            waiting_since = None
            timeout_warned = False
            if tracking is not None:
                tracking.submit(snapshot.payload)
            units = ready_units(snapshot, robot_model, client.client_id)
            keys = set() if keyboard is None else keyboard.read()
            if keys is None:
                break
            safety = snapshot.payload["_safety"]
            current_context = (safety.get("host_session_id"), safety.get("control_epoch"))
            if context is not None and current_context != context:
                # The old lease is no longer ours: cleanup must not claim a new one.
                identity = None
                if safety.get("host_session_id") != context[0]:
                    raise RuntimeError("Host 已重启或会话已更换，请重新启动遥操。")
                if units is None or not _own_watchdog_release(previous, snapshot, client.client_id):
                    raise RuntimeError(
                        f"Host 控制状态已改变：{context} → {current_context}；"
                        f"owner={safety.get('control_owner')}，请检查 Host 日志。"
                    )
                mapper.reset()
                # Source teleoperation resumes on valid feedback. Sample the
                # current leader/keys below, never replay the pre-timeout action.
                logger.info("Host 响应恢复，重新采样主臂和按键，继续遥操。")
                context = current_context
            previous = snapshot
            # Source AlohaMiniClient replenishes before local input processing.
            # Refill the bounded window if transport backpressure delayed a send.
            # The real client already replenishes before decoding the response.
            client.prefetch()
            sent = False
            if units is None:
                mapper.reset()
            else:
                action = {} if leader is None else leader.read(units)
                keys = set() if keyboard is None else keyboard.read()
                if keys is None:
                    break
                # Source teleoperation's feedback_fresh gate is application-local:
                # a long leader retry discards this input, then observes/resamples.
                # Policy inference intentionally has no such age gate in HostClient.
                timeout = min(0.25, safety.get("command_watchdog_timeout_s", 0.25))
                if time.monotonic() - snapshot.request_started_s >= timeout:
                    mapper.reset()
                    monitor.update(snapshot.payload, sent=False, fresh=False, permitted=False)
                    stop.wait(max(1.0 / fps - (time.perf_counter() - t0), 0.0))
                    continue
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
    tracking=False,
):
    if no_leader and no_keyboard:
        raise ValueError("不能同时关闭主臂与键盘。")
    if tracking and no_robot:
        raise ValueError("--tracking 需要真实 Host 反馈，不能与 --no_robot 一起使用。")
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
                HostClient(
                    host,
                    expected_model=robot_model,
                    timeout_s=0.2,
                    request_window=3,
                    prefetch_before_decode=True,
                )
            )
            client.connect_control()  # Verify both channels before touching leaders or preview.
        on_frame = None
        if not no_preview:
            from alohamini.apps.visualization import init_rerun, shutdown_rerun

            cleanup.callback(shutdown_rerun)
            init_rerun(session_name="alohamini_teleop")
        if leader is not None:
            cleanup.enter_context(leader)
        if not no_preview:
            from alohamini.apps.visualization import TeleopPreview

            preview = cleanup.enter_context(
                TeleopPreview(None if no_robot else host, robot_model, fps=camera_fps)
            )
            on_frame = preview.submit
        if no_leader:
            print(
                "🧪 NO_LEADER mode enabled: leader arms will not connect; "
                "keyboard controls base and lift."
            )
        tracking_log = None
        if tracking:
            from alohamini.apps.tracking import TrackingLog

            tracking_log = TrackingLog()
            cleanup.callback(tracking_log.close)
        run_loop(
            client,
            robot_model,
            leader,
            keyboard,
            fps=fps,
            camera_fps=camera_fps,
            on_frame=on_frame,
            tracking=tracking_log,
        )
