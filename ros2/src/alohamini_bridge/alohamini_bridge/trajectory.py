# SPDX-License-Identifier: Apache-2.0
# Migrated from alohamini_lerobot_bridge/control.py.
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

LEFT_ARM_JOINTS = (
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_yaw_joint",
    "left_wrist_roll",
)
RIGHT_ARM_JOINTS = tuple(name.replace("left_", "right_", 1) for name in LEFT_ARM_JOINTS)
LEFT_GRIPPER_JOINTS = ("left_gripper",)
RIGHT_GRIPPER_JOINTS = ("right_gripper",)
LIFT_JOINTS = ("vertical_move",)


class TerminalState(Enum):
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"
    PREEMPTED = "preempted"
    ABORTED = "aborted"
    GOAL_TOLERANCE = "goal_tolerance"
    STALE = "stale"


@dataclass(frozen=True)
class TrajectorySample:
    time_from_start: float
    positions: dict[str, float]


@dataclass(frozen=True)
class TerminalEvent:
    goal_id: int
    state: TerminalState
    message: str


class TrajectoryResource:
    """One independently activated joint resource with no implicit zero targets."""

    def __init__(
        self,
        name: str,
        joints: Iterable[str],
        tracking_error: float,
        goal_tolerance: float,
        goal_time_tolerance: float,
        hold_duration: float,
    ) -> None:
        self.name = name
        self.joints = tuple(joints)
        self.tracking_error = float(tracking_error)
        self.goal_tolerance = float(goal_tolerance)
        self.goal_time_tolerance = float(goal_time_tolerance)
        self.hold_duration = float(hold_duration)
        if not math.isfinite(self.tracking_error) or self.tracking_error <= 0.0:
            raise ValueError(f"{name} tracking_error must be finite and positive")
        if not math.isfinite(self.goal_tolerance) or self.goal_tolerance <= 0.0:
            raise ValueError(f"{name} goal_tolerance must be finite and positive")
        if not math.isfinite(self.goal_time_tolerance) or self.goal_time_tolerance < 0.0:
            raise ValueError(f"{name} goal_time_tolerance must be finite and non-negative")
        if self.hold_duration != math.inf and (
            not math.isfinite(self.hold_duration) or self.hold_duration < 0.0
        ):
            raise ValueError(f"{name} hold_duration must be finite and non-negative")
        self.goal_id = 0
        self.active_goal_id: int | None = None
        self.start_time = 0.0
        self.start_positions: dict[str, float] = {}
        self.samples: tuple[TrajectorySample, ...] = ()
        self.hold_positions: dict[str, float] | None = None
        self.hold_until = 0.0
        self.stream_positions: dict[str, float] | None = None
        self.stream_until = 0.0
        self.desired: dict[str, float] | None = None
        self.terminals: dict[int, TerminalEvent] = {}
        self.path_limits = dict.fromkeys(self.joints, self.tracking_error)
        self.goal_limits = dict.fromkeys(self.joints, self.goal_tolerance)

    @property
    def active(self) -> bool:
        return (
            self.active_goal_id is not None
            or self.hold_positions is not None
            or self.stream_positions is not None
        )

    def accept_stream_target(
        self,
        positions: dict[str, float],
        measured: dict[str, float],
        now: float,
        timeout: float,
    ) -> None:
        """Replace the latest realtime setpoint without creating an Action goal."""
        if set(positions) != set(self.joints):
            raise ValueError(f"{self.name} stream must contain exactly {self.joints}")
        if any(not math.isfinite(value) for value in positions.values()):
            raise ValueError(f"{self.name} stream positions must be finite")
        if any(joint not in measured for joint in self.joints):
            raise ValueError(f"fresh measured state lacks {self.joints}")
        if self.active_goal_id is not None:
            self._finish(
                TerminalState.PREEMPTED,
                "preempted by realtime JointJog",
                measured,
                now,
                hold=False,
            )
        self.hold_positions = None
        self.stream_positions = dict(positions)
        self.stream_until = now + timeout

    def validate(self, joint_names: Iterable[str], samples: Iterable[TrajectorySample]) -> None:
        names = tuple(joint_names)
        points = tuple(samples)
        if not names or len(names) != len(set(names)):
            raise ValueError("trajectory joint_names must be non-empty and unique")
        unsupported = set(names) - set(self.joints)
        if unsupported:
            raise ValueError(f"{self.name} does not own joints {sorted(unsupported)}")
        if not points:
            raise ValueError("trajectory must contain at least one point")
        previous_time = -1.0
        for point in points:
            if not math.isfinite(point.time_from_start) or point.time_from_start < 0.0:
                raise ValueError("trajectory time_from_start must be finite and non-negative")
            if point.time_from_start <= previous_time:
                raise ValueError("trajectory times must be strictly increasing")
            if set(point.positions) != set(names):
                raise ValueError("every trajectory point must contain exactly joint_names")
            if any(not math.isfinite(value) for value in point.positions.values()):
                raise ValueError("trajectory positions must be finite")
            previous_time = point.time_from_start

    def activate(
        self,
        joint_names: Iterable[str],
        samples: Iterable[TrajectorySample],
        measured: dict[str, float],
        now: float,
    ) -> int:
        samples = tuple(samples)
        self.validate(joint_names, samples)
        missing = set(self.joints) - set(measured)
        if missing:
            raise ValueError(f"fresh measured state lacks {sorted(missing)}")
        if self.active_goal_id is not None:
            self._finish(
                TerminalState.PREEMPTED,
                "preempted by a newer trajectory",
                measured,
                now,
            )
        self.goal_id += 1
        self.active_goal_id = self.goal_id
        self.start_time = now
        self.start_positions = {joint: float(measured[joint]) for joint in self.joints}
        self.samples = samples
        self.hold_positions = None
        self.stream_positions = None
        self.desired = dict(self.start_positions)
        return self.goal_id

    def _finish(
        self,
        state: TerminalState,
        message: str,
        measured: dict[str, float] | None,
        now: float,
        hold: bool = True,
    ) -> None:
        if self.active_goal_id is not None:
            self.terminals[self.active_goal_id] = TerminalEvent(self.active_goal_id, state, message)
        self.active_goal_id = None
        self.samples = ()
        self.desired = None
        if hold and measured is not None and all(joint in measured for joint in self.joints):
            self.hold_positions = {joint: float(measured[joint]) for joint in self.joints}
            self.hold_until = now + self.hold_duration
        else:
            self.hold_positions = None
        self.stream_positions = None

    def cancel(self, goal_id: int, measured: dict[str, float], fresh: bool, now: float) -> bool:
        if goal_id != self.active_goal_id:
            return False
        self._finish(
            TerminalState.CANCELED,
            "trajectory canceled",
            measured if fresh else None,
            now,
            hold=fresh,
        )
        return True

    def invalidate_stale(self, now: float) -> None:
        self._finish(
            TerminalState.STALE,
            "Host observation became stale",
            None,
            now,
            hold=False,
        )

    def terminal(self, goal_id: int) -> TerminalEvent | None:
        # Each action execute callback consumes its terminal event exactly once. This
        # bounds memory when teleoperation continuously preempts short trajectories.
        return self.terminals.pop(goal_id, None)

    def _interpolate(self, elapsed: float) -> dict[str, float]:
        previous_time = 0.0
        previous = self.start_positions
        for point in self.samples:
            if elapsed <= point.time_from_start:
                duration = point.time_from_start - previous_time
                ratio = 1.0 if duration <= 0.0 else (elapsed - previous_time) / duration
                ratio = min(1.0, max(0.0, ratio))
                target = dict(previous)
                for joint, value in point.positions.items():
                    target[joint] = previous[joint] + ratio * (value - previous[joint])
                return target
            previous_time = point.time_from_start
            previous = {**previous, **point.positions}
        return dict(previous)

    def update(
        self, measured: dict[str, float], fresh: bool, now: float, *, allow_success=True
    ) -> dict[str, float] | None:
        if not fresh:
            if self.active:
                self.invalidate_stale(now)
            return None
        if self.stream_positions is not None:
            if now > self.stream_until:
                self._finish(TerminalState.CANCELED, "JointJog timed out", measured, now)
                return dict(self.hold_positions)
            error = max(
                abs(self.stream_positions[joint] - measured[joint]) for joint in self.joints
            )
            if error > self.tracking_error:
                self._finish(TerminalState.ABORTED, "JointJog tracking error", measured, now)
                return dict(self.hold_positions)
            return dict(self.stream_positions)
        if self.active_goal_id is not None:
            desired = self._interpolate(now - self.start_time)
            errors = {joint: abs(desired[joint] - measured[joint]) for joint in self.joints}
            error = max(errors.values())
            violation = next(
                (joint for joint in self.joints if errors[joint] > self.path_limits[joint]), None
            )
            if violation is not None:
                self._finish(
                    TerminalState.ABORTED,
                    f"{violation} tracking error {errors[violation]:.6f} exceeds "
                    f"{self.path_limits[violation]:.6f}",
                    measured,
                    now,
                )
                return dict(self.hold_positions) if self.hold_positions else None
            self.desired = desired
            elapsed = now - self.start_time
            end_time = self.samples[-1].time_from_start
            if elapsed >= end_time:
                violation = next(
                    (joint for joint in self.joints if errors[joint] > self.goal_limits[joint]),
                    None,
                )
                if allow_success and violation is None:
                    goal_id = self.active_goal_id
                    self.terminals[goal_id] = TerminalEvent(
                        goal_id,
                        TerminalState.SUCCEEDED,
                        f"goal reached with error {error:.6f}",
                    )
                    self.active_goal_id = None
                    self.samples = ()
                    self.hold_positions = desired
                    self.hold_until = now + self.hold_duration
                elif elapsed > end_time + self.goal_time_tolerance:
                    message = "Goal was not submitted or remains blocked at the goal deadline"
                    state = TerminalState.ABORTED
                    if violation is not None:
                        message = (
                            f"{violation} goal error {errors[violation]:.6f} exceeds "
                            f"{self.goal_limits[violation]:.6f} after "
                            f"{self.goal_time_tolerance:.3f}s grace period"
                        )
                        state = TerminalState.GOAL_TOLERANCE
                    self._finish(
                        state,
                        message,
                        measured,
                        now,
                    )
                    return dict(self.hold_positions) if self.hold_positions else None
            return desired
        if self.hold_positions is not None:
            if now <= self.hold_until:
                return dict(self.hold_positions)
            self.hold_positions = None
        return None
