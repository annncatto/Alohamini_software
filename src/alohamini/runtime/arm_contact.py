# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AlohaMini joint/gripper current limiters; SI interface and explicit calibration.
"""Advisory joint stall detection and gripper holds; no hardware access."""

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

from alohamini._validation import finite_number, identifier
from alohamini.calibration import EncoderCalibration
from alohamini.model import ActuatorSpec
from alohamini.runtime.current_protection import collision_current_limit_a


@dataclass(frozen=True)
class JointContactCalibration:
    """Legacy joint contact calibration retained for installed configurations.

    Joint stalls are advisory; release_margin_rad no longer gates joint targets.
    """

    release_margin_rad: float

    def __post_init__(self) -> None:
        finite_number(self.release_margin_rad, "release_margin_rad")
        if self.release_margin_rad <= 0:
            raise ValueError("Release margin must be positive")


@dataclass(frozen=True)
class GripperContactCalibration:
    """Physical endpoints corresponding to legacy 0% closed and 100% open."""

    closed_position_rad: float
    open_position_rad: float

    def __post_init__(self) -> None:
        finite_number(self.closed_position_rad, "closed_position_rad")
        finite_number(self.open_position_rad, "open_position_rad")
        finite_number(self.open_position_rad - self.closed_position_rad, "gripper stroke")
        if self.closed_position_rad == self.open_position_rad:
            raise ValueError("Gripper endpoints must differ")


@dataclass(frozen=True)
class ArmJointSpec:
    actuator: ActuatorSpec
    calibration: EncoderCalibration
    contact: JointContactCalibration | GripperContactCalibration

    def __post_init__(self) -> None:
        if not isinstance(self.actuator, ActuatorSpec):
            raise TypeError("Expected ActuatorSpec")
        identifier(self.actuator.name, "joint name")
        collision_current_limit_a(self.actuator.motor_model)
        if not isinstance(self.calibration, EncoderCalibration):
            raise TypeError("Expected EncoderCalibration")
        if self.calibration.position_min_rad is None:
            raise ValueError("Arm commands require explicit device joint limits")
        if isinstance(self.contact, GripperContactCalibration):
            for position in (self.contact.closed_position_rad, self.contact.open_position_rad):
                self.calibration.position_to_tick(position)
        elif not isinstance(self.contact, JointContactCalibration):
            raise TypeError("Expected a joint or gripper contact calibration")


@dataclass(frozen=True)
class _StallCandidate:
    started_s: float
    position_rad: float
    direction: float


class ArmContactGuard:
    """Report joint stalls without changing targets; hold grippers on contact.

    Gripper holds survive current falling and owner changes until a retreat.
    Missing feedback raises. Sustained overcurrent is independently enforced
    by HostSupervisor before this limiter runs.
    """

    def __init__(self, joints: Mapping[str, ArmJointSpec]) -> None:
        if not joints:
            raise ValueError("Provide at least one arm joint")
        for name, spec in joints.items():
            if not isinstance(spec, ArmJointSpec) or spec.actuator.name != name:
                raise ValueError("Joint keys must match their actuator specification")
        self._joints = dict(joints)
        self._candidates: dict[str, _StallCandidate] = {}
        self._holds: dict[str, float] = {}
        self._release_directions: dict[str, float] = {}
        self._last_time: float | None = None
        self._hold_events = 0
        self._joint_stall_currents_a: dict[str, float] = {}
        self._last_warning_s: dict[str, float] = {}

    @property
    def holds(self) -> dict[str, float]:
        return dict(self._holds)

    @property
    def hold_events(self) -> int:
        return self._hold_events

    @property
    def joint_hold_events(self) -> int:
        """Compatibility counter: advisory stalls are not joint hold events."""
        return 0

    @property
    def joint_stall_currents_a(self) -> dict[str, float]:
        """Present advisory stalls and their measured current magnitudes, in A."""
        return dict(self._joint_stall_currents_a)

    def cancel_candidates(self) -> None:
        """Cancel pending timers on stop without silently releasing contact holds."""
        self._candidates.clear()
        self._joint_stall_currents_a.clear()

    def limit(
        self,
        goals_rad: Mapping[str, float],
        positions_rad: Mapping[str, float],
        currents_a: Mapping[str, float],
        *,
        now: float,
    ) -> dict[str, float]:
        try:
            finite_number(now, "contact sample time")
            if now < 0 or (self._last_time is not None and now <= self._last_time):
                raise ValueError("Contact sample times must increase")
            if not goals_rad.keys() <= self._joints.keys():
                raise ValueError("Unknown arm target")
            for name, goal in goals_rad.items():
                self._joints[name].calibration.position_to_tick(goal)
                finite_number(positions_rad[name], "joint position")
                finite_number(currents_a[name], "joint current")
        except (KeyError, ValueError, TypeError):
            self.cancel_candidates()
            raise
        self._last_time = now
        self._candidates = {k: v for k, v in self._candidates.items() if k in goals_rad}
        self._joint_stall_currents_a.clear()
        limited = dict(goals_rad)
        for name, goal in goals_rad.items():
            spec = self._joints[name]
            present, current = positions_rad[name], abs(currents_a[name])
            contact = spec.contact
            if isinstance(contact, GripperContactCalibration):
                span = contact.open_position_rad - contact.closed_position_rad
                open_direction = 1.0 if span > 0 else -1.0
                release_margin = 0.01 * abs(span)
                if name not in self._holds:
                    if current < 0.5:
                        continue
                    delta = goal - present
                    release = -math.copysign(1.0, delta) if delta else open_direction
                    hold = present
                    if delta * open_direction < 0:
                        hold = present - 0.03 * span
                        lower, upper = sorted(
                            (contact.closed_position_rad, contact.open_position_rad)
                        )
                        hold = min(upper, max(lower, hold))
                    self._holds[name] = hold
                    self._release_directions[name] = release
                    self._hold_events += 1
            else:
                # Preserve the source's encoder-space trigger, but never latch a joint target.
                calibration = spec.calibration
                ratio = calibration.joint_per_encoder_ratio * (
                    (calibration.ticks_per_revolution - 1) / calibration.ticks_per_revolution
                )
                error = (goal - present) / ratio
                if current < collision_current_limit_a(spec.actuator.motor_model) or abs(
                    error
                ) < math.radians(2):
                    self._candidates.pop(name, None)
                    continue
                direction = math.copysign(1.0, error)
                candidate = self._candidates.get(name)
                if candidate is None or candidate.direction != direction:
                    self._candidates[name] = _StallCandidate(now, present, direction)
                    continue
                progress = (present - candidate.position_rad) * direction / ratio
                if progress >= math.radians(0.2):
                    self._candidates[name] = _StallCandidate(now, present, direction)
                    continue
                if now - candidate.started_s >= 0.150:
                    self._joint_stall_currents_a[name] = current
                    if now - self._last_warning_s.get(name, -math.inf) >= 5.0:
                        logging.getLogger(__name__).warning(
                            "Joint stall suspected: %s, current=%.3f A; "
                            "target unchanged, overload protection remains active",
                            name,
                            current,
                        )
                        self._last_warning_s[name] = now
                continue
            hold = self._holds[name]
            if (goal - hold) * self._release_directions[name] >= release_margin:
                self._holds.pop(name)
                self._release_directions.pop(name)
            else:
                limited[name] = hold
        return limited
