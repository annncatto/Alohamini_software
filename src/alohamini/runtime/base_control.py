# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AlohaMini._body_to_wheel_raw; signed values remain unencoded here.
"""Omni-base target conversion and measured velocity, with deployed wheel limiting."""

import math
from dataclasses import dataclass

from alohamini.kinematics import OmniBaseKinematics
from alohamini.schema import BodyVelocity


@dataclass(frozen=True)
class BaseTarget:
    """Target after proportional wheel limiting and encoder quantization, not feedback."""

    wheel_ticks_s: tuple[int, int, int]
    body_velocity: BodyVelocity


class BaseDrive:
    """Wheel order left/back/right; cap all three proportionally at 3000 ticks/s."""

    def __init__(self, kinematics: OmniBaseKinematics) -> None:
        if not isinstance(kinematics, OmniBaseKinematics):
            raise TypeError("Expected OmniBaseKinematics")
        self._kinematics = kinematics

    def target(self, velocity: BodyVelocity) -> BaseTarget:
        if not isinstance(velocity, BodyVelocity):
            raise TypeError("Expected BodyVelocity in m/s and rad/s")
        wheels = self._kinematics.body_to_wheels(velocity)
        # Scale angular velocities before conversion to avoid overflowing ticks/s.
        peak = max(abs(value) for value in wheels)
        limit_rad_s = 3000 * math.tau / 4096
        scale = limit_rad_s / peak if peak > limit_rad_s else 1.0
        raw = tuple(round(value * scale * 4096 / math.tau) for value in wheels)
        return BaseTarget(raw, self.measured_velocity(raw))

    def measured_velocity(self, wheel_ticks_s: tuple[int, int, int]) -> BodyVelocity:
        """Convert all three measured signed registers; never fill a missing wheel."""
        if len(wheel_ticks_s) != 3:
            raise ValueError("Expected exactly three measured wheel speeds")
        if any(type(value) is not int or abs(value) > 32767 for value in wheel_ticks_s):
            raise ValueError("Expected decoded STS signed velocity registers")
        return self._kinematics.wheels_to_body(tuple(v * math.tau / 4096 for v in wheel_ticks_s))
