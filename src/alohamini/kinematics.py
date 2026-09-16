# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AlohaMini Host base kinematics and alohamini_lerobot_bridge/protocol.py.
# Changes: SI-only interface, standard-library inverse, explicit geometry validation.
"""Three-wheel omni-base kinematics; no register encoding or command limiting."""

import math
from dataclasses import dataclass

from alohamini._validation import finite_number
from alohamini.schema import BodyVelocity

_ANGLES = tuple(math.radians(a) for a in (150.0, -90.0, 30.0))


@dataclass(frozen=True)
class OmniBaseKinematics:
    """Wheel order: left, back, right; angular velocities in rad/s.

    Wheel signs follow the Host drive geometry and the ROS wheel1/2/3 axes.
    This conversion neither enforces actuator limits nor sends commands.
    """

    wheel_radius_m: float
    base_radius_m: float

    def __post_init__(self) -> None:
        for name in ("wheel_radius_m", "base_radius_m"):
            value = getattr(self, name)
            finite_number(value, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    def body_to_wheels(self, velocity: BodyVelocity) -> tuple[float, float, float]:
        wheels = tuple(
            (
                -math.cos(angle) * velocity.x_m_s
                - math.sin(angle) * velocity.y_m_s
                + self.base_radius_m * velocity.yaw_rad_s
            )
            / self.wheel_radius_m
            for angle in _ANGLES
        )
        for value in wheels:
            finite_number(value, "wheel velocity")
        return wheels

    def wheels_to_body(self, wheels_rad_s: tuple[float, float, float]) -> BodyVelocity:
        if len(wheels_rad_s) != 3:
            raise ValueError("Expected left, back and right wheel velocities")
        for value in wheels_rad_s:
            finite_number(value, "wheel velocity")
        linear = tuple(value * self.wheel_radius_m for value in wheels_rad_s)
        # Equally spaced wheel axes: sum(cos²) = sum(sin²) = 3/2.
        return BodyVelocity(
            x_m_s=-2 / 3 * sum(v * math.cos(a) for v, a in zip(linear, _ANGLES, strict=True)),
            y_m_s=-2 / 3 * sum(v * math.sin(a) for v, a in zip(linear, _ANGLES, strict=True)),
            yaw_rad_s=sum(linear) / (3 * self.base_radius_m),
        )
