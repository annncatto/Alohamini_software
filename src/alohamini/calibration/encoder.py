# SPDX-License-Identifier: Apache-2.0
# Adapted from alohamini_lerobot_bridge.protocol.JointMapper.
# Changes: named physical-unit inputs, explicit per-stream continuity, strict validation.
"""Single-turn encoder conversion independent of normalization and ROS joint names."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from alohamini._validation import finite_number, identifier


@dataclass(frozen=True)
class HostPositionUnits:
    """Deployed Host normalization, separate from physical encoder calibration.

    The formulas match FeetechMotorsBus, including truncation and normalized
    saturation. Raw feedback remains available separately without saturation.
    drive_mode is the effective software inversion, not an EEPROM write.
    """

    normalization: str
    range_min: int
    range_max: int
    drive_mode: int

    def __post_init__(self) -> None:
        if self.normalization not in ("range_m100_100", "range_0_100", "degrees"):
            raise ValueError("Unsupported Host position normalization")
        if (
            type(self.range_min) is not int
            or type(self.range_max) is not int
            or not 0 <= self.range_min < self.range_max <= 4095
        ):
            raise ValueError("Host ranges must lie within one STS encoder revolution")
        if type(self.drive_mode) is not int or self.drive_mode not in (0, 1):
            raise ValueError("drive_mode must be 0 or 1")

    def to_tick(self, value: float) -> int:
        finite_number(value, "position target")
        lower, upper = self.range_min, self.range_max
        if self.normalization == "range_m100_100":
            value = -value if self.drive_mode else value
            tick = int((min(100.0, max(-100.0, value)) + 100) / 200 * (upper - lower) + lower)
        elif self.normalization == "range_0_100":
            value = 100 - value if self.drive_mode else value
            tick = int(min(100.0, max(0.0, value)) / 100 * (upper - lower) + lower)
        else:
            # Reject before integer conversion if finite input arithmetic overflows.
            raw = value * 4095 / 360 + (lower + upper) / 2
            finite_number(raw, "position tick")
            tick = int(raw)
        if not lower <= tick <= upper:
            raise ValueError("Position target exceeds the installed encoder range")
        return tick

    def from_tick(self, tick: int) -> float:
        if type(tick) is not int or not 0 <= tick <= 4095:
            raise ValueError("Expected a single-turn STS position")
        lower, upper = self.range_min, self.range_max
        if self.normalization == "degrees":
            return (tick - (lower + upper) / 2) * 360 / 4095
        fraction = (min(upper, max(lower, tick)) - lower) / (upper - lower)
        if self.normalization == "range_m100_100":
            value = fraction * 200 - 100
            return -value if self.drive_mode else value
        value = fraction * 100
        return 100 - value if self.drive_mode else value


@dataclass(frozen=True)
class EncoderCalibration:
    """Mapping for one explicitly calibrated joint, in post-offset encoder ticks.

    Feedback conversion does not clamp measured positions. A single-turn encoder
    cannot recover turns lost while disconnected. Successive observations must
    move by less than half a joint encoder period for nearest-branch continuity.
    Target limits are supplied per device, not inferred from a robot model.
    """

    ticks_per_revolution: int
    reference_tick: int
    reference_position_rad: float
    direction: int
    joint_per_encoder_ratio: float = 1.0
    position_min_rad: float | None = None
    position_max_rad: float | None = None

    def __post_init__(self) -> None:
        if type(self.ticks_per_revolution) is not int or self.ticks_per_revolution < 2:
            raise ValueError("ticks_per_revolution must be an integer >= 2")
        self._check_tick(self.reference_tick)
        if type(self.direction) is not int or self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        finite_number(self.reference_position_rad, "reference_position_rad")
        finite_number(self.joint_per_encoder_ratio, "joint_per_encoder_ratio")
        if self.joint_per_encoder_ratio <= 0:
            raise ValueError("joint_per_encoder_ratio must be positive")
        finite_number(self.joint_period_rad, "joint_period_rad")
        try:
            step = self.joint_period_rad / self.ticks_per_revolution
        except OverflowError as exc:
            raise ValueError("ticks_per_revolution is too large") from exc
        if step == 0:
            raise ValueError("Encoder angular resolution underflows")
        if (self.position_min_rad is None) != (self.position_max_rad is None):
            raise ValueError("Provide both position limits or neither")
        if self.position_min_rad is not None:
            finite_number(self.position_min_rad, "position_min_rad")
            finite_number(self.position_max_rad, "position_max_rad")
            if self.position_min_rad >= self.position_max_rad:
                raise ValueError("position_max_rad must exceed position_min_rad")

    @property
    def joint_period_rad(self) -> float:
        return math.tau * self.joint_per_encoder_ratio

    def _check_tick(self, tick: int) -> None:
        if type(tick) is not int or not 0 <= tick < self.ticks_per_revolution:
            raise ValueError("Encoder tick must be an integer within one revolution")

    def position_from_tick(self, tick: int, *, previous_position_rad: float | None = None) -> float:
        """Read a position; an ambiguous initial calibrated branch is rejected."""
        self._check_tick(tick)
        period = self.ticks_per_revolution
        delta = (tick - self.reference_tick + period // 2) % period - period // 2
        value = (
            self.reference_position_rad
            + self.direction * delta * math.tau / period * self.joint_per_encoder_ratio
        )
        finite_number(value, "decoded position")
        span = self.joint_period_rad
        if previous_position_rad is not None:
            finite_number(previous_position_rad, "previous_position_rad")
            turns = (previous_position_rad - value) / span
            finite_number(turns, "encoder turn offset")
            value += round(turns) * span
        elif self.position_min_rad is not None:
            lower = (self.position_min_rad - value) / span
            upper = (self.position_max_rad - value) / span
            finite_number(lower, "lower encoder branch")
            finite_number(upper, "upper encoder branch")
            first, last = math.ceil(lower), math.floor(upper)
            if first < last:
                raise ValueError(
                    "Ambiguous initial encoder branch; supply a known previous position"
                )
            if first == last:
                value += first * span
            # No equivalent in the interval: retain the actual principal reading,
            # including an out-of-range reading, rather than fabricate an endpoint.
        finite_number(value, "decoded position")
        return value

    def position_to_tick(self, position_rad: float) -> int:
        """Quantize a target within configured joint limits; this does not send it.

        The 1e-4 rad boundary tolerance preserves the existing mapping's treatment
        of floating-point drift at a calibrated endpoint. Bus-range checks, current
        protection and command authorization remain the Host's responsibility.
        """
        finite_number(position_rad, "position_rad")
        if self.position_min_rad is None:
            raise ValueError("Target conversion requires explicit position limits")
        lower, upper = self.position_min_rad, self.position_max_rad
        if position_rad < lower - 1e-4 or position_rad > upper + 1e-4:
            raise ValueError("Target is outside calibrated position limits")
        position = max(lower, min(upper, position_rad))
        delta = (position - self.reference_position_rad) * self.ticks_per_revolution
        delta /= self.direction * math.tau * self.joint_per_encoder_ratio
        finite_number(delta, "target encoder delta")
        return (self.reference_tick + round(delta)) % self.ticks_per_revolution

    def velocity_from_ticks_per_second(self, velocity: float) -> float:
        """Convert signed encoder ticks/s into joint rad/s, not an encoded register."""
        finite_number(velocity, "encoder velocity")
        result = self.direction * velocity * self.joint_period_rad / self.ticks_per_revolution
        finite_number(result, "joint velocity")
        return result


class JointPositionDecoder:
    """One sequential feedback stream; reset after reconnect, restart or a data gap.

    Calibrations are keyed by caller-defined joint names; there is no left/right
    fallback. Missing joints stay missing and lose their continuity reference.
    A malformed batch clears continuity so a later frame cannot reuse stale turns.
    """

    def __init__(self, calibrations: Mapping[str, EncoderCalibration]) -> None:
        if not isinstance(calibrations, Mapping) or not calibrations:
            raise ValueError("Provide at least one named encoder calibration")
        for name, calibration in calibrations.items():
            identifier(name, "joint name")
            if not isinstance(calibration, EncoderCalibration):
                raise TypeError("Expected EncoderCalibration values")
        self._calibrations = MappingProxyType(dict(calibrations))
        self._previous: dict[str, float] = {}

    def reset(self) -> None:
        self._previous.clear()

    def decode(self, ticks: Mapping[str, int]) -> dict[str, float]:
        try:
            if not isinstance(ticks, Mapping):
                raise TypeError("Expected a mapping of joint names to encoder ticks")
            result = {}
            for name, tick in ticks.items():
                if name not in self._calibrations:
                    raise ValueError(f"No calibration for joint {name!r}")
                result[name] = self._calibrations[name].position_from_tick(
                    tick, previous_position_rad=self._previous.get(name)
                )
        except (ValueError, TypeError):
            self.reset()
            raise
        self._previous = dict(result)
        return result
