# SPDX-License-Identifier: Apache-2.0
# Adapted from alohamini_lerobot_bridge.protocol.JointMapper lift conversion.
# Changes: explicit SI units; display-only feedback clamping is not applied here.
"""Affine physical-height to model-joint conversion, not homing or lift control."""

from dataclasses import dataclass

from alohamini._validation import finite_number


@dataclass(frozen=True)
class LiftCalibration:
    """Explicit physical and model endpoints, all in meters.

    Loading this mapping does not establish a hardware home reference and does
    not validate the machine's physical travel. No calibration is loaded globally.
    """

    physical_min_m: float
    physical_max_m: float
    position_min_m: float
    position_max_m: float

    def __post_init__(self) -> None:
        for name in ("physical_min_m", "physical_max_m", "position_min_m", "position_max_m"):
            finite_number(getattr(self, name), name)
        for name, span in (
            ("physical range", self.physical_max_m - self.physical_min_m),
            ("position range", self.position_max_m - self.position_min_m),
        ):
            finite_number(span, name)
            if span <= 0:
                raise ValueError(f"{name} must be positive")

    def height_to_position(self, height_m: float) -> float:
        """Transform feedback without hiding physical-range overrun by clamping."""
        finite_number(height_m, "height_m")
        ratio = (height_m - self.physical_min_m) / (self.physical_max_m - self.physical_min_m)
        position = self.position_min_m + ratio * (self.position_max_m - self.position_min_m)
        finite_number(position, "lift position")
        return position

    def position_to_height(self, position_m: float) -> float:
        """Map an in-range target to physical height; never command a homing motion."""
        finite_number(position_m, "position_m")
        if not self.position_min_m <= position_m <= self.position_max_m:
            raise ValueError("Lift target is outside calibrated position limits")
        ratio = (position_m - self.position_min_m) / (self.position_max_m - self.position_min_m)
        height = self.physical_min_m + ratio * (self.physical_max_m - self.physical_min_m)
        finite_number(height, "physical height")
        return height
