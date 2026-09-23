# SPDX-License-Identifier: Apache-2.0
# Adapted from alohamini_lerobot_bridge/protocol.py.
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from alohamini._validation import finite_number as validate_number
from alohamini.calibration.encoder import (
    EncoderCalibration,
    HostPositionUnits,
    JointPositionDecoder,
)
from alohamini.calibration.lift import LiftCalibration

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)


def finite_number(value: Any, field: str) -> float:
    validate_number(value, field)
    return float(value)


class JointMapper:
    """Convert between Host motor normalization and authoritative URDF space."""

    def __init__(self, calibration: dict[str, Any], lift_calibration: dict[str, Any]) -> None:
        if set(calibration) != {"left", "right"}:
            raise ValueError("Provide independent left and right arm mappings")
        self.period = 4096
        self.joints_by_side = {side: calibration[side]["joints"] for side in ("left", "right")}
        encoders = {}
        for side in ("left", "right"):
            if calibration[side]["ticks_per_revolution"] != self.period:
                raise ValueError("STS mappings require 4096 ticks per revolution")
            if set(self.joints_by_side[side]) != set(ARM_JOINTS):
                raise ValueError(f"Incomplete arm mapping: {side}")
            for joint in ARM_JOINTS:
                entry = self._entry(side, joint)
                name = f"{side}_{'wrist_yaw_joint' if joint == 'wrist_yaw' else joint}"
                encoders[name] = EncoderCalibration(
                    self.period,
                    entry["reference_tick"],
                    entry["reference_q_rad"],
                    entry["sign"],
                    entry.get("joint_per_encoder_ratio", 1.0),
                    entry.get("safe_q_min_rad"),
                    entry.get("safe_q_max_rad"),
                )
        self.decoder = JointPositionDecoder(encoders)
        self.encoders = encoders
        mechanism, urdf = lift_calibration["mechanism"], lift_calibration["urdf"]
        self.lift = LiftCalibration(
            mechanism["physical_min_mm"] / 1000,
            mechanism["physical_max_mm"] / 1000,
            urdf["q_at_physical_min_m"],
            urdf["q_at_physical_max_m"],
        )

    def reset(self) -> None:
        self.decoder.reset()

    def _entry(self, side: str, joint: str) -> dict[str, Any]:
        try:
            return self.joints_by_side[side][joint]
        except KeyError as error:
            raise ValueError(f"missing calibration for {side}_{joint}") from error

    def lift_height_to_urdf(self, height_mm: float) -> float:
        return self.lift.height_to_position(finite_number(height_mm, "lift height") / 1000)

    def lift_urdf_to_height(self, position_m: float) -> float:
        return self.lift.position_to_height(position_m) * 1000

    def host_to_tick(self, value: float, metadata: dict[str, Any]) -> int:
        return HostPositionUnits(
            **{
                key: metadata[key]
                for key in ("normalization", "drive_mode", "range_min", "range_max")
            }
        ).to_tick(value)

    def tick_to_host(self, tick: int, metadata: dict[str, Any]) -> float:
        return HostPositionUnits(
            **{
                key: metadata[key]
                for key in ("normalization", "drive_mode", "range_min", "range_max")
            }
        ).from_tick(tick)

    def urdf_to_host(self, name: str, position: float, metadata: dict) -> tuple[str, float]:
        side, _, suffix = name.partition("_")
        joint = "wrist_yaw" if suffix == "wrist_yaw_joint" else suffix
        entry = self._entry(side, joint)
        encoder = self.encoders[name]
        if encoder.position_min_rad is None:
            # Preserve the original wrist-roll and gripper command bounds.
            limits = (
                (-math.pi, math.pi)
                if joint == "wrist_roll"
                else (entry["urdf_open_rad"], entry["urdf_closed_rad"])
            )
            encoder = replace(encoder, position_min_rad=min(limits), position_max_rad=max(limits))
        tick = encoder.position_to_tick(position)
        motor = f"arm_{side}_{joint}"
        units = metadata["motors"][motor]
        if not units["range_min"] <= tick <= units["range_max"]:
            raise ValueError(f"{name} target exceeds installed Host encoder range")
        return f"{motor}.pos", self.tick_to_host(tick, units)

    def observation_to_joint_positions(
        self, observation: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, float]:
        ticks = {}
        for side in ("left", "right"):
            for joint in ARM_JOINTS:
                motor_name = f"arm_{side}_{joint}"
                suffix = "wrist_yaw_joint" if joint == "wrist_yaw" else joint
                ticks[f"{side}_{suffix}"] = self.host_to_tick(
                    observation[f"{motor_name}.pos"], metadata["motors"][motor_name]
                )
        height = finite_number(observation["lift_axis.height_mm"], "lift_axis.height_mm")
        lift = self.lift_height_to_urdf(height)
        positions = self.decoder.decode(ticks)
        positions["vertical_move"] = lift
        return positions
