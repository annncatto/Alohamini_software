# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable model descriptions, separate from installed-device calibration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ActuatorSpec:
    """Motor IDs are unique within one bus, not across the robot."""

    name: str
    bus: str
    motor_id: int
    motor_model: str


def asset_path(directory: Path, relative: str) -> Path:
    """Resolve a model-local file without following paths outside the asset root."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Asset path must be a nonempty relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Asset path must stay inside the model directory")
    root = directory.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Asset path escapes the model directory")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


@dataclass(frozen=True)
class RobotModel:
    """Nominal layout and description assets; not a physical safety configuration."""

    model_id: str
    actuators: tuple[ActuatorSpec, ...]
    wheel_radius_m: float
    base_radius_m: float
    lift_lead_m_per_rev: float
    asset_revision: int
    descriptions: Mapping[str, str]
    directory: Path = field(repr=False, compare=False)

    def asset_path(self, relative: str) -> Path:
        return asset_path(self.directory, relative)

    def description_path(self, name: str) -> Path:
        """Return a description file; no fallback to another model or description."""
        if name not in self.descriptions:
            raise ValueError(f"Model {self.model_id!r} has no description {name!r}")
        return self.asset_path(self.descriptions[name])
