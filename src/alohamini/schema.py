"""Framework-independent values; no transport or hardware dependencies."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from alohamini._validation import finite_number, identifier


@dataclass(frozen=True)
class BodyVelocity:
    """Velocity in the body frame: +x forward, +y left, +yaw counterclockwise."""

    x_m_s: float = 0.0
    y_m_s: float = 0.0
    yaw_rad_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x_m_s", "y_m_s", "yaw_rad_s"):
            finite_number(getattr(self, name), name)


@dataclass(frozen=True)
class CommandIdentity:
    """Identity of a validated command, not an acknowledgement of execution."""

    client_id: str
    sequence: int
    host_session_id: str
    control_epoch: int

    def __post_init__(self) -> None:
        identifier(self.client_id, "client_id")
        identifier(self.host_session_id, "host_session_id")
        for name in ("sequence", "control_epoch"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class RobotCommand:
    """Partial targets in controller coordinates; omitted subsystems retain targets.

    positions_rad uses the controller's EncoderCalibration, not an implicit URDF
    frame. Host startup uses post-offset servo output angle (tick 2048 = 0,
    increasing ticks = positive). A ROS/model adapter must apply its own installed
    joint reference and sign. Base velocity is body-frame SI; lift height is metres
    above this Host session's established reference.
    lift_stop zeros only the lift, then holds subsequent Host-local feedback after
    a settling interval; it cannot be combined with a lift height target.
    """

    positions_rad: Mapping[str, float] = field(default_factory=dict)
    base_velocity: BodyVelocity | None = None
    lift_height_m: float | None = None
    lift_stop: bool = False

    def __post_init__(self) -> None:
        positions = dict(self.positions_rad)
        for name, value in positions.items():
            identifier(name, "joint name")
            finite_number(value, name)
        if self.base_velocity is not None and not isinstance(self.base_velocity, BodyVelocity):
            raise TypeError("Expected BodyVelocity")
        if self.lift_height_m is not None:
            finite_number(self.lift_height_m, "lift_height_m")
        if type(self.lift_stop) is not bool:
            raise TypeError("lift_stop must be a bool")
        if self.lift_stop and self.lift_height_m is not None:
            raise ValueError("Lift stop and height target are mutually exclusive")
        if (
            not positions
            and self.base_velocity is None
            and self.lift_height_m is None
            and not self.lift_stop
        ):
            raise ValueError("Robot command must contain a target")
        object.__setattr__(self, "positions_rad", MappingProxyType(positions))
