"""Base/lift cycle controller; height control requires an explicit physical reference."""

from collections.abc import Mapping
from typing import Protocol

from alohamini._validation import identifier
from alohamini.hardware.feedback import FeedbackBatch
from alohamini.kinematics import OmniBaseKinematics
from alohamini.model import ActuatorSpec
from alohamini.runtime.base_control import BaseDrive, BaseTarget
from alohamini.runtime.lift_control import (
    LiftAxisSpec,
    LiftHeightTracker,
    LiftHoming,
    LiftOutput,
    lift_height_target,
)
from alohamini.schema import BodyVelocity


class VelocityDevice(Protocol):
    @property
    def actuators(self) -> tuple[ActuatorSpec, ...]: ...

    @property
    def velocity_limits(self) -> Mapping[str, int]: ...

    def write_targets(
        self, *, positions_rad: Mapping[str, float], velocities_raw: Mapping[str, int]
    ) -> None: ...

    def stop(self) -> None: ...


class BaseLiftController:
    """CycleControl for velocity axes, with wheel order left/back/right.

    Feedback and targets remain separate. Missing lift reference does not invent
    height zero: base-only commands work, height commands are rejected. Accepted
    height targets are recomputed each cycle until stopped or replaced. Sensorless
    homing is an explicit local startup operation, not a remote velocity bypass.
    """

    def __init__(
        self,
        devices: Mapping[str, VelocityDevice],
        *,
        wheels: tuple[ActuatorSpec, ActuatorSpec, ActuatorSpec],
        kinematics: OmniBaseKinematics,
        lift: LiftAxisSpec,
    ) -> None:
        if len(wheels) != 3 or not all(isinstance(w, ActuatorSpec) for w in wheels):
            raise ValueError("Provide left/back/right wheel actuators")
        if not isinstance(lift, LiftAxisSpec):
            raise TypeError("Expected LiftAxisSpec")
        self._wheels = tuple(wheels)
        self._lift_spec = lift
        self._actuators = (*self._wheels, lift.actuator)
        if len({motor.name for motor in self._actuators}) != 4:
            raise ValueError("Wheel and lift names must be distinct")
        for motor in self._actuators:
            if motor.bus not in devices or motor not in devices[motor.bus].actuators:
                raise ValueError(f"Velocity actuator does not match its device: {motor.name}")
            expected = 1300 if motor == lift.actuator else 3000
            if devices[motor.bus].velocity_limits.get(motor.name) != expected:
                raise ValueError(f"Velocity limit does not match deployed policy: {motor.name}")
        self._devices = dict(devices)
        self._sources = {motor.bus for motor in self._actuators}
        self._base = BaseDrive(kinematics)
        self._lift = LiftHeightTracker(lift)
        self._session: str | None = None
        self._sequences: dict[str, int] = {}
        self._received: dict[str, float] = {}
        self._base_target: BaseTarget | None = None
        self._measured_base: BodyVelocity | None = None
        self._lift_target: float | None = None
        self._lift_output: LiftOutput | None = None
        self._lift_stop_pending = False
        self._sent: dict[str, int] = {}
        self._fresh = False
        self._new_command = False
        self._homing: LiftHoming | None = None
        self._homing_stop_confirmed = False

    @property
    def measured_base_velocity(self) -> BodyVelocity | None:
        return self._measured_base

    @property
    def base_target(self) -> BaseTarget | None:
        return self._base_target

    @property
    def lift_height_m(self) -> float | None:
        return self._lift.height_m

    @property
    def lift_target_height_m(self) -> float | None:
        return self._lift_target

    @property
    def lift_output(self) -> LiftOutput | None:
        return self._lift_output

    @property
    def lift_homing_phase(self) -> str | None:
        return self._homing.phase if self._homing is not None else None

    @property
    def lift_homing(self) -> bool:
        return self.lift_homing_phase in ("seeking", "settling")

    def begin_lift_homing(self) -> None:
        """Schedule local descent; caller must ensure the path is unobstructed.

        No I/O or torque enable occurs here. Execution remains in supervised Host
        cycles, and all ordinary motion commands are rejected until completion.
        """
        if self._session is None or self._homing is not None:
            raise RuntimeError("Homing requires a new, connected Host session")
        self.clear_targets()
        self._lift.invalidate()
        self._homing = LiftHoming(self._lift_spec)
        self._homing_stop_confirmed = False

    def bind_session(self, session: str) -> None:
        identifier(session, "host_session_id")
        if self._session is not None:
            raise RuntimeError("Base/lift controller cannot be rebound")
        self._lift.bind_session(session)
        self._session = session

    def clear_targets(self) -> None:
        """Cancel motion, not a continuously observed height reference."""
        if self.lift_homing:
            self._homing.phase = "failed"
            self._homing.velocity_raw = 0
            self._lift.invalidate()
        self._base_target = None
        self._lift_target = None
        self._lift_output = None
        self._lift_stop_pending = False
        self._sent.clear()
        self._measured_base = None
        self._fresh = False
        self._new_command = False

    def observe(self, batches: tuple[FeedbackBatch, ...]) -> None:
        self._fresh = False
        self._measured_base = None
        try:
            if self._session is None:
                raise RuntimeError("Base/lift controller has no Host session")
            by_source = {batch.source_id: batch for batch in batches}
            if len(by_source) != len(batches) or not self._sources <= by_source.keys():
                raise ValueError("Missing or duplicated velocity-axis feedback bus")
            for source in self._sources:
                batch = by_source[source]
                if batch.clock_id != self._session or batch.sequence <= self._sequences.get(
                    source, -1
                ):
                    raise ValueError("Wrong session or repeated velocity-axis feedback")
                if batch.request_started_s < self._received.get(source, 0):
                    raise ValueError("Velocity-axis feedback intervals overlap")
                self._sequences[source] = batch.sequence
                self._received[source] = batch.received_s
            wheel_speeds = []
            for motor in self._wheels:
                batch = by_source[motor.bus]
                sample = batch.samples.get(motor.name)
                if sample is None or motor.name in batch.failures or sample.packet_error:
                    raise ValueError(f"Wheel feedback unavailable: {motor.name}")
                wheel_speeds.append(sample.registers.get("velocity_raw"))
            measured = self._base.measured_velocity(tuple(wheel_speeds))
            self._lift.observe(by_source[self._lift_spec.actuator.bus])
            if self.lift_homing:
                self._homing.update(by_source[self._lift_spec.actuator.bus])
                if self._homing.phase == "complete":
                    if not self._homing_stop_confirmed:
                        raise RuntimeError("Homing stop target has not been confirmed")
                    self._lift.establish_reference(0.0)
                    self._lift_stop_pending = True
            self._measured_base = measured
            self._fresh = True
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            self.clear_targets()
            self._lift.invalidate()
            raise

    def establish_lift_reference(self, height_m: float) -> None:
        """Internal calibration task after independent height/rest verification.

        Must run inside a current Host cycle, not using a previously cached sample.
        Stages zero velocities; it does not move to a stop to discover the height.
        """
        if not self._fresh:
            raise RuntimeError("Lift reference requires the current Host feedback cycle")
        if self.lift_homing:
            raise RuntimeError("Cannot replace the reference during lift homing")
        self._lift.establish_reference(height_m)
        self._base_target = self._base.target(BodyVelocity())
        self._lift_target = None
        self._lift_stop_pending = True
        self._new_command = True

    def validate_targets(
        self, base_velocity: BodyVelocity, *, lift_height_m: float | None = None
    ) -> None:
        """Validate against this cycle without changing either target."""
        if not self._fresh:
            raise RuntimeError("Velocity targets require the current Host feedback cycle")
        if self.lift_homing:
            raise RuntimeError("Lift homing is in progress")
        self._base.target(base_velocity)
        if lift_height_m is not None:
            if self._lift.height_m is None:
                raise RuntimeError("Lift has no verified physical height reference")
            lift_height_target(
                lift_height_m, self._lift.height_m, direction=self._lift_spec.direction
            )

    def set_targets(
        self, base_velocity: BodyVelocity, *, lift_height_m: float | None = None
    ) -> None:
        """Stage an accepted command; validate both targets before changing either."""
        self.validate_targets(base_velocity, lift_height_m=lift_height_m)
        base = self._base.target(base_velocity)
        lift_target = self._lift_target
        if lift_height_m is not None:
            lift_target = lift_height_target(
                lift_height_m, self._lift.height_m, direction=self._lift_spec.direction
            ).target_height_m
        self._base_target, self._lift_target = base, lift_target
        self._new_command = True

    def supervise(self) -> None:
        if (
            self._base_target is None
            and self._lift_target is None
            and not self._lift_stop_pending
            and not self.lift_homing
        ):
            self._fresh = False
            return
        if not self._fresh:
            raise RuntimeError("Velocity supervision requires a new feedback batch")
        self._fresh = False
        planned = {}
        if self._base_target is not None:
            planned.update(
                zip((m.name for m in self._wheels), self._base_target.wheel_ticks_s, strict=True)
            )
        if self.lift_homing:
            planned.update({motor.name: 0 for motor in self._wheels})
            planned[self._lift_spec.actuator.name] = self._homing.velocity_raw
            if self._homing.phase == "settling" and not self._homing_stop_confirmed:
                # stop verifies target registers; successful group transmission
                # alone cannot confirm removal of the descent target at contact.
                for source in sorted(self._sources):
                    self._devices[source].stop()
                self._homing_stop_confirmed = True
                self._sent.update({motor.name: 0 for motor in self._actuators})
        elif self._lift_stop_pending:
            planned[self._lift_spec.actuator.name] = 0
        elif self._lift_target is not None:
            if self._lift.height_m is None:
                raise RuntimeError("Lift reference lost; cannot continue the height target")
            self._lift_output = lift_height_target(
                self._lift_target, self._lift.height_m, direction=self._lift_spec.direction
            )
            planned[self._lift_spec.actuator.name] = self._lift_output.velocity_raw
        # Build and check all bus groups before the first write.
        groups: dict[str, dict[str, int]] = {}
        for motor in self._actuators:
            if motor.name not in planned:
                continue
            value = planned[motor.name]
            if abs(value) > self._devices[motor.bus].velocity_limits[motor.name]:
                raise ValueError(f"Velocity target exceeds device limit: {motor.name}")
            if self._new_command or self._sent.get(motor.name) != value:
                groups.setdefault(motor.bus, {})[motor.name] = value
        for source, values in groups.items():
            self._devices[source].write_targets(positions_rad={}, velocities_raw=values)
            self._sent.update(values)
        self._lift_stop_pending = False
        self._new_command = False
