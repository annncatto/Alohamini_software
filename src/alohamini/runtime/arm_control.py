"""Per-cycle arm supervision using the same feedback sampled by the Host."""

import time
from collections.abc import Mapping, Sequence
from typing import Protocol

from alohamini._validation import identifier
from alohamini.calibration import EncoderCalibration
from alohamini.hardware.feedback import FeedbackBatch
from alohamini.model import ActuatorSpec
from alohamini.runtime.arm_contact import ArmContactGuard, ArmJointSpec


class ArmDevice(Protocol):
    @property
    def actuators(self) -> tuple[ActuatorSpec, ...]: ...

    @property
    def position_calibrations(self) -> Mapping[str, EncoderCalibration]: ...

    def write_targets(
        self, *, positions_rad: Mapping[str, float], velocities_raw: Mapping[str, int]
    ) -> None: ...


class ArmController:
    """Arm-only cycle controller, supplied to HostSupervisor as its control dependency.

    CommandSubmission stages validated targets via set_targets; supervise writes
    them only after current/feedback checks. With no new command, only changed
    safety corrections are sent. Base and lift are not controlled by this class.
    A controller is bound to exactly one Host session.
    """

    def __init__(self, devices: Mapping[str, ArmDevice], joints: Sequence[ArmJointSpec]) -> None:
        joints = tuple(joints)
        if not all(isinstance(spec, ArmJointSpec) for spec in joints):
            raise TypeError("Expected ArmJointSpec values")
        self._joints = {spec.actuator.name: spec for spec in joints}
        if not joints or len(self._joints) != len(joints):
            raise ValueError("Provide distinct arm joint specifications")
        self._devices = dict(devices)
        self._groups: dict[str, dict[str, ArmJointSpec]] = {}
        for name, spec in self._joints.items():
            source = spec.actuator.bus
            if source not in devices or spec.actuator not in devices[source].actuators:
                raise ValueError(f"Arm actuator does not match device: {name}")
            if devices[source].position_calibrations.get(name) != spec.calibration:
                raise ValueError(f"Arm calibration does not match device: {name}")
            self._groups.setdefault(source, {})[name] = spec
        self._guard = ArmContactGuard(self._joints)
        self._sequences: dict[str, int] = {}
        self._received: dict[str, float] = {}
        self._session: str | None = None
        self._positions: dict[str, float] = {}
        self._currents: dict[str, float] = {}
        self._targets: dict[str, float] = {}
        self._sent: dict[str, float] = {}
        self._new_command = False
        self._fresh = False

    @property
    def requested_targets(self) -> dict[str, float]:
        return dict(self._targets)

    @property
    def measured_positions(self) -> dict[str, float]:
        return dict(self._positions)

    @property
    def sent_targets(self) -> dict[str, float]:
        return dict(self._sent)

    @property
    def contact_holds(self) -> dict[str, float]:
        return self._guard.holds

    @property
    def contact_hold_events(self) -> int:
        return self._guard.hold_events

    @property
    def joint_hold_events(self) -> int:
        return self._guard.joint_hold_events

    def bind_session(self, host_session_id: str) -> None:
        identifier(host_session_id, "host_session_id")
        if self._session is not None:
            raise RuntimeError("Arm controller cannot be reused across Host sessions")
        self._session = host_session_id

    def clear_targets(self) -> None:
        """Cancel motion on stop; preserve contact holds until explicit retreat."""
        self._targets.clear()
        self._sent.clear()
        self._positions.clear()
        self._currents.clear()
        self._new_command = False
        self._fresh = False
        self._guard.cancel_candidates()

    def observe(self, batches: tuple[FeedbackBatch, ...]) -> None:
        self._fresh = False
        self._positions.clear()
        self._currents.clear()
        if self._session is None:
            raise RuntimeError("Arm controller has no Host session")
        sources = [batch.source_id for batch in batches]
        if len(set(sources)) != len(sources) or not self._groups.keys() <= set(sources):
            raise ValueError("Missing or duplicated arm feedback bus")
        positions, currents = {}, {}
        for batch in batches:
            if batch.source_id not in self._groups:
                continue
            source = batch.source_id
            if (
                batch.clock_id != self._session
                or batch.sequence <= self._sequences.get(source, -1)
                or batch.request_started_s < self._received.get(source, 0)
            ):
                raise ValueError("Wrong session or out-of-order arm feedback")
            self._sequences[source] = batch.sequence
            self._received[source] = batch.received_s
            for name in self._groups[batch.source_id]:
                sample = batch.samples.get(name)
                if (
                    sample is None
                    or name in batch.failures
                    or sample.packet_error
                    or "position_raw" not in sample.registers
                    or sample.current_a is None
                ):
                    raise ValueError(f"Arm position/current feedback unavailable: {name}")
                # Native position-mode goals and feedback share one encoder branch.
                # ROS continuous-angle unwrapping belongs to the model adapter;
                # applying it here creates a full-turn error after a wrist wrap.
                positions[name] = self._joints[name].calibration.position_from_tick(
                    sample.registers["position_raw"]
                )
                currents[name] = sample.current_a
        self._positions, self._currents = positions, currents
        self._fresh = True

    def validate_targets(self, positions_rad: Mapping[str, float]) -> None:
        """Validate against this cycle without changing targets or writing devices."""
        if not self._fresh:
            raise RuntimeError("Targets require fresh feedback in the current Host cycle")
        positions = dict(positions_rad)
        if not positions or not positions.keys() <= self._joints.keys():
            raise ValueError("Expected nonempty known arm targets")
        for name, position in positions.items():
            self._joints[name].calibration.position_to_tick(position)

    def set_targets(self, positions_rad: Mapping[str, float]) -> None:
        """Stage one accepted command; validate all joints before changing any target."""
        positions = dict(positions_rad)
        self.validate_targets(positions)
        self._targets.update(positions)
        self._new_command = True

    def supervise(self) -> None:
        # A watchdog can clear targets after observe() and before this call.
        if not self._targets:
            self._fresh = False
            return
        if not self._fresh:
            raise RuntimeError("Arm supervision requires a new feedback batch")
        self._fresh = False
        limited = self._guard.limit(
            self._targets, self._positions, self._currents, now=time.monotonic()
        )
        # Check both arms before either bus is written. A hold outside calibrated
        # target bounds faults the Host; its stop backend can hold actual raw position.
        for name, position in limited.items():
            self._joints[name].calibration.position_to_tick(position)
        for source, group in self._groups.items():
            changed = {
                name: position
                for name, position in limited.items()
                if name in group and (self._new_command or self._sent.get(name) != position)
            }
            if changed:
                self._devices[source].write_targets(positions_rad=changed, velocities_raw={})
                self._sent.update(changed)
        self._new_command = False
