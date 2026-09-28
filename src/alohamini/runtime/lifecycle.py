# SPDX-License-Identifier: Apache-2.0
# Adapted from AlohaMini Host watchdog and shutdown ordering.
"""Host-thread lifecycle and command gate, independent of transports and target units."""

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Protocol
from uuid import uuid4

from alohamini._validation import identifier
from alohamini.errors import CommandRejectedError
from alohamini.hardware.feedback import FeedbackBatch
from alohamini.model import ActuatorSpec
from alohamini.runtime.command_owner import CommandOwner
from alohamini.runtime.current_protection import CurrentProtection, CurrentTrip
from alohamini.schema import CommandIdentity


class HostDevice(Protocol):
    """One exclusively owned bus; all methods must have bounded I/O.

    connect opens resources only: no torque enable, homing or target writes.
    read_feedback uses the supplied session ID as its monotonic clock_id.
    stop cancels active targets, holds positional axes using valid feedback and
    zeros velocity-controlled axes. It must raise if any required stop fails.
    stop_velocity only zeros velocity axes before overcurrent torque release;
    it must attempt all such axes and raise on failure without holding joints.
    stop_motion is the runtime watchdog path, using this cycle's feedback rather
    than maintenance register readback. Its group writes have no servo ACK.
    disable_torque and close must also handle partially completed connections.
    These contracts do not prove that physical motion has stopped.
    """

    def connect(self, host_session_id: str) -> None: ...
    def read_feedback(self) -> FeedbackBatch: ...
    def stop_motion(self, feedback: FeedbackBatch) -> None: ...
    def stop(self) -> None: ...
    def stop_velocity(self) -> None: ...
    def disable_torque(self) -> None: ...
    def close(self) -> None: ...


class CycleControl(Protocol):
    """Internal motion controller; supervision follows the Host's current checks."""

    def bind_session(self, host_session_id: str) -> None: ...
    def observe(self, batches: tuple[FeedbackBatch, ...]) -> None: ...
    def supervise(self) -> None: ...
    def clear_targets(self) -> None: ...


@dataclass(frozen=True)
class CommandSubmission:
    """In-process submission, not a wire command or a target-validation bypass.

    The Host's command decoder must validate target names, units and device
    limits before constructing this value. write captures those immutable targets
    and either stages them in the cycle controller or performs bounded actuator
    I/O; never policy inference or camera work. With CycleControl, targets must
    be staged so supervision runs before the hardware write.
    """

    identity: CommandIdentity
    write: Callable[[], None]
    validate: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CommandIdentity) or not callable(self.write):
            raise TypeError("Expected a command identity and a bounded write operation")
        if self.validate is not None and not callable(self.validate):
            raise TypeError("Command validation must be callable")


class HostPhase(str, Enum):
    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    ACTIVE = "active"
    FAULT = "fault"
    CLOSED = "closed"


@dataclass(frozen=True)
class CleanupFailure:
    source_id: str
    operation: str
    error: str


@dataclass(frozen=True)
class HostStatus:
    phase: HostPhase
    host_session_id: str
    control_owner: str | None
    control_epoch: int
    watchdog_events: int
    fault: str | None
    current_trip: CurrentTrip | None
    cleanup_failures: tuple[CleanupFailure, ...]


@dataclass(frozen=True)
class CycleResult:
    feedback: tuple[FeedbackBatch, ...]
    command_applied: bool
    status: HostStatus
    command_error: str | None = None


class HostSupervisor:
    """One Host session; every cycle samples protection before any new command.

    Faults are latched and close all attempted connections. Recovery requires a
    new supervisor/session, not an automatic torque re-enable or homing attempt.
    A watchdog stop retains connections and revokes ownership only on success.
    The 2 s command lease is independent of client inference latency:
    neither an observation-request age limit nor a cross-machine clock is used.

    This component does not schedule 50 Hz, validate targets, or implement motor
    writes. HostDevice supplies those hardware operations; callbacks are internal
    to the Host, never accepted from remote clients.
    """

    COMMAND_WATCHDOG_TIMEOUT_S = 2.0

    def __init__(
        self,
        devices: Mapping[str, HostDevice],
        actuators: Sequence[ActuatorSpec],
        *,
        control: CycleControl | None = None,
    ) -> None:
        self._devices = dict(devices)
        self._control = control
        self._control_bound = False
        self._expected: dict[str, set[str]] = {source: set() for source in devices}
        for source in devices:
            identifier(source, "source_id")
        models = {}
        for motor in actuators:
            if not isinstance(motor, ActuatorSpec):
                raise TypeError("Expected ActuatorSpec values")
            if motor.bus not in devices or motor.name in models:
                raise ValueError("Actuators must have unique names and an assigned bus")
            self._expected[motor.bus].add(motor.name)
            models[motor.name] = motor.motor_model
        if not devices or not all(self._expected.values()):
            raise ValueError("Every device must have configured actuators")
        self._protection = CurrentProtection(models)
        self._owner = CommandOwner(uuid4().hex)
        self._phase = HostPhase.NEW
        self._attempted: list[str] = []
        self._sequences: dict[str, int] = {}
        self._last_command_s: float | None = None
        self._watchdog_events = 0
        self._fault_reason: str | None = None
        self._trip: CurrentTrip | None = None
        self._cleanup_failures: list[CleanupFailure] = []
        self._thread_id = threading.get_ident()
        self._busy = False
        self.timing_ms: dict[str, float] = {}

    @property
    def status(self) -> HostStatus:
        return HostStatus(
            self._phase,
            self._owner.host_session_id,
            self._owner.owner,
            self._owner.epoch,
            self._watchdog_events,
            self._fault_reason,
            self._trip,
            tuple(self._cleanup_failures),
        )

    def __enter__(self) -> "HostSupervisor":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _enter(self) -> None:
        if threading.get_ident() != self._thread_id or self._busy:
            raise RuntimeError("Host lifecycle requires non-reentrant use on its owning thread")
        self._busy = True

    def _perform(self, operation: str) -> list[BaseException]:
        errors = []
        for source in self._attempted:
            try:
                getattr(self._devices[source], operation)()
            except BaseException as exc:
                # Finish other devices even if cleanup itself receives an interrupt.
                errors.append(exc)
                self._cleanup_failures.append(
                    CleanupFailure(source, operation, f"{type(exc).__name__}: {exc}")
                )
        return errors

    def _shutdown(self, *, already_stopped: bool = False) -> None:
        errors = []
        if not already_stopped:
            # Source overcurrent shutdown stops base/lift then unloads torque;
            # do not spend position hold transactions on overloaded arm joints.
            stop_errors = self._stop(velocity_only=self._trip is not None)
            errors.extend(stop_errors)
            if not stop_errors and self._owner.owner is not None:
                self._owner.release()
        errors.extend(self._perform("disable_torque"))
        errors.extend(self._perform("close"))
        self._attempted.clear()
        if errors:
            self._phase = HostPhase.FAULT
            self._fault_reason = self._fault_reason or "Device cleanup failed"
        for error in errors:
            if not isinstance(error, Exception):
                raise error

    def _stop(
        self, feedback: tuple[FeedbackBatch, ...] | None = None, *, velocity_only: bool = False
    ) -> list[BaseException]:
        errors = []
        if self._control is not None and self._control_bound:
            try:
                self._control.clear_targets()
            except BaseException as exc:
                errors.append(exc)
                self._cleanup_failures.append(
                    CleanupFailure("control", "clear_targets", f"{type(exc).__name__}: {exc}")
                )
        if feedback is None:
            errors.extend(self._perform("stop_velocity" if velocity_only else "stop"))
        else:
            by_source = {batch.source_id: batch for batch in feedback}
            for source in self._attempted:
                try:
                    self._devices[source].stop_motion(by_source[source])
                except BaseException as exc:
                    errors.append(exc)
                    self._cleanup_failures.append(
                        CleanupFailure(source, "stop_motion", f"{type(exc).__name__}: {exc}")
                    )
        return errors

    def _fault(self, reason: str, *, already_stopped: bool = False) -> None:
        self._phase = HostPhase.FAULT
        self._fault_reason = reason
        self._shutdown(already_stopped=already_stopped)

    def start(self) -> HostStatus:
        self._enter()
        try:
            if self._phase is not HostPhase.NEW:
                raise RuntimeError("A Host session can only be started once")
            self._phase = HostPhase.STARTING
            try:
                if self._control is not None:
                    self._control.bind_session(self._owner.host_session_id)
                    self._control_bound = True
                for source, device in self._devices.items():
                    self._attempted.append(source)
                    device.connect(self._owner.host_session_id)
            except BaseException as exc:
                self._fault(f"Device connection failed: {type(exc).__name__}: {exc}")
                raise
            return self.status
        finally:
            self._busy = False

    def _expire_command(self, feedback: tuple[FeedbackBatch, ...]) -> bool:
        if (
            self._last_command_s is None
            or time.monotonic() - self._last_command_s <= self.COMMAND_WATCHDOG_TIMEOUT_S
        ):
            return False
        self._watchdog_events += 1
        self._last_command_s = None
        errors = self._stop(feedback)
        if errors:
            self._fault("Command watchdog stop failed", already_stopped=True)
            for error in errors:
                if not isinstance(error, Exception):
                    raise error
        else:
            self._owner.release()
            self._phase = HostPhase.READY
        return True

    def _validate_feedback(self, source: str, batch: FeedbackBatch, started: float) -> None:
        if not isinstance(batch, FeedbackBatch):
            raise TypeError("Expected FeedbackBatch")
        if batch.source_id != source or batch.clock_id != self._owner.host_session_id:
            raise ValueError("Feedback bus or Host session mismatch")
        if batch.request_started_s < started or batch.received_s > time.monotonic():
            raise ValueError("Feedback is outside the current read transaction")
        if batch.sequence <= self._sequences.get(source, -1):
            raise ValueError("Repeated or out-of-order feedback sequence")
        self._sequences[source] = batch.sequence
        if batch.failures or batch.samples.keys() != self._expected[source]:
            raise ValueError("Feedback contains missing or failed motors")
        for sample in batch.samples.values():
            if sample.packet_error or sample.current_a is None:
                raise ValueError("Feedback has a servo fault or missing current")

    def cycle(
        self,
        command: CommandSubmission | None = None,
        *,
        poll_command: Callable[[], CommandSubmission | None] | None = None,
    ) -> CycleResult:
        self._enter()
        self.timing_ms = {}
        batches = []
        applied = False
        command_error = None
        try:
            if self._phase not in (HostPhase.STARTING, HostPhase.READY, HostPhase.ACTIVE):
                raise RuntimeError("Host must be started and free of latched faults")
            try:
                if command is not None and not isinstance(command, CommandSubmission):
                    raise TypeError("Expected CommandSubmission")
                if command is not None and poll_command is not None:
                    raise ValueError("Provide a command or a command poller, not both")
                if self._phase is not HostPhase.FAULT:
                    observation_started = time.perf_counter()
                    currents = {}
                    for source, device in self._devices.items():
                        started = time.monotonic()
                        read_started = time.perf_counter()
                        batch = device.read_feedback()
                        self.timing_ms[f"{source}_bus"] = (time.perf_counter() - read_started) * 1e3
                        if isinstance(batch, FeedbackBatch):
                            batches.append(batch)
                        self._validate_feedback(source, batch, started)
                        currents.update({name: s.current_a for name, s in batch.samples.items()})
                    self._trip = self._protection.update(currents, now=time.monotonic())
                    if self._trip is not None:
                        self._fault(f"{self._trip.cause}: {self._trip.motor}")
                    else:
                        if self._control is not None:
                            self._control.observe(tuple(batches))
                        observation_done = time.perf_counter()
                        self.timing_ms["robot_observation"] = (
                            observation_done - observation_started
                        ) * 1e3
                        if self._phase is HostPhase.STARTING:
                            self._phase = HostPhase.READY
                        if poll_command is not None:
                            command = poll_command()
                            if command is not None and not isinstance(command, CommandSubmission):
                                raise TypeError("Expected CommandSubmission from poller")
                        if command is not None and command.validate is not None:
                            try:
                                command.validate()
                            except CommandRejectedError as exc:
                                command_error = str(exc)
                                command = None
                        if (
                            self._phase is not HostPhase.FAULT
                            and command is not None
                            and self._owner.accept(command.identity)
                        ):
                            command.write()
                            applied = True
                        if applied:
                            self._last_command_s = time.monotonic()
                            self._phase = HostPhase.ACTIVE
                        # Only a validated, accepted and written command renews inactivity.
                        expired = self._expire_command(tuple(batches))
                        if (
                            self._control is not None
                            and self._phase is not HostPhase.FAULT
                            and not expired
                        ):
                            self._control.supervise()
                        self.timing_ms["robot_action"] = (
                            time.perf_counter() - observation_done
                        ) * 1e3
            except BaseException as exc:
                applied = False
                if self._phase is not HostPhase.FAULT:
                    self._fault(f"Host cycle failed: {type(exc).__name__}: {exc}")
                if not isinstance(exc, Exception):
                    raise
            return CycleResult(tuple(batches), applied, self.status, command_error)
        finally:
            self._busy = False

    def close(self) -> HostStatus:
        self._enter()
        try:
            if self._phase in (HostPhase.CLOSED, HostPhase.FAULT):
                return self.status
            self._phase = HostPhase.CLOSED
            self._shutdown()
            return self.status
        finally:
            self._busy = False
