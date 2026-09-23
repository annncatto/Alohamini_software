# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Register layout and command encoding adapted from LeRobot FeetechMotorsBus.
"""Host-owned STS bus with explicit connection, preparation and torque enable."""

import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from types import MappingProxyType

from alohamini._validation import identifier
from alohamini.calibration import EncoderCalibration
from alohamini.calibration.servo import MotorCalibration
from alohamini.hardware.feedback import FeedbackBatch, MotorFeedback
from alohamini.hardware.feetech import _FIELDS, _START, FeetechFeedbackReader, _PacketPort
from alohamini.model import ActuatorSpec

_MODEL_NUMBERS = {"sts3215": 777, "sts3250": 2825, "sts3095": 2569}
_IO_BUDGET_S = 0.008
# LeRobot's Feetech PortHandler patch allows 50 ms plus packet wire time.
# Acknowledged register transactions retain the source driver's 50 ms allowance.
# Group writes use this bounded USB allowance too; cyclic feedback and its
# explicitly bounded fallback retain their shorter deadlines.
_REGISTER_IO_BUDGET_S = 0.050
_REGISTER_ATTEMPTS = 4  # Source driver: initial attempt plus three transport retries.
_GOAL_POSITION = 42
_GOAL_VELOCITY = 46
_TORQUE_ENABLE = 40


class CalibrationMismatchError(ValueError):
    """Installed motor calibration differs from the explicitly selected file."""


class DeviceOperationError(ConnectionError):
    """Some device operations failed; errors retain every attempted motor's result."""

    def __init__(self, errors: Mapping[str, str], *, context: str | None = None) -> None:
        self.errors = dict(errors)
        detail = "; ".join(f"{name}: {error}" for name, error in errors.items())
        super().__init__(f"{context}: {detail}" if context else detail)


def _finish_cleanup(errors: Mapping[str, str], interrupts: Sequence[BaseException]) -> None:
    if interrupts:
        raise interrupts[0]
    if errors:
        raise DeviceOperationError(errors)


class FeetechBusDevice:
    """Concrete HostDevice with an exclusive 1 Mbaud, 8N1 serial connection.

    Position joints require explicit device calibration and limits. Velocity
    limits are signed encoder ticks/s magnitudes, supplied by the robot layer
    (existing base: 3000; lift: 1300), not inferred from model names.

    connect checks identities and inspects existing modes/Profile without writes.
    Unconfigured buses can be read but cannot receive motion targets. prepare
    requires torque off, verifies installed calibration and initializes settings.
    Empty position/velocity maps select unconfigured calibration access; after
    calibration, reconnect with the saved parameters before preparing for motion.
    enable_torque replaces stale goals before enabling; it never homes the lift.
    write_targets is internal actuator I/O: the Host must apply current,
    collision, gripper and lift protection before invoking it.
    """

    def __init__(
        self,
        port: str,
        actuators: Sequence[ActuatorSpec],
        *,
        position_calibrations: Mapping[str, EncoderCalibration],
        velocity_limits: Mapping[str, int],
    ) -> None:
        if not isinstance(port, str) or not port or "\x00" in port:
            raise ValueError("Provide a serial device path")
        self._actuators = tuple(actuators)
        names, ids, buses = set(), set(), set()
        for motor in self._actuators:
            if not isinstance(motor, ActuatorSpec):
                raise TypeError("Expected ActuatorSpec values")
            identifier(motor.name, "motor name")
            identifier(motor.bus, "bus name")
            if motor.motor_model not in _MODEL_NUMBERS:
                raise ValueError("Unsupported STS motor model")
            if type(motor.motor_id) is not int or not 1 <= motor.motor_id <= 253:
                raise ValueError("Invalid motor ID")
            if motor.name in names or motor.motor_id in ids:
                raise ValueError("Duplicate motor name or ID")
            names.add(motor.name)
            ids.add(motor.motor_id)
            buses.add(motor.bus)
        # SDK packets allow 250 bytes: 8-byte header plus 3 bytes per 16-bit target.
        if not names or len(buses) != 1 or len(names) > 80:
            raise ValueError("Provide 1 to 80 motors on one bus")
        if set(position_calibrations) & set(velocity_limits):
            raise ValueError("A motor cannot be both position and velocity controlled")
        if (position_calibrations or velocity_limits) and (
            set(position_calibrations) | set(velocity_limits) != names
        ):
            raise ValueError("Every motor needs an explicit control mode")
        for calibration in position_calibrations.values():
            if not isinstance(calibration, EncoderCalibration):
                raise TypeError("Expected EncoderCalibration")
            if calibration.ticks_per_revolution != 4096 or calibration.position_min_rad is None:
                raise ValueError(
                    "STS position commands require 4096 ticks and explicit joint limits"
                )
        for limit in velocity_limits.values():
            if type(limit) is not int or not 1 <= limit <= 32767:
                raise ValueError("Velocity limits must fit the signed STS register")
        self._path = port
        self._positions = dict(position_calibrations)
        self._velocities = dict(velocity_limits)
        self._position_io_budget_s = (
            _IO_BUDGET_S if self._positions or self._velocities else _REGISTER_IO_BUDGET_S
        )
        self._ids = {motor.name: motor.motor_id for motor in self._actuators}
        self._serial = None
        self._reader = None
        self._handler = None
        self._verified: set[str] = set()
        self._modes: set[str] = set()
        self._ready = False
        self._prepared = False
        self._calibrating = False
        self._passive = False
        self._released_velocities: set[str] = set()
        self._feedback_retry: dict[str, tuple[float, float]] = {}
        self._used = False
        self._thread = threading.get_ident()

    @property
    def actuators(self) -> tuple[ActuatorSpec, ...]:
        return self._actuators

    @property
    def position_calibrations(self) -> Mapping[str, EncoderCalibration]:
        return MappingProxyType(self._positions)

    @property
    def velocity_limits(self) -> Mapping[str, int]:
        return MappingProxyType(self._velocities)

    def _check_thread(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("Serial operations require the owning Host thread")

    def _packet_port(self, deadline: float | None = None, *, register: bool = False) -> _PacketPort:
        self._check_thread()
        if self._serial is None or not self._serial.is_open:
            raise ConnectionError("Serial device is not open")
        budget = _REGISTER_IO_BUDGET_S if register else _IO_BUDGET_S
        limit = time.monotonic() + budget
        return _PacketPort(
            self._serial,
            limit if deadline is None else min(limit, deadline),
            write_timeout_s=budget if register else None,
        )

    def _read_register(
        self, name: str, address: int, width: int, *, deadline: float | None = None
    ) -> int:
        data = self._register_transaction(name, address, width, deadline=deadline)
        return int.from_bytes(data, "little")

    def _write_register(self, name: str, address: int, width: int, value: int) -> None:
        self._register_transaction(name, address, width, value=value)

    def _register_transaction(
        self,
        name: str,
        address: int,
        width: int,
        *,
        value: int | None = None,
        deadline: float | None = None,
    ) -> bytes:
        from scservo_sdk.scservo_def import COMM_RX_CORRUPT, COMM_RX_TIMEOUT, COMM_TX_FAIL
        from serial import SerialTimeoutException

        # Repeat only RAM target/Profile writes, or torque OFF. Never blindly
        # repeat EEPROM, EEPROM lock, torque enable, or calibration commands.
        repeatable = (
            value is None
            or address in (41, 42, 44, 46, 85)
            or (address == _TORQUE_ENABLE and value == 0)
        )
        attempts = _REGISTER_ATTEMPTS if deadline is None and repeatable else 1
        operation = f"Read {address}" if value is None else f"Write {address}={value}"
        context = f"{self._path} {name} (ID {self._ids[name]}): {operation}"
        for attempt in range(attempts):
            port = self._packet_port(deadline, register=True)
            result, error = 0, 0
            try:
                if value is None:
                    result = self._handler.readTx(port, self._ids[name], address, width)
                else:
                    result = self._handler.writeTxOnly(
                        port, self._ids[name], address, width, list(value.to_bytes(width, "little"))
                    )
                expected = width if value is None else 0
                while result == 0:
                    packet, result = self._handler.rxPacket(port)
                    if result != 0:
                        break
                    if packet[2] != self._ids[name]:
                        continue
                    error = packet[4]
                    # A voltage-only alarm does not invalidate proof that torque
                    # is OFF. Accept only a complete one-byte zero readback;
                    # torque ON, other registers and combined faults stay strict.
                    if (
                        error == 0x01
                        and value is None
                        and address == _TORQUE_ENABLE
                        and width == 1
                        and len(packet) == 7
                        and packet[3] == 3
                        and packet[5] == 0
                    ):
                        return b"\x00"
                    if error:
                        break
                    # An old write ACK must not be decoded as a one-byte read
                    # (the SDK readTxRx otherwise slices its checksum as data).
                    if len(packet) == expected + 6 and packet[3] == expected + 2:
                        return bytes(packet[5:-1])
                reason = f"transport={result}, servo_error=0x{error:02x}"
                if error:
                    reason += f" ({self._handler.getRxPacketError(error)})"
                retry = not error and result in (COMM_RX_TIMEOUT, COMM_RX_CORRUPT, COMM_TX_FAIL)
            except OSError as exc:
                # Disconnected/closed ports and other OS errors are not transient
                # packet loss; do not spend retries on them.
                if not isinstance(exc, (TimeoutError, SerialTimeoutException)):
                    raise ConnectionError(f"{context}: {exc}") from exc
                reason, retry = str(exc), True
            finally:
                port.is_using = False
            if not retry or attempt + 1 == attempts:
                raise ConnectionError(f"{context}: {reason} (attempts={attempt + 1})")
        raise AssertionError("Register transaction exhausted without a result")

    def connect(self, host_session_id: str) -> None:
        self._connect(host_session_id, passive=False)

    def connect_passive(self, host_session_id: str) -> None:
        """Verify a leader bus without reading follower motion Profile/PID fields."""
        if self._velocities or not self._positions:
            raise ValueError("Passive leaders must contain only calibrated position joints")
        self._connect(host_session_id, passive=True)

    def _connect(self, host_session_id: str, *, passive: bool) -> None:
        self._check_thread()
        identifier(host_session_id, "host_session_id")
        if self._used:
            raise RuntimeError("Create a new bus device for each Host session")
        self._used = True
        from scservo_sdk.protocol_packet_handler import protocol_packet_handler
        from serial import Serial

        self._handler = protocol_packet_handler()
        # Retain the object before open, so partial connection cleanup can close it.
        self._serial = Serial(
            port=None,
            baudrate=1_000_000,
            timeout=0,
            write_timeout=0.005,
            bytesize=8,
            parity="N",
            stopbits=1,
            exclusive=True,
        )
        self._serial.port = self._path
        try:
            self._serial.open()
            self._reader = FeetechFeedbackReader(
                self._serial, self._actuators, clock_id=host_session_id
            )
            profile_valid = True
            for motor in self._actuators:
                name = motor.name
                if self._read_register(name, 3, 2) != _MODEL_NUMBERS[motor.motor_model]:
                    raise ConnectionError(f"Motor model mismatch: {name}")
                self._verified.add(name)
                if passive or (not self._positions and not self._velocities):
                    continue  # Passive/unconfigured access has no motion readiness.
                mode = 0 if name in self._positions else 1
                if self._read_register(name, 33, 1) == mode:
                    self._modes.add(name)
                else:
                    continue
                if mode == 0:
                    for address, width, expected in (
                        (46, 2, 2000),
                        (41, 1, 100),
                        (21, 1, 16),
                        (23, 1, 0),
                        (22, 1, 32),
                    ):
                        if self._read_register(name, address, width) != expected:
                            profile_valid = False
            # Source Feetech handshake applies to followers and calibration too.
            versions = {
                name: (self._read_register(name, 0, 1), self._read_register(name, 1, 1))
                for name in self._ids
            }
            if len(set(versions.values())) != 1:
                raise ConnectionError(f"{self._path}: Motor firmware versions differ: {versions}")
            self._ready = not passive and profile_valid and self._modes == self._ids.keys()
        except BaseException:
            # HostSupervisor owns stop/disable/close sequencing, including partial open.
            self._ready = False
            raise

    def read_feedback(self) -> FeedbackBatch:
        """Retain healthy blocks; retry failed optional telemetry with backoff.

        Adapted from AlohaMini._read_current_feedback: a failed full block must
        not make temperature/load availability a condition for control. Missing
        blocks use fresh critical-register reads, never cached values. Servo
        error flags remain fatal and cannot be cleared by a successful fallback.
        """
        self._check_thread()
        if self._reader is None or self._verified != self._ids.keys():
            raise ConnectionError("Bus identities have not passed connection validation")
        now = time.monotonic()
        retry = self._feedback_retry
        attempted = [name for name in self._ids if name not in retry]
        due = [name for name in self._ids if name in retry and now >= retry[name][0]]
        if due:
            attempted.append(min(due, key=lambda name: retry[name][0]))
        batch = self._reader.read(attempted)
        samples, failures = dict(batch.samples), dict(batch.failures)
        for name in attempted:
            if name in samples:  # Includes faulted packets: never mask their errors.
                retry.pop(name, None)
            else:
                delay = min(retry.get(name, (0.0, 0.5))[1] * 2, 30.0)
                retry[name] = (batch.received_s + delay, delay)
        missing = [name for name in self._ids if name not in samples]
        # One shared fallback budget, not N independent retry timeouts.
        deadline = time.monotonic() + _IO_BUDGET_S + len(missing) * 0.001
        fields = {name: (offset + _START, width, sign) for name, offset, width, sign in _FIELDS}
        for name in missing:
            required = ["current_raw"]
            if name in self._positions or name == "lift_axis":
                required.append("position_raw")
            if name in self._velocities:
                required.append("velocity_raw")
            if name == "lift_axis":
                required.append("moving")
            values = {}
            try:
                for field in required:
                    address, width, sign = fields[field]
                    value = self._read_register(name, address, width, deadline=deadline)
                    if sign is not None:
                        value = (value & ((1 << sign) - 1)) * (-1 if value & (1 << sign) else 1)
                    values[field] = value
                samples[name] = MotorFeedback(
                    values,
                    current_a=values["current_raw"] * 0.0065,
                    field_errors={
                        field: "Full feedback block unavailable"
                        for field in fields
                        if field not in values
                    },
                )
                failures.pop(name, None)
            except (OSError, ValueError) as exc:
                failures[name] = f"Critical feedback unavailable: {exc}"
        return replace(batch, received_s=time.monotonic(), samples=samples, failures=failures)

    def _write_verified(self, name: str, address: int, width: int, value: int) -> None:
        self._write_register(name, address, width, value)
        if self._read_register(name, address, width) != value:
            raise ConnectionError(f"Register write not accepted: {name}, address {address}")

    def verify_calibration(self, calibrations: Mapping[str, MotorCalibration]) -> None:
        """Check torque-off and installed coordinates without configuring motion."""
        self._check_thread()
        self._prepared = self._ready = False
        self._passive = False
        if not self._positions and not self._velocities:
            raise RuntimeError("Reconnect with installed calibration before enabling motion")
        if self._verified != self._ids.keys() or calibrations.keys() != self._ids.keys():
            raise ValueError("Preparation requires every verified motor's calibration")
        for name, calibration in calibrations.items():
            if not isinstance(calibration, MotorCalibration) or calibration.id != self._ids[name]:
                raise ValueError(f"Calibration identity mismatch: {name}")
            if (
                name in self._positions
                and calibration.encoder_calibration() != self._positions[name]
            ):
                raise ValueError(f"Encoder coordinate mismatch: {name}")
            if self._read_register(name, _TORQUE_ENABLE, 1) != 0:
                raise ConnectionError(f"Disable every motor before preparation: {name}")
            for address, expected in (
                (31, calibration.offset_register),
                (9, calibration.range_min),
                (11, calibration.range_max),
            ):
                actual = self._read_register(name, address, 2)
                if actual != expected:
                    raise CalibrationMismatchError(
                        f"Installed calibration mismatch: {name}, register {address} "
                        f"(raw actual={actual}, expected={expected})"
                    )

    def prepare_passive(self, calibrations: Mapping[str, MotorCalibration]) -> None:
        """Restore SOLeader sampling configuration, preserving installed calibration.

        Both leaders must have torque disabled first. Changed return delay,
        single-turn Phase and mode fields are written and relocked; offsets,
        ranges, PID, protection settings and motion targets are never rewritten.
        """
        if self._velocities:
            raise ValueError("Passive leaders must contain only position joints")
        self.verify_calibration(calibrations)
        self._modes.clear()
        for motor in self._actuators:
            name = motor.name
            settings = [(7, 1, 0), (33, 1, 0)]
            if motor.motor_model == "sts3215":
                settings.append((18, 1, self._read_register(name, 18, 1) & ~0x10))
            changed = [s for s in settings if self._read_register(name, s[0], s[1]) != s[2]]
            if changed:
                self._calibration_write(name, changed)
            # Match SOLeader.configure_motors: check the write ACK for these
            # passive-arm Profile fields, without an additional register read.
            self._write_register(name, 85, 1, 254)
            self._write_register(name, 41, 1, 254)
            self._modes.add(name)
        self._passive = True

    def read_positions(self, names: Sequence[str] | None = None) -> dict[str, int]:
        """Return fresh positions; passive sampling tolerates only the voltage flag.

        The source GroupSyncRead discarded servo error bytes. For torque-off
        calibration and verified passive leaders, a voltage-only flag does not
        invalidate encoder data. Other faults and transport failures still fail;
        full feedback and acknowledged register operations remain strict.
        """
        self._check_thread()
        if self._reader is None or self._verified != self._ids.keys():
            raise ConnectionError("Connect and verify the bus before reading positions")
        selected = list(self._ids) if names is None else list(names)
        if len(set(selected)) != len(selected) or not set(selected) <= self._ids.keys():
            raise ValueError("Position selection must contain distinct configured motors")
        passive = self._calibrating or self._passive
        # SOLeader.get_action used sync_read's 50 ms allowance and three retries.
        # Only torque-off leader/calibration reads use that policy. Host feedback
        # and position holds retain their separate, shorter cyclic budgets.
        timeout = _REGISTER_IO_BUDGET_S if passive else self._position_io_budget_s
        attempts = _REGISTER_ATTEMPTS if passive else 1
        context = (
            f"{self._path} [{self._actuators[0].bus}] SyncRead Present_Position (56), "
            f"IDs {[self._ids[name] for name in selected]}"
        )
        for attempt in range(attempts):
            try:
                batch = self._reader.read_positions(selected, timeout_s=timeout)
            except (OSError, ValueError) as exc:
                raise ConnectionError(f"{context}: {type(exc).__name__}: {exc}") from exc
            failures = {
                name: reason
                for name, reason in batch.failures.items()
                if not (
                    passive and name in batch.samples and batch.samples[name].packet_error == 0x01
                )
            }
            servo_fault = any(
                sample.packet_error and not (passive and sample.packet_error == 0x01)
                for sample in batch.samples.values()
            )
            if not failures:
                break
            if servo_fault or attempt + 1 == attempts:
                raise DeviceOperationError(
                    failures, context=f"{context}, after {attempt + 1} attempt(s)"
                )
            # Query the entire group again. Never merge positions from different
            # attempts or turn partial replies into a complete leader action.
        positions = {}
        for name in selected:
            sample = batch.samples.get(name)
            value = None if sample is None else sample.registers.get("position_raw")
            if (
                sample is None
                or (sample.packet_error and not (passive and sample.packet_error == 0x01))
                or type(value) is not int
                or not 0 <= value < 4096
            ):
                raise ConnectionError(f"Missing or invalid position: {name}")
            positions[name] = value
        return positions

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read verified motors' EEPROM offsets/ranges without writing registers.

        Software drive_mode is not stored in these registers; callers retain it
        from their installed JSON rather than infer inversion from the servo.
        """
        self._check_thread()
        if self._reader is None or self._verified != self._ids.keys():
            raise ConnectionError("Connect and verify every motor before reading calibration")
        result = {}
        for name, motor_id in self._ids.items():
            raw_offset = self._read_register(name, 31, 2)
            if raw_offset & ~0xFFF:
                raise ValueError(f"Invalid STS homing offset: {name}")
            offset = raw_offset & 0x7FF
            result[name] = MotorCalibration(
                motor_id,
                0,
                -offset if raw_offset & 0x800 else offset,
                self._read_register(name, 9, 2),
                self._read_register(name, 11, 2),
            )
        return result

    def _calibration_write(self, name: str, settings: Sequence[tuple[int, int, int]]) -> None:
        """Changed EEPROM fields only, each acknowledged and read back, then locked."""
        primary = None
        try:
            changed = [s for s in settings if self._read_register(name, s[0], s[1]) != s[2]]
            if changed:
                self._write_verified(name, 55, 1, 0)
                for address, width, value in changed:
                    self._write_verified(name, address, width, value)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            try:
                self._write_verified(name, 55, 1, 1)
            except Exception as exc:
                if primary is None:
                    raise
                if not isinstance(primary, Exception):
                    primary.add_note(f"Relock failed: {exc}")
                else:
                    raise DeviceOperationError(
                        {"register update": str(primary), "relock": str(exc)}
                    ) from primary

    @contextmanager
    def calibration_session(self):
        """Explicit, torque-off calibration transaction; no Host motion configuration.

        Restore the original calibration/mode on failure, attempting every motor.
        Torque remains disabled, including after success; callers must reconnect.
        """
        self._check_thread()
        if self._positions or self._velocities or self._calibrating:
            raise RuntimeError("Use a separate unconfigured bus for calibration")
        with self._calibration_transaction():
            yield self

    @contextmanager
    def calibration_restore_session(self):
        """Restore startup calibration with rollback; require preparation afterwards."""
        self._check_thread()
        self._ready = self._prepared = self._passive = False
        with self._calibration_transaction():
            yield self

    @contextmanager
    def _calibration_transaction(self):
        self._check_thread()
        if self._calibrating:
            raise RuntimeError("Calibration transaction already active")
        if self._verified != self._ids.keys():
            raise ConnectionError("Verify every motor before calibration")
        fields = ((7, 1), (9, 2), (11, 2), (18, 1), (31, 2), (33, 1))
        original = {}
        for name in self._ids:
            if self._read_register(name, _TORQUE_ENABLE, 1) != 0:
                raise ConnectionError("Disable both buses before calibration")
            original[name] = [(a, w, self._read_register(name, a, w)) for a, w in fields]
        self._calibrating = True
        try:
            yield self
        except BaseException as exc:
            errors = {}
            for name, settings in original.items():
                try:
                    self._calibration_write(name, settings)
                except BaseException as restore_error:
                    errors[name] = f"{type(restore_error).__name__}: {restore_error}"
            if errors:
                raise DeviceOperationError(
                    {"calibration": str(exc), **{f"restore {k}": v for k, v in errors.items()}}
                ) from exc
            raise
        finally:
            self._calibrating = False

    def configure_calibration(self, position_names: Sequence[str]) -> None:
        """Source position mode and STS3215 single-turn feedback; velocity axes stay off."""
        if not self._calibrating:
            raise RuntimeError("Open a torque-off calibration session first")
        names = set(position_names)
        if not names <= self._ids.keys():
            raise ValueError("Unknown calibration motor")
        for motor in self._actuators:
            settings = [(7, 1, 0), (33, 1, 0 if motor.name in names else 1)]
            if motor.motor_model == "sts3215":
                settings.append((18, 1, self._read_register(motor.name, 18, 1) & ~0x10))
            self._calibration_write(motor.name, settings)

    def write_calibration(self, calibrations: Mapping[str, MotorCalibration]) -> None:
        """Write source Homing_Offset / Min_Position_Limit / Max_Position_Limit fields."""
        if not self._calibrating:
            raise RuntimeError("Open a torque-off calibration session first")
        if not calibrations or not calibrations.keys() <= self._ids.keys():
            raise ValueError("Expected known calibration motors")
        for name, entry in calibrations.items():
            if not isinstance(entry, MotorCalibration) or entry.id != self._ids[name]:
                raise ValueError(f"Calibration identity mismatch: {name}")
        for name, entry in calibrations.items():
            self._calibration_write(
                name,
                [(31, 2, entry.offset_register), (9, 2, entry.range_min), (11, 2, entry.range_max)],
            )

    def set_half_turn_homings(self, names: Sequence[str]) -> dict[str, int]:
        """Source reset -> raw position -> position minus 2047 -> EEPROM offset."""
        if not names or len(set(names)) != len(names) or not set(names) <= self._ids.keys():
            raise ValueError("Expected unique configured calibration motors")
        reset = {name: MotorCalibration(self._ids[name], 0, 0, 0, 4095) for name in names}
        self.write_calibration(reset)
        positions = self.read_positions(names)
        offsets = {name: positions[name] - int(4095 / 2) for name in names}
        # Validate the entire batch before writing: +2048 cannot fit sign bit 11.
        entries = {
            name: MotorCalibration(self._ids[name], 0, offset, 0, 4095)
            for name, offset in offsets.items()
        }
        self.write_calibration(entries)
        return offsets

    def prepare(self, calibrations: Mapping[str, MotorCalibration]) -> None:
        """Verify calibration without rewriting it; configure only while torque is off.

        The robot startup must disable BOTH buses before preparing either one.
        EEPROM writes are limited to changed mode/PID/return-delay settings;
        current, overload and unloading protections are left untouched.
        """
        self.verify_calibration(calibrations)
        self._modes.clear()
        for motor in self._actuators:
            name = motor.name
            position = name in self._positions
            settings = [(7, 0), (33, 0 if position else 1)]
            if motor.motor_model == "sts3215":
                settings.append((18, self._read_register(name, 18, 1) & ~0x10))
            if position:
                settings.extend(((21, 16), (23, 0), (22, 32)))
            self._calibration_write(name, [(address, 1, value) for address, value in settings])
            self._modes.add(name)
            self._write_verified(name, 85, 1, 254)
            self._write_verified(name, 41, 1, 100 if position else 254)
            self._write_verified(name, _GOAL_VELOCITY, 2, 2000 if position else 0)
            self._write_verified(name, 44, 2, 0)
        self._prepared = self._ready = True

    def enable_torque(self) -> None:
        """Refresh and read back every safe target before enabling any motor.

        The caller must prepare all buses first and clean up all of them after
        any failure. Sequential enabling is not an atomic multi-bus operation.
        """
        self._check_thread()
        if not self._prepared or not self._ready:
            raise RuntimeError("Prepare the bus before enabling torque")
        self.stop()
        for name in self._ids:
            self._write_verified(name, _TORQUE_ENABLE, 1, 1)
            self._released_velocities.discard(name)

    def release_velocity(self, name: str) -> None:
        """Zero and unload one verified velocity axis without disabling its bus."""
        self._check_thread()
        if name not in self._velocities or name not in self._verified or name not in self._modes:
            raise ValueError("Release requires a verified velocity-controlled motor")
        errors, interrupts = {}, []
        # Attempt torque-off even if the zero-speed write fails.
        for address, width in ((_GOAL_VELOCITY, 2), (_TORQUE_ENABLE, 1)):
            try:
                self._write_verified(name, address, width, 0)
            except BaseException as exc:
                errors[f"{name} register {address}"] = f"{type(exc).__name__}: {exc}"
                if not isinstance(exc, Exception):
                    interrupts.append(exc)
        _finish_cleanup(errors, interrupts)
        self._released_velocities.add(name)

    def _sync_write(self, address: int, values: Mapping[str, int]) -> None:
        if not values:
            return
        params = []
        for name, value in values.items():
            params.extend([self._ids[name], *value.to_bytes(2, "little")])
        # USB scheduling can exceed the serial port's 5 ms feedback default.
        # Allow the same 50 ms ceiling as register I/O, without a fixed delay,
        # ACK wait, or retry of a possibly partially transmitted motion packet.
        port = self._packet_port(register=True)
        register = {_GOAL_POSITION: "Goal_Position", _GOAL_VELOCITY: "Goal_Velocity"}[address]
        motors = ", ".join(f"{name} (ID {self._ids[name]})" for name in values)
        context = f"{self._path} SyncWrite {register} ({address}) [{motors}]"
        try:
            result = self._handler.syncWriteTxOnly(port, address, 2, params, len(params))
        except OSError as exc:
            raise ConnectionError(f"{context}: {type(exc).__name__}: {exc}") from exc
        finally:
            port.is_using = False
        if result != 0:
            raise ConnectionError(f"{context}: transport={result}")

    def write_targets(
        self, *, positions_rad: Mapping[str, float], velocities_raw: Mapping[str, int]
    ) -> None:
        """Validate the entire submission before I/O; group writes have no servo ACK.

        Missing targets are not synthesized. Velocity conversion/saturation and
        task-level safety belong to the robot layer, not this register transport.
        """
        self._check_thread()
        if not self._ready:
            raise ConnectionError("Bus has not passed connection validation")
        if not positions_rad and not velocities_raw:
            raise ValueError("Empty motor command")
        if not positions_rad.keys() <= self._positions.keys():
            raise ValueError("Unknown position-controlled motor")
        if not velocities_raw.keys() <= self._velocities.keys():
            raise ValueError("Unknown velocity-controlled motor")
        positions = {
            name: self._positions[name].position_to_tick(value)
            for name, value in positions_rad.items()
        }
        velocities = {}
        for name, value in velocities_raw.items():
            if type(value) is not int or abs(value) > self._velocities[name]:
                raise ValueError(f"Velocity target exceeds configured limit: {name}")
            velocities[name] = abs(value) | (0x8000 if value < 0 else 0)
        for name in tuple(velocities):
            if name not in self._released_velocities:
                continue
            if velocities[name] == 0:
                # Do not rewrite an idle goal after explicitly unloading the axis.
                del velocities[name]
                continue
            self._write_verified(name, _GOAL_VELOCITY, 2, 0)
            self._write_verified(name, _TORQUE_ENABLE, 1, 1)
            self._released_velocities.remove(name)
        self._sync_write(_GOAL_POSITION, positions)
        self._sync_write(_GOAL_VELOCITY, velocities)

    def stop_motion(self, feedback: FeedbackBatch) -> None:
        """Watchdog: source-cycle arm hold, grouped base zero and lift zero.

        Mirrors alohamini_host.py hold_action / AlohaMini.stop_motion. This is
        not the maintenance stop/readback used for startup and fault cleanup.
        Group writes confirm transmission, not servo acceptance or standstill.
        """
        self._check_thread()
        if self._modes != self._ids.keys() or feedback.source_id != self._actuators[0].bus:
            raise ConnectionError("Watchdog stop requires verified modes and matching feedback")
        positions = {}
        for name in self._positions:
            sample = feedback.samples.get(name)
            tick = None if sample is None else sample.registers.get("position_raw")
            if (
                sample is None
                or sample.packet_error
                or name in feedback.failures
                or type(tick) is not int
                or not 0 <= tick < 4096
            ):
                raise ConnectionError(f"Invalid watchdog position feedback: {name}")
            positions[name] = tick
        velocities = {name: 0 for name in self._velocities if name not in self._released_velocities}
        lift = velocities.pop("lift_axis", None)
        errors, interrupts = {}, []
        operations = [
            ("arms", lambda: self._sync_write(_GOAL_POSITION, positions)),
            ("base", lambda: self._sync_write(_GOAL_VELOCITY, velocities)),
        ]
        if lift is not None:
            operations.append(
                ("lift_axis", lambda: self._write_register("lift_axis", _GOAL_VELOCITY, 2, 0))
            )
        for name, operation in operations:
            try:
                operation()
            except BaseException as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
                if not isinstance(exc, Exception):
                    interrupts.append(exc)
        _finish_cleanup(errors, interrupts)

    def stop_velocity(self) -> None:
        """Zero base/lift targets without holding overloaded position joints."""
        self._check_thread()
        errors = {}
        interrupts = []
        for name in self._velocities:
            try:
                if name not in self._modes:
                    if name in self._verified and self._read_register(name, _TORQUE_ENABLE, 1) == 0:
                        continue  # Cancelled startup: unloaded axis needs no motion target.
                    raise ConnectionError("Operating mode not verified; no target written")
                if name in self._released_velocities:
                    self.release_velocity(name)
                    continue
                self._write_register(name, _GOAL_VELOCITY, 2, 0)
                if self._read_register(name, _GOAL_VELOCITY, 2) != 0:
                    raise ConnectionError("Velocity stop target was not accepted")
            except BaseException as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
                if not isinstance(exc, Exception):
                    interrupts.append(exc)
        _finish_cleanup(errors, interrupts)

    def stop(self) -> None:
        """Zero velocity axes, then replace position targets using a fresh read.

        Each target is read back. This confirms the target register, not physical
        standstill. Missing/faulted positions are never replaced with cached goals.
        """
        self._check_thread()
        errors, interrupts = {}, []
        try:
            self.stop_velocity()
        except DeviceOperationError as exc:
            errors.update(exc.errors)
        except BaseException as exc:
            errors["velocity stop"] = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, Exception):
                interrupts.append(exc)
        if self._positions:
            try:
                batch = (
                    self._reader.read_positions(
                        [name for name in self._positions if name in self._modes],
                        timeout_s=_REGISTER_IO_BUDGET_S,
                    )
                    if self._reader is not None
                    else None
                )
            except BaseException as exc:
                batch = None
                errors["feedback"] = f"{type(exc).__name__}: {exc}"
                if not isinstance(exc, Exception):
                    interrupts.append(exc)
            for name in self._positions:
                try:
                    if name not in self._modes or batch is None:
                        if (
                            name in self._verified
                            and self._read_register(name, _TORQUE_ENABLE, 1) == 0
                        ):
                            continue
                        raise ConnectionError(
                            "No verified mode and fresh feedback for position hold"
                        )
                    sample = batch.samples.get(name)
                    if sample is not None and (sample.packet_error or name in batch.failures):
                        raise ConnectionError("Position feedback missing or faulted")
                    # A missing group reply may still have a fresh single-register
                    # response. Never replace servo fault flags or use cached goals.
                    position = (
                        self._read_register(name, _START, 2)
                        if sample is None
                        else sample.registers.get("position_raw")
                    )
                    if type(position) is not int or not 0 <= position < 4096:
                        raise ConnectionError("Invalid single-turn position feedback")
                    self._write_register(name, _GOAL_POSITION, 2, position)
                    if self._read_register(name, _GOAL_POSITION, 2) != position:
                        raise ConnectionError("Position hold target was not accepted")
                except BaseException as exc:
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    if not isinstance(exc, Exception):
                        interrupts.append(exc)
        _finish_cleanup(errors, interrupts)

    def disable_torque(self) -> None:
        """Attempt each verified motor; never unlock EEPROM or enable torque here."""
        self._check_thread()
        self._ready = False
        self._prepared = False
        self._passive = False
        errors = {}
        interrupts = []
        for name in self._ids:
            if name not in self._verified:
                errors[name] = "Motor identity not verified; no register written"
                continue
            try:
                self._write_register(name, _TORQUE_ENABLE, 1, 0)
                if self._read_register(name, _TORQUE_ENABLE, 1) != 0:
                    raise ConnectionError("Torque disable was not accepted")
            except BaseException as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
                if not isinstance(exc, Exception):
                    interrupts.append(exc)
        _finish_cleanup(errors, interrupts)

    def close(self) -> None:
        """Release the port only; HostSupervisor must stop and disable first."""
        self._check_thread()
        self._ready = False
        self._passive = False
        if self._serial is not None:
            self._serial.close()
