# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Feedback layout/sign decoding adapted from LeRobot FeetechMotorsBus.
"""Read-only STS feedback, using a caller-owned nonblocking serial connection."""

import threading
import time
from collections.abc import Sequence
from typing import Protocol

from alohamini._validation import finite_number, identifier
from alohamini.hardware.feedback import FeedbackBatch, MotorFeedback
from alohamini.model import ActuatorSpec

_START = 56
_SIZE = 15
_MODELS = frozenset({"sts3215", "sts3250", "sts3095"})
# name, offset within the block, width, sign-magnitude bit (if encoded)
_FIELDS = (
    ("position_raw", 0, 2, 15),
    ("velocity_raw", 2, 2, 15),
    ("load_raw", 4, 2, 10),
    ("voltage_raw", 6, 1, None),
    ("temperature_raw", 7, 1, None),
    ("status_raw", 9, 1, None),
    ("moving", 10, 1, None),
    ("current_raw", 13, 2, None),
)


def decode_feedback(block: bytes, *, packet_error: int = 0) -> MotorFeedback:
    """Decode the shared STS block; current is unsigned, load is not joint torque."""
    if not isinstance(block, bytes) or len(block) != _SIZE:
        raise ValueError("Expected exactly 15 feedback bytes")
    values = {}
    for name, offset, width, sign_bit in _FIELDS:
        value = int.from_bytes(block[offset : offset + width], "little")
        if sign_bit is not None:
            value = (value & ((1 << sign_bit) - 1)) * (-1 if value & (1 << sign_bit) else 1)
        values[name] = value
    return MotorFeedback(values, packet_error, current_a=values["current_raw"] * 0.0065)


class SerialConnection(Protocol):
    """Subset of an already-open pyserial connection, owned by one Host thread."""

    baudrate: int
    bytesize: int
    parity: str
    stopbits: float
    timeout: float | None
    write_timeout: float | None
    is_open: bool

    def read(self, size: int) -> bytes: ...
    def write(self, data: bytes) -> int: ...
    def reset_input_buffer(self) -> None: ...


class _PacketPort:
    """SDK adapter with one monotonic deadline; no blocking flush or timeout reset."""

    def __init__(self, serial: SerialConnection, deadline: float) -> None:
        self.serial = serial
        self.deadline = deadline
        self.is_using = False

    def _check_deadline(self) -> None:
        if self.isPacketTimeout():
            raise TimeoutError("Feedback transaction deadline exceeded")

    def readPort(self, length: int) -> bytes:
        self._check_deadline()  # Also bounds continuous malformed traffic inside SDK rxPacket.
        if length < 0:
            raise ValueError("Invalid packet length")
        return self.serial.read(length)

    def writePort(self, packet: list[int]) -> int:
        self._check_deadline()
        return self.serial.write(bytes(packet))

    def clearPort(self) -> None:
        self._check_deadline()
        self.serial.reset_input_buffer()

    def setPacketTimeout(self, _length: int) -> None:
        pass  # SDK cannot extend the transaction's original deadline.

    def isPacketTimeout(self) -> bool:
        return time.monotonic() >= self.deadline


class FeetechFeedbackReader:
    """Collect one fresh block per configured motor without retries or actuation.

    The caller opens/configures/owns the port exclusively and supplies a unique
    clock_id for each Host session. Replies are accepted in arrival order, so a
    missing ID cannot hide later healthy motors. Serial has no transaction IDs:
    clearing pending input cannot prove that a delayed reply is newly sampled.

    The budget is wire time plus 8 ms, as in the deployed block reader. This is a
    bounded software wait, not a hard real-time guarantee of OS/USB scheduling.
    """

    def __init__(
        self, serial: SerialConnection, actuators: Sequence[ActuatorSpec], *, clock_id: str
    ) -> None:
        identifier(clock_id, "clock_id")
        actuators = tuple(actuators)
        if not actuators or len(actuators) > 242:
            raise ValueError("Motor count exceeds the SDK sync-read packet capacity")
        names, ids, buses = set(), set(), set()
        for motor in actuators:
            if not isinstance(motor, ActuatorSpec):
                raise TypeError("Expected ActuatorSpec values")
            identifier(motor.name, "motor name")
            identifier(motor.bus, "bus name")
            if motor.motor_model not in _MODELS:
                raise ValueError(f"Unsupported feedback model: {motor.motor_model}")
            if type(motor.motor_id) is not int or not 1 <= motor.motor_id <= 253:
                raise ValueError("Motor ID must be between 1 and 253")
            if motor.name in names or motor.motor_id in ids:
                raise ValueError("Duplicate motor name or bus ID")
            names.add(motor.name)
            ids.add(motor.motor_id)
            buses.add(motor.bus)
        if len(buses) != 1:
            raise ValueError("A feedback reader must own exactly one bus")
        self._serial = serial
        self._motors = {motor.motor_id: motor.name for motor in actuators}
        self._source = actuators[0].bus
        self._clock_id = clock_id
        self._sequence = 0
        self._thread_id = threading.get_ident()
        self._validate_port()

    def _validate_port(self) -> float:
        port = self._serial
        if not port.is_open or port.timeout != 0:
            raise ValueError("Feedback requires an open serial port with timeout=0")
        if type(port.baudrate) is not int or port.baudrate <= 0:
            raise ValueError("Invalid serial baud rate")
        if (port.bytesize, port.parity, port.stopbits) != (8, "N", 1):
            raise ValueError("STS feedback requires 8N1 serial framing")
        # Request: 8 + N bytes; replies: (6 + 15) * N bytes; 8N1 framing.
        budget = (8 + 22 * len(self._motors)) * 10 / port.baudrate + 0.008
        finite_number(port.write_timeout, "serial write_timeout")
        if not 0 < port.write_timeout <= budget:
            raise ValueError("Serial write_timeout must be positive and within the read budget")
        return budget

    def read(self, names: Sequence[str] | None = None) -> FeedbackBatch:
        return self._read(names, position_only=False)

    def read_positions(self, names: Sequence[str] | None = None) -> FeedbackBatch:
        """Read only Present_Position, as in the deployed passive leader loop."""
        return self._read(names, position_only=True)

    def _read(self, names: Sequence[str] | None, *, position_only: bool) -> FeedbackBatch:
        if threading.get_ident() != self._thread_id:
            raise RuntimeError("Serial feedback must be read by its owning thread")
        budget = self._validate_port()
        selected = list(self._motors.values()) if names is None else list(names)
        if len(set(selected)) != len(selected) or not set(selected) <= set(self._motors.values()):
            raise ValueError("Feedback selection must contain distinct configured motors")
        # Optional dependency is loaded only when hardware feedback is requested.
        from scservo_sdk.protocol_packet_handler import protocol_packet_handler
        from scservo_sdk.scservo_def import COMM_RX_CORRUPT, COMM_SUCCESS

        handler = protocol_packet_handler()
        started = time.monotonic()
        port = _PacketPort(self._serial, started + budget)
        samples, failures = {}, {}
        missing_reason = "Feedback reply missing before deadline"
        ids_by_name = {name: motor_id for motor_id, name in self._motors.items()}
        ids = [ids_by_name[name] for name in selected]
        sequence = self._sequence
        self._sequence += 1
        size = 2 if position_only else _SIZE
        try:
            result = handler.syncReadTx(port, _START, size, ids, len(ids)) if ids else COMM_SUCCESS
            if result != COMM_SUCCESS:
                missing_reason = handler.getTxRxResult(result)
            else:
                while len(samples) < len(ids) and not port.isPacketTimeout():
                    packet, result = handler.rxPacket(port)
                    if result == COMM_RX_CORRUPT:
                        continue
                    if result != COMM_SUCCESS:
                        missing_reason = handler.getTxRxResult(result)
                        break
                    if len(packet) != size + 6 or packet[3] != size + 2:
                        continue
                    name = self._motors.get(packet[2])
                    if name not in selected or name in samples:
                        continue
                    sample = (
                        MotorFeedback(
                            {"position_raw": int.from_bytes(bytes(packet[5:-1]), "little")},
                            packet_error=packet[4],
                        )
                        if position_only
                        else decode_feedback(bytes(packet[5:-1]), packet_error=packet[4])
                    )
                    samples[name] = sample
                    if sample.packet_error:
                        failures[name] = f"Servo packet error 0x{sample.packet_error:02x}"
        except (OSError, ValueError) as exc:
            missing_reason = str(exc) or type(exc).__name__
        finally:
            port.is_using = False
        for name in selected:
            if name not in samples:
                failures[name] = missing_reason
        return FeedbackBatch(
            self._source, self._clock_id, sequence, started, time.monotonic(), samples, failures
        )
