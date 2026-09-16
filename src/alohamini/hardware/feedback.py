"""Fresh feedback values, independent of datasets and transport serialization."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from alohamini._validation import finite_number, identifier


@dataclass(frozen=True)
class MotorFeedback:
    """Raw values are decoded registers, not normalized positions or torque.

    packet_error is the servo status-packet error byte, distinct from status_raw.
    A sample with an error still contains measurements; it is not a safety permit.
    Unavailable physical quantities remain None, never synthetic zeros.
    """

    registers: Mapping[str, int]
    packet_error: int = 0
    current_a: float | None = None
    position_rad: float | None = None
    velocity_rad_s: float | None = None
    field_errors: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in self.registers.items():
            identifier(name, "register name")
            if type(value) is not int:
                raise ValueError("Decoded registers must be integers")
        if type(self.packet_error) is not int or not 0 <= self.packet_error <= 127:
            raise ValueError("Invalid status-packet error byte")
        for name in ("current_a", "position_rad", "velocity_rad_s"):
            value = getattr(self, name)
            if value is not None:
                finite_number(value, name)
        for name, error in self.field_errors.items():
            identifier(name, "field name")
            if not isinstance(error, str) or not error:
                raise ValueError("Field errors must be nonempty strings")
        object.__setattr__(self, "registers", MappingProxyType(dict(self.registers)))
        object.__setattr__(self, "field_errors", MappingProxyType(dict(self.field_errors)))


@dataclass(frozen=True)
class FeedbackBatch:
    """One bus read, including partial failures; no last-observation fallback.

    Times bound the Host transaction on clock_id's monotonic clock. They are not
    servo sampling times, nor estimates of another machine's clock. Sequence
    increases per read, including failed reads, within a source/clock stream.
    """

    source_id: str
    clock_id: str
    sequence: int
    request_started_s: float
    received_s: float
    samples: Mapping[str, MotorFeedback]
    failures: Mapping[str, str]

    def __post_init__(self) -> None:
        identifier(self.source_id, "source_id")
        identifier(self.clock_id, "clock_id")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sequence must be a nonnegative integer")
        for name in ("request_started_s", "received_s"):
            finite_number(getattr(self, name), name)
        if self.request_started_s < 0 or self.received_s < self.request_started_s:
            raise ValueError("Invalid feedback transaction interval")
        for name, sample in self.samples.items():
            identifier(name, "motor name")
            if not isinstance(sample, MotorFeedback):
                raise TypeError("Expected MotorFeedback samples")
        for name, error in self.failures.items():
            identifier(name, "motor name")
            if not isinstance(error, str) or not error:
                raise ValueError("Failures must be nonempty strings")
        object.__setattr__(self, "samples", MappingProxyType(dict(self.samples)))
        object.__setattr__(self, "failures", MappingProxyType(dict(self.failures)))
