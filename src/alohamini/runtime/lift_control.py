# SPDX-License-Identifier: Apache-2.0
# Adapted from AlohaMini LiftAxis feedback accumulation and height proportional control.
"""Feedback-driven lift height control and current-contact homing; no device I/O."""

import math
import statistics
from collections import deque
from dataclasses import dataclass
from itertools import pairwise

from alohamini._validation import finite_number, identifier
from alohamini.hardware.feedback import FeedbackBatch
from alohamini.model import ActuatorSpec


@dataclass(frozen=True)
class LiftAxisSpec:
    """Installed mechanics and the accepted speed-register range.

    lead_m_per_revolution includes gearing. The register range is not a physical
    speed bound or a sampling deadline. Encoder accumulation assumes less than
    half a turn between valid samples, as in the source LiftAxis.
    """

    actuator: ActuatorSpec
    lead_m_per_revolution: float
    direction: int
    max_encoder_speed_ticks_s: float = 32767

    def __post_init__(self) -> None:
        if not isinstance(self.actuator, ActuatorSpec):
            raise TypeError("Expected ActuatorSpec")
        identifier(self.actuator.name, "lift name")
        identifier(self.actuator.bus, "lift bus")
        finite_number(self.lead_m_per_revolution, "lift lead")
        finite_number(self.max_encoder_speed_ticks_s, "encoder speed bound")
        if self.lead_m_per_revolution <= 0:
            raise ValueError("Lift lead must be positive")
        if not 1300 <= self.max_encoder_speed_ticks_s <= 32767:
            raise ValueError("Encoder speed bound must cover the 1300 ticks/s command limit")
        if type(self.direction) is not int or self.direction not in (-1, 1):
            raise ValueError("Lift direction must be -1 or 1")


@dataclass(frozen=True)
class LiftOutput:
    target_height_m: float
    velocity_raw: int
    reason: str | None


class LiftHoming:
    """Local sensorless homing over fresh Host feedback, never an endstop sensor.

    The operator must clear the descent path. Current plus standstill can detect
    contact, but cannot distinguish the bottom from an obstruction. The deployed
    1300 ticks/s, 0.3 A and 30 s limits are retained. No-motion alone is a fault,
    not a zero reference. The source's one-second unloaded settling interval
    follows the stop and torque-off commands before establishing zero.
    """

    def __init__(self, spec: LiftAxisSpec) -> None:
        if not isinstance(spec, LiftAxisSpec):
            raise TypeError("Expected LiftAxisSpec")
        self.spec = spec
        self.phase = "seeking"
        self.velocity_raw = 0
        self._started: float | None = None
        self._previous: FeedbackBatch | None = None
        self._tick: int | None = None
        self._stationary_since: float | None = None
        self._contact_since: float | None = None
        self._stop_started: float | None = None
        self._stationary_tick: int | None = None
        self._travel_ticks = 0
        self._released = False
        self._settle_position = 0
        self._settle_samples: deque[tuple[float, int, int]] = deque(maxlen=128)

    def update(self, batch: FeedbackBatch) -> None:
        try:
            self._update(batch)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            self.phase = "failed"
            self.velocity_raw = 0
            raise

    def confirm_release(self, now: float) -> None:
        """Start the settling interval after the torque-off write was verified."""
        finite_number(now, "lift release time")
        if self.phase != "settling" or now < self._stop_started:
            raise RuntimeError("Lift release must follow contact detection")
        self._stop_started = now
        self._released = True
        self._settle_position = 0
        self._settle_samples.clear()

    def _update(self, batch: FeedbackBatch) -> None:
        if self.phase not in ("seeking", "settling"):
            raise RuntimeError("Homing is already finished or faulted")
        sample = batch.samples.get(self.spec.actuator.name)
        if (
            batch.source_id != self.spec.actuator.bus
            or self.spec.actuator.name in batch.failures
            or sample is None
            or sample.packet_error
        ):
            raise ValueError("Lift homing feedback missing or faulted")
        current = sample.current_a
        finite_number(current, "homing current")
        tick = sample.registers.get("position_raw")
        speed = sample.registers.get("velocity_raw")
        moving = sample.registers.get("moving")
        if type(tick) is not int or not 0 <= tick < 4096:
            raise ValueError("Invalid lift homing position")
        if type(speed) is not int or abs(speed) > self.spec.max_encoder_speed_ticks_s:
            raise ValueError("Invalid lift homing speed")
        if type(moving) is not int or moving not in (0, 1):
            raise ValueError("Invalid lift homing moving flag")
        now = batch.received_s
        if self._previous is not None:
            previous = self._previous
            if (
                batch.clock_id != previous.clock_id
                or batch.sequence <= previous.sequence
                or batch.request_started_s < previous.received_s
            ):
                raise ValueError("Lift homing feedback is not continuous")
            delta = (tick - self._tick + 2048) % 4096 - 2048
            if delta == -2048:
                raise ValueError("Lift homing encoder half-turn is ambiguous")
            self._travel_ticks += abs(delta)
            if self._travel_ticks * self.spec.lead_m_per_revolution / 4096 > 0.65:
                raise RuntimeError("Lift homing exceeded the travel allowance")
            if self.phase == "settling" and self._released:
                self._settle_position += delta
        else:
            self._started = now
        self._previous, self._tick = batch, tick
        if now - self._started >= 30:
            raise RuntimeError("Lift homing timed out; zero reference not established")

        if self.phase == "settling":
            self._settle(batch, tick, speed, moving)
            return

        stationary = speed == 0 and moving == 0
        displacement = (
            None
            if self._stationary_tick is None
            else (tick - self._stationary_tick + 2048) % 4096 - 2048
        )
        if not stationary or displacement is None or abs(displacement) > 2:
            self._stationary_since = now if stationary else None
            self._stationary_tick = tick if stationary else None
            self._contact_since = None
        if self.phase == "seeking":
            self.velocity_raw = -self.spec.direction * 1300
            if stationary and abs(current) >= 0.3:
                if self._contact_since is None:
                    self._contact_since = now
                if now - self._contact_since >= 0.1:
                    self.phase = "settling"
                    self.velocity_raw = 0
                    self._stop_started = now
                    self._stationary_since = self._stationary_tick = None
            else:
                self._contact_since = None
                if self._stationary_since is not None and now - self._stationary_since >= 0.5:
                    raise RuntimeError("Lift stopped without contact current; homing failed")

    def _settle(self, batch: FeedbackBatch, tick: int, speed: int, moving: int) -> None:
        """Verify unloaded rest from a full second of position feedback.

        Use the source's 10-tick position tolerance over a longer, one-second
        window, not its 50 ms contact interval. Moving is diagnostic here, not
        proof of physical motion with torque off. A speed peak accompanying
        encoder jitter must not veto the whole window; sustained reported
        velocity or position drift still prevents zeroing.
        """
        self.velocity_raw = 0
        if not self._released:
            raise RuntimeError("Lift homing release has not been confirmed")
        now = batch.received_s
        if batch.request_started_s >= self._stop_started:
            self._settle_samples.append((now, self._settle_position, abs(speed)))
        samples = self._settle_samples
        while len(samples) > 2 and samples[1][0] <= now - 1:
            samples.popleft()
        duration = samples[-1][0] - samples[0][0] if samples else 0.0
        positions = [item[1] for item in samples]
        span = max(positions) - min(positions) if positions else 0
        max_speed = max((s[2] for s in samples), default=0)
        median_speed = statistics.median(s[2] for s in samples) if samples else 0
        tail_speed = (
            statistics.median(s[2] for s in samples if s[0] >= now - 0.25) if samples else 0
        )
        max_gap = max((b[0] - a[0] for a, b in pairwise(samples)), default=0.0)
        drift = None
        if duration >= 1 - 1e-9:
            first = [p for t, p, _ in samples if t <= samples[0][0] + 0.25]
            last = [p for t, p, _ in samples if t >= now - 0.25]
            drift = statistics.median(last) - statistics.median(first)
            if (
                max_gap <= 0.1
                and span <= 10
                and abs(drift) <= 2
                and median_speed <= 10
                and tail_speed <= 10
            ):
                self.phase = "complete"
                return
        if now - self._stop_started >= 2:
            raise RuntimeError(
                "Lift did not confirm standstill after homing stop: "
                f"position={tick} speed={speed} moving={moving} "
                f"window_s={duration:.3f} span_ticks={span} drift_ticks={drift} "
                f"max_speed={max_speed} median_speed={median_speed} tail_speed={tail_speed} "
                f"max_gap_s={max_gap:.3f} samples={len(samples)}"
            )


def lift_height_target(target_m: float, measured_m: float, *, direction: int) -> LiftOutput:
    """Existing P control in SI units: 300 ticks/s/mm, 1300 limit, 1 mm tolerance."""
    finite_number(target_m, "lift target")
    finite_number(measured_m, "lift measured height")
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("Lift direction must be -1 or 1")
    target = min(0.6, max(0.0, target_m))
    error = target - measured_m
    # Unit conversion must not turn a floating-point endpoint into a motion command.
    if abs(error) <= 0.001 or math.isclose(abs(error), 0.001, rel_tol=0, abs_tol=1e-12):
        return LiftOutput(target, 0, "at_target")
    velocity = min(1300.0, max(-1300.0, 300_000 * error))
    # Preserve integer-tick boundaries through metre/millimetre conversion.
    nearest_tick = round(velocity)
    if math.isclose(velocity, nearest_tick, rel_tol=0, abs_tol=1e-9):
        velocity = nearest_tick
    if velocity < 0 and measured_m <= 0.005:
        return LiftOutput(target, 0, "descent_floor")
    if (velocity > 0 and measured_m >= 0.6) or (velocity < 0 and measured_m <= 0):
        return LiftOutput(target, 0, "travel_limit")
    return LiftOutput(target, int(direction * velocity), None)


class LiftHeightTracker:
    """Single-session encoder unwrapping anchored to an independently known height.

    A reference must be established against the currently observed stationary
    sample. As in LiftAxis._update_extended_ticks, each delta is wrapped to half
    a revolution. This assumes less than half a turn between valid samples;
    neither timestamps nor the signed velocity register establish a physical
    speed bound or recover turns lost during an interruption.
    """

    def __init__(self, spec: LiftAxisSpec) -> None:
        if not isinstance(spec, LiftAxisSpec):
            raise TypeError("Expected LiftAxisSpec")
        self.spec = spec
        self._session: str | None = None
        self._sequence = -1
        self._tick: int | None = None
        self._received_s: float | None = None
        self._height_m: float | None = None
        self._reference_height_m: float | None = None
        self._reference_tick: int | None = None
        self._reference_sequence = 0
        self._displacement_ticks = 0
        self._stationary = False

    @property
    def height_m(self) -> float | None:
        return self._height_m

    @property
    def calibration_feedback(self) -> dict:
        """Same Host-local unwrap as height control, not a PC-side reconstruction."""
        result = {
            "homed": self._height_m is not None,
            "reference_sequence": self._reference_sequence,
        }
        if self._height_m is not None:
            result.update(
                raw_tick=self._tick,
                extended_ticks=self._reference_tick + self._displacement_ticks,
                zero_extended_ticks=self._reference_tick
                - (
                    self._reference_height_m
                    * 4096
                    / (self.spec.direction * self.spec.lead_m_per_revolution)
                ),
            )
        return result

    def bind_session(self, session: str) -> None:
        identifier(session, "host_session_id")
        if self._session is not None:
            raise RuntimeError("Lift tracker cannot be rebound")
        self._session = session

    def invalidate(self) -> None:
        self._height_m = None
        self._reference_height_m = None
        self._reference_tick = None
        self._displacement_ticks = 0
        self._stationary = False

    def observe(self, batch: FeedbackBatch) -> None:
        try:
            if batch.source_id != self.spec.actuator.bus or batch.clock_id != self._session:
                raise ValueError("Lift feedback belongs to another bus or Host session")
            if batch.sequence <= self._sequence:
                raise ValueError("Repeated or out-of-order lift feedback")
            sample = batch.samples.get(self.spec.actuator.name)
            if sample is None or self.spec.actuator.name in batch.failures or sample.packet_error:
                raise ValueError("Lift feedback missing or faulted")
            tick = sample.registers.get("position_raw")
            speed = sample.registers.get("velocity_raw")
            moving = sample.registers.get("moving")
            if type(tick) is not int or not 0 <= tick < 4096:
                raise ValueError("Lift requires single-turn position feedback")
            if type(speed) is not int or abs(speed) > self.spec.max_encoder_speed_ticks_s:
                raise ValueError("Lift speed missing or exceeds the accepted register range")
            if type(moving) is not int or moving not in (0, 1):
                raise ValueError("Lift moving flag unavailable")
            if self._received_s is not None and batch.request_started_s < self._received_s:
                raise ValueError("Lift feedback intervals overlap or run backwards")
            if self._height_m is not None:
                # Retain the deployed LiftAxis shortest-delta accumulation.
                delta = tick - self._tick
                if abs(delta) == 2048:
                    raise ValueError("Ambiguous half-revolution lift change")
                if delta > 2048:
                    delta -= 4096
                elif delta < -2048:
                    delta += 4096
                displacement = self._displacement_ticks + delta
                height = self._reference_height_m + (
                    self.spec.direction * displacement * self.spec.lead_m_per_revolution / 4096
                )
                finite_number(height, "lift height")
                self._height_m = height
                self._displacement_ticks = displacement
            self._sequence = batch.sequence
            self._tick = tick
            self._received_s = batch.received_s
            self._stationary = speed == 0 and moving == 0
        except (AttributeError, TypeError, ValueError):
            self.invalidate()
            raise

    def establish_reference(self, height_m: float) -> None:
        """Caller must independently verify height and mechanical rest before calling."""
        finite_number(height_m, "known physical height")
        if not 0 <= height_m <= 0.6:
            raise ValueError("Known height must be inside the installed travel range")
        if self._tick is None or not self._stationary:
            raise RuntimeError("Reference requires a valid stationary feedback sample")
        self._set_reference(height_m)

    def establish_homing_reference(self, homing: LiftHoming) -> None:
        """Use completed torque-off homing evidence for this exact feedback sample."""
        if (
            not isinstance(homing, LiftHoming)
            or homing.spec.actuator != self.spec.actuator
            or homing.spec.lead_m_per_revolution != self.spec.lead_m_per_revolution
            or homing.spec.direction != self.spec.direction
            or homing.phase != "complete"
            or not homing._released
            or homing._previous is None
            or homing._previous.clock_id != self._session
            or homing._previous.sequence != self._sequence
            or homing._previous.received_s != self._received_s
            or homing._tick != self._tick
        ):
            raise RuntimeError("Reference requires completed homing on the current feedback sample")
        self._set_reference(0.0)

    def _set_reference(self, height_m: float) -> None:
        self._height_m = height_m
        self._reference_height_m = height_m
        self._reference_tick = self._tick
        self._reference_sequence += 1
        self._displacement_ticks = 0
