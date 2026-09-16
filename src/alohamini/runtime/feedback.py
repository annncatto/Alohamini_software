"""Project bus feedback into calibrated joint units without losing raw measurements."""

from collections.abc import Mapping
from dataclasses import replace

from alohamini._validation import identifier
from alohamini.calibration import EncoderCalibration
from alohamini.hardware.feedback import FeedbackBatch


class FeedbackProjector:
    """One bus/session stream with explicit per-joint encoder calibration.

    Missing, faulted or invalid samples and skipped batches break continuity.
    Call reset after reconnect or a long sampling gap even if no sequence was
    skipped: a single-turn encoder cannot infer motion while unobserved. Lift
    height and body velocity are not inferred from uncalibrated motor registers.
    """

    def __init__(
        self,
        calibrations: Mapping[str, EncoderCalibration],
        *,
        source_id: str,
        clock_id: str,
    ) -> None:
        identifier(source_id, "source_id")
        identifier(clock_id, "clock_id")
        for name, calibration in calibrations.items():
            identifier(name, "joint name")
            if not isinstance(calibration, EncoderCalibration):
                raise TypeError("Expected EncoderCalibration values")
        self._calibrations = dict(calibrations)
        self._source_id = source_id
        self._clock_id = clock_id
        self._previous: dict[str, float] = {}
        self._sequence: int | None = None
        self._received_s: float | None = None

    def reset(self) -> None:
        """Forget encoder turns without allowing old batches to be replayed."""
        self._previous.clear()

    def project(self, batch: FeedbackBatch) -> FeedbackBatch:
        if not isinstance(batch, FeedbackBatch):
            raise TypeError("Expected FeedbackBatch")
        if batch.source_id != self._source_id or batch.clock_id != self._clock_id:
            raise ValueError("Feedback belongs to another bus or Host session")
        if self._sequence is not None:
            if batch.sequence <= self._sequence or batch.request_started_s < self._received_s:
                self.reset()
                raise ValueError("Feedback batches must be ordered and nonoverlapping")
            if batch.sequence != self._sequence + 1:
                self.reset()
        self._sequence = batch.sequence
        self._received_s = batch.received_s
        samples, previous = {}, {}
        for name, sample in batch.samples.items():
            calibration = self._calibrations.get(name)
            if calibration is None:
                samples[name] = sample
                continue
            position, velocity = None, None
            errors = dict(sample.field_errors)
            for field in ("position_rad", "velocity_rad_s"):
                errors.pop(field, None)
            try:
                position = calibration.position_from_tick(
                    sample.registers["position_raw"],
                    previous_position_rad=(
                        self._previous.get(name)
                        if name not in batch.failures and not sample.packet_error
                        else None
                    ),
                )
                if name not in batch.failures and not sample.packet_error:
                    previous[name] = position
            except (KeyError, ValueError) as exc:
                errors["position_rad"] = str(exc)
            try:
                velocity = calibration.velocity_from_ticks_per_second(
                    sample.registers["velocity_raw"]
                )
            except (KeyError, ValueError) as exc:
                errors["velocity_rad_s"] = str(exc)
            samples[name] = replace(
                sample, position_rad=position, velocity_rad_s=velocity, field_errors=errors
            )
        self._previous = previous
        return replace(batch, samples=samples)
