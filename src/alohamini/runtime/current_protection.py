# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AlohaMini.read_and_check_currents; units changed from mA to A.
"""Time-based overload detection; no bus writes, process exit or collision shaping."""

from collections.abc import Mapping
from dataclasses import dataclass

from alohamini._validation import finite_number, identifier

# Deployed protection policy, not a substitute for device safety validation.
_RATINGS_MA = {"sts3215": (900, 2700), "sts3095": (2200, 9800), "sts3250": (1400, 4200)}


def collision_current_limit_a(motor_model: str) -> float:
    """Existing joint contact threshold: 1.5 times rated current."""
    if motor_model not in _RATINGS_MA:
        raise ValueError(f"Missing current ratings for motor model {motor_model!r}")
    return 1.5 * _RATINGS_MA[motor_model][0] / 1000


@dataclass(frozen=True)
class CurrentTrip:
    motor: str
    cause: str
    current_a: float
    limit_a: float
    duration_s: float


class CurrentProtection:
    """Complete fresh current samples only; missing data cannot count as low current.

    Near-stall: 80% of stall current for 80 ms. Sustained overload: twice rated
    current for 650 ms. Threshold equality counts as overcurrent, matching Host.
    Callers must supervise acquisition failures and gaps separately.
    """

    def __init__(self, motor_models: Mapping[str, str]) -> None:
        if not motor_models:
            raise ValueError("Current protection requires at least one motor")
        self._limits: dict[str, tuple[float, float]] = {}
        for name, model in motor_models.items():
            identifier(name, "motor name")
            if model not in _RATINGS_MA:
                raise ValueError(f"Missing current ratings for motor model {model!r}")
            rated, stall = _RATINGS_MA[model]
            # Convert the original mA thresholds once, retaining equality boundaries.
            self._limits[name] = (0.8 * stall / 1000, 2.0 * rated / 1000)
        self._started: dict[tuple[str, str], float] = {}
        self._last_sample_s: float | None = None

    def update(self, currents_a: Mapping[str, float], *, now: float) -> CurrentTrip | None:
        finite_number(now, "current sample time")
        if now < 0 or (self._last_sample_s is not None and now <= self._last_sample_s):
            raise ValueError("Current sample times must increase")
        if currents_a.keys() != self._limits.keys():
            raise ValueError("Current protection requires exactly the configured motors")
        for value in currents_a.values():
            finite_number(value, "current_a")
        self._last_sample_s = now
        first_trip = None
        for name, (near_stall, sustained) in self._limits.items():
            current = abs(currents_a[name])
            for cause, limit, duration in (
                ("near_stall_current", near_stall, 0.080),
                ("sustained_overload", sustained, 0.650),
            ):
                key = (name, cause)
                if current < limit:
                    self._started.pop(key, None)
                    continue
                started = self._started.setdefault(key, now)
                if now - started >= duration and first_trip is None:
                    first_trip = CurrentTrip(name, cause, current, limit, duration)
        return first_trip
