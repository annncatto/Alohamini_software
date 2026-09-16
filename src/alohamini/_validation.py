"""Scalar validation shared by framework-independent interfaces."""

import math


def finite_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")


def identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 64 or value.isspace():
        raise ValueError(f"{name} must contain 1 to 64 characters")
