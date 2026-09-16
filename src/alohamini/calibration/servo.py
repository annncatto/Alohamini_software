# SPDX-License-Identifier: Apache-2.0
# Calibration fields and normalization adapted from LeRobot MotorCalibration.
"""Installed STS calibration, retaining the deployed per-motor JSON format."""

import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from alohamini.calibration.encoder import EncoderCalibration, HostPositionUnits
from alohamini.model import ActuatorSpec


@dataclass(frozen=True)
class MotorCalibration:
    """EEPROM offset/ranges plus software inversion; IDs are local to each bus.

    Present_Position already includes homing_offset. The offset must not be
    subtracted again in software. These values do not define a URDF joint zero.
    """

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int

    def __post_init__(self) -> None:
        if type(self.id) is not int or not 1 <= self.id <= 253:
            raise ValueError("Calibration motor ID must be in [1, 253]")
        if type(self.homing_offset) is not int or not -2047 <= self.homing_offset <= 2047:
            raise ValueError("STS homing_offset must fit sign-magnitude bit 11")
        self.position_units()

    def position_units(self, normalization: str = "range_m100_100") -> HostPositionUnits:
        return HostPositionUnits(normalization, self.range_min, self.range_max, self.drive_mode)

    @property
    def offset_register(self) -> int:
        return abs(self.homing_offset) | (0x800 if self.homing_offset < 0 else 0)

    def encoder_calibration(self) -> EncoderCalibration:
        """Post-offset servo output angle: tick 2048 = 0, increasing ticks = +.

        This device coordinate is independent of normalized command inversion
        and of ROS/URDF installation transforms. One output turn is 4096 ticks;
        the deployed degree-mode wire conversion separately uses 4095.
        """
        scale = math.tau / 4096
        return EncoderCalibration(
            4096,
            2048,
            0,
            1,
            position_min_rad=(self.range_min - 2048) * scale,
            position_max_rad=(self.range_max - 2048) * scale,
        )


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate calibration key: {name}")
        result[name] = value
    return result


def save_motor_calibration(
    path: str | Path,
    calibrations: dict[str, MotorCalibration],
    actuators: Sequence[ActuatorSpec],
) -> None:
    """Validate the deployed JSON, retain an existing file, and replace atomically."""
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError("Provide the calibration file itself, not a symbolic link")
    payload = json.dumps({k: asdict(v) for k, v in calibrations.items()}, indent=4) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        load_motor_calibration(temporary, actuators)
        if path.exists():
            backup = path.with_name(f"{path.stem}.{time.time_ns()}.backup.json")
            with path.open("rb") as source, backup.open("xb") as destination:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_motor_calibration(
    path: str | Path, actuators: Sequence[ActuatorSpec]
) -> dict[str, MotorCalibration]:
    """Read one explicit file; reject incomplete or mismatched robots before I/O."""
    path = Path(path).expanduser()
    with path.open("rb") as stream:
        data = stream.read(128 * 1024 + 1)
    if len(data) > 128 * 1024:
        raise ValueError("Calibration file exceeds 128 KiB")
    raw = json.loads(data, object_pairs_hook=_unique_object)
    motors = {motor.name: motor for motor in actuators}
    if len(motors) != len(actuators) or not motors:
        raise ValueError("Expected unique configured actuator names")
    if not isinstance(raw, dict) or raw.keys() != motors.keys():
        raise ValueError("Calibration must contain exactly the configured robot's motors")
    result = {}
    fields = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
    for name, values in raw.items():
        if not isinstance(values, dict) or values.keys() != fields:
            raise ValueError(f"Invalid calibration fields: {name}")
        calibration = MotorCalibration(**values)
        if calibration.id != motors[name].motor_id:
            raise ValueError(f"Calibration ID mismatch: {name}")
        result[name] = calibration
    return result
