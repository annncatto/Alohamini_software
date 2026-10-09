"""Explicit transfer of normalized arm coordinates between calibrated robots."""

from copy import deepcopy

from alohamini.apps.replay import check_calibration
from alohamini.calibration.encoder import HostPositionUnits
from alohamini.model import get_robot_model


def normalized_transfer(trained, snapshot):
    """Keep normalized values; use the new robot's own ranges and encoder offsets.

    This transfers fractional joint positions, not Cartesian poses. Identities,
    units, inversion, base and lift calibration retain the strict contract.
    No calibration file, register, checkpoint or model statistic is modified.
    """
    live = snapshot.payload["_robot_metadata"]
    candidate = deepcopy(trained)
    differences = {}
    allowed = ("range_min", "range_max", "homing_offset")
    unit_fields = ("normalization", "range_min", "range_max", "drive_mode")
    for actuator in get_robot_model(snapshot.robot_model).actuators:
        name = actuator.name
        if not name.startswith("arm_"):
            continue
        old = candidate.get("motors", {}).get(name, {})
        new = live.get("motors", {}).get(name, {})
        for entry in (old, new):
            if entry.get("normalization") not in ("range_m100_100", "range_0_100"):
                raise ValueError(f"Normalized transfer requires normalized arm positions: {name}")
            if any(key not in entry for key in (*allowed, *unit_fields)):
                raise ValueError(f"Incomplete calibration for normalized transfer: {name}")
            HostPositionUnits(**{key: entry[key] for key in unit_fields})
            if (
                type(entry["homing_offset"]) is not int
                or not -2047 <= entry["homing_offset"] <= 2047
            ):
                raise ValueError(f"Invalid homing offset for normalized transfer: {name}")
        changes = {
            key: {"trained": old[key], "Host": new[key]} for key in allowed if old[key] != new[key]
        }
        if changes:
            differences[name] = changes
        old.update({key: new[key] for key in allowed})
    check_calibration(candidate, snapshot)
    return deepcopy(live), dict(
        mode="normalized",
        meaning=(
            "Same normalized arm values in the destination joint ranges; "
            "not Cartesian pose preservation"
        ),
        trained_metadata=deepcopy(trained),
        host_metadata=deepcopy(live),
        differences=differences,
    )
