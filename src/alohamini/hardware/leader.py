# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from SOLeader and BiSOLeader; local calibration and native STS transport.
"""Passive bimanual leaders using the deployed per-arm calibration files."""

from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

from alohamini.calibration.servo import load_motor_calibration
from alohamini.hardware.feetech_device import FeetechBusDevice
from alohamini.model import ActuatorSpec, get_robot_model
from alohamini.paths import WorkspacePaths


class BimanualLeader:
    """SO 5-DoF or AM 6-DoF leaders; grippers are additional position channels.

    Both profiles use STS3215. Joint names/order come from the selected whole
    robot, but ranges and inversion come exclusively from each leader's JSON.
    Connection only disables torque and verifies existing calibration/mode.
    """

    def __init__(
        self,
        robot_model: str,
        *,
        leader_id: str | None = None,
        calibration_dir: str | Path | None = None,
        left_port: str = "/dev/am_arm_leader_left",
        right_port: str = "/dev/am_arm_leader_right",
    ) -> None:
        model = get_robot_model(robot_model)
        if Path(left_port).resolve() == Path(right_port).resolve():
            raise ValueError("Leader arms must use different serial devices")
        leader_id = leader_id or (
            "so101_leader_bi" if robot_model == "alohamini1" else "am_leader_bi"
        )
        paths = WorkspacePaths()
        self.calibrations, self.devices = {}, {}
        self._resources = ExitStack()
        self._connected = self._used = False
        for side, port in (("left", left_port), ("right", right_port)):
            # The old BiSOLeader stores <id>_left/right.json with unprefixed joints.
            filename = paths.calibration_file("teleoperators", f"{leader_id}_{side}")
            if calibration_dir is not None:
                filename = Path(calibration_dir).expanduser() / filename.name
            actuators = tuple(
                ActuatorSpec(m.name.removeprefix(f"arm_{side}_"), side, m.motor_id, "sts3215")
                for m in model.actuators
                if m.name.startswith(f"arm_{side}_")
            )
            calibration = load_motor_calibration(filename, actuators)
            self.calibrations[side] = calibration
            self.devices[side] = FeetechBusDevice(
                port,
                actuators,
                position_calibrations={k: c.encoder_calibration() for k, c in calibration.items()},
                velocity_limits={},
            )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_exc):
        self.close()

    def connect(self) -> None:
        if self._used:
            raise RuntimeError("Create a new leader for each connection")
        self._used = True
        session = uuid4().hex
        try:
            for device in self.devices.values():
                self._resources.callback(device.close)
                device.connect(session)
                self._resources.callback(device.disable_torque)
            for device in self.devices.values():
                device.disable_torque()
            for side, device in self.devices.items():
                device.prepare_passive(self.calibrations[side])
            self._connected = True
        except BaseException:
            self.close()
            raise

    def read(self, normalizations: Mapping[str, str]) -> dict[str, float]:
        """Map fresh leader positions into the Host's declared wire units."""
        if not self._connected:
            raise RuntimeError("Connect the leader before reading")
        units = {}
        for side, calibration in self.calibrations.items():
            for name, entry in calibration.items():
                key = f"arm_{side}_{name}.pos"
                normalization = normalizations[key]
                expected = ("range_0_100",) if name == "gripper" else ("range_m100_100", "degrees")
                if normalization not in expected:
                    raise ValueError(f"Incompatible Host joint units: {key}")
                units[key] = entry.position_units(normalization)
        result = {}
        for side, device in self.devices.items():
            for name, tick in device.read_positions().items():
                key = f"arm_{side}_{name}.pos"
                result[key] = units[key].from_tick(tick)
        return result

    def close(self) -> None:
        self._connected = False
        self._resources.close()
