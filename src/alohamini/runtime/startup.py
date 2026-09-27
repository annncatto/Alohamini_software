# SPDX-License-Identifier: Apache-2.0
# Device defaults and startup sequence adapted from AlohaMini.configure/connect.
"""Assemble and start the robot from installed motor calibration."""

import logging
import math
import shlex
from collections.abc import Mapping
from pathlib import Path

from alohamini.calibration.procedure import ensure_calibration
from alohamini.calibration.servo import load_motor_calibration
from alohamini.hardware.camera import CameraConfig, OpenCVCamera
from alohamini.hardware.feetech_device import CalibrationMismatchError, FeetechBusDevice
from alohamini.model import get_robot_model
from alohamini.paths import WorkspacePaths
from alohamini.runtime.arm_contact import (
    ArmJointSpec,
    GripperContactCalibration,
    JointContactCalibration,
)
from alohamini.runtime.host import NativeHost
from alohamini.runtime.lift_control import LiftAxisSpec

logger = logging.getLogger(__name__)


def open_host(
    robot_model: str,
    *,
    calibration_file: str | Path | None = None,
    left_port: str = "/dev/am_arm_follower_left",
    right_port: str = "/dev/am_arm_follower_right",
    cameras: Mapping[str, CameraConfig] | None = None,
    use_degrees: bool = False,
    bind_host: str = "0.0.0.0",
    command_port: int = 5555,
    state_port: int = 5556,
    camera_port: int = 5557,
) -> NativeHost:
    """Open, prepare and enable both buses, then schedule protected lift homing.

    Arms must be supported at rest and the lift descent path must be clear.
    Homing advances in run/step; ordinary commands are rejected until it finishes.
    The caller owns the returned Host and must run it or close it immediately.
    Bind only on a trusted robot network; ZMQ here does not authenticate clients.

    Calibration is read from the visible workspace or one explicitly supplied
    file. A mismatch requires interactive confirmation before restoring EEPROM
    from that file. cameras={} explicitly selects camera-free operation.
    """
    for _ in range(2):
        host = _open_host(
            robot_model,
            calibration_file=calibration_file,
            left_port=left_port,
            right_port=right_port,
            cameras=cameras,
            use_degrees=use_degrees,
            bind_host=bind_host,
            command_port=command_port,
            state_port=state_port,
            camera_port=camera_port,
        )
        if host is not None:
            return host
    raise CalibrationMismatchError("Calibration changed again during Host startup")


def _open_host(
    robot_model,
    *,
    calibration_file,
    left_port,
    right_port,
    cameras,
    use_degrees,
    bind_host,
    command_port,
    state_port,
    camera_port,
) -> NativeHost | None:
    logger.info("Configuring AlohaMini")
    model = get_robot_model(robot_model)
    if type(use_degrees) is not bool:
        raise ValueError("use_degrees must be a boolean")
    if Path(left_port).resolve() == Path(right_port).resolve():
        raise ValueError("Left and right buses must use different serial devices")
    path = (
        WorkspacePaths().calibration_file("robots", "AlohaMiniRobot")
        if calibration_file is None
        else Path(calibration_file).expanduser()
    )
    try:
        calibrations = load_motor_calibration(path, model.actuators)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Calibration not found: {path}. Supply the installed robot's existing JSON "
            f"with --calibration, or run alohamini calibrate robot --robot_model {robot_model}; "
            "do not substitute another robot's calibration."
        ) from exc
    joints, units = [], {}
    for motor in model.actuators:
        if not motor.name.startswith("arm_"):
            continue
        calibration = calibrations[motor.name]
        gripper = motor.name.endswith("_gripper")
        normalization = "range_0_100" if gripper else "degrees" if use_degrees else "range_m100_100"
        wire = calibration.position_units(normalization)
        encoder = calibration.encoder_calibration()
        # Inversion belongs to wire conversion, not a second encoder inversion.
        contact = (
            GripperContactCalibration(
                encoder.position_from_tick(wire.to_tick(0)),
                encoder.position_from_tick(wire.to_tick(100)),
            )
            if gripper
            else JointContactCalibration(
                math.tau
                / 4096
                * (4095 / 360 if use_degrees else (wire.range_max - wire.range_min) / 200)
            )
        )
        joints.append(ArmJointSpec(motor, encoder, contact))
        units[motor.name] = wire
    positions = {joint.actuator.name: joint.calibration for joint in joints}
    devices = {}
    for source, port in (("left", left_port), ("right", right_port)):
        motors = [motor for motor in model.actuators if motor.bus == source]
        devices[source] = FeetechBusDevice(
            port,
            motors,
            position_calibrations={
                m.name: positions[m.name] for m in motors if m.name in positions
            },
            velocity_limits={
                m.name: 1300 if m.name == "lift_axis" else 3000
                for m in motors
                if m.name not in positions
            },
        )
    if cameras is None:
        cameras = {
            name: CameraConfig(f"/dev/am_camera_{name}") for name in ("forward", "wrist_right")
        }
    if not all(isinstance(config, CameraConfig) for config in cameras.values()):
        raise TypeError("Expected named CameraConfig values")
    lift = next(m for m in model.actuators if m.name == "lift_axis")
    host = NativeHost(
        model,
        devices,
        joints,
        LiftAxisSpec(lift, model.lift_lead_m_per_rev, -1),
        units,
        cameras={name: OpenCVCamera(config) for name, config in cameras.items()},
        bind_host=bind_host,
        command_port=command_port,
        state_port=state_port,
        camera_port=camera_port,
        motor_calibrations=calibrations,
    )
    logger.info("Connecting AlohaMini")
    try:
        host.start()
        for device in devices.values():
            device.disable_torque()
        selected = {
            side: {m.name: calibrations[m.name] for m in device.actuators}
            for side, device in devices.items()
        }
        recalibrated = ensure_calibration(devices, selected, {side: path for side in devices})
        if recalibrated:
            # The Host, controllers and buses all captured the old coordinates.
            # Close this torque-off instance before constructing the replacement.
            host.close()
            return None
        for device in devices.values():
            device.prepare({m.name: calibrations[m.name] for m in device.actuators})
        # No bus is enabled until every bus has a verified safe target.
        for device in devices.values():
            device.stop()
        for device in devices.values():
            device.enable_torque()
        host.begin_lift_homing()
    except BaseException as exc:
        try:
            host.close()
        except BaseException as cleanup:
            raise RuntimeError(f"Host startup failed: {exc}; cleanup failed: {cleanup}") from exc
        if isinstance(exc, CalibrationMismatchError):
            recovery = "Restore the selected calibration before restarting Host."
            if path.suffix == ".json":
                command = shlex.join(
                    [
                        "alohamini",
                        "calibrate",
                        "robot",
                        "--robot_model",
                        robot_model,
                        "--calibration_dir",
                        str(path.parent),
                        "--id",
                        path.stem,
                    ]
                )
                recovery += (
                    f"\nSupport both arms and the lift, then run: {command}"
                    "\nPress ENTER to use the existing file; do not choose 'c'."
                )
            raise CalibrationMismatchError(f"{exc}\nCalibration file: {path}\n{recovery}") from exc
        raise
    return host
