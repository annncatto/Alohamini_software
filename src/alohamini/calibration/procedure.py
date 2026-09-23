# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Migrated from SOLeader.calibrate, AlohaMini.calibrate and MotorsBus.record_ranges_of_motion.
"""Interactive local calibration; no Host startup, torque enable or automatic movement."""

import logging
import select
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from pprint import pformat
from uuid import uuid4

from alohamini.calibration.servo import (
    MotorCalibration,
    load_motor_calibration,
    save_motor_calibration,
)
from alohamini.hardware.feetech_device import (
    CalibrationMismatchError,
    DeviceOperationError,
    FeetechBusDevice,
)
from alohamini.model import ActuatorSpec, get_robot_model
from alohamini.paths import WorkspacePaths

logger = logging.getLogger(__name__)


def enter_pressed() -> bool:
    """Original terminal polling; a disconnected terminal aborts, not confirms."""
    if sys.platform == "win32":
        import msvcrt

        return msvcrt.kbhit() and msvcrt.getch() in (b"\r", b"\n")
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        if line == "":
            raise EOFError("Calibration terminal disconnected")
        return line.strip() == ""
    return False


def record_ranges_of_motion(
    bus, motors: list[str], display_values: bool = True
) -> tuple[dict[str, int], dict[str, int]]:
    """Interactively record the min/max encoder values of each motor.

    Move the joints by hand (with torque disabled) while the method streams live positions. Press
    :kbd:`Enter` to finish.

    Args:
        motors: Configured joint names to record.
        display_values: Print the original live position table when True.

    Returns:
        tuple[dict[str, int], dict[str, int]]: Two dictionaries *mins* and *maxes* with the
            extreme values observed for each motor.
    """
    motor_names = list(motors)

    start_positions = bus.read_positions(motor_names)
    mins = start_positions.copy()
    maxes = start_positions.copy()

    user_pressed_enter = False
    rendered = False
    refresh = display_values and sys.stdout.isatty()
    while not user_pressed_enter:
        positions = bus.read_positions(motor_names)
        mins = {motor: min(positions[motor], min_) for motor, min_ in mins.items()}
        maxes = {motor: max(positions[motor], max_) for motor, max_ in maxes.items()}

        if display_values:
            lines = ["", "-------------------------------------------"]
            lines.append(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
            for motor in motor_names:
                lines.append(
                    f"{motor:<15} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}"
                )
            # Move up only after a successful read, then leave the cursor below
            # the table so faults and the shell prompt cannot overwrite it.
            prefix = f"\033[{len(lines)}A" if refresh and rendered else ""
            print(prefix + "\n".join(lines), flush=True)
            rendered = True

        if enter_pressed():
            user_pressed_enter = True

    same_min_max = [motor for motor in motor_names if mins[motor] == maxes[motor]]
    if same_min_max:
        raise ValueError(f"Some motors have the same min and max values:\n{pformat(same_min_max)}")

    return mins, maxes


def _reuse(calibration, device_id):
    if not calibration:
        return False
    user_input = input(
        f"Press ENTER to use provided calibration file associated with the id {device_id}, "
        "or type 'c' and press ENTER to run calibration: "
    )
    return user_input.strip().lower() != "c"


def ensure_calibration(buses, calibrations, paths):
    """Restore or recalibrate torque-off buses; return whether coordinates changed."""
    mismatches = {}
    for side, bus in buses.items():
        try:
            bus.verify_calibration(calibrations[side])
        except CalibrationMismatchError as exc:
            mismatches.setdefault(paths[side], exc)
    if not mismatches:
        return False
    if not sys.stdin.isatty():
        path, error = next(iter(mismatches.items()))
        raise CalibrationMismatchError(
            f"{error}\nCalibration file: {path}\n"
            "Interactive confirmation required to restore calibration."
        ) from error
    support = (
        "both arms and the lift"
        if any(m.name == "lift_axis" for bus in buses.values() for m in bus.actuators)
        else "both leader arms"
    )
    print(f"Support {support}; torque is disabled.", flush=True)
    # Confirm every affected file before changing either bus.
    recalibrate = set()
    for path in mismatches:
        print(f"Calibration file: {path}", flush=True)
        response = input(
            f"Press ENTER to use provided calibration file associated with the id {path.stem}, "
            "or type 'c' and press ENTER to run calibration: "
        )
        response = response.strip().lower()
        if response == "c":
            recalibrate.add(path)
        elif response:
            raise InterruptedError("Calibration restore cancelled")
    restoring = {side: bus for side, bus in buses.items() if paths[side] not in recalibrate}
    if restoring:
        logger.info("Writing existing calibration to motors")
    with ExitStack() as transactions:
        for bus in restoring.values():
            transactions.enter_context(bus.calibration_restore_session())
        for side, bus in restoring.items():
            bus.write_calibration(calibrations[side])
        for side, bus in restoring.items():
            bus.verify_calibration(calibrations[side])
    for path in mismatches:
        if path not in recalibrate:
            continue
        selected = {side: bus for side, bus in buses.items() if paths[side] == path}
        if len(selected) == 1:
            _calibrate_leader(next(iter(selected.values())), path.stem, path, {})
        else:
            _calibrate_robot(selected, path.stem, path, {})
    # Callers must rebuild devices and coordinate conversions from the saved files.
    return bool(recalibrate)


def _calibrate_leader(bus, device_id, path, existing):
    # SOLeader.calibrate: each leader retains its own unprefixed joint file.
    names = [motor.name for motor in bus.actuators]
    with bus.calibration_restore_session():
        bus.configure_calibration(names)
        if _reuse(existing, device_id):
            logger.info(
                f"Writing calibration file associated with the id {device_id} to the motors"
            )
            bus.write_calibration(existing)
            return

        logger.info(f"\nRunning calibration of {device_id}")
        input(f"Move {device_id} to the middle of its range of motion and press ENTER....")
        homing_offsets = bus.set_half_turn_homings(names)

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in names if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = record_ranges_of_motion(bus, unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        calibration = {}
        for motor in bus.actuators:
            calibration[motor.name] = MotorCalibration(
                id=motor.motor_id,
                drive_mode=0,
                homing_offset=homing_offsets[motor.name],
                range_min=range_mins[motor.name],
                range_max=range_maxes[motor.name],
            )

        bus.write_calibration(calibration)
        save_motor_calibration(path, calibration, bus.actuators)
        print(f"Calibration saved to {path}")


def _calibrate_robot(buses, device_id, path, existing):
    # AlohaMini.calibrate: left arm first, right arm second; one whole-robot file.
    with ExitStack() as transactions:
        for bus in buses.values():
            transactions.enter_context(bus.calibration_restore_session())
        for bus in buses.values():
            bus.configure_calibration([m.name for m in bus.actuators if m.name.startswith("arm_")])
        if _reuse(existing, device_id):
            logger.info("Writing existing calibration to both buses")
            for bus in buses.values():
                bus.write_calibration({m.name: existing[m.name] for m in bus.actuators})
            return

        logger.info(f"\nRunning calibration of {device_id}")
        calibration = {}
        for side, bus in buses.items():
            arm_motors = [m.name for m in bus.actuators if m.name.startswith("arm_")]
            input(
                f"Move {side.upper()} arm to the middle of its range of motion, then press ENTER..."
            )
            homing = bus.set_half_turn_homings(arm_motors)
            full_turn_motor = f"arm_{side}_wrist_roll"
            unknown = [m for m in arm_motors if m != full_turn_motor]
            print(
                f"Move {side.upper()} arm joints sequentially through full ROM "
                f"(except '{full_turn_motor}'). Press ENTER to stop..."
            )
            mins, maxes = record_ranges_of_motion(bus, unknown)
            mins[full_turn_motor], maxes[full_turn_motor] = 0, 4095
            for motor in bus.actuators:
                if motor.name.startswith("arm_"):
                    calibration[motor.name] = MotorCalibration(
                        motor.motor_id, 0, homing[motor.name], mins[motor.name], maxes[motor.name]
                    )
                else:
                    # Source wheel/lift convention: no offset or hand-driven ROM.
                    calibration[motor.name] = MotorCalibration(motor.motor_id, 0, 0, 0, 4095)

        for bus in buses.values():
            bus.write_calibration({m.name: calibration[m.name] for m in bus.actuators})
        actuators = tuple(m for bus in buses.values() for m in bus.actuators)
        save_motor_calibration(path, calibration, actuators)
        print("Calibration saved to", path)


def _calibrate_one_arm(bus, side, previous, rehome):
    """Adapt calibrate_arms.calibrate_one_arm; use the same passive range recorder."""
    motor_names = [motor.name for motor in bus.actuators]
    if rehome:
        bus.configure_calibration(motor_names)
        input(
            f"Move the {side.upper()} arm to the middle of every joint's usable "
            "range, then press ENTER to rewrite homing offsets: "
        )
        homings = bus.set_half_turn_homings(motor_names)
    else:
        homings = {name: previous[name].homing_offset for name in motor_names}

    full_turn_name = f"arm_{side}_wrist_roll"
    ranged_names = [name for name in motor_names if name != full_turn_name]
    print(
        f"Move every {side.upper()} arm joint through its complete safe range. "
        "Include the gripper; wrist_roll is treated as a full turn. Press ENTER to finish."
    )
    mins, maxes = record_ranges_of_motion(bus, ranged_names)
    mins[full_turn_name], maxes[full_turn_name] = 0, 4095
    result = {
        name: replace(
            previous[name], homing_offset=homings[name], range_min=mins[name], range_max=maxes[name]
        )
        for name in motor_names
    }
    bus.write_calibration(result)
    return result


def _calibrate_arms(buses, path, existing, model, rehome):
    """One transaction across both arm-only buses and the installed robot JSON."""
    with ExitStack() as transactions:
        for bus in buses.values():
            transactions.enter_context(bus.calibration_session())
        previous = {}
        print("Read Homing_Offset directly from connected motor EEPROM:")
        for side, bus in buses.items():
            previous[side] = {
                name: replace(value, drive_mode=existing[name].drive_mode)
                for name, value in bus.read_calibration().items()
            }
            print(f"  {side.upper()}")
            for name, entry in previous[side].items():
                print(
                    f"    {name}: homing_offset={entry.homing_offset} "
                    f"range=[{entry.range_min}, {entry.range_max}]"
                )
        calibration = dict(existing)
        for side, bus in buses.items():
            calibration.update(_calibrate_one_arm(bus, side, previous[side], rehome))
        save_motor_calibration(path, calibration, model.actuators)
    print(f"Calibration saved to {path}; previous file retained as a backup.")
    print("Arm torque remains disabled. Check the ROS joint mapping before commanding motion.")


@contextmanager
def _connected_bus(bus, session):
    """Keep the original failure visible while attempting torque-off and close."""
    primary = None
    try:
        bus.connect(session)
        yield bus
    except BaseException as exc:
        primary = exc
        raise
    finally:
        errors, interrupts = {}, []
        for label, operation in (("disable torque", bus.disable_torque), ("close", bus.close)):
            try:
                operation()
            except BaseException as exc:
                errors[label] = f"{type(exc).__name__}: {exc}"
                if not isinstance(exc, Exception):
                    interrupts.append(exc)
        if errors:
            if primary is not None and not isinstance(primary, Exception):
                primary.add_note(str(DeviceOperationError(errors)))
                logger.error("Calibration cleanup failed: %s", DeviceOperationError(errors))
            elif interrupts:
                if primary is not None:
                    interrupts[0].add_note(f"Calibration failed: {primary}")
                raise interrupts[0]
            else:
                if primary is not None:
                    errors = {"calibration": f"{type(primary).__name__}: {primary}", **errors}
                raise DeviceOperationError(errors) from primary


def calibrate(
    target,
    robot_model,
    *,
    device_id=None,
    calibration_dir=None,
    left_port=None,
    right_port=None,
    arm_profile=None,
    rehome=False,
):
    """Calibrate local leaders, the full robot, or arms only using the deployed JSON."""
    model = get_robot_model(robot_model)
    if target not in ("leader", "robot", "arms"):
        raise ValueError("Calibration target must be leader, robot or arms")
    if type(rehome) is not bool or (rehome and target != "arms"):
        raise ValueError("--rehome is only valid for arms calibration")
    expected_profile = "so-arm-5dof" if robot_model == "alohamini1" else "am-leader-6dof"
    if arm_profile is not None and (target != "leader" or arm_profile != expected_profile):
        raise ValueError(
            f"Leader profile must match {expected_profile}; only valid for leader calibration"
        )
    if not sys.stdin.isatty():
        raise RuntimeError("Calibration requires an interactive terminal")
    leader = target == "leader"
    device_id = device_id or (
        ("so101_leader_bi" if robot_model == "alohamini1" else "am_leader_bi")
        if leader
        else "AlohaMiniRobot"
    )
    role = "teleoperators" if leader else "robots"
    ports = (
        left_port or f"/dev/am_arm_{'leader' if leader else 'follower'}_left",
        right_port or f"/dev/am_arm_{'leader' if leader else 'follower'}_right",
    )
    if Path(ports[0]).resolve() == Path(ports[1]).resolve():
        raise ValueError("Left and right buses must use different serial devices")
    paths, actuators, existing = {}, {}, {}
    workspace = WorkspacePaths()
    for side in ("left", "right"):
        names = tuple(m for m in model.actuators if m.bus == side)
        if target == "arms":
            names = tuple(m for m in names if m.name.startswith("arm_"))
        if leader:
            names = tuple(
                ActuatorSpec(m.name.removeprefix(f"arm_{side}_"), side, m.motor_id, "sts3215")
                for m in names
                if m.name.startswith("arm_")
            )
        actuators[side] = names
        file_id = f"{device_id}_{side}" if leader else device_id
        path = workspace.calibration_file(role, file_id)
        paths[side] = (
            path if calibration_dir is None else Path(calibration_dir).expanduser() / path.name
        )
    for side, path in paths.items():
        expected = actuators[side] if leader else model.actuators
        existing[side] = load_motor_calibration(path, expected) if path.exists() else {}
        if target == "arms" and (not existing[side] or path.is_symlink()):
            raise ValueError("Arms calibration requires an existing full robot JSON, not a symlink")

    if target == "arms":
        operation = (
            "rewrites homing offsets and ranges"
            if rehome
            else "preserves homing offsets and rewrites only ranges"
        )
        print(
            f"DANGER: real arm EEPROM calibration; {operation}.\n"
            "Support both arms; stop Host, teleoperation and every other serial owner.\n"
            "No torque enable, motion command, lift homing or base/lift register writes.",
            file=sys.stderr,
        )
        if (
            input("Type exactly 'CALIBRATE BOTH ARMS' to continue: ").strip()
            != "CALIBRATE BOTH ARMS"
        ):
            raise InterruptedError("Calibration cancelled; no serial port was opened")
    with ExitStack() as cleanup:
        buses = {}
        session = uuid4().hex
        for side, port in zip(("left", "right"), ports, strict=True):
            bus = FeetechBusDevice(
                port, actuators[side], position_calibrations={}, velocity_limits={}
            )
            buses[side] = bus
            cleanup.enter_context(_connected_bus(bus, session))
        # Disable every verified motor on both buses before modifying either bus.
        for bus in buses.values():
            bus.disable_torque()
        if leader:
            for side, bus in buses.items():
                _calibrate_leader(bus, f"{device_id}_{side}", paths[side], existing[side])
        elif target == "arms":
            _calibrate_arms(buses, paths["left"], existing["left"], model, rehome)
        else:
            _calibrate_robot(buses, device_id, paths["left"], existing["left"])
