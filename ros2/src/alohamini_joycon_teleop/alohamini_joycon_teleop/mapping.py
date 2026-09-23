from __future__ import annotations

import math


def normalize_stick(raw: float, deadzone: float) -> float:
    """Normalize a Joy-Con 0..4095 stick channel with a radial-style deadzone."""
    if not 0.0 <= deadzone < 1.0:
        raise ValueError("deadzone must be in [0, 1)")
    value = max(-1.0, min(1.0, (float(raw) - 2000.0) / 2000.0))
    if abs(value) <= deadzone:
        return 0.0
    return math.copysign((abs(value) - deadzone) / (1.0 - deadzone), value)


def quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    """Multiply XYZW quaternions."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    result = [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]
    norm = math.sqrt(sum(value * value for value in result))
    if norm <= 1.0e-12:
        raise ValueError("zero quaternion")
    return [value / norm for value in result]


def quaternion_conjugate(quaternion: list[float]) -> list[float]:
    x, y, z, w = quaternion
    norm_squared = sum(value * value for value in quaternion)
    if norm_squared <= 1.0e-12:
        raise ValueError("zero quaternion")
    return [-x / norm_squared, -y / norm_squared, -z / norm_squared, w / norm_squared]


def quaternion_rotate_vector(quaternion: list[float], vector: list[float]) -> list[float]:
    """Rotate a 3-vector by an XYZW quaternion without allocating matrices."""
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def quaternion_axis_signs(quaternion: list[float], signs: list[float]) -> list[float]:
    """Apply configured roll/pitch/yaw axis signs to a relative quaternion."""
    if len(signs) != 3 or any(abs(float(sign)) != 1.0 for sign in signs):
        raise ValueError("quaternion axis signs must contain three +/-1 values")
    x, y, z, w = quaternion
    result = [x * signs[0], y * signs[1], z * signs[2], w]
    norm = math.sqrt(sum(value * value for value in result))
    if norm <= 1.0e-12:
        raise ValueError("zero quaternion")
    return [value / norm for value in result]


def relative_quaternion(current: list[float], anchor: list[float]) -> list[float]:
    """Return the local/controller-frame rotation since the gesture latch."""
    return quaternion_multiply(quaternion_conjugate(anchor), current)


def base_relative_quaternion(current: list[float], anchor: list[float]) -> list[float]:
    """Return the active rotation in the fixed arm-base coordinate axes."""
    return quaternion_multiply(current, quaternion_conjugate(anchor))


def faucet_translation_velocity(
    stick_horizontal: float,
    stick_vertical: float,
    relative_attitude: list[float],
    speed: float,
    vertical_input: float = 0.0,
) -> list[float]:
    """Map Joy-Con motion to TCP XYZ velocity in the arm-base frame.

    At the clutch latch, stick-up is robot-forward (``-Y``), stick-right is
    robot-right (``-X``), and the auxiliary input is up (``+Z``). Tilting the
    hand rotates these three translation directions: the controller behaves
    like a faucet/nozzle while the command remains an XYZ displacement.
    """
    nominal = [
        -float(stick_horizontal),
        -float(stick_vertical),
        float(vertical_input),
    ]
    length = math.sqrt(sum(value * value for value in nominal))
    if length > 1.0:
        nominal = [value / length for value in nominal]
    return [speed * value for value in quaternion_rotate_vector(relative_attitude, nominal)]


def quaternion_slerp(start: list[float], goal: list[float], t: float) -> list[float]:
    """Spherical linear interpolation between XYZW quaternions."""
    dot = sum(a * b for a, b in zip(start, goal, strict=True))
    dot = max(-1.0, min(1.0, dot))
    if dot < 0.0:
        goal = [-value for value in goal]
        dot = -dot
    if dot > 1.0 - 1.0e-9:
        return [a + t * (b - a) for a, b in zip(start, goal, strict=True)]
    theta = math.acos(dot)
    start_weight = math.sin((1.0 - t) * theta) / math.sin(theta)
    goal_weight = math.sin(t * theta) / math.sin(theta)
    return [start_weight * a + goal_weight * b for a, b in zip(start, goal, strict=True)]


def step_orientation_toward(
    current: list[float], commanded: list[float], max_step_rad: float
) -> list[float]:
    """Rotate ``current`` toward ``commanded`` by at most ``max_step_rad``.

    A fixed commanded attitude computed from the latch delta would otherwise be
    rejected forever by the per-request joint-step filter; stepping the target
    keeps each IK request small and reachable.
    """
    dot = sum(a * b for a, b in zip(current, commanded, strict=True))
    dot = max(-1.0, min(1.0, dot))
    angle = 2.0 * math.acos(abs(dot))
    if angle <= max_step_rad or angle <= 1.0e-9:
        return list(commanded)
    return quaternion_slerp(current, commanded, max_step_rad / angle)


def euler_delta_quaternion(roll: float, pitch: float, yaw: float) -> list[float]:
    """Return an XYZW quaternion for intrinsic XYZ Euler increments."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


# TCP orientation in the {side}_Base frame of the classic "level grasp" rest
# pose (arm extended forward, gripper opening straight down, jaws horizontal).
# Derived from the validated MoveIt FK golden reference transform
# (alohamini_validation/config/fk_golden.yaml, sample "reference").
LEVEL_TCP_QUATERNION = [math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0]


def tcp_button_delta(buttons: dict[str, bool], speed: float, dt: float) -> list[float]:
    """Map face buttons to a TCP translation step in the {side}_Base frame.

    Both arm base frames are root-aligned (+x = robot left, -y = robot
    forward, +z = up), so the same mapping reads naturally for either hand:

    - X (left Joy-Con: d-pad up): forward  (-y)
    - B (left Joy-Con: d-pad down): backward (+y)
    - Y (left Joy-Con: d-pad left): left (+x)
    - A (left Joy-Con: d-pad right): right (-x)
    - shoulder + X: up (+z)
    - shoulder + B: down (-z)
    """
    shoulder = bool(buttons.get("shoulder"))
    forward = bool(buttons.get("up")) and not shoulder
    backward = bool(buttons.get("down")) and not shoulder
    # A shoulder-modified face-button gesture is a distinct command.  Do not
    # also emit the unmodified X displacement from left/right while shoulder
    # is held (those combinations are currently intentionally unassigned).
    left = bool(buttons.get("left")) and not shoulder
    right = bool(buttons.get("right")) and not shoulder
    up = shoulder and bool(buttons.get("up"))
    down = shoulder and bool(buttons.get("down"))
    step = speed * dt
    return [
        step * (float(left) - float(right)),
        step * (float(backward) - float(forward)),
        step * (float(up) - float(down)),
    ]


def lift_stick_requested(buttons: dict[str, bool], stick_vertical: float, deadzone: float) -> bool:
    """Reserve the left stick for lift only while L and the stick are active."""
    clutch = bool(buttons.get("sl") or buttons.get("sr"))
    return bool(
        buttons.get("shoulder")
        and not clutch
        and abs(normalize_stick(stick_vertical, deadzone)) > 1.0e-9
    )


def next_lift_stick_latch(
    latched: bool,
    buttons: dict[str, bool],
    stick_vertical: float,
    deadzone: float,
) -> bool:
    """Keep one L+stick gesture in lift mode until the stick is centered."""
    clutch = bool(buttons.get("sl") or buttons.get("sr"))
    stick_active = abs(normalize_stick(stick_vertical, deadzone)) > 1.0e-9
    if clutch or not stick_active:
        return False
    return bool(latched or buttons.get("shoulder"))


def base_command(
    sticks: list[tuple[float, float]],
    linear_speed: float,
    angular_speed: float,
) -> tuple[float, float, float]:
    """Map one or two sticks to a LeRobot body velocity command.

    ``sticks`` holds (horizontal, vertical) normalized inputs summed with
    clamping: +x forward, +y left. Yaw is always zero here; the caller adds
    the shoulder-modified turn stick separately.
    """
    del angular_speed
    horizontal = 0.0
    vertical = 0.0
    for stick_h, stick_v in sticks:
        horizontal += stick_h
        vertical += stick_v
    horizontal = max(-1.0, min(1.0, horizontal))
    vertical = max(-1.0, min(1.0, vertical))
    return (
        linear_speed * vertical,
        -linear_speed * horizontal,
        0.0,
    )


def integrate_base_preview(
    pose: list[float], vx: float, vy: float, yaw_rate: float, dt: float
) -> list[float]:
    """Integrate a preview pose in REP-103 ``base_link`` coordinates."""
    return [
        float(pose[0]) + float(vx) * float(dt),
        float(pose[1]) + float(vy) * float(dt),
        float(pose[2]) + float(yaw_rate) * float(dt),
    ]
