#!/usr/bin/env python3
"""Read paired Joy-Cons and publish input-only JSON over localhost ZMQ.

This process deliberately imports neither LeRobot nor ROS. It never connects to
the robot Host and cannot send a robot command.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import signal
import time
from pathlib import Path
from urllib.parse import urlsplit

import zmq

from alohamini.paths import WorkspacePaths


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class ComplementaryAttitudeEstimator:
    """Relative yaw gyro integration with gravity-corrected roll and pitch."""

    def __init__(self, gravity_gain: float = 0.02) -> None:
        self.gravity_gain = float(gravity_gain)
        self.rpy = [0.0, 0.0, 0.0]
        self.initialized = False
        self.last_update: float | None = None

    @staticmethod
    def gravity_angles(accel: list[float]) -> tuple[float, float]:
        ax, ay, az = accel
        return (
            math.atan2(ay, -az),
            math.atan2(ax, math.sqrt(ay * ay + az * az)),
        )

    def update(
        self,
        gyro_samples: list[list[float]],
        accel_samples: list[list[float]],
        now: float,
    ) -> list[float]:
        if not gyro_samples or len(gyro_samples) != len(accel_samples):
            return list(self.rpy)
        elapsed = (
            1.0 / 50.0
            if self.last_update is None
            else min(0.05, max(0.001, now - self.last_update))
        )
        self.last_update = now
        sample_dt = elapsed / len(gyro_samples)
        for gyro, accel in zip(gyro_samples, accel_samples, strict=True):
            roll_acc, pitch_acc = self.gravity_angles(accel)
            if not self.initialized:
                self.rpy[:2] = [roll_acc, pitch_acc]
                self.initialized = True
            gx, gy, gz = gyro
            self.rpy[0] = wrap_angle(self.rpy[0] - gx * sample_dt)
            self.rpy[1] = wrap_angle(self.rpy[1] + gy * sample_dt)
            self.rpy[2] = wrap_angle(self.rpy[2] - gz * sample_dt)
            self.rpy[0] += self.gravity_gain * wrap_angle(roll_acc - self.rpy[0])
            self.rpy[1] += self.gravity_gain * wrap_angle(pitch_acc - self.rpy[1])
        return [wrap_angle(value) for value in self.rpy]


class SingleHidJoyCon:
    """One physical HID reader shared by buttons, sticks and IMU."""

    def __init__(self, side: str, skip_calibration: bool) -> None:
        from joyconrobotics.device import get_L_id, get_R_id
        from joyconrobotics.gyro import GyroTrackingJoyCon

        joycon_id = get_L_id() if side == "left" else get_R_id()
        self.joycon = GyroTrackingJoyCon(*joycon_id)
        self.gyro = self.joycon
        self.attitude = ComplementaryAttitudeEstimator()
        self.last_report_counter: int | None = None
        if not skip_calibration:
            self.joycon.calibrate(seconds=2)
            time.sleep(2.1)
            self.joycon.reset_orientation()

    def orientation(self, report_counter: int) -> list[float]:
        if report_counter == self.last_report_counter:
            return list(self.attitude.rpy)
        self.last_report_counter = report_counter
        return self.attitude.update(
            _vectors(self.gyro.gyro_in_rad),
            _vectors(self.gyro.accel_in_g),
            time.monotonic(),
        )

    def disconnnect(self) -> None:
        self.joycon._close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5567")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--sides", nargs="+", choices=("left", "right"), default=("left", "right"))
    parser.add_argument(
        "--skip-imu-calibration",
        action="store_true",
        help="Do not run the native two-second stationary IMU calibration.",
    )
    parser.add_argument(
        "--stick-log",
        default=str(WorkspacePaths().logs / "joycon_sticks.log"),
        help="Append-only raw stick log; defaults to the platform workspace logs.",
    )
    return parser.parse_args()


def button_state(controller, side: str) -> dict[str, bool]:
    joycon = controller.joycon
    common = {
        "stick": bool(
            joycon.get_button_l_stick() if side == "left" else joycon.get_button_r_stick()
        ),
        "shoulder": bool(joycon.get_button_l() if side == "left" else joycon.get_button_r()),
        "trigger": bool(joycon.get_button_zl() if side == "left" else joycon.get_button_zr()),
        "sl": bool(joycon.get_button_left_sl() if side == "left" else joycon.get_button_right_sl()),
        "sr": bool(joycon.get_button_left_sr() if side == "left" else joycon.get_button_right_sr()),
    }
    if side == "left":
        common.update(
            {
                "menu": bool(joycon.get_button_minus()),
                "relatch": bool(joycon.get_button_capture()),
                "up": bool(joycon.get_button_up()),
                "down": bool(joycon.get_button_down()),
                "left": bool(joycon.get_button_left()),
                "right": bool(joycon.get_button_right()),
            }
        )
    else:
        common.update(
            {
                "menu": bool(joycon.get_button_plus()),
                "relatch": bool(joycon.get_button_home()),
                "up": bool(joycon.get_button_x()),
                "down": bool(joycon.get_button_b()),
                "left": bool(joycon.get_button_y()),
                "right": bool(joycon.get_button_a()),
            }
        )
    return common


def euler_xyz_quaternion(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def _vectors(values) -> list[list[float]]:
    return [[float(component) for component in value] for value in values]


def sample(controller, side: str, sequence: int) -> dict:
    joycon = controller.joycon
    if side == "left":
        stick_vertical = joycon.get_stick_left_vertical()
        stick_horizontal = joycon.get_stick_left_horizontal()
    else:
        stick_vertical = joycon.get_stick_right_vertical()
        stick_horizontal = joycon.get_stick_right_horizontal()
    # The native report contains three IMU samples. Preserve all of them so a
    # recorded ZMQ stream can be replayed through a better attitude estimator;
    # do not reduce the control input to an already-filtered Euler triple.
    gyro = controller.gyro
    report = bytes(gyro._input_report)
    report_counter = int(report[1]) if len(report) > 1 else -1
    orientation_rpy = controller.orientation(report_counter)
    return {
        "schema_version": 2,
        "side": side,
        "sequence": sequence,
        "monotonic_ns": time.monotonic_ns(),
        "report_counter": report_counter,
        "stick": [float(stick_horizontal), float(stick_vertical)],
        "imu": {
            "accel_g": _vectors(gyro.accel_in_g),
            "gyro_rad_s": _vectors(gyro.gyro_in_rad),
        },
        "orientation_rpy": orientation_rpy,
        "orientation_xyzw": euler_xyz_quaternion(*orientation_rpy),
        "buttons": button_state(controller, side),
    }


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.rate_hz) or args.rate_hz <= 0.0:
        raise ValueError("rate-hz must be positive")
    endpoint = urlsplit(args.endpoint)
    if (
        endpoint.scheme != "tcp"
        or endpoint.hostname not in ("127.0.0.1", "localhost")
        or not endpoint.port
    ):
        raise ValueError("Joy-Con input must bind a localhost TCP endpoint")
    args.stick_log = Path(args.stick_log).expanduser()
    args.stick_log.parent.mkdir(parents=True, exist_ok=True)
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind(args.endpoint)
    controllers = {}
    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        for side in args.sides:
            print(
                f"[{side}] opening Joy-Con; native gyro-bias calibration takes "
                "about 2 s. Keep the controller STILL in any orientation "
                "(flat on the desk is fine). The calibration pose does not "
                "set the robot gripper pose: TCP orientation is controlled "
                "relative to the latch point of that controller.",
                flush=True,
            )
            controllers[side] = SingleHidJoyCon(side, args.skip_imu_calibration)
        period = 1.0 / args.rate_hz
        sequence = 0
        report_counters = dict.fromkeys(controllers)
        frozen_since = dict.fromkeys(controllers)
        reconnect_cooldown = dict.fromkeys(controllers, 0.0)
        stick_log_baseline = dict.fromkeys(controllers)
        stick_log_time = dict.fromkeys(controllers, 0.0)
        log_baseline = dict.fromkeys(controllers)
        log_deadline = dict.fromkeys(controllers, 0.0)
        log_write_time = dict.fromkeys(controllers, 0.0)
        with open(args.stick_log, "a", encoding="utf-8") as stick_log:
            stick_log.write(f"# reader restarted at {time.time():.1f}\n")
            while not stop:
                started = time.monotonic()
                sequence += 1
                for side, controller in controllers.items():
                    payload = sample(controller, side, sequence)
                    now = time.monotonic()
                    stick = tuple(payload["stick"])
                    # Append raw sticks to the on-disk log whenever they move
                    # (or every 10 s), so stick health can be inspected after
                    # the fact without timing coordination.
                    log_last = log_baseline[side]
                    if (
                        log_last is None
                        or (
                            max(
                                abs(stick[0] - log_last[0]),
                                abs(stick[1] - log_last[1]),
                            )
                            >= 2
                            and now - log_write_time[side] >= 0.2
                        )
                        or now >= log_deadline[side]
                    ):
                        log_baseline[side] = stick
                        log_write_time[side] = now
                        log_deadline[side] = now + 10.0
                        stick_log.write(
                            f"{time.time():.2f} {side} H={stick[0]:.0f} "
                            f"V={stick[1]:.0f} sh={int(payload['buttons']['shoulder'])}"
                            f" sl={int(payload['buttons']['sl'])} "
                            f"sr={int(payload['buttons']['sr'])}\n"
                        )
                    # Console print whenever the stick moves, so stick health
                    # is visible directly in the terminal (rate-limited).
                    baseline = stick_log_baseline[side]
                    if baseline is None:
                        stick_log_baseline[side] = stick
                    elif (
                        max(
                            abs(stick[0] - baseline[0]),
                            abs(stick[1] - baseline[1]),
                        )
                        > 200
                        and now - stick_log_time[side] > 1.0
                    ):
                        stick_log_time[side] = now
                        stick_log_baseline[side] = stick
                        print(
                            f"[{side}] stick H={stick[0]:.0f} V={stick[1]:.0f}",
                            flush=True,
                        )
                    # A stationary controller legitimately has unchanged sticks,
                    # buttons and a nearly constant attitude. Detect a dead HID
                    # reader from Nintendo's incrementing report counter instead
                    # of interpreting stationary data as a disconnect.
                    report_counter = payload["report_counter"]
                    if report_counter == report_counters[side]:
                        if frozen_since[side] is None:
                            frozen_since[side] = now
                        elif (
                            now - frozen_since[side] > 1.5 and now - reconnect_cooldown[side] > 5.0
                        ):
                            # The HID read thread can die silently on a
                            # Bluetooth hiccup (the native library has no
                            # read-loop error handling), leaving the input
                            # report frozen. Reopen the device so stale
                            # sticks/buttons cannot be sent.
                            reconnect_cooldown[side] = now
                            print(
                                f"[{side}] Joy-Con input report frozen; reopening the device",
                                flush=True,
                            )
                            try:
                                controller.disconnnect()
                                controllers[side] = SingleHidJoyCon(side, skip_calibration=True)
                                report_counters[side] = None
                                frozen_since[side] = None
                                stick_log_baseline[side] = None
                                log_baseline[side] = None
                                payload = sample(controllers[side], side, sequence)
                            except Exception as exc:
                                print(
                                    f"[{side}] reconnect failed ({exc}); will retry",
                                    flush=True,
                                )
                    else:
                        frozen_since[side] = None
                    if report_counter == report_counters[side]:
                        continue
                    report_counters[side] = report_counter
                    publisher.send_string(json.dumps(payload, separators=(",", ":")))
                time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        for controller in controllers.values():
            with contextlib.suppress(Exception):
                controller.disconnnect()
        publisher.close()
        context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
