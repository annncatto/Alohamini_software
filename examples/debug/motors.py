# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Migrated from examples/debug/motors.py; retain raw-register table semantics.
"""Local STS diagnostics and torque-off maintenance; stop Host first."""

import argparse
import sys
import time
from collections import namedtuple

from alohamini.hardware.feetech import _PacketPort
from alohamini.hardware.feetech_device import _MODEL_NUMBERS

DEFAULT_PORT = "/dev/ttyACM0"
HALF_TURN_DEGREE = 180
SCAN_START = 1
SCAN_END = 22
Motor = namedtuple("Motor", "id model")

# Addresses and sign bits retained from Feetech STS_SMS_SERIES_CONTROL_TABLE.
REGISTERS = {
    "Model_Number": (3, 2, None),
    "ID": (5, 1, None),
    "Present_Position": (56, 2, 15),
    "Maximum_Acceleration": (85, 1, None),
    "Present_Voltage": (62, 1, None),
    "Present_Current": (69, 2, None),
    "Homing_Offset": (31, 2, 11),
    "Present_Temperature": (63, 1, None),
    "Phase": (18, 1, None),
    "Torque_Enable": (40, 1, None),
    "Lock": (55, 1, None),
}


class MotorStateReader:
    """One exclusive connection; only READ packets, including device discovery."""

    def __init__(self, port):
        self.port = port
        self.serial = None

    def __enter__(self):
        from scservo_sdk.protocol_packet_handler import protocol_packet_handler
        from serial import Serial

        self.handler = protocol_packet_handler()
        self.serial = Serial(
            port=None,
            baudrate=1_000_000,
            timeout=0,
            write_timeout=0.05,
            bytesize=8,
            parity="N",
            stopbits=1,
            exclusive=True,
        )
        self.serial.port = self.port
        try:
            self.serial.open()
        except BaseException:
            self.serial.close()
            raise
        print(f"Connected on port {self.port}")
        return self

    def __exit__(self, *_):
        self.serial.close()  # No torque or EEPROM writes, even on failure.

    def read(self, motor_id, register, *, num_retry=0):
        address, width, sign = REGISTERS[register]
        for attempt in range(num_retry + 1):
            # Source SDK allowance: 50 ms plus the reply's wire time at 1 Mbps.
            port = _PacketPort(self.serial, time.monotonic() + 0.05 + (width + 6) * 10e-6)
            try:
                data, result, error = self.handler.readTxRx(port, motor_id, address, width)
                if result != 0 or len(data) != width:
                    raise OSError(f"ID={motor_id} {register}: transport={result}")
                value = int.from_bytes(bytes(data), "little")
                if sign is not None and value & (1 << sign):
                    value = -(value & ~(1 << sign))
                # Fault flags accompany readable values; display them, never clear them.
                return value, error
            except OSError:
                if attempt == num_retry:
                    raise
            finally:
                port.is_using = False


class MotorMaintenance(MotorStateReader):
    """Explicit torque-off maintenance; never enable torque or select a motor mode."""

    def checked_read(self, motor_id, register):
        value, error = self.read(motor_id, register)
        if error:
            raise OSError(f"ID={motor_id} {register}: servo_error=0x{error:02x}")
        return value

    def identify(self, motor_id):
        number = self.checked_read(motor_id, "Model_Number")
        if number not in _MODEL_NUMBERS.values():
            raise ValueError(f"ID={motor_id}: unsupported model {number}")
        if self.checked_read(motor_id, "ID") != motor_id:
            raise OSError(f"ID={motor_id}: identity readback mismatch")
        return number

    def write(self, motor_id, register, value):
        address, width, _ = REGISTERS[register]
        # ID changes cannot be retried at the old address after a lost ACK.
        attempts = 1 if register == "ID" else 4
        for attempt in range(attempts):
            port = _PacketPort(self.serial, time.monotonic() + 0.05006)
            try:
                result, error = self.handler.writeTxRx(
                    port, motor_id, address, width, list(value.to_bytes(width, "little"))
                )
                if result:
                    raise OSError(f"ID={motor_id} {register}: transport={result}")
            except OSError:
                if attempt + 1 == attempts:
                    raise
                continue
            finally:
                port.is_using = False
            if error:
                raise OSError(
                    f"ID={motor_id} {register}: transport={result}, servo_error=0x{error:02x}"
                )
            return

    def write_verified(self, motor_id, register, value):
        self.write(motor_id, register, value)
        if self.checked_read(motor_id, register) != value:
            raise OSError(f"ID={motor_id} {register}: write verification failed")

    def relock_identity(self, motor_ids, model_number):
        # An ID write may take effect even if its acknowledgement is lost.
        found, errors = False, []
        for motor_id in motor_ids:
            try:
                number, _ = self.read(motor_id, "Model_Number")
            except OSError:
                continue
            found = True
            try:
                if number != model_number:
                    raise OSError(f"Unexpected model at ID={motor_id}; not writing Lock")
                self.write_verified(motor_id, "Lock", 1)
            except OSError as exc:
                errors.append(str(exc))
        if not found or errors:
            raise OSError("EEPROM lock unconfirmed; inspect before use: " + "; ".join(errors))


def _check_id(motor_id):
    if type(motor_id) is not int or not 1 <= motor_id <= 253:
        raise ValueError("Motor ID must be in [1, 253]")


def configure_motor_id(port, current_id, new_id):
    """Original ID change sequence, on one connection with verified torque-off/lock."""
    _check_id(current_id)
    _check_id(new_id)
    if current_id == new_id:
        raise ValueError("Current ID and new ID must differ")
    with MotorMaintenance(port) as bus:
        number = bus.identify(current_id)
        try:
            bus.read(new_id, "Model_Number")
        except OSError:
            pass
        else:
            raise ValueError(f"ID={new_id} is already occupied")
        bus.identify(current_id)  # Recheck communication after the unoccupied-ID probe.
        bus.write_verified(current_id, "Torque_Enable", 0)
        try:
            bus.write_verified(current_id, "Lock", 0)
            try:
                bus.write(current_id, "ID", new_id)
            finally:
                time.sleep(0.2)
            if bus.identify(new_id) != number:
                raise OSError("Model changed during ID verification")
            try:
                bus.read(current_id, "Model_Number")
            except OSError:
                pass
            else:
                raise OSError(f"Old ID={current_id} still responds")
        finally:
            bus.relock_identity((current_id, new_id), number)
        print(f"[OK] ID {current_id} -> {new_id}; torque disabled, EEPROM locked")


def configure_motor_phase(port, motor_id, new_phase):
    _check_id(motor_id)
    if type(new_phase) is not int or not 0 <= new_phase <= 255:
        raise ValueError("Phase must be in [0, 255]")
    with MotorMaintenance(port) as bus:
        number = bus.identify(motor_id)
        before = bus.checked_read(motor_id, "Phase")
        bus.write_verified(motor_id, "Torque_Enable", 0)
        try:
            if before != new_phase:
                bus.write_verified(motor_id, "Lock", 0)
                bus.write_verified(motor_id, "Phase", new_phase)
        finally:
            bus.relock_identity((motor_id,), number)
        print(f"[ID {motor_id}] Phase {before} -> {new_phase}; torque disabled, EEPROM locked")


def reset_motors_torque(port, motor_id=None):
    if motor_id is not None:
        _check_id(motor_id)
    with MotorMaintenance(port) as bus:
        ids = [motor_id] if motor_id is not None else [m.id for m in probe_scan_ids(bus).values()]
        if not ids:
            raise OSError("No supported motors found")
        errors = []
        for current_id in ids:
            try:
                # Faulted motors must still receive torque-off; never clear fault bits.
                number, _ = bus.read(current_id, "Model_Number")
                if number not in _MODEL_NUMBERS.values():
                    raise ValueError(f"ID={current_id}: unsupported model {number}")
                write_error = None
                try:
                    bus.write(current_id, "Torque_Enable", 0)
                except OSError as exc:
                    write_error = exc
                value, flags = bus.read(current_id, "Torque_Enable")
                if value != 0:
                    raise OSError(f"ID={current_id}: torque-off unconfirmed: {write_error}")
                print(f"------- motor_{current_id} reset torque complete!------")
                if flags:
                    print(f"ID={current_id}: servo_error=0x{flags:02x} remains set")
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
        if errors:
            raise OSError("; ".join(errors))


def probe_scan_ids(bus):
    """Inspect model numbers at the original local bus IDs 1–22."""
    models = {number: name for name, number in _MODEL_NUMBERS.items()}
    found = {}
    for motor_id in range(SCAN_START, SCAN_END + 1):
        try:
            number, error = bus.read(motor_id, "Model_Number", num_retry=1)
        except (OSError, TimeoutError):
            continue
        if number not in models:
            print(f"ID={motor_id}: unsupported model {number}; skipping STS register reads")
            continue
        if error:
            print(f"ID={motor_id}: servo_error=0x{error:02x}")
        found[f"motor_{motor_id}"] = Motor(motor_id, models[number])
    return found


def _motor_angle_from_position(position):
    return position / (4096 // 2) * HALF_TURN_DEGREE


def collect_states(bus, motors):
    rows, errors = [], []
    fields = {
        "Position": "Present_Position",
        "Acceleration": "Maximum_Acceleration",
        "Voltage": "Present_Voltage",
        "Current(mA)": "Present_Current",
        "Offset": "Homing_Offset",
        "Temperature": "Present_Temperature",
        "Phase": "Phase",
    }
    for name, motor in motors.items():
        state = {"ID": motor.id, "Model": motor.model}
        flags = 0
        for key, register in fields.items():
            try:
                value, error = bus.read(motor.id, register)
                flags |= error
                state[key] = value * 6.5 if key == "Current(mA)" else value
            except (OSError, TimeoutError) as exc:
                state[key] = None
                errors.append(str(exc))
        position = state["Position"]
        state["Angle"] = (
            None if position is None else round(_motor_angle_from_position(position), 1)
        )
        if flags:
            errors.append(f"{name}: servo_error=0x{flags:02x}")
        rows.append((name, state))
    return rows, errors


def get_motors_states(port):
    """
    Display the original live motor table, using online IDs in [SCAN_START, SCAN_END].
    """
    import shutil

    # Enable ANSI escape codes on Windows (no-op on other platforms).
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

    # ---------- ANSI helpers ----------
    CSI = "\x1b["

    def _hide_cursor():
        sys.stdout.write(f"{CSI}?25l")
        sys.stdout.flush()

    def _show_cursor():
        sys.stdout.write(f"{CSI}?25h")
        sys.stdout.flush()

    def _move_up(n):
        sys.stdout.write(f"{CSI}{n}A") if n > 0 else None
        sys.stdout.flush()

    def _clear_line():
        sys.stdout.write(f"{CSI}2K\r")

    def _term_width():
        try:
            return max(60, min(200, shutil.get_terminal_size().columns))
        except Exception:
            return 100

    def _format_row(name: str, st: dict, maxw: int) -> str:
        def F(v, w, a=">"):
            s = "-" if v is None else str(v)
            if len(s) > w:
                s = s[:w]
            return f"{s:{a}{w}}"

        row = (
            f"{F(name, 15, '<')} | {F(st.get('ID'), 3)} | {F(st.get('Model'), 8, '<')} | "
            f"{F(st.get('Position'), 6)} | "
            f"{F(st.get('Offset'), 6)} | {F(st.get('Angle'), 6)} | "
            f"{F(st.get('Acceleration'), 6)} | "
            f"{F(st.get('Voltage'), 4)} | {F(st.get('Current(mA)'), 8)} | "
            f"{F(st.get('Temperature'), 4)} | {F(st.get('Phase'), 5)}"
        )
        return row[:maxw] if len(row) > maxw else row

    with MotorStateReader(port) as bus:
        motors = probe_scan_ids(bus)
        if not motors:
            print(f"No motors found in ID range [{SCAN_START}, {SCAN_END}] on {port}.")
            return
        try:
            _hide_cursor()
            ids_str = ", ".join(str(m.id) for m in motors.values())
            print(f"Online IDs: [{ids_str}] on {port}")
            printed_lines = 0
            interval_s = 0.1
            while True:
                rows, errors = collect_states(bus, motors)
                maxw = _term_width()
                sep = "-" * min(maxw, 140)
                header = (
                    f"{'NAME':<15} | {'ID':>3} | {'MODEL':<8} | {'POS':>6} | "
                    f"{'OFF':>6} | {'ANG':>6} | "
                    f"{'ACC':>6} | {'VOLT':>4} | {'CURR(MA)':>8} | {'TEMP':>4} | {'PHASE':>5}"
                )
                header = header[:maxw] if len(header) > maxw else header
                frame_lines = (
                    [sep, header]
                    + [_format_row(n, s, maxw) for (n, s) in rows]
                    + [sep, f"Updated: {time.strftime('%H:%M:%S')}   (Ctrl+C 退出)", *errors]
                )

                _move_up(printed_lines)
                for ln in frame_lines:
                    _clear_line()
                    sys.stdout.write(ln + "\n")

                extra = printed_lines - len(frame_lines)
                for _ in range(max(0, extra)):
                    _clear_line()
                    sys.stdout.write("\n")

                sys.stdout.flush()
                printed_lines = len(frame_lines)
                time.sleep(interval_s)

        except KeyboardInterrupt:
            pass
        finally:
            _show_cursor()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="STS diagnostics and torque-off maintenance; stop Host first",
        epilog=(
            "Support the arm/lift before disabling torque. Isolate a motor before changing its ID."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("get_motors_states", "Read-only live register table"),
        ("configure_motor_id", "Change one isolated motor's ID; leave torque disabled"),
        ("configure_motor_phase", "Change one motor's Phase; leave torque disabled"),
        ("reset_motors_torque", "Disable torque; all supported IDs 1–22 if --id is omitted"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--port", default=DEFAULT_PORT)
        if name != "get_motors_states":
            command.add_argument("--id", type=int, required=name != "reset_motors_torque")
        if name == "configure_motor_id":
            command.add_argument("--set_id", type=int, required=True)
        if name == "configure_motor_phase":
            command.add_argument("--set_phase", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "get_motors_states":
            get_motors_states(args.port)
        elif args.command == "configure_motor_id":
            configure_motor_id(args.port, args.id, args.set_id)
        elif args.command == "configure_motor_phase":
            configure_motor_phase(args.port, args.id, args.set_phase)
        else:
            reset_motors_torque(args.port, args.id)
        return 0
    except (OSError, RuntimeError, ImportError, ValueError) as exc:
        print(
            f"motors: {exc}; check the port, permissions, and whether Host is running",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
