# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Migrated from src/lerobot/scripts/lerobot_find_port.py.
"""Identify a controller by USB disconnection, without opening serial ports."""

import time


def find_available_ports() -> list[str]:
    from serial.tools import list_ports

    # Exclude symlink aliases: one controller must not appear as several ports.
    return sorted({port.device for port in list_ports.comports(include_links=False)})


def find_port() -> str:
    print("拔插 USB 前先停止遥操、数采或 Host，并支撑可能下落的部件。")
    print("Finding all available ports for the MotorsBus.")
    ports_before = find_available_ports()
    print("Ports before disconnecting:", ports_before)
    if not ports_before:
        raise OSError("No serial ports found. Connect the USB controller and try again.")

    print("Remove the USB cable from your MotorsBus and press Enter when done.")
    try:
        input()  # Disconnect one controller at a time, as in the original tool.
        time.sleep(0.5)
        ports_after = find_available_ports()
        ports_diff = sorted(set(ports_before) - set(ports_after))

        if len(ports_diff) == 1:
            port = ports_diff[0]
            print(f"The port of this MotorsBus is '{port}'")
            return port
        elif len(ports_diff) == 0:
            raise OSError(f"Could not detect the port. No difference was found ({ports_diff}).")
        else:
            raise OSError(
                f"Could not detect the port. More than one port was found ({ports_diff})."
            )
    finally:
        print("Reconnect the USB cable.")
