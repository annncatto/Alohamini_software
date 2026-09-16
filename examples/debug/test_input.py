# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import platform
import sys
import time
from contextlib import suppress


def main(argv=None):
    argparse.ArgumentParser(
        description="Keyboard input and desktop session diagnostics"
    ).parse_args(argv)
    print("=== Env Check ===")
    print("platform:", platform.platform())
    print("XDG_SESSION_TYPE:", os.environ.get("XDG_SESSION_TYPE"))
    print("DISPLAY:", os.environ.get("DISPLAY"))
    print("WAYLAND_DISPLAY:", os.environ.get("WAYLAND_DISPLAY"))
    print("SSH_CONNECTION:", os.environ.get("SSH_CONNECTION"))
    print("TTY:", os.ttyname(0) if sys.stdin.isatty() else "not a tty")

    print("\n=== Pynput Keyboard Listener Smoke Test (5s) ===")
    fired = {"press": False, "release": False}
    err = None
    listener = None

    try:
        from pynput import keyboard

        def on_press(key):
            fired["press"] = True
            with suppress(AttributeError):
                print(f"[on_press] {getattr(key, 'char', key)}")

        def on_release(key):
            fired["release"] = True
            print(f"[on_release] {key}")
            if key == keyboard.Key.esc:
                print("Esc detected → stopping listener")
                return False

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        print("Listening... press some keys (Esc to stop). Waiting up to 5s...")
        t0 = time.time()
        while time.time() - t0 < 5 and listener.is_alive():
            time.sleep(0.05)
        if listener.is_alive():
            listener.stop()
        time.sleep(0.2)

    except Exception as e:
        err = e
    finally:
        if listener is not None:
            listener.stop()
            if listener.ident is not None:
                listener.join(timeout=1)

    print("\n=== Result ===")
    print("callbacks fired:", fired, "| error:", repr(err))

    print("\n=== Diagnosis Hint ===")
    if err:
        print("• 初始化报错，多为环境/依赖问题。")
    elif not fired["press"] and not fired["release"]:
        print("• 5 秒内未捕获事件：")
        print("  - 若 XDG_SESSION_TYPE=wayland → 请改用“Ubuntu on Xorg”登录。")
        print("  - 若在 SSH/TTY/tmux 中 → 请在本机图形终端里运行。")
        print("  - 若 DISPLAY 为空 → 没有连接到 X Server。")
    else:
        print("• 已成功捕获事件：pynput 正常，问题可能在你的业务脚本（回调、线程或主循环阻塞）。")
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
