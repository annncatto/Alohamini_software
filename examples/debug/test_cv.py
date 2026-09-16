# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect OpenCV and capture one local camera without desktop GUI dependencies."""

import argparse
import platform
import sys

from alohamini.hardware.find_cameras import save_images_from_cameras


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="0", help="Local /dev path or video index (default: 0)")
    parser.add_argument(
        "--output-dir", help="New snapshot directory; default: workspace logs/debug/"
    )
    parser.add_argument("--record-time-s", type=float, default=6.0)
    args = parser.parse_args(argv)
    import cv2

    print("Python:", sys.version.split()[0])
    print("OS:", platform.system(), platform.release())
    print("cv2 path:", cv2.__file__)
    print("cv2 version:", cv2.__version__)
    device = f"/dev/video{int(args.camera)}" if args.camera.isdecimal() else args.camera
    try:
        save_images_from_cameras(
            [{"type": "OpenCV", "id": device}], args.output_dir, args.record_time_s
        )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Camera test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
