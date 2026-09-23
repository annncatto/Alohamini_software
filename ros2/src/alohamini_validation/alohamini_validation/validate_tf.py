# SPDX-License-Identifier: Apache-2.0
# Migrated from alohamini_ros2/alohamini_validation.
from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from alohamini.model import get_robot_model


def translation(transform) -> tuple[float, float, float]:
    value = transform.transform.translation
    return float(value.x), float(value.y), float(value.z)


def main() -> None:
    root = ET.parse(get_robot_model("alohamini2pro").description_path("collision")).getroot()
    frames = {link.get("name") for link in root.findall("link")}

    rclpy.init()
    node = Node(
        "alohamini_tf_validation",
        namespace="/alohamini_plan_only",
        cli_args=["--ros-args", "-r", "/tf:=tf", "-r", "/tf_static:=tf_static"],
    )
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    deadline = time.monotonic() + 8.0
    missing = set(frames) - {"root"}
    try:
        while missing and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            missing = {
                frame
                for frame in missing
                if not buffer.can_transform(
                    "root", frame, rclpy.time.Time(), timeout=Duration(seconds=0.0)
                )
            }
        assert not missing, f"missing TF frames: {sorted(missing)}"
        left = translation(buffer.lookup_transform("root", "left_tcp", rclpy.time.Time()))
        right = translation(buffer.lookup_transform("root", "right_tcp", rclpy.time.Time()))
        assert all(math.isfinite(value) for value in (*left, *right))
        # Live poses may be asymmetric, including the model's folded Home.
        for frame in frames - {"root"}:
            transform = buffer.lookup_transform("root", frame, rclpy.time.Time())
            assert all(math.isfinite(value) for value in translation(transform)), frame
            q = transform.transform.rotation
            assert math.isclose(
                sum(value * value for value in (q.x, q.y, q.z, q.w)), 1.0, abs_tol=1e-6
            ), frame
        print(f"[PASS] TF tree: {len(frames)} frames, root -> both TCPs available")
        print(f"       left_tcp={left}, right_tcp={right}")
    finally:
        del listener
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
