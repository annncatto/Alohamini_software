from __future__ import annotations

import math
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from alohamini.paths import WorkspacePaths


def load_extrinsic(path: Path, *, allow_candidate: bool = False):
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"{path}: camera extrinsic must be an object")
    status = str(document.get("status", ""))
    if not status.startswith("accepted_") and not allow_candidate:
        raise ValueError(f"refusing non-accepted camera extrinsic {path}: {status}")
    transform = document.get("T_mount_link_from_camera_optical")
    if not isinstance(transform, dict):
        raise ValueError(f"{path} lacks T_mount_link_from_camera_optical")
    xyz = transform.get("xyz_m")
    orders = [key for key in ("quaternion_xyzw", "quaternion_wxyz") if key in transform]
    if len(orders) != 1:
        raise ValueError(f"{path}: provide exactly one explicit quaternion order")
    quaternion = transform[orders[0]]
    for value, size in ((xyz, 3), (quaternion, 4)):
        if (
            not isinstance(value, list)
            or len(value) != size
            or any(
                type(component) not in (int, float) or not math.isfinite(component)
                for component in value
            )
        ):
            raise ValueError(f"{path}: translation/quaternion must be finite vectors")
    if not math.isclose(sum(component**2 for component in quaternion), 1.0, abs_tol=1e-6):
        raise ValueError(f"{path}: quaternion must have unit length")
    if orders[0] == "quaternion_wxyz":
        quaternion = [*quaternion[1:], quaternion[0]]
    parent, child = document.get("mount_link"), document.get("optical_frame")
    if (
        any(
            not isinstance(frame, str)
            or not frame
            or frame.startswith("/")
            or any(c.isspace() for c in frame)
            for frame in (parent, child)
        )
        or parent == child
    ):
        raise ValueError(f"{path}: provide distinct valid mount_link and optical_frame")
    return parent, child, list(map(float, xyz)), list(map(float, quaternion))


class CameraExtrinsicsNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("alohamini_camera_extrinsics", **kwargs)
        self.declare_parameter("extrinsics_csv", "")
        self.declare_parameter("allow_candidate", False)
        self.declare_parameter("calibration_dir", str(WorkspacePaths().calibration / "cameras"))
        paths = [
            value.strip()
            for value in str(self.get_parameter("extrinsics_csv").value).split(",")
            if value.strip()
        ]
        allow_candidate = bool(self.get_parameter("allow_candidate").value)
        calibration_dir = Path(str(self.get_parameter("calibration_dir").value)).expanduser()
        if not calibration_dir.is_absolute():
            raise ValueError("calibration_dir must be absolute")
        broadcaster = StaticTransformBroadcaster(self)
        transforms = []
        parents = {}
        for configured in paths:
            path = Path(configured).expanduser()
            if not path.is_absolute():
                path = calibration_dir / "extrinsics" / path
            parent, child, xyz, quaternion = load_extrinsic(path, allow_candidate=allow_candidate)
            if child in parents:
                raise ValueError(f"duplicate optical TF child: {child}")
            parents[child] = parent
            chain = {child}
            while parent in parents:
                if parent in chain:
                    raise ValueError("camera extrinsics form a TF cycle")
                chain.add(parent)
                parent = parents[parent]
            message = TransformStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = parents[child]
            message.child_frame_id = child
            message.transform.translation.x = float(xyz[0])
            message.transform.translation.y = float(xyz[1])
            message.transform.translation.z = float(xyz[2])
            message.transform.rotation.x = float(quaternion[0])
            message.transform.rotation.y = float(quaternion[1])
            message.transform.rotation.z = float(quaternion[2])
            message.transform.rotation.w = float(quaternion[3])
            transforms.append(message)
        if transforms:
            broadcaster.sendTransform(transforms)
        self._broadcaster = broadcaster
        self.get_logger().info(f"Published {len(transforms)} calibrated camera transforms")


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = CameraExtrinsicsNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
