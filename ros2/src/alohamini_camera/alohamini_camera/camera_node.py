from __future__ import annotations

import io
import math
import time
from collections import deque
from pathlib import Path

import rclpy
import zmq
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from PIL import Image as PilImage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from alohamini.paths import WorkspacePaths

from .camera_info import (
    camera_info_is_accepted,
    load_camera_info_document,
    validate_frame_against_camera_info,
)
from .protocol import CameraFrame, parse_camera_message

CAMERA_DEFAULTS = {
    "forward": ("forward_camera_optical", "forward.yaml"),
    "backward": ("backward_camera_optical", "backward.yaml"),
    "chest": ("chest_camera_optical", "chest.yaml"),
    "wrist_right": ("right_camera_optical", "wrist_right.yaml"),
    "wrist_left": ("left_camera_optical", "wrist_left.yaml"),
}


def ros_stamp_from_unix_ns(unix_ns: int):
    from builtin_interfaces.msg import Time

    return Time(sec=unix_ns // 1_000_000_000, nanosec=unix_ns % 1_000_000_000)


class AlohaMiniCameraNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("alohamini_camera", **kwargs)
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 5557)
        self.declare_parameter("camera_names", ["auto"])
        self.declare_parameter("calibration_dir", str(WorkspacePaths().calibration / "cameras"))
        self.declare_parameter("publish_raw", True)
        self.declare_parameter("timestamp_mode", "receipt")
        self.declare_parameter("max_transport_latency_ms", 250.0)
        self.declare_parameter("poll_rate_hz", 200.0)
        calibration_dir = Path(str(self.get_parameter("calibration_dir").value)).expanduser()
        if not calibration_dir.is_absolute():
            raise ValueError("calibration_dir must be an absolute directory")
        requested = self.get_parameter("camera_names").value or []
        if not isinstance(requested, (list, tuple)) or any(
            not isinstance(name, str) for name in requested
        ):
            raise ValueError("camera_names must be a list of camera names")
        if len(set(requested)) != len(requested):
            raise ValueError("camera_names must not contain duplicates")
        self.auto_cameras = requested == ["auto"]
        if not requested:
            raise ValueError("Use ['auto'] to follow Host cameras, or provide camera names")
        self.camera_names = tuple(CAMERA_DEFAULTS) if self.auto_cameras else tuple(requested)
        unknown = set(self.camera_names) - set(CAMERA_DEFAULTS)
        if unknown:
            raise ValueError(f"unsupported camera names: {sorted(unknown)}")
        self.publish_raw = bool(self.get_parameter("publish_raw").value)
        self.timestamp_mode = str(self.get_parameter("timestamp_mode").value)
        if self.timestamp_mode not in ("host_wall", "receipt"):
            raise ValueError("timestamp_mode must be host_wall or receipt")
        self.max_transport_latency_ms = float(self.get_parameter("max_transport_latency_ms").value)
        poll_rate = float(self.get_parameter("poll_rate_hz").value)
        if not math.isfinite(poll_rate) or not 0 < poll_rate <= 1000:
            raise ValueError("poll_rate_hz must be in (0, 1000]")
        if not math.isfinite(self.max_transport_latency_ms) or self.max_transport_latency_ms <= 0:
            raise ValueError("max_transport_latency_ms must be positive and finite")
        host = str(self.get_parameter("host").value)
        port = self.get_parameter("port").value
        if not host or any(c.isspace() or c in "/:[]" for c in host):
            raise ValueError("host must be an IPv4 address or hostname")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("port must be an integer in [1, 65535]")

        self.camera_info_documents = {}
        self.frame_ids = {}
        self.raw_publishers = {}
        self.compressed_publishers = {}
        self.info_publishers = {}
        for camera_name in self.camera_names:
            default_frame, default_file = CAMERA_DEFAULTS[camera_name]
            self.declare_parameter(f"{camera_name}.frame_id", default_frame)
            self.declare_parameter(f"{camera_name}.namespace", f"/alohamini/cameras/{camera_name}")
            self.declare_parameter(f"{camera_name}.camera_info_url", "")
            frame_id = str(self.get_parameter(f"{camera_name}.frame_id").value)
            self.frame_ids[camera_name] = frame_id
            configured_path = str(
                self.get_parameter(f"{camera_name}.camera_info_url").value
            ).strip()
            default_path = calibration_dir / "intrinsics" / default_file
            info_path = Path(configured_path).expanduser() if configured_path else default_path
            if configured_path or info_path.is_file():
                document = load_camera_info_document(info_path)
                if document["frame_id"] != frame_id or document["camera_name"] != camera_name:
                    raise ValueError(
                        f"{info_path}: camera name or optical frame differs from configuration"
                    )
                if not camera_info_is_accepted(document):
                    if configured_path:
                        raise ValueError(f"{info_path}: camera intrinsics are not accepted")
                    document = None
            else:
                # Intrinsics must never be fabricated merely to make a new camera
                # stream available for calibration. Images remain usable, while
                # CameraInfo is intentionally absent until that camera is solved.
                document = None
            self.camera_info_documents[camera_name] = document
            namespace = str(self.get_parameter(f"{camera_name}.namespace").value).rstrip("/")
            if self.publish_raw:
                self.raw_publishers[camera_name] = self.create_publisher(
                    Image, f"{namespace}/image_raw", 2
                )
            self.compressed_publishers[camera_name] = self.create_publisher(
                CompressedImage, f"{namespace}/image_raw/compressed", 2
            )
            if document is not None:
                self.info_publishers[camera_name] = self.create_publisher(
                    CameraInfo, f"{namespace}/camera_info", 2
                )

        self.last_sequence = dict.fromkeys(self.camera_names, 0)
        self.received = dict.fromkeys(self.camera_names, 0)
        self.raw_published = dict.fromkeys(self.camera_names, 0)
        self.dropped = dict.fromkeys(self.camera_names, 0)
        self.invalid = 0
        self.last_latency_ms = dict.fromkeys(self.camera_names, float("nan"))
        self.last_frame_monotonic = dict.fromkeys(self.camera_names, None)
        self.host_session_id = None
        self.retired_sessions = deque(maxlen=8)
        self._last_invalid_warning = -math.inf
        self.diagnostic_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.create_timer(1.0 / poll_rate, self.poll_messages)
        self.create_timer(1.0, self.publish_diagnostics)
        self.context_zmq = zmq.Context()
        self.socket = None
        try:
            self.socket = self.context_zmq.socket(zmq.SUB)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.MAXMSGSIZE, 8 * 1024 * 1024)
            self.socket.setsockopt(zmq.RCVHWM, max(4, len(self.camera_names) * 2))
            if self.auto_cameras:
                self.socket.setsockopt(zmq.SUBSCRIBE, b"camera/")
            else:
                for camera_name in self.camera_names:
                    self.socket.setsockopt(zmq.SUBSCRIBE, f"camera/{camera_name}".encode())
            self.socket.connect(f"tcp://{host}:{port}")
        except BaseException:
            self.destroy_node()
            raise
        self.get_logger().info(
            f"Subscribing to AlohaMini Host cameras at tcp://{host}:{port}: {self.camera_names}"
        )

    def destroy_node(self):
        if getattr(self, "socket", None) is not None:
            self.socket.close(linger=0)
            self.socket = None
        if getattr(self, "context_zmq", None) is not None:
            self.context_zmq.term()
            self.context_zmq = None
        return super().destroy_node()

    def poll_messages(self) -> None:
        for _ in range(32):
            try:
                parts = self.socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            try:
                frame = parse_camera_message(parts)
                self.publish_frame(frame)
            except Exception as error:
                self.invalid += 1
                now = time.monotonic()
                if now - self._last_invalid_warning >= 1:
                    self.get_logger().warning(f"Rejected camera frame: {error}")
                    self._last_invalid_warning = now

    def frame_stamp(self, frame: CameraFrame):
        if self.timestamp_mode == "receipt":
            return self.get_clock().now().to_msg()
        return ros_stamp_from_unix_ns(frame.capture_unix_ns)

    def publish_frame(self, frame: CameraFrame) -> None:
        if frame.camera_name not in self.camera_names:
            return
        if frame.host_session_id in self.retired_sessions or (
            self.host_session_id is not None and frame.host_session_id is None
        ):
            self.dropped[frame.camera_name] += 1
            return
        changed_session = frame.host_session_id != self.host_session_id
        previous = 0 if changed_session else self.last_sequence[frame.camera_name]
        if frame.sequence <= previous:
            self.dropped[frame.camera_name] += 1
            return
        document = self.camera_info_documents[frame.camera_name]
        if document is not None:
            validate_frame_against_camera_info(
                document, frame.camera_name, frame.width, frame.height
            )
        stamp = self.frame_stamp(frame)
        frame_id = self.frame_ids[frame.camera_name]

        raw_publisher = self.raw_publishers.get(frame.camera_name)
        raw_data = None
        with PilImage.open(io.BytesIO(frame.jpeg)) as decoded:
            if decoded.format != "JPEG" or decoded.size != (frame.width, frame.height):
                raise ValueError("decoded JPEG format/dimensions do not match metadata")
            if raw_publisher is not None and raw_publisher.get_subscription_count() > 0:
                raw_data = decoded.convert("RGB").tobytes()
        if changed_session:
            if self.host_session_id is not None:
                self.retired_sessions.append(self.host_session_id)
            self.host_session_id = frame.host_session_id
            self.last_sequence = dict.fromkeys(self.camera_names, 0)
            self.last_frame_monotonic = dict.fromkeys(self.camera_names, None)
        if previous and frame.sequence > previous + 1:
            self.dropped[frame.camera_name] += frame.sequence - previous - 1

        compressed = CompressedImage()
        compressed.header.stamp = stamp
        compressed.header.frame_id = frame_id
        compressed.format = "jpeg"
        compressed.data = frame.jpeg
        self.compressed_publishers[frame.camera_name].publish(compressed)

        if raw_data is not None:
            image = Image()
            image.header = compressed.header
            image.height = frame.height
            image.width = frame.width
            image.encoding = "rgb8"
            image.is_bigendian = False
            image.step = frame.width * 3
            image.data = raw_data
            raw_publisher.publish(image)
            self.raw_published[frame.camera_name] += 1

        info_publisher = self.info_publishers.get(frame.camera_name)
        if document is not None and info_publisher is not None:
            info = CameraInfo()
            info.header = compressed.header
            info.height = int(document["image_height"])
            info.width = int(document["image_width"])
            info.distortion_model = str(document["distortion_model"])
            info.d = [float(value) for value in document["distortion_coefficients"]["data"]]
            info.k = [float(value) for value in document["camera_matrix"]["data"]]
            info.r = [float(value) for value in document["rectification_matrix"]["data"]]
            info.p = [float(value) for value in document["projection_matrix"]["data"]]
            info_publisher.publish(info)

        receipt_unix_ns = time.time_ns()
        self.last_latency_ms[frame.camera_name] = (receipt_unix_ns - frame.capture_unix_ns) / 1e6
        self.last_sequence[frame.camera_name] = frame.sequence
        self.received[frame.camera_name] += 1
        self.last_frame_monotonic[frame.camera_name] = time.monotonic()

    def publish_diagnostics(self) -> None:
        now = self.get_clock().now().to_msg()
        array = DiagnosticArray()
        array.header.stamp = now
        if self.auto_cameras and not any(self.last_frame_monotonic.values()):
            array.status.append(
                DiagnosticStatus(
                    level=DiagnosticStatus.WARN,
                    name="alohamini_camera/stream",
                    hardware_id="host_camera_stream",
                    message="waiting for Host camera streams",
                )
            )
        for camera_name in self.camera_names:
            if self.auto_cameras and self.last_frame_monotonic[camera_name] is None:
                continue
            status = DiagnosticStatus()
            status.name = f"alohamini_camera/{camera_name}"
            status.hardware_id = camera_name
            age = (
                float("inf")
                if self.last_frame_monotonic[camera_name] is None
                else time.monotonic() - self.last_frame_monotonic[camera_name]
            )
            latency = self.last_latency_ms[camera_name]
            if age > 0.5:
                status.level = DiagnosticStatus.ERROR
                status.message = "camera stream stale"
            elif (
                self.timestamp_mode == "host_wall" and abs(latency) > self.max_transport_latency_ms
            ):
                status.level = DiagnosticStatus.WARN
                status.message = "clock offset or transport latency too large"
            elif self.camera_info_documents[camera_name] is None:
                status.level = DiagnosticStatus.WARN
                status.message = "image stream healthy; intrinsic calibration unavailable"
            else:
                status.level = DiagnosticStatus.OK
                status.message = "camera stream healthy"
            status.values = [
                KeyValue(key="received", value=str(self.received[camera_name])),
                KeyValue(
                    key="raw_published",
                    value=str(self.raw_published[camera_name]),
                ),
                KeyValue(key="dropped", value=str(self.dropped[camera_name])),
                KeyValue(key="last_sequence", value=str(self.last_sequence[camera_name])),
                KeyValue(key="host_session_id", value=self.host_session_id or "legacy"),
                KeyValue(key="age_sec", value=f"{age:.3f}"),
                KeyValue(key="latency_ms", value=f"{latency:.3f}"),
                KeyValue(key="timestamp_mode", value=self.timestamp_mode),
                KeyValue(
                    key="camera_info",
                    value=(
                        "loaded"
                        if self.camera_info_documents[camera_name] is not None
                        else "unavailable_calibration_mode"
                    ),
                ),
            ]
            array.status.append(status)
        if self.invalid:
            array.status.append(
                DiagnosticStatus(
                    level=DiagnosticStatus.WARN,
                    name="alohamini_camera/protocol",
                    message="invalid camera messages received",
                    hardware_id="host_camera_stream",
                    values=[KeyValue(key="invalid", value=str(self.invalid))],
                )
            )
        self.diagnostic_pub.publish(array)


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = AlohaMiniCameraNode()
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
