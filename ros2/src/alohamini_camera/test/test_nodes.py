import io
import json
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import rclpy
import yaml
import zmq
from alohamini_camera.camera_node import AlohaMiniCameraNode
from alohamini_camera.extrinsics_node import CameraExtrinsicsNode, load_extrinsic
from alohamini_camera.protocol import parse_camera_message
from PIL import Image as PilImage
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from test_protocol import camera_info

from alohamini.hardware.camera import JpegSnapshot
from alohamini.model import get_robot_model
from alohamini.paths import WorkspacePaths
from alohamini.runtime.camera_stream import encode_camera_stream_message


@pytest.fixture
def ros(monkeypatch):
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_DOMAIN_ID", "83")
    rclpy.init(args=[])
    nodes = []

    def create(cls=AlohaMiniCameraNode, **parameters):
        node = cls(
            parameter_overrides=[Parameter(name, value=value) for name, value in parameters.items()]
        )
        nodes.append(node)
        return node

    yield create
    for node in reversed(nodes):
        node.destroy_node()
    rclpy.shutdown()


@pytest.fixture
def publisher():
    with zmq.Context() as context, context.socket(zmq.PUB) as socket:
        socket.linger = 0
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        yield socket, port


def parts(session="session-a", sequence=1, camera="forward"):
    output = io.BytesIO()
    PilImage.new("RGB", (16, 8), (255, 0, 0)).save(output, format="JPEG", quality=70)
    return encode_camera_stream_message(
        camera,
        JpegSnapshot(time.monotonic(), output.getvalue(), 16, 8),
        sequence,
        host_session_id=session,
    )


def camera(ros, publisher, tmp_path, **parameters):
    return ros(host="127.0.0.1", port=publisher[1], calibration_dir=str(tmp_path), **parameters)


def test_native_model_assets_and_storage_paths_work_in_ros_install():
    model = get_robot_model("alohamini2pro")
    assert model.description_path("collision").is_file()
    assert model.actuators[0].name == "arm_left_shoulder_pan"
    assert WorkspacePaths().calibration.name == "calibration"


@pytest.mark.parametrize("selection", ["[auto]", "[forward, wrist_right]"])
def test_launch_entry_accepts_default_and_explicit_camera_selection(
    publisher, tmp_path, monkeypatch, selection
):
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription, LaunchService
    from launch.actions import (
        EmitEvent,
        IncludeLaunchDescription,
        RegisterEventHandler,
        TimerAction,
    )
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown
    from launch.launch_description_sources import PythonLaunchDescriptionSource

    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_DOMAIN_ID", "83")
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "logs"))
    entry = Path(get_package_share_directory("alohamini_camera")) / "launch/camera.launch.py"
    exits = []
    service = LaunchService()
    service.include_launch_description(
        LaunchDescription(
            [
                RegisterEventHandler(
                    OnProcessExit(on_exit=lambda event, context: exits.append(event.returncode))
                ),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(str(entry)),
                    launch_arguments={
                        "host": "127.0.0.1",
                        "port": str(publisher[1]),
                        "cameras": selection,
                        "calibration_dir": str(tmp_path),
                    }.items(),
                ),
                TimerAction(
                    period=1.0, actions=[EmitEvent(event=Shutdown(reason="test complete"))]
                ),
            ]
        )
    )
    assert service.run() == 0
    assert exits == [0]


def test_host_restart_resets_all_camera_sequences_and_rejects_retired_messages(
    ros, publisher, tmp_path
):
    node = camera(ros, publisher, tmp_path)
    node.publish_frame(parse_camera_message(parts(sequence=100)))
    node.publish_frame(parse_camera_message(parts(sequence=200, camera="wrist_right")))
    node.publish_frame(parse_camera_message(parts(session="session-b")))
    assert node.last_sequence["forward"] == 1
    assert node.last_sequence["wrist_right"] == 0
    assert node.last_frame_monotonic["wrist_right"] is None
    node.publish_frame(parse_camera_message(parts(sequence=201, camera="wrist_right")))
    assert node.host_session_id == "session-b"
    assert node.last_sequence["wrist_right"] == 0
    node.publish_frame(parse_camera_message(parts(session="session-b", camera="wrist_right")))
    assert node.last_sequence["wrist_right"] == 1
    assert node.dropped["wrist_right"] == 1


def test_duplicate_or_malformed_frames_do_not_advance_session_or_publish(ros, publisher, tmp_path):
    node = camera(ros, publisher, tmp_path)
    frame = parse_camera_message(parts(sequence=5))
    node.publish_frame(frame)
    node.publish_frame(frame)
    assert node.received["forward"] == 1
    with pytest.raises(ValueError):
        node.publish_frame(replace(frame, host_session_id="bad-session", width=32))
    assert node.host_session_id == "session-a"
    assert node.received["forward"] == 1


def test_raw_decode_is_demand_driven_and_unused_cameras_are_not_stale_errors(
    ros, publisher, tmp_path
):
    node = camera(ros, publisher, tmp_path)
    with patch.object(PilImage.Image, "convert", side_effect=AssertionError("unrequested decode")):
        node.publish_frame(parse_camera_message(parts()))
    diagnostics = Mock()
    node.diagnostic_pub = diagnostics
    node.publish_diagnostics()
    statuses = diagnostics.publish.call_args.args[0].status
    assert [status.name for status in statuses] == ["alohamini_camera/forward"]
    assert node.raw_published["forward"] == 0
    assert not node.info_publishers


def test_candidate_default_does_not_publish_camera_info_but_explicit_candidate_fails(
    ros, publisher, tmp_path
):
    path = tmp_path / "intrinsics/forward.yaml"
    path.parent.mkdir()
    document = camera_info()
    document["status"] = "candidate_intrinsics_requires_review"
    path.write_text(yaml.safe_dump(document))
    node = camera(ros, publisher, tmp_path)
    node.publish_frame(parse_camera_message(parts()))
    assert "forward" not in node.info_publishers
    with pytest.raises(ValueError, match="not accepted"):
        camera(ros, publisher, tmp_path, **{"forward.camera_info_url": str(path)})


def test_wrong_optical_frame_is_not_silently_overwritten(ros, publisher, tmp_path):
    path = tmp_path / "wrong.yaml"
    document = camera_info()
    document["frame_id"] = "other_camera_optical"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="optical frame"):
        camera(ros, publisher, tmp_path, **{"forward.camera_info_url": str(path)})


def test_zmq_to_real_ros_messages_preserves_rgb_topics_and_capture_time(ros, publisher, tmp_path):
    path = tmp_path / "intrinsics/forward.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(camera_info()))
    node = camera(ros, publisher, tmp_path, timestamp_mode="host_wall")
    observer = Node("camera_test_observer")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(observer)
    received = {"compressed": [], "raw": [], "info": []}
    for message_type, topic, key in (
        (CompressedImage, "/alohamini/cameras/forward/image_raw/compressed", "compressed"),
        (Image, "/alohamini/cameras/forward/image_raw", "raw"),
        (CameraInfo, "/alohamini/cameras/forward/camera_info", "info"),
    ):
        observer.create_subscription(message_type, topic, received[key].append, 2)
    try:
        deadline = time.monotonic() + 3
        while (
            node.raw_publishers["forward"].get_subscription_count() == 0
            and time.monotonic() < deadline
        ):
            executor.spin_once(timeout_sec=0.02)
        assert node.raw_publishers["forward"].get_subscription_count() > 0
        message = parts()
        publisher[0].send_multipart(message)
        while not all(received.values()) and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        assert all(received.values())
        compressed, raw, info = (received[key][0] for key in ("compressed", "raw", "info"))
        assert bytes(compressed.data) == message[2]
        assert raw.encoding == "rgb8" and (raw.width, raw.height, raw.step) == (16, 8, 48)
        assert raw.data[0] > 240 and raw.data[2] < 10
        assert compressed.header == raw.header == info.header
        assert info.header.frame_id == "forward_camera_optical"
        ns = compressed.header.stamp.sec * 1_000_000_000 + compressed.header.stamp.nanosec
        assert ns == json.loads(message[1])["capture_unix_ns"]
    finally:
        executor.shutdown()
        observer.destroy_node()


def extrinsic_document():
    return {
        "status": "accepted_hand_eye",
        "mount_link": "front_camera",
        "optical_frame": "forward_camera_optical",
        "T_mount_link_from_camera_optical": {
            "xyz_m": [0.01, 0.02, 0.03],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
    }


def test_extrinsics_preserve_parent_child_translation_and_explicit_quaternion_order(ros, tmp_path):
    path = tmp_path / "forward.yaml"
    document = extrinsic_document()
    path.write_text(yaml.safe_dump(document))
    assert load_extrinsic(path) == (
        "front_camera",
        "forward_camera_optical",
        [0.01, 0.02, 0.03],
        [0, 0, 0, 1],
    )
    transform = document["T_mount_link_from_camera_optical"]
    transform.pop("quaternion_xyzw")
    transform["quaternion_wxyz"] = [1, 0, 0, 0]
    path.write_text(yaml.safe_dump(document))
    assert load_extrinsic(path)[-1] == [0, 0, 0, 1]
    with patch("alohamini_camera.extrinsics_node.StaticTransformBroadcaster") as broadcaster:
        ros(CameraExtrinsicsNode, extrinsics_csv=str(path))
    message = broadcaster.return_value.sendTransform.call_args.args[0][0]
    assert (
        message.header.frame_id == "front_camera"
        and message.child_frame_id == "forward_camera_optical"
    )
    assert message.transform.translation.x == 0.01 and message.transform.rotation.w == 1


@pytest.mark.parametrize(
    "change",
    [
        {"status": "manual_candidate"},
        {"optical_frame": "front_camera"},
        {"T_mount_link_from_camera_optical": {"xyz_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 0]}},
    ],
)
def test_invalid_or_unaccepted_extrinsics_are_rejected(tmp_path, change):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({**extrinsic_document(), **change}))
    with pytest.raises(ValueError):
        load_extrinsic(path)


def test_duplicate_tf_children_are_rejected_before_publishing(ros, tmp_path):
    path = tmp_path / "forward.yaml"
    path.write_text(yaml.safe_dump(extrinsic_document()))
    with patch("alohamini_camera.extrinsics_node.StaticTransformBroadcaster") as broadcaster:
        with pytest.raises(ValueError, match="duplicate"):
            ros(CameraExtrinsicsNode, extrinsics_csv=f"{path},{path}")
        broadcaster.return_value.sendTransform.assert_not_called()
