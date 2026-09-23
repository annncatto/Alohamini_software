import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from alohamini_gazebo.lift_pick_place_demo import (
    HOME,
    PICK,
    PICK_PRE,
    PLACE,
    PLACE_PRE,
    RIGHT_JOINTS,
    LiftPickPlaceDemo,
    parse_attachment_state,
)
from alohamini_gazebo.omni_base_adapter import OmniBaseAdapter
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

from alohamini.kinematics import OmniBaseKinematics
from alohamini.model import get_robot_model
from alohamini.schema import BodyVelocity

PACKAGE = Path(__file__).resolve().parents[1]
DESCRIPTION = get_robot_model("alohamini2pro").directory / "urdf/alohamini2pro.urdf"


@pytest.fixture
def ros_context():
    rclpy.init(args=["--ros-args", "-r", "__ns:=/alohamini_sim", "-p", "use_sim_time:=true"])
    yield
    rclpy.shutdown()


@pytest.mark.parametrize("node_class", [OmniBaseAdapter, LiftPickPlaceDemo])
def test_command_nodes_reject_default_real_robot_namespace(node_class):
    rclpy.init(args=[])
    try:
        with pytest.raises(ValueError, match="/alohamini_sim"):
            node_class()
    finally:
        rclpy.shutdown()


def test_command_nodes_require_simulated_clock():
    rclpy.init(args=["--ros-args", "-r", "__ns:=/alohamini_sim"])
    try:
        for node_class in (OmniBaseAdapter, LiftPickPlaceDemo):
            with pytest.raises(ValueError, match="use_sim_time"):
                node_class()
    finally:
        rclpy.shutdown()


def test_canceled_trajectory_is_failure_even_with_zero_error_code(ros_context):
    node = LiftPickPlaceDemo()
    try:
        node._motion_next = "complete"
        future = Mock()
        future.result.return_value = SimpleNamespace(
            status=GoalStatus.STATUS_CANCELED,
            result=FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.SUCCESSFUL),
        )
        node._goal_result(future)
        assert node._done and node._exit_code == 1
        assert node._phase.startswith("failed")
        handle = Mock(accepted=True)
        future.result.return_value = handle
        node._goal_response(future)
        handle.cancel_goal_async.assert_called_once()
    finally:
        node.destroy_node()


def test_simulation_topics_and_controller_configuration_are_isolated(ros_context):
    nodes = [OmniBaseAdapter(), LiftPickPlaceDemo()]
    try:
        for node in nodes:
            for publisher in node.publishers:
                assert publisher.topic_name.startswith(
                    "/alohamini_sim/"
                ) or publisher.topic_name in (
                    "/rosout",
                    "/parameter_events",
                )
            for subscription in node.subscriptions:
                # /clock is remapped by the launch file; these nodes were constructed directly.
                assert (
                    subscription.topic_name.startswith("/alohamini_sim/")
                    or subscription.topic_name == "/clock"
                )
        controllers = yaml.safe_load((PACKAGE / "config/ros2_controllers.yaml").read_text())
        assert all(name.startswith("/alohamini_sim/") for name in controllers)
        bridge = yaml.safe_load((PACKAGE / "config/bridge.yaml").read_text())
        assert all(row["ros_topic_name"].startswith("/alohamini_sim/") for row in bridge)
    finally:
        for node in nodes:
            node.destroy_node()


def test_base_uses_native_wheel_convention_and_does_not_integrate_stale_pose(ros_context):
    node = OmniBaseAdapter()
    try:
        expected = OmniBaseKinematics(0.063, 0.195).body_to_wheels(BodyVelocity(0.1, -0.2, 0.3))
        assert node._wheel_velocity(0.1, -0.2, 0.3) == list(expected)
        node._base_pub = Mock()
        node._wheel_pub = Mock()
        with patch("alohamini_gazebo.omni_base_adapter.monotonic", return_value=10.0):
            node._last_update = 10.0
            node._on_joint_state(
                JointState(
                    name=["root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"],
                    position=[0.0, 0.0, 0.0],
                )
            )
            command = Twist()
            command.linear.x = 0.1
            node._on_cmd_vel(command)
            node._update()
            first = node._base_pub.publish.call_args.args[0].data
            node._update()
            assert node._base_pub.publish.call_args.args[0].data == first
            assert node._pose == [0.0, 0.0, 0.0]
        node._base_pub.reset_mock()
        with patch("alohamini_gazebo.omni_base_adapter.monotonic", return_value=11.0):
            node._update()
        node._base_pub.publish.assert_not_called()
        assert list(node._wheel_pub.publish.call_args.args[0].data) == [0.0, 0.0, 0.0]
    finally:
        node.destroy_node()


def test_demo_phase_timeout_is_terminal_and_stale_object_cannot_pass(ros_context):
    node = LiftPickPlaceDemo()
    try:
        with patch("alohamini_gazebo.lift_pick_place_demo.monotonic", return_value=100.0):
            node._phase = "verify"
            node._phase_started = 94.0
            node._object_pose = (0.70, 0.72, 0.1525, 1.0)
            node._object_last_moved = 90.0
            node._object_last_seen = 90.0
            node._update()
            assert node._done and node._exit_code == 1
            node._goal_result(Mock(side_effect=AssertionError("late result must not revive demo")))
            assert node._phase.startswith("failed")
            node._done = False
            node._phase = "wait"
            node._phase_started = 0.0
            node._update()
            assert node._phase == "failed_phase_timeout"
    finally:
        node.destroy_node()


def test_all_demo_sdf_assets_parse():
    assets = [PACKAGE / "worlds/lift_pick_place.sdf", *PACKAGE.glob("models/*/model.sdf")]
    assert len(assets) == 4
    for asset in assets:
        assert ET.parse(asset).getroot().tag == "sdf"


def test_object_is_gripper_sized_and_target_marker_is_complete():
    object_root = ET.parse(PACKAGE / "models/grasp_object/model.sdf").getroot()
    size_text = object_root.findtext(".//collision/geometry/box/size")
    size = [float(value) for value in size_text.split()]
    assert size == [0.05, 0.02, 0.065]

    target_root = ET.parse(PACKAGE / "models/aruco_drop_zone/model.sdf").getroot()
    marker_cells = {
        visual.get("name")
        for visual in target_root.findall(".//visual")
        if visual.get("name", "").startswith("marker_r")
    }
    assert marker_cells == {
        "marker_r1c3",
        "marker_r2c3",
        "marker_r3c2",
        "marker_r3c3",
        "marker_r3c4",
        "marker_r4c1",
        "marker_r4c2",
        "marker_r4c4",
    }


def test_pick_platform_is_one_600_mm_high_stage():
    root = ET.parse(PACKAGE / "models/high_plinth/model.sdf").getroot()
    collisions = root.findall(".//collision")
    assert len(collisions) == 1
    assert collisions[0].get("name") == "platform_collision"
    size = [float(value) for value in collisions[0].findtext("geometry/box/size").split()]
    pose = [float(value) for value in collisions[0].findtext("pose").split()]
    assert size == [0.48, 0.38, 0.60]
    assert pose[2] + size[2] / 2 == 0.60


def test_demo_arm_waypoints_respect_authoritative_limits():
    root = ET.parse(DESCRIPTION).getroot()
    limits = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("lower") is not None:
            limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    for waypoint in (HOME, PICK_PRE, PICK, PLACE_PRE, PLACE):
        assert len(waypoint) == len(RIGHT_JOINTS)
        for name, value in zip(RIGHT_JOINTS, waypoint, strict=True):
            lower, upper = limits[name]
            assert lower <= value <= upper


def test_detachable_joint_string_state_is_strict():
    assert parse_attachment_state("attached") is True
    assert parse_attachment_state(" detached ") is False
    assert parse_attachment_state("true") is None
    assert parse_attachment_state("") is None


def test_bridge_uses_detachable_joint_string_message():
    bridge = (PACKAGE / "config/bridge.yaml").read_text()
    block = bridge.split("ros_topic_name: /alohamini_sim/demo_object/attached", 1)[1]
    assert "ros_type_name: std_msgs/msg/String" in block
    assert "gz_type_name: gz.msgs.StringMsg" in block


def test_demo_launch_propagates_failed_exit_code():
    launch_path = PACKAGE / "launch/lift_pick_place_demo.launch.py"
    spec = importlib.util.spec_from_file_location("lift_pick_place_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert len(module._shutdown_after_demo(SimpleNamespace(returncode=0), None)) == 1
    with pytest.raises(RuntimeError, match="exit code 1"):
        module._shutdown_after_demo(SimpleNamespace(returncode=1), None)
