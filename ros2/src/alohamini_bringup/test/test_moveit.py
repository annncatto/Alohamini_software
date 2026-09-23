import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetMotionPlan
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from test_hardware import host as host
from test_hardware import ros as ros
from test_trajectory import robot as robot
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectoryPoint


def test_offline_rviz_targets_only_offline_move_group():
    package = Path(get_package_share_directory("alohamini_moveit_config"))
    config = yaml.safe_load((package / "config/plan_only.rviz").read_text())
    planning = next(
        display
        for display in config["Visualization Manager"]["Displays"]
        if display["Class"] == "moveit_rviz_plugin/MotionPlanning"
    )
    assert planning["Move Group Namespace"] == "/alohamini_plan_only"
    assert planning["Robot Description"].startswith("/alohamini_plan_only/")
    assert planning["Planning Scene Topic"].startswith("/alohamini_plan_only/")


def test_moveit_executes_through_the_native_host_bridge(robot, tmp_path):
    node, _, state, until = robot
    client = ActionClient(node, ExecuteTrajectory, "/execute_trajectory")
    log_path = tmp_path / "moveit_execution.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                "ros2",
                "launch",
                "alohamini_moveit_config",
                "move_group.launch.py",
                "use_rviz:=false",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            until(client.server_is_ready, timeout=15)
            for resource, target in (
                ("left_arm", 0.1),
                ("right_arm", 0.1),
                ("lift", 0.01),
                ("left_gripper", -0.1),
                ("right_gripper", -0.1),
            ):
                goal = ExecuteTrajectory.Goal()
                joints = node.commands.resources[resource].joints
                start = [node.commands.positions[joint] for joint in joints]
                end = [target, *start[1:]]
                goal.trajectory.joint_trajectory.joint_names = list(joints)
                goal.trajectory.joint_trajectory.points = [
                    JointTrajectoryPoint(positions=start, time_from_start=Duration()),
                    JointTrajectoryPoint(
                        positions=end,
                        time_from_start=Duration(nanosec=100_000_000),
                    ),
                ]
                request = client.send_goal_async(goal)
                until(request.done)
                handle = request.result()
                assert handle.accepted, log_path.read_text()
                result = handle.get_result_async()
                until(result.done, timeout=10)
                assert result.result().status == GoalStatus.STATUS_SUCCEEDED, log_path.read_text()
                assert result.result().result.error_code.val == MoveItErrorCodes.SUCCESS
                assert node.commands.positions[joints[0]] == pytest.approx(target, abs=0.03)
            assert state["commands"]
            assert len({command["_command"]["client_id"] for command in state["commands"]}) == 1
        finally:
            client.destroy()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
                pytest.fail(log_path.read_text())
    assert process.returncode == 0, log_path.read_text()


def test_offline_moveit_plans_without_publishing_hardware_state(ros, tmp_path):
    node = Node("moveit_migration_test")
    states, hardware_states, hardware_tf = [], [], []
    node.create_subscription(JointState, "/alohamini_plan_only/joint_states", states.append, 10)
    node.create_subscription(JointState, "/joint_states", hardware_states.append, 10)
    node.create_subscription(TFMessage, "/tf", hardware_tf.append, 10)
    planner = node.create_client(GetMotionPlan, "/alohamini_plan_only/plan_kinematic_path")
    parameters = node.create_client(GetParameters, "/alohamini_plan_only/move_group/get_parameters")
    log_path = tmp_path / "moveit.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            ["ros2", "launch", "alohamini_moveit_config", "plan_only.launch.py", "use_rviz:=false"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        def until(predicate):
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and process.poll() is None:
                if predicate():
                    return
                rclpy.spin_once(node, timeout_sec=0.05)
            pytest.fail(log_path.read_text())

        try:
            until(lambda: states and planner.service_is_ready() and parameters.service_is_ready())
            setting = parameters.call_async(
                GetParameters.Request(names=["allow_trajectory_execution"])
            )
            until(setting.done)
            assert setting.result().values[0].bool_value is False
            state = states[-1]
            request = GetMotionPlan.Request()
            request.motion_plan_request.group_name = "left_arm"
            request.motion_plan_request.num_planning_attempts = 1
            request.motion_plan_request.allowed_planning_time = 3.0
            request.motion_plan_request.max_velocity_scaling_factor = 0.1
            request.motion_plan_request.max_acceleration_scaling_factor = 0.1
            request.motion_plan_request.start_state.joint_state = state
            positions = dict(zip(state.name, state.position, strict=True))
            constraint = Constraints()
            for joint in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_yaw_joint",
                "wrist_roll",
            ):
                name = f"left_{joint}"
                constraint.joint_constraints.append(
                    JointConstraint(
                        joint_name=name,
                        position=positions[name],
                        tolerance_above=0.001,
                        tolerance_below=0.001,
                        weight=1.0,
                    )
                )
            request.motion_plan_request.goal_constraints = [constraint]
            result = planner.call_async(request)
            until(result.done)
            response = result.result().motion_plan_response
            assert response.error_code.val == MoveItErrorCodes.SUCCESS, log_path.read_text()
            assert response.trajectory.joint_trajectory.points
            for executable in ("validate_moveit", "validate_tf"):
                validation = subprocess.run(
                    ["ros2", "run", "alohamini_validation", executable],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert validation.returncode == 0, validation.stdout + validation.stderr
                assert "[PASS]" in validation.stdout
            assert not hardware_states and not hardware_tf
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
                pytest.fail(f"MoveIt launch did not stop: {log_path.read_text()}")
            finally:
                node.destroy_node()
    assert process.returncode == 0, log_path.read_text()
