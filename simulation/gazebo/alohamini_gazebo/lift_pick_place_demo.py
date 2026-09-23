"""Deterministic lift / drive / pick / place acceptance sequence."""

from __future__ import annotations

import math
from time import monotonic

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64MultiArray, String
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectoryPoint

RIGHT_JOINTS = [
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_yaw_joint",
    "right_wrist_roll",
]

HOME = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0]
PICK_PRE = [0.228767, -0.372980, 1.025548, 0.987686, 0.179765, -0.131600]
PICK = [0.229499, -0.333693, 0.736790, 0.944129, 0.178597, -0.130306]
# These place poses preserve the grasp orientation, so the carton remains
# upright. Their endpoints were checked against the current MoveIt model.
PLACE_PRE = [-0.951726, -2.320815, 1.752219, 0.703553, 1.011836, 0.985839]
PLACE = [-0.951727, -2.771713, 2.062544, 0.562982, 1.011836, 0.985841]
PICK_BASE_TARGET = (0.203, -0.121, 0.0)


def parse_attachment_state(value: str) -> bool | None:
    """Translate the Gazebo DetachableJoint StringMsg state."""
    state = value.strip().lower()
    if state == "attached":
        return True
    if state == "detached":
        return False
    return None


class LiftPickPlaceDemo(Node):
    def __init__(self) -> None:
        super().__init__("alohamini_lift_pick_place_demo")
        if self.get_namespace() != "/alohamini_sim" or not self.get_parameter("use_sim_time").value:
            self.destroy_node()
            raise ValueError("Gazebo nodes require /alohamini_sim and use_sim_time:=true")
        self._joints: dict[str, float] = {}
        self._phase = "wait"
        self._phase_started = monotonic()
        self._motion_active = False
        self._motion_next = ""
        self._goal_handle = None
        self._attached = False
        self._done = False
        self._exit_code = 1
        self._object_pose: tuple[float, float, float, float] | None = None
        self._object_last_moved = monotonic()
        self._object_last_seen = -math.inf

        self._cmd_vel = self.create_publisher(Twist, "cmd_vel", 1)
        self._gripper = self.create_publisher(
            Float64MultiArray, "right_gripper_controller/commands", 1
        )
        self._attach = self.create_publisher(Empty, "demo_object/attach", 1)
        self._detach = self.create_publisher(Empty, "demo_object/detach", 1)
        self.create_subscription(JointState, "joint_states", self._joint_state, 5)
        self.create_subscription(String, "demo_object/attached", self._attach_state, 5)
        self.create_subscription(TFMessage, "gazebo/model_poses", self._model_poses, 5)

        self._lift = ActionClient(
            self, FollowJointTrajectory, "lift_controller/follow_joint_trajectory"
        )
        self._arm = ActionClient(
            self, FollowJointTrajectory, "right_arm_controller/follow_joint_trajectory"
        )
        self.create_timer(0.02, self._update)

    def _joint_state(self, message: JointState) -> None:
        if len(message.name) != len(message.position) or not all(
            math.isfinite(value) for value in message.position
        ):
            return
        self._joints.update(zip(message.name, message.position, strict=False))

    def _attach_state(self, message: String) -> None:
        state = parse_attachment_state(message.data)
        if state is None:
            self.get_logger().warning(f"ignored unknown detachable-joint state: {message.data!r}")
            return
        self._attached = state

    def _model_poses(self, message: TFMessage) -> None:
        for transform in message.transforms:
            if transform.child_frame_id.split("::")[-1] != "demo_object":
                continue
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            pose = (
                translation.x,
                translation.y,
                translation.z,
                1.0 - 2.0 * (rotation.x**2 + rotation.y**2),
            )
            if not all(math.isfinite(value) for value in pose):
                continue
            if self._object_pose is None or math.dist(pose[:3], self._object_pose[:3]) > 0.002:
                self._object_last_moved = monotonic()
            self._object_pose = pose
            self._object_last_seen = monotonic()
            break

    def _set_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_started = monotonic()
        self.get_logger().info(f"demo phase: {phase}")
        if phase == "complete":
            self._done = True
            self._exit_code = 0
        elif phase.startswith("failed"):
            self._done = True
            self._exit_code = 1
            self._cmd_vel.publish(Twist())
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal_async()

    def _publish_gripper(self, position: float) -> None:
        self._gripper.publish(Float64MultiArray(data=[position]))

    def _start_trajectory(
        self,
        client: ActionClient,
        joints: list[str],
        positions: list[float],
        duration: float,
        next_phase: str,
    ) -> None:
        if self._motion_active:
            return
        if not client.wait_for_server(timeout_sec=0.05):
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joints
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        goal.trajectory.points = [point]
        self._motion_active = True
        self._motion_next = next_phase
        future = client.send_goal_async(goal)
        future.add_done_callback(self._goal_response)

    def _goal_response(self, future) -> None:
        handle = future.result()
        if self._done:
            if handle.accepted:
                handle.cancel_goal_async()
            return
        if not handle.accepted:
            self._motion_active = False
            self._set_phase("failed_goal_rejected")
            return
        self._goal_handle = handle
        result = handle.get_result_async()
        result.add_done_callback(self._goal_result)

    def _goal_result(self, future) -> None:
        if self._done:
            return
        response = future.result()
        result = response.result
        self._goal_handle = None
        self._motion_active = False
        if (
            response.status != GoalStatus.STATUS_SUCCEEDED
            or result.error_code != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            self._set_phase(f"failed_trajectory_{result.error_code}")
            return
        self._set_phase(self._motion_next)

    def _drive_to(self, target: tuple[float, float, float], next_phase: str) -> None:
        required = ("root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint")
        if not all(name in self._joints for name in required):
            return
        x, y, yaw = (self._joints[name] for name in required)
        ex, ey, eyaw = target[0] - x, target[1] - y, target[2] - yaw
        eyaw = math.atan2(math.sin(eyaw), math.cos(eyaw))
        if math.hypot(ex, ey) < 0.008 and abs(eyaw) < 0.015:
            self._cmd_vel.publish(Twist())
            self._set_phase(next_phase)
            return
        # Convert the world-frame position correction to base_link body axes.
        command = Twist()
        command.linear.x = max(-0.25, min(0.25, 0.9 * (math.cos(yaw) * ex + math.sin(yaw) * ey)))
        command.linear.y = max(-0.25, min(0.25, 0.9 * (-math.sin(yaw) * ex + math.cos(yaw) * ey)))
        command.angular.z = max(-0.5, min(0.5, 1.2 * eyaw))
        self._cmd_vel.publish(command)

    def _update(self) -> None:
        elapsed = monotonic() - self._phase_started
        if self._phase.startswith("failed") or self._phase == "complete":
            return
        if elapsed > 60.0:
            self._cmd_vel.publish(Twist())
            self._set_phase("failed_phase_timeout")
            return
        if self._phase == "wait":
            required = {"vertical_move", *RIGHT_JOINTS, "root_x_axis_joint", "root_y_axis_joint"}
            servers_ready = self._lift.server_is_ready() and self._arm.server_is_ready()
            if required.issubset(self._joints) and servers_ready:
                self._set_phase("release_initial_attachment")
        elif self._phase == "release_initial_attachment":
            self._detach.publish(Empty())
            self._publish_gripper(-1.0)
            if elapsed > 1.0:
                self._set_phase("drive_pick")
        elif self._phase == "drive_pick":
            # Centre the object at Fixed_Jaw x=+2.5 mm, the calibrated centre
            # of the 25 mm gripper aperture.  The previous base target left it
            # around x=-7 mm and the fixed fingertip blocked the pick descent.
            self._drive_to(PICK_BASE_TARGET, "lift_pick")
        elif self._phase == "lift_pick":
            self._start_trajectory(self._lift, ["vertical_move"], [-0.1], 8.0, "pick_pre")
        elif self._phase == "pick_pre":
            self._start_trajectory(self._arm, RIGHT_JOINTS, PICK_PRE, 5.0, "pick")
        elif self._phase == "pick":
            self._start_trajectory(self._arm, RIGHT_JOINTS, PICK, 3.0, "close")
        elif self._phase == "close":
            self._publish_gripper(0.1)
            if elapsed > 1.0:
                self._set_phase("attach")
        elif self._phase == "attach":
            self._attach.publish(Empty())
            if self._attached:
                self._set_phase("pick_retreat")
            elif elapsed > 2.0:
                self._set_phase("failed_attach_timeout")
        elif self._phase == "pick_retreat":
            self._start_trajectory(self._arm, RIGHT_JOINTS, PICK_PRE, 3.0, "drive_place")
        elif self._phase == "drive_place":
            self._drive_to((0.25, 0.72, 0.0), "lift_place")
        elif self._phase == "lift_place":
            self._start_trajectory(self._lift, ["vertical_move"], [-0.3], 8.0, "place_pre")
        elif self._phase == "place_pre":
            self._start_trajectory(self._arm, RIGHT_JOINTS, PLACE_PRE, 6.0, "place")
        elif self._phase == "place":
            self._start_trajectory(self._arm, RIGHT_JOINTS, PLACE, 4.0, "release")
        elif self._phase == "release":
            self._publish_gripper(-1.0)
            self._detach.publish(Empty())
            if not self._attached:
                self._set_phase("place_retreat")
            elif elapsed > 2.0:
                self._set_phase("failed_detach_timeout")
        elif self._phase == "place_retreat":
            self._start_trajectory(self._arm, RIGHT_JOINTS, PLACE_PRE, 4.0, "home")
        elif self._phase == "home":
            self._start_trajectory(self._arm, RIGHT_JOINTS, HOME, 7.0, "verify")
        elif self._phase == "verify":
            if self._object_pose is None or monotonic() - self._object_last_seen > 1.0:
                if elapsed > 5.0:
                    self._set_phase("failed_no_object_pose")
                return
            x, y, z, upright_cosine = self._object_pose
            inside = abs(x - 0.70) <= 0.09 and abs(y - 0.72) <= 0.105
            height_ok = abs(z - 0.1525) <= 0.01
            upright = upright_cosine >= math.cos(math.radians(10.0))
            stable = monotonic() - self._object_last_moved >= 1.0
            if inside and height_ok and upright and stable:
                self.get_logger().info(
                    f"demo accepted: object=({x:.3f}, {y:.3f}, {z:.3f}) m, "
                    "inside tape frame and stable"
                )
                self._set_phase("complete")
            elif elapsed > 5.0:
                self.get_logger().error(
                    f"demo rejected: object=({x:.3f}, {y:.3f}, {z:.3f}) m, "
                    f"inside={inside}, height={height_ok}, upright={upright}, stable={stable}"
                )
                self._set_phase("failed_final_object_state")


def main() -> int:
    rclpy.init()
    node = LiftPickPlaceDemo()
    exit_code = 130
    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.1)
        exit_code = node._exit_code
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._cmd_vel.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code
