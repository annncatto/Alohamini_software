"""Deterministic Gazebo-only planar base adapter for the first demo milestone."""

from __future__ import annotations

import math
from time import monotonic

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from alohamini.kinematics import OmniBaseKinematics
from alohamini.schema import BodyVelocity


class OmniBaseAdapter(Node):
    """
    Integrate body velocity and keep simulated wheel animation consistent.

    The real robot does not use this node. It controls the description's planar
    joints only inside Gazebo so scene/task work is deterministic before a
    calibrated omni-wheel contact model is introduced.
    """

    def __init__(self) -> None:
        super().__init__("alohamini_gazebo_omni_base_adapter")
        if self.get_namespace() != "/alohamini_sim" or not self.get_parameter("use_sim_time").value:
            self.destroy_node()
            raise ValueError("Gazebo nodes require /alohamini_sim and use_sim_time:=true")
        self.declare_parameter("command_rate_hz", 50.0)
        self.declare_parameter("command_timeout_sec", 0.25)
        self.declare_parameter("position_command_horizon_sec", 0.10)
        self.declare_parameter("wheel_radius_m", 0.063)
        self.declare_parameter("base_radius_m", 0.195)
        self.declare_parameter("max_linear_speed_mps", 0.35)
        self.declare_parameter("max_yaw_speed_radps", 0.7)

        self._timeout = float(self.get_parameter("command_timeout_sec").value)
        self._command_horizon = float(self.get_parameter("position_command_horizon_sec").value)
        self._wheel_radius = float(self.get_parameter("wheel_radius_m").value)
        self._base_radius = float(self.get_parameter("base_radius_m").value)
        self._max_linear = float(self.get_parameter("max_linear_speed_mps").value)
        self._max_yaw = float(self.get_parameter("max_yaw_speed_radps").value)
        rate = float(self.get_parameter("command_rate_hz").value)
        if not all(
            math.isfinite(value) and value > 0
            for value in (
                self._timeout,
                self._command_horizon,
                self._max_linear,
                self._max_yaw,
                rate,
            )
        ):
            raise ValueError("Simulation rates, timeouts and speed limits must be positive")
        self._kinematics = OmniBaseKinematics(self._wheel_radius, self._base_radius)

        self._pose = [0.0, 0.0, 0.0]
        self._have_actual_pose = False
        self._velocity = [0.0, 0.0, 0.0]
        self._last_command = -math.inf
        self._last_update = monotonic()
        self._last_feedback = -math.inf
        self._base_pub = self.create_publisher(
            Float64MultiArray, "base_planar_controller/commands", 1
        )
        self._wheel_pub = self.create_publisher(
            Float64MultiArray, "wheel_velocity_controller/commands", 1
        )
        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, 1)
        self.create_subscription(JointState, "joint_states", self._on_joint_state, 5)
        self.create_timer(1.0 / rate, self._update)

    @staticmethod
    def _clip(value: float, magnitude: float) -> float:
        return max(-magnitude, min(magnitude, value))

    def _on_cmd_vel(self, message: Twist) -> None:
        values = (message.linear.x, message.linear.y, message.angular.z)
        if not all(math.isfinite(value) for value in values):
            self.get_logger().warning("ignored non-finite /cmd_vel")
            return
        self._velocity = [
            self._clip(values[0], self._max_linear),
            self._clip(values[1], self._max_linear),
            self._clip(values[2], self._max_yaw),
        ]
        self._last_command = monotonic()

    def _on_joint_state(self, message: JointState) -> None:
        if len(message.name) != len(message.position):
            return
        state = dict(zip(message.name, message.position, strict=False))
        names = ("root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint")
        if all(name in state and math.isfinite(state[name]) for name in names):
            self._pose = [state[name] for name in names]
            self._have_actual_pose = True
            self._last_feedback = monotonic()

    def _wheel_velocity(self, vx: float, vy: float, yaw_rate: float) -> list[float]:
        return list(self._kinematics.body_to_wheels(BodyVelocity(vx, vy, yaw_rate)))

    def _update(self) -> None:
        now = monotonic()
        dt = min(now - self._last_update, 0.1)
        self._last_update = now
        vx, vy, yaw_rate = self._velocity
        if now - self._last_command > self._timeout:
            vx = vy = yaw_rate = 0.0

        if not self._have_actual_pose or now - self._last_feedback > self._timeout:
            self._wheel_pub.publish(Float64MultiArray(data=[0.0, 0.0, 0.0]))
            return
        yaw = self._pose[2]
        horizon = max(dt, self._command_horizon)
        target = [
            self._pose[0] + (math.cos(yaw) * vx - math.sin(yaw) * vy) * horizon,
            self._pose[1] + (math.sin(yaw) * vx + math.cos(yaw) * vy) * horizon,
            self._pose[2] + yaw_rate * horizon,
        ]
        self._base_pub.publish(Float64MultiArray(data=target))
        self._wheel_pub.publish(Float64MultiArray(data=self._wheel_velocity(vx, vy, yaw_rate)))


def main() -> None:
    rclpy.init()
    node = OmniBaseAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
