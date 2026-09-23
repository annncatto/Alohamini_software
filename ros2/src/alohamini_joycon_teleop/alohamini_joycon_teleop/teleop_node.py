from __future__ import annotations

import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import partial
from urllib.parse import urlsplit

import numpy as np
import rclpy
import zmq
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from control_msgs.msg import JointJog
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import Point, Twist
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from alohamini.model import get_robot_model

from .kinematics import (
    AlohaMiniArmKinematics,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from .mapping import (
    LEVEL_TCP_QUATERNION,
    base_command,
    base_relative_quaternion,
    euler_delta_quaternion,
    faucet_translation_velocity,
    integrate_base_preview,
    next_lift_stick_latch,
    normalize_stick,
    quaternion_axis_signs,
    quaternion_multiply,
    relative_quaternion,
    step_orientation_toward,
    tcp_button_delta,
)

ARM_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw_joint",
    "wrist_roll",
)
ROOT_JOINTS = (
    "root_x_axis_joint",
    "root_y_axis_joint",
    "root_z_rotation_joint",
)


def arm_names(side: str) -> list[str]:
    return [f"{side}_{suffix}" for suffix in ARM_SUFFIXES]


def duration(seconds: float) -> Duration:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    return Duration(sec=whole, nanosec=int((seconds - whole) * 1_000_000_000))


def pose_to_dict(pose) -> dict[str, list[float]]:
    return {
        "position": [pose.position.x, pose.position.y, pose.position.z],
        "orientation": [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
    }


@dataclass
class ArmControl:
    active: bool = False
    pending_fk: bool = False
    pending_ik: bool = False
    kinematics_generation: int = 0
    target_pose: dict | None = None
    orientation_reference: list[float] | None = None
    joy_anchor_rpy: list[float] | None = None
    joy_anchor_orientation: list[float] | None = None
    last_ik_request: float = 0.0
    goal_busy: bool = False
    goal_handle: object | None = None
    queued_positions: list[float] | None = None
    cancel_after_accept: bool = False
    fault_latched: bool = False
    relatch_was_pressed: bool = False
    trigger_was_pressed: bool = False
    trigger_release_started: float | None = None
    last_gripper_toggle: float = -1.0e9
    # None means no command has been issued in this process yet. After the first
    # command, toggling follows accepted command state instead of measured travel:
    # a loaded/stalled gripper must still reverse on the next press.
    gripper_open_commanded: bool | None = None
    rejection_streak: int = 0
    hand_anchor_orientation: list[float] | None = None
    tcp_anchor_orientation: list[float] | None = None


@dataclass
class InputSample:
    received_at: float
    payload: dict = field(default_factory=dict)


class JoyConTeleop(Node):
    def __init__(self) -> None:
        super().__init__("alohamini_joycon_teleop")
        defaults = {
            "hardware_mode": False,
            "input_endpoint": "tcp://127.0.0.1:5568",
            "input_timeout_sec": 0.25,
            "measured_state_timeout_sec": 0.25,
            "control_rate_hz": 50.0,
            "ik_rate_hz": 20.0,
            "arm_control_mode": "differential",
            "ik_timeout_sec": 0.3,
            "deadzone": 0.25,
            "tcp_speed_m_s": 0.15,
            "orientation_scale": 1.0,
            "orientation_deadband_rad": 0.02,
            # Per-axis signs of the Joy-Con attitude increments applied to the
            # TCP (roll, pitch, yaw). The native library reports roll in the
            # opposite direction to the gripper, so roll is negated.
            "orientation_axis_signs": [-1.0, 1.0, -1.0],
            "max_orientation_delta_rad": 1.2,
            "orientation_speed_rad_s": 2.0,
            "imu_latch_reference": "current",
            "preview_home": "home",
            "max_joint_step_rad": 0.10,
            "max_ik_jump_rad": 0.5,
            "arm_goal_max_velocity_rad_s": 1.5,
            "dls_position_gain": 8.0,
            "dls_orientation_gain": 5.0,
            "dls_orientation_weight": 0.35,
            "dls_damping": 0.04,
            "dls_max_joint_velocity_rad_s": 1.5,
            "dls_max_joint_step_rad": 0.10,
            "dls_command_horizon_sec": 0.12,
            "dls_max_target_error_m": 0.025,
            "dls_joint_limit_margin_rad": 0.03,
            "avoid_collisions": True,
            "arm_trajectory_duration_sec": 0.08,
            "finish_last_target": True,
            "base_linear_speed_m_s": 0.10,
            "base_angular_speed_rad_s": 0.5,
            "lift_speed_m_s": 0.02,
            "button_release_grace_sec": 0.12,
            "gripper_button_debounce_sec": 0.30,
            "preview_joint_states_topic": "/alohamini_plan_only/joint_states",
            "measured_joint_states_topic": "/alohamini_lerobot_bridge/measured_joint_states",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.hardware_mode = bool(self.get_parameter("hardware_mode").value)
        endpoint = urlsplit(str(self.get_parameter("input_endpoint").value))
        if (
            endpoint.scheme != "tcp"
            or endpoint.hostname not in ("127.0.0.1", "localhost")
            or not endpoint.port
        ):
            raise ValueError("Joy-Con input must use a localhost TCP endpoint")
        self.input_timeout = float(self.get_parameter("input_timeout_sec").value)
        self.measured_state_timeout = float(self.get_parameter("measured_state_timeout_sec").value)
        self.deadzone = float(self.get_parameter("deadzone").value)
        self.arm_control_mode = str(self.get_parameter("arm_control_mode").value)
        if self.arm_control_mode not in ("differential", "moveit"):
            raise ValueError("arm_control_mode must be differential or moveit")
        self.ik_period = 1.0 / float(self.get_parameter("ik_rate_hz").value)
        self.ik_timeout = float(self.get_parameter("ik_timeout_sec").value)
        self.orientation_deadband = float(self.get_parameter("orientation_deadband_rad").value)
        self.imu_latch_reference = str(self.get_parameter("imu_latch_reference").value)
        if self.imu_latch_reference not in ("level", "current"):
            raise ValueError("imu_latch_reference must be 'level' or 'current'")
        self.callback_group = MutuallyExclusiveCallbackGroup()

        semantic = ET.parse(get_robot_model("alohamini2pro").description_path("semantic"))
        self.positions = {
            joint.attrib["name"]: float(joint.attrib["value"])
            for group in semantic.findall("group_state")
            if group.attrib["name"] in ("home", "closed")
            for joint in group.findall("joint")
        }
        if str(self.get_parameter("preview_home").value) == "reference":
            # Teleop rest pose: arm extended forward with the gripper level
            # (approach axis straight down), matching the MoveIt FK golden
            # "reference" sample. Hardware mode replaces this with measured
            # /joint_states anyway; this only shapes the offline preview.
            reference = {
                "shoulder_pan": 0.0,
                "shoulder_lift": -1.571,
                "elbow_flex": 1.571,
                "wrist_flex": 0.0,
                "wrist_yaw_joint": 0.0,
                "wrist_roll": 0.0,
            }
            for side in ("left", "right"):
                for joint, value in reference.items():
                    self.positions[f"{side}_{joint}"] = value
        self.positions.update(
            {
                "root_x_axis_joint": 0.0,
                "root_y_axis_joint": 0.0,
                "root_z_rotation_joint": 0.0,
                "vertical_move": 0.0,
                "wheel1_joint": 0.0,
                "wheel2_joint": 0.0,
                "wheel3_joint": 0.0,
            }
        )
        self.samples: dict[str, InputSample] = {}
        self.arms = {side: ArmControl() for side in ("left", "right")}
        self.kinematics = {
            side: AlohaMiniArmKinematics.from_description(side) for side in ("left", "right")
        }
        self.preview_base = [0.0, 0.0, 0.0]
        self.lift_target = self.positions["vertical_move"]
        self.measured_lift = self.lift_target
        self.lift_active = False
        self.left_lift_stick_latched = False
        self.commands_enabled = False
        self.last_bridge_status = -math.inf
        self.input_generation = 0
        self.hardware_was_ready = False
        self.require_neutral = dict.fromkeys(("left", "right"), self.hardware_mode)
        self.base_was_active = False
        self.last_readiness_log = 0.0
        self.readiness_announced = False
        self.last_loop = time.monotonic()
        self.last_stale_warning = 0.0
        self.last_measured_state = None
        self.last_ik_warning: dict[str, float] = {}
        self.last_ik_ok: dict[str, float] = {}
        self.input_sequence: dict[str, int] = {}
        self.input_dropped = dict.fromkeys(("left", "right"), 0)
        self.input_max_age_ms = dict.fromkeys(("left", "right"), 0.0)
        self.button_last_true = {"left": {}, "right": {}}
        self.last_timing_log = time.monotonic()
        self.max_control_dt_ms = 0.0
        self.max_dls_ms = dict.fromkeys(("left", "right"), 0.0)

        self.zmq_context = zmq.Context()
        self.input_socket = self.zmq_context.socket(zmq.SUB)
        self.input_socket.setsockopt(zmq.LINGER, 0)
        self.input_socket.setsockopt(zmq.RCVHWM, 100)
        self.input_socket.setsockopt(zmq.MAXMSGSIZE, 16384)
        self.input_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.input_socket.connect(str(self.get_parameter("input_endpoint").value))

        prefix = "" if self.hardware_mode else "/alohamini_plan_only"
        measured_topic = str(self.get_parameter("measured_joint_states_topic").value)
        if self.hardware_mode:
            self.create_subscription(JointState, measured_topic, self.on_joint_state, 1)
            self.create_subscription(DiagnosticArray, "/diagnostics", self.on_bridge_status, 1)
        self.preview_pub = self.create_publisher(
            JointState,
            str(self.get_parameter("preview_joint_states_topic").value),
            10,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, prefix + "/alohamini/joycon_tcp_markers", 10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, prefix + "/cmd_vel", 1)
        self.fk_client = self.create_client(
            GetPositionFK, prefix + "/compute_fk", callback_group=self.callback_group
        )
        self.ik_client = self.create_client(
            GetPositionIK, prefix + "/compute_ik", callback_group=self.callback_group
        )
        self.arm_clients = {
            side: ActionClient(
                self,
                FollowJointTrajectory,
                prefix + f"/{side}_arm_controller/follow_joint_trajectory",
                callback_group=self.callback_group,
            )
            for side in ("left", "right")
        }
        self.gripper_clients = {
            side: ActionClient(
                self,
                GripperCommand,
                prefix + f"/{side}_gripper_controller/gripper_cmd",
                callback_group=self.callback_group,
            )
            for side in ("left", "right")
        }
        self.lift_jog_pub = self.create_publisher(
            JointJog, prefix + "/lift_controller/joint_jog", 1
        )
        self.arm_jog_pubs = {
            side: self.create_publisher(JointJog, prefix + f"/{side}_arm_controller/joint_jog", 1)
            for side in ("left", "right")
        }
        rate = float(self.get_parameter("control_rate_hz").value)
        self.timer = self.create_timer(
            1.0 / rate, self.on_timer, callback_group=self.callback_group
        )
        self.readiness_timer = self.create_timer(
            2.0, self.publish_readiness, callback_group=self.callback_group
        )
        mode = "HARDWARE" if self.hardware_mode else "PREVIEW"
        self.get_logger().info(
            f"Joy-Con teleop started in {mode} mode; it never enables bridge commands"
        )

    def destroy_node(self):
        if self.hardware_mode and rclpy.ok():
            self.invalidate_inputs()
            if self.base_was_active:
                self.cmd_vel_pub.publish(Twist())
            self.update_lift(None, False, time.monotonic(), 0.0)
        self.input_socket.close()
        self.zmq_context.term()
        return super().destroy_node()

    def on_bridge_status(self, message: DiagnosticArray) -> None:
        for status in message.status:
            if (
                status.name != "AlohaMini Host state bridge"
                or status.hardware_id != "alohamini2pro"
            ):
                continue
            values = {value.key: value.value for value in status.values}
            enabled = (
                status.level != DiagnosticStatus.ERROR
                and values.get("command_enabled") == "true"
                and values.get("stop_pending") == "false"
            )
            if enabled != self.commands_enabled:
                self.invalidate_inputs()
            self.commands_enabled = enabled
            self.last_bridge_status = time.monotonic()

    def invalidate_inputs(self) -> None:
        self.input_generation += 1
        self.samples.clear()
        self.button_last_true = {"left": {}, "right": {}}
        self.require_neutral = dict.fromkeys(("left", "right"), True)
        for side in self.arms:
            self.deactivate_arm(side, cancel_goal=True)
            self.arms[side].pending_fk = False
            self.arms[side].pending_ik = False
            self.arms[side].goal_busy = False
            self.arms[side].gripper_open_commanded = None
            self.arms[side].trigger_was_pressed = False
            self.arms[side].trigger_release_started = None

    def hardware_ready(self, now: float) -> bool:
        return (
            self.commands_enabled
            and now - self.last_bridge_status <= 2.0
            and self.last_measured_state is not None
            and now - self.last_measured_state <= self.measured_state_timeout
        )

    def on_joint_state(self, message: JointState) -> None:
        if not self.hardware_mode:
            return
        required = {
            *arm_names("left"),
            *arm_names("right"),
            "left_gripper",
            "right_gripper",
            "vertical_move",
        }
        if (
            len(message.name) != len(message.position)
            or len(set(message.name)) != len(message.name)
            or not required <= set(message.name)
            or not all(math.isfinite(value) for value in message.position)
        ):
            self.last_measured_state = None
            return
        for name, value in zip(message.name, message.position, strict=False):
            if math.isfinite(value):
                self.positions[name] = float(value)
        if "vertical_move" in message.name:
            self.measured_lift = self.positions["vertical_move"]
            if not self.lift_active:
                self.lift_target = self.measured_lift
        if any(name in message.name for name in arm_names("left")) or any(
            name in message.name for name in arm_names("right")
        ):
            self.last_measured_state = time.monotonic()

    def receive_inputs(self) -> None:
        for _ in range(100):
            try:
                payload = json.loads(self.input_socket.recv_string(flags=zmq.NOBLOCK))
            except zmq.Again:
                return
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.get_logger().warning(f"Rejected malformed Joy-Con input: {exc}")
                continue
            if not isinstance(payload, dict):
                continue
            side = payload.get("side")
            if (
                payload.get("schema_version") not in (1, 2)
                or not isinstance(side, str)
                or side not in self.arms
            ):
                self.get_logger().warning("Rejected incompatible Joy-Con input")
                continue
            sequence = payload.get("sequence")
            source_ns = payload.get("monotonic_ns")
            vectors = (("stick", 2), ("orientation_rpy", 3), ("orientation_xyzw", 4))
            if (
                type(sequence) is not int
                or sequence < 0
                or type(source_ns) is not int
                or source_ns <= 0
                or not 0 <= time.monotonic_ns() - source_ns <= int(self.input_timeout * 1e9)
                or (self.hardware_mode and payload.get("replay", False))
                or not isinstance(payload.get("buttons"), dict)
                or not {"trigger", "relatch", "shoulder"} <= payload["buttons"].keys()
                or not all(type(value) is bool for value in payload["buttons"].values())
                or any(
                    not isinstance(payload.get(key), list)
                    or len(payload[key]) != count
                    or not all(
                        type(value) in (int, float) and math.isfinite(value)
                        for value in payload[key]
                    )
                    for key, count in vectors
                    if key != "orientation_xyzw" or key in payload
                )
                or (
                    "orientation_xyzw" in payload
                    and not math.isclose(
                        math.hypot(*payload["orientation_xyzw"]), 1.0, abs_tol=1e-3
                    )
                )
                or not all(0 <= value <= 4095 for value in payload["stick"])
            ):
                continue
            previous = self.input_sequence.get(side)
            previous_sample = self.samples.get(side)
            if previous_sample is not None and source_ns <= previous_sample.payload["monotonic_ns"]:
                continue
            if previous is not None and sequence > previous + 1:
                self.input_dropped[side] += sequence - previous - 1
            self.input_sequence[side] = sequence
            age_ms = max(0.0, (time.monotonic_ns() - source_ns) / 1.0e6)
            self.input_max_age_ms[side] = max(self.input_max_age_ms[side], age_ms)
            self.samples[side] = InputSample(source_ns / 1e9, payload)

    def sample(self, side: str, now: float) -> dict | None:
        item = self.samples.get(side)
        if item is None or now - item.received_at > self.input_timeout:
            return None
        payload = dict(item.payload)
        buttons = dict(payload["buttons"])
        if self.hardware_mode and self.require_neutral[side]:
            if not any(buttons.values()) and all(
                abs(normalize_stick(value, self.deadzone)) < 1e-9 for value in payload["stick"]
            ):
                self.require_neutral[side] = False
            return None
        release_grace = float(self.get_parameter("button_release_grace_sec").value)
        last_true = self.button_last_true[side]
        # Joy-Con shoulder/d-pad reports can contain isolated false samples
        # while a button is physically held. Accept presses immediately, but
        # require a continuous release before handing the stick to another
        # control mode. This prevents L+stick from alternating lift/base at
        # the report rate and gives held d-pad TCP commands a continuous edge.
        for name in ("shoulder", "sl", "sr", "up", "down", "left", "right"):
            if bool(buttons.get(name)):
                last_true[name] = now
            elif now - last_true.get(name, -1.0e9) < release_grace:
                buttons[name] = True
        payload["buttons"] = buttons
        return payload

    def fill_robot_state(self, state) -> None:
        state.joint_state.name = list(self.positions)
        state.joint_state.position = list(self.positions.values())

    def arm_seed(self, side: str) -> list[float] | None:
        names = arm_names(side)
        if any(name not in self.positions for name in names):
            return None
        return [self.positions[name] for name in names]

    def request_fk(self, side: str, sample: dict | None = None) -> None:
        arm = self.arms[side]
        seed = self.arm_seed(side)
        if arm.pending_fk or seed is None:
            return
        if self.arm_control_mode == "differential":
            transform = self.kinematics[side].forward(seed)
            arm.target_pose = {
                "position": transform[:3, 3].tolist(),
                "orientation": matrix_to_quaternion(transform[:3, :3]),
            }
            arm.orientation_reference = (
                list(LEVEL_TCP_QUATERNION) if self.imu_latch_reference == "level" else None
            )
            arm.joy_anchor_rpy = None
            arm.joy_anchor_orientation = None
            arm.hand_anchor_orientation = None
            arm.tcp_anchor_orientation = None
            self.publish_markers(side, arm.target_pose)
            return
        request = GetPositionFK.Request()
        # Keep the complete Cartesian control loop in the arm's local frame.
        # Because {side}_Base moves with vertical_link, lift motion then moves
        # the whole TCP naturally and never creates a world-frame compensation
        # target for the arm joints.
        request.header.frame_id = f"{side}_Base"
        request.fk_link_names = [f"{side}_tcp"]
        self.fill_robot_state(request.robot_state)
        arm.pending_fk = True
        future = self.fk_client.call_async(request)
        future.add_done_callback(
            partial(
                self.on_fk,
                side,
                generation=(self.input_generation, arm.kinematics_generation),
            )
        )

    def on_fk(self, side: str, future, *, generation=None) -> None:
        arm = self.arms[side]
        if generation is not None and generation != (
            self.input_generation,
            arm.kinematics_generation,
        ):
            return
        arm.pending_fk = False
        try:
            response = future.result()
            if (
                response.error_code.val != response.error_code.SUCCESS
                or len(response.pose_stamped) != 1
            ):
                raise RuntimeError(f"FK error {response.error_code.val}")
            arm.target_pose = pose_to_dict(response.pose_stamped[0].pose)
            if self.imu_latch_reference == "level":
                # Rotate toward the level-grasp orientation gradually instead
                # of jumping the target, so the first IK requests stay within
                # the joint-step filter (the preview already starts level).
                arm.orientation_reference = list(LEVEL_TCP_QUATERNION)
            else:
                arm.orientation_reference = None
            arm.joy_anchor_rpy = None
            arm.joy_anchor_orientation = None
            self.publish_markers(side, arm.target_pose)
        except Exception as exc:
            self.get_logger().warning(f"{side} arm FK latch failed: {exc}")

    def request_ik(
        self,
        side: str,
        proposed_pose: dict,
        now: float,
        allow_position_fallback: bool = False,
        mode: str = "kdl",
    ) -> None:
        arm = self.arms[side]
        seed = self.arm_seed(side)
        if arm.pending_ik or seed is None:
            return
        group_suffix = {"kdl": "", "lma": "_lma", "position": "_pos"}[mode]
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = f"{side}_arm{group_suffix}"
        ik.ik_link_name = f"{side}_tcp"
        ik.pose_stamped.header.frame_id = f"{side}_Base"
        (
            ik.pose_stamped.pose.position.x,
            ik.pose_stamped.pose.position.y,
            ik.pose_stamped.pose.position.z,
        ) = proposed_pose["position"]
        (
            ik.pose_stamped.pose.orientation.x,
            ik.pose_stamped.pose.orientation.y,
            ik.pose_stamped.pose.orientation.z,
            ik.pose_stamped.pose.orientation.w,
        ) = proposed_pose["orientation"]
        self.fill_robot_state(ik.robot_state)
        ik.avoid_collisions = bool(self.get_parameter("avoid_collisions").value)
        ik.timeout = duration(self.ik_timeout)
        arm.pending_ik = True
        arm.last_ik_request = now
        future = self.ik_client.call_async(request)
        future.add_done_callback(
            partial(
                self.on_ik,
                side,
                proposed_pose,
                seed,
                mode,
                allow_position_fallback,
                generation=(self.input_generation, arm.kinematics_generation),
            )
        )

    def warn_ik(self, side: str, message: str) -> None:
        """Rate-limited per-side warning so IK failures are visible without
        flooding the terminal at control rate."""
        now = time.monotonic()
        if now - self.last_ik_warning.get(side, 0.0) < 1.0:
            return
        self.last_ik_warning[side] = now
        self.get_logger().warning(f"[{side}] {message}")

    def retry_ik_mode(self, side: str, proposed_pose: dict, reason: str) -> None:
        self.get_logger().info(
            f"[{side}] {reason}; retrying with LMA IK "
            f"pos={[round(v, 4) for v in proposed_pose['position']]} "
            f"orient={[round(v, 4) for v in proposed_pose['orientation']]} "
            f"seed={[round(v, 4) for v in self.arm_seed(side) or []]}"
        )
        self.request_ik(
            side,
            proposed_pose,
            time.monotonic(),
            allow_position_fallback=False,
            mode="lma",
        )

    def retry_position_only(self, side: str, proposed_pose: dict, reason: str) -> None:
        self.get_logger().info(
            f"[{side}] {reason}; retrying with position-only IK ({side}_arm_pos group)"
        )
        self.request_ik(
            side,
            proposed_pose,
            time.monotonic(),
            allow_position_fallback=False,
            mode="position",
        )

    def on_ik(
        self,
        side: str,
        proposed_pose: dict,
        seed: list[float],
        mode: str,
        allow_position_fallback: bool,
        future,
        *,
        generation=None,
    ) -> None:
        arm = self.arms[side]
        if generation is not None and generation != (
            self.input_generation,
            arm.kinematics_generation,
        ):
            return
        arm.pending_ik = False
        try:
            response = future.result()
            if response.error_code.val != response.error_code.SUCCESS:
                code = response.error_code.val
                hint = {
                    -31: "NO_IK_SOLUTION (unreachable pose)",
                    -6: "TIMED_OUT (solver timeout)",
                    -21: "FRAME_TRANSFORM_FAILURE",
                }.get(code, f"error {code}")
                self.warn_ik(side, f"IK failed: {hint} [{mode}]")
                if mode == "kdl":
                    self.retry_ik_mode(side, proposed_pose, hint)
                elif mode == "lma" and allow_position_fallback:
                    self.retry_position_only(side, proposed_pose, hint)
                return
            solution = dict(
                zip(
                    response.solution.joint_state.name,
                    response.solution.joint_state.position,
                    strict=True,
                )
            )
            names = arm_names(side)
            candidate = [float(solution[name]) for name in names]
            max_step = float(self.get_parameter("max_joint_step_rad").value)
            worst = max(
                range(len(names)),
                key=lambda index: abs(candidate[index] - seed[index]),
            )
            worst_step = abs(candidate[worst] - seed[worst])
            had_rejections = arm.rejection_streak > 0
            if worst_step > max_step:
                arm.rejection_streak += 1
                max_jump = float(self.get_parameter("max_ik_jump_rad").value)
                if arm.rejection_streak >= 2 and worst_step <= max_jump:
                    # The solver occasionally needs a branch change (elbow
                    # flip) near the reference pose; after repeated identical
                    # rejections, allow one bounded larger step so the arm
                    # does not stick. The goal duration is scaled by the step.
                    self.get_logger().info(
                        f"[{side}] accepting larger IK step after repeated "
                        f"rejections ({names[worst]} {worst_step:.3f} rad)"
                    )
                    arm.rejection_streak = 0
                elif mode == "kdl":
                    self.retry_ik_mode(
                        side,
                        proposed_pose,
                        f"step {worst_step:.3f} rad > {max_step} rad",
                    )
                    return
                elif mode == "lma" and allow_position_fallback:
                    self.retry_position_only(
                        side,
                        proposed_pose,
                        f"step {worst_step:.3f} rad > {max_step} rad",
                    )
                    return
                else:
                    self.warn_ik(
                        side,
                        f"IK solution rejected by max_joint_step_rad "
                        f"({names[worst]} step {worst_step:.3f} rad > {max_step}) "
                        f"[{mode}]",
                    )
                    return
            else:
                arm.rejection_streak = 0
            if arm.cancel_after_accept:
                # A stale/disconnected input invalidates an IK request that
                # was already in flight. Do not let its late result advance
                # the TCP latch or create a hardware goal.
                return
            now = time.monotonic()
            if now - self.last_ik_ok.get(side, 0.0) > 1.0:
                self.last_ik_ok[side] = now
                recovered = " (recovered)" if had_rejections else ""
                self.get_logger().info(
                    f"[{side}] IK OK [{mode}] max step {worst_step:.3f} rad{recovered}"
                )
            arm.target_pose = proposed_pose
            for name, value in zip(names, candidate, strict=True):
                if not self.hardware_mode:
                    self.positions[name] = value
            self.publish_markers(side, proposed_pose)
            if self.hardware_mode and not arm.cancel_after_accept:
                self.send_arm_goal(side, candidate)
            if mode == "position":
                # A position-only solution ignores the requested orientation;
                # re-latch FK so the target pose and markers show the true
                # achieved pose.
                self.request_fk(side)
        except Exception as exc:
            self.get_logger().warning(f"{side} arm IK failed: {exc}")

    def send_arm_goal(self, side: str, positions: list[float]) -> None:
        arm = self.arms[side]
        if arm.goal_busy:
            # IK may run faster than the physical trajectory. Keep only the
            # newest solution instead of preempting the active Action.
            arm.queued_positions = list(positions)
            return
        client = self.arm_clients[side]
        if not client.server_is_ready():
            return
        names = arm_names(side)
        worst_step = max(
            (
                abs(target - self.positions.get(name, target))
                for name, target in zip(names, positions, strict=True)
            ),
            default=0.0,
        )
        goal_duration = max(
            float(self.get_parameter("arm_trajectory_duration_sec").value),
            worst_step / float(self.get_parameter("arm_goal_max_velocity_rad_s").value),
        )
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = duration(goal_duration)
        goal.trajectory.points = [point]
        arm.queued_positions = None
        arm.goal_busy = True
        arm.cancel_after_accept = False
        future = client.send_goal_async(goal)
        future.add_done_callback(
            partial(self.on_arm_goal_response, side, generation=self.input_generation)
        )

    def on_arm_goal_response(self, side: str, future, *, generation=None) -> None:
        if generation is not None and generation != self.input_generation:
            try:
                handle = future.result()
                if handle.accepted:
                    handle.cancel_goal_async()
            except Exception as exc:
                self.get_logger().warning(f"[{side}] obsolete arm goal cleanup failed: {exc}")
            return
        arm = self.arms[side]
        try:
            handle = future.result()
            if not handle.accepted:
                arm.goal_busy = False
                self.warn_ik(
                    side,
                    "bridge rejected the arm goal (commands disabled?); "
                    "enable with: ros2 service call "
                    "/alohamini_lerobot_bridge/command_enable "
                    "std_srvs/srv/SetBool '{data: true}'",
                )
                return
            arm.goal_handle = handle
            # ``goal_busy`` covers only the asynchronous goal-request handshake.
            # Once accepted, immediately stream the newest queued IK solution. The
            # bridge preempts the previous short trajectory, producing a continuous
            # target stream instead of waiting for every action result.
            arm.goal_busy = False
            if arm.cancel_after_accept or (
                not arm.active and not bool(self.get_parameter("finish_last_target").value)
            ):
                handle.cancel_goal_async()
            result = handle.get_result_async()
            result.add_done_callback(partial(self.on_arm_result, side, handle))
            pending = arm.queued_positions
            arm.queued_positions = None
            if (
                pending is not None
                and not arm.cancel_after_accept
                and (arm.active or bool(self.get_parameter("finish_last_target").value))
            ):
                self.send_arm_goal(side, pending)
        except Exception as exc:
            arm.goal_busy = False
            arm.goal_handle = None
            self.get_logger().warning(f"[{side}] arm goal request failed: {exc}")

    def on_arm_result(self, side: str, handle, future) -> None:
        arm = self.arms[side]
        if arm.goal_handle is not handle:
            return
        arm.goal_handle = None
        try:
            response = future.result()
            result = response.result
            succeeded = response.status == GoalStatus.STATUS_SUCCEEDED
            if not succeeded:
                if response.status == GoalStatus.STATUS_CANCELED:
                    return
                # Streaming replaces the current short goal before it completes. A
                # preempted result is expected, including the small race where the old
                # result arrives while the next goal request is still being accepted.
                if arm.goal_busy or "preempted by a newer trajectory" in result.error_string:
                    return
                arm.queued_positions = None
                arm.cancel_after_accept = False
                self.get_logger().warning(
                    f"[{side}] arm goal ended with code "
                    f"{result.error_code}: {result.error_string or 'unknown'}"
                )
                arm.active = False
                arm.fault_latched = True
                return
            pending = arm.queued_positions
            arm.queued_positions = None
            canceled = arm.cancel_after_accept
            arm.cancel_after_accept = False
            if (
                pending is not None
                and not canceled
                and (arm.active or bool(self.get_parameter("finish_last_target").value))
            ):
                self.send_arm_goal(side, pending)
            elif not arm.active:
                self.get_logger().info(f"[{side}] finished the last commanded TCP target")
        except Exception as exc:
            arm.queued_positions = None
            arm.cancel_after_accept = False
            self.get_logger().warning(f"[{side}] arm result failed: {exc}")

    def deactivate_arm(self, side: str, cancel_goal: bool = False) -> None:
        """Stop generating new arm targets; keep the last latch so Home
        re-latch and the TCP marker persist while the arm is idle."""
        arm = self.arms[side]
        if (
            getattr(self, "arm_control_mode", "moveit") == "differential"
            and arm.active
            and self.hardware_mode
        ):
            self.send_arm_jog(side, [0.0] * 6, 0.0)
        if cancel_goal:
            arm.kinematics_generation += 1
            arm.pending_fk = False
            arm.pending_ik = False
            arm.queued_positions = None
            arm.cancel_after_accept = True
            arm.target_pose = None
            if self.hardware_mode and arm.goal_handle is not None:
                arm.goal_handle.cancel_goal_async()
        if arm.active and self.hardware_mode and arm.goal_handle is not None:
            if not cancel_goal and not bool(self.get_parameter("finish_last_target").value):
                arm.queued_positions = None
                arm.cancel_after_accept = True
                arm.goal_handle.cancel_goal_async()
            elif not cancel_goal:
                self.get_logger().info(
                    f"[{side}] released; finishing the last commanded TCP target"
                )
        arm.active = False

    def update_arm(
        self,
        side: str,
        sample: dict | None,
        now: float,
        dt: float,
    ) -> None:
        if getattr(self, "arm_control_mode", "moveit") == "differential":
            JoyConTeleop.update_arm_differential(self, side, sample, now, dt)
        else:
            JoyConTeleop.update_arm_moveit(self, side, sample, now, dt)

    def send_arm_jog(self, side: str, displacements: list[float], dt: float) -> None:
        message = JointJog()
        message.header.stamp = self.get_clock().now().to_msg()
        message.joint_names = arm_names(side)
        message.displacements = [float(value) for value in displacements]
        message.duration = max(0.0, float(dt))
        self.arm_jog_pubs[side].publish(message)

    def update_arm_differential(
        self,
        side: str,
        sample: dict | None,
        now: float,
        dt: float,
    ) -> None:
        arm = self.arms[side]
        if sample is None:
            self.deactivate_arm(side, cancel_goal=True)
            arm.hand_anchor_orientation = None
            return
        buttons = sample["buttons"]
        clutch = bool(buttons.get("sl") or buttons.get("sr"))
        fixed_buttons = buttons
        if clutch and buttons.get("shoulder"):
            # In full faucet mode L already provides +Z. Keep the face buttons
            # on their ordinary XY mapping so L+face can compose axes instead
            # of adding a duplicate shoulder-modified Z command.
            fixed_buttons = dict(buttons)
            fixed_buttons["shoulder"] = False
        fixed_delta = tcp_button_delta(
            fixed_buttons,
            float(self.get_parameter("tcp_speed_m_s").value),
            dt,
        )
        fixed_active = any(abs(value) > 1.0e-9 for value in fixed_delta)
        engaged = clutch or fixed_active
        relatch = bool(buttons["relatch"])
        relatch_pressed = relatch and not arm.relatch_was_pressed
        arm.relatch_was_pressed = relatch
        if relatch_pressed:
            self.request_fk(side, sample)
        JoyConTeleop.update_gripper_button(self, side, bool(buttons["trigger"]), now)
        if not engaged:
            self.deactivate_arm(side)
            arm.fault_latched = False
            arm.hand_anchor_orientation = None
            arm.tcp_anchor_orientation = None
            return
        if not arm.active:
            arm.active = True
            # Always latch from measured FK on a new gesture. Otherwise an old
            # look-ahead survives release and appears as an unrelated command.
            self.request_fk(side, sample)
        if arm.target_pose is None:
            return

        hand_orientation = sample.get("orientation_xyzw")
        if hand_orientation is None:
            hand_orientation = euler_delta_quaternion(
                *[float(value) for value in sample["orientation_rpy"]]
            )
        hand_orientation = [float(value) for value in hand_orientation]
        proposed = {
            "position": [float(value) for value in arm.target_pose["position"]],
            "orientation": list(arm.target_pose["orientation"]),
        }
        for index, value in enumerate(fixed_delta):
            proposed["position"][index] += value

        # Both fixed Cartesian buttons and the full faucet gesture may change
        # TCP translation and orientation together. Each newly engaged gesture
        # treats the current hand attitude as its neutral orientation.
        if clutch or fixed_active:
            if arm.hand_anchor_orientation is None:
                arm.hand_anchor_orientation = hand_orientation
                arm.tcp_anchor_orientation = list(arm.target_pose["orientation"])
            local_relative = relative_quaternion(hand_orientation, arm.hand_anchor_orientation)
            orientation_signs = [
                float(value) for value in self.get_parameter("orientation_axis_signs").value
            ]
            local_relative = quaternion_axis_signs(local_relative, orientation_signs)
            if clutch:
                horizontal = normalize_stick(sample["stick"][0], self.deadzone)
                vertical = normalize_stick(sample["stick"][1], self.deadzone)
                vertical_input = float(bool(buttons.get("shoulder"))) - float(
                    bool(buttons.get("stick"))
                )
                base_relative = base_relative_quaternion(
                    hand_orientation, arm.hand_anchor_orientation
                )
                base_relative = quaternion_axis_signs(base_relative, orientation_signs)
                velocity = faucet_translation_velocity(
                    horizontal,
                    vertical,
                    base_relative,
                    float(self.get_parameter("tcp_speed_m_s").value),
                    vertical_input,
                )
                for index, value in enumerate(velocity):
                    proposed["position"][index] += value * dt
            commanded_orientation = quaternion_multiply(arm.tcp_anchor_orientation, local_relative)
            proposed["orientation"] = step_orientation_toward(
                proposed["orientation"],
                commanded_orientation,
                float(self.get_parameter("orientation_speed_rad_s").value) * dt,
            )
        else:
            arm.hand_anchor_orientation = None
            arm.tcp_anchor_orientation = None

        seed = self.arm_seed(side)
        if seed is None:
            return
        current = self.kinematics[side].forward(seed)
        target_lead = np.asarray(proposed["position"]) - current[:3, 3]
        target_lead_m = float(np.linalg.norm(target_lead))
        max_target_lead = float(self.get_parameter("dls_max_target_error_m").value)
        target_limited = target_lead_m > max_target_lead
        if target_limited:
            # This is an outer Cartesian tracking gate, not an IK rejection.
            # Keep the target at a bounded look-ahead while continuing to emit
            # joint steps that let the measured arm catch up.
            proposed["position"] = (
                current[:3, 3] + target_lead * (max_target_lead / target_lead_m)
            ).tolist()
        target = np.eye(4)
        target[:3, 3] = proposed["position"]
        target[:3, :3] = quaternion_to_matrix(proposed["orientation"])
        solve_started = time.perf_counter()
        # The control timer stays at the configured realtime rate, but a
        # slightly longer position
        # look-ahead prevents every command from collapsing to a few servo
        # ticks when Bridge anchors JointJog displacements at fresh feedback.
        # Preview needs no such look-ahead because its joint state accepts the
        # candidate exactly and has no static friction or bus latency.
        command_dt = (
            max(
                dt,
                float(self.get_parameter("dls_command_horizon_sec").value),
            )
            if self.hardware_mode
            else dt
        )
        candidate, metrics = self.kinematics[side].step(
            seed,
            target,
            command_dt,
            position_gain=float(self.get_parameter("dls_position_gain").value),
            orientation_gain=float(self.get_parameter("dls_orientation_gain").value),
            orientation_weight=float(self.get_parameter("dls_orientation_weight").value),
            damping=float(self.get_parameter("dls_damping").value),
            max_joint_velocity=float(self.get_parameter("dls_max_joint_velocity_rad_s").value),
            max_joint_step=float(self.get_parameter("dls_max_joint_step_rad").value),
            joint_limit_margin=float(self.get_parameter("dls_joint_limit_margin_rad").value),
        )
        self.max_dls_ms[side] = max(
            self.max_dls_ms[side],
            (time.perf_counter() - solve_started) * 1.0e3,
        )
        achieved = self.kinematics[side].forward(candidate)
        residual = float(np.linalg.norm(target[:3, 3] - achieved[:3, 3]))
        arm.target_pose = proposed
        limit_hits = [arm_names(side)[index] for index in metrics["joint_limit_hits"]]
        joint_delta = candidate - np.asarray(seed)
        delta_text = (
            ",".join(
                f"{name.removeprefix(side + '_')}={value:+.4f}"
                for name, value in zip(arm_names(side), joint_delta, strict=True)
                if abs(value) >= 1.0e-5
            )
            or "zero"
        )
        if target_limited:
            self.warn_ik(
                side,
                f"Cartesian target lead limited to {max_target_lead:.3f} m; "
                f"residual={residual:.3f} m "
                f"step={metrics['max_joint_step_rad']:.4f} rad "
                f"sigma_min={metrics['minimum_singular_value']:.4f} "
                f"limits={limit_hits or 'none'} dq=[{delta_text}]",
            )
        elif metrics["max_joint_step_rad"] < 1.0e-5 and residual > 0.005:
            self.warn_ik(
                side,
                f"DLS stalled with residual={residual:.3f} m; "
                f"sigma_min={metrics['minimum_singular_value']:.4f} "
                f"limits={limit_hits or 'none'} dq=[{delta_text}]",
            )
        if self.hardware_mode:
            self.send_arm_jog(
                side,
                (candidate - np.asarray(seed)).tolist(),
                dt,
            )
        else:
            for name, value in zip(arm_names(side), candidate, strict=True):
                self.positions[name] = float(value)
        self.publish_markers(side, arm.target_pose)
        if now - self.last_ik_ok.get(side, 0.0) >= 1.0:
            self.last_ik_ok[side] = now
            self.get_logger().info(
                f"[{side}] DLS {float(self.get_parameter('control_rate_hz').value):g} Hz: "
                f"xyz_err={metrics['position_error_m']:.4f}m "
                f"rot_err={metrics['orientation_error_rad']:.3f}rad "
                f"sigma_min={metrics['minimum_singular_value']:.4f} "
                f"step={metrics['max_joint_step_rad']:.4f}rad "
                f"limits={limit_hits or 'none'} dq=[{delta_text}]"
            )

    def update_arm_moveit(
        self,
        side: str,
        sample: dict | None,
        now: float,
        dt: float,
    ) -> None:
        arm = self.arms[side]
        if sample is None:
            # Stale/disconnected input still stops in-flight motion: the
            # operator is not in control.
            self.deactivate_arm(side, cancel_goal=True)
            arm.joy_anchor_rpy = None
            return
        buttons = sample["buttons"]
        delta = tcp_button_delta(
            buttons,
            float(self.get_parameter("tcp_speed_m_s").value),
            dt,
        )
        position_changed = any(abs(value) > 1.0e-8 for value in delta)
        # IMU tilt only adjusts TCP orientation while an SL/SR rail button of
        # that controller is held, so normal position moves never drift.
        orientation_mode = bool(buttons.get("sl") or buttons.get("sr"))
        engaged = position_changed or orientation_mode
        relatch = bool(buttons["relatch"])
        relatch_pressed = relatch and not arm.relatch_was_pressed
        arm.relatch_was_pressed = relatch
        if relatch_pressed and not arm.pending_fk:
            self.get_logger().info(f"{side} arm relatch requested (Home/Capture)")
            self.request_fk(side, sample)
        if not engaged:
            self.deactivate_arm(side)
            arm.fault_latched = False
            arm.joy_anchor_rpy = None
        elif arm.fault_latched:
            return
        elif not arm.active:
            arm.active = True
            if not arm.goal_busy:
                arm.cancel_after_accept = False
            # Only the first-ever engage (or an explicit Home relatch) runs
            # FK and resets the orientation reference; later engages keep the
            # adjusted TCP pose so releasing and re-pressing a button never
            # undoes the operator's orientation or triggers a large, rejected
            # level correction.
            if arm.target_pose is None and not arm.pending_fk:
                self.request_fk(side, sample)
        if arm.active and arm.target_pose is None and not arm.pending_fk:
            self.request_fk(side, sample)
        if not orientation_mode:
            arm.joy_anchor_rpy = None

        JoyConTeleop.update_gripper_button(self, side, bool(buttons["trigger"]), now)
        if not arm.active or arm.target_pose is None or arm.pending_fk:
            return
        if now - arm.last_ik_request < self.ik_period:
            return
        # The control timer is faster than IK. Using the timer's dt here loses
        # every step skipped by the IK rate limiter (30 Hz -> 12 Hz made a
        # configured 30 mm/s behave like roughly 12 mm/s). Integrate over the
        # actual interval since the last IK request, with a bounded first step.
        ik_step_dt = (
            dt if arm.last_ik_request <= 0.0 else min(0.1, max(dt, now - arm.last_ik_request))
        )
        delta = tcp_button_delta(
            buttons,
            float(self.get_parameter("tcp_speed_m_s").value),
            ik_step_dt,
        )
        position_changed = any(abs(value) > 1.0e-8 for value in delta)
        proposed = {
            "position": [arm.target_pose["position"][index] + delta[index] for index in range(3)],
            "orientation": list(arm.target_pose["orientation"]),
        }
        orientation_changed = False
        if orientation_mode:
            # Anchor on press: the commanded TCP attitude is the pose at
            # press time rotated by the Joy-Con attitude change since then,
            # stepped at orientation_speed_rad_s. The controller's absolute
            # attitude is irrelevant, so re-pressing SL/SR continues from the
            # current gripper orientation without returning to any pose.
            if arm.joy_anchor_rpy is None:
                arm.joy_anchor_rpy = [float(value) for value in sample["orientation_rpy"]]
                arm.joy_anchor_orientation = list(arm.target_pose["orientation"])
            else:
                max_delta = float(self.get_parameter("max_orientation_delta_rad").value)
                scale = float(self.get_parameter("orientation_scale").value)
                signs = [
                    float(value) for value in self.get_parameter("orientation_axis_signs").value
                ]
                deltas = [
                    max(
                        -max_delta,
                        min(max_delta, sign * scale * (current - anchor)),
                    )
                    for sign, current, anchor in zip(
                        signs,
                        sample["orientation_rpy"],
                        arm.joy_anchor_rpy,
                        strict=True,
                    )
                ]
                if max(abs(value) for value in deltas) > self.orientation_deadband:
                    commanded = quaternion_multiply(
                        arm.joy_anchor_orientation,
                        euler_delta_quaternion(*deltas),
                    )
                    proposed["orientation"] = step_orientation_toward(
                        proposed["orientation"],
                        commanded,
                        float(self.get_parameter("orientation_speed_rad_s").value) * dt,
                    )
                    orientation_changed = True
        if arm.orientation_reference is not None:
            # Gradual leveling after the first latch: step the target toward
            # the level-grasp reference at the configured speed, then stop.
            proposed["orientation"] = step_orientation_toward(
                proposed["orientation"],
                arm.orientation_reference,
                float(self.get_parameter("orientation_speed_rad_s").value) * dt,
            )
            dot = sum(
                a * b
                for a, b in zip(proposed["orientation"], arm.orientation_reference, strict=True)
            )
            if 2.0 * math.acos(max(-1.0, min(1.0, abs(dot)))) < 0.005:
                arm.orientation_reference = None
            else:
                orientation_changed = True
        if position_changed or orientation_changed:
            self.request_ik(
                side,
                proposed,
                now,
                allow_position_fallback=position_changed,
            )

    def update_gripper_button(self, side: str, pressed: bool, now: float) -> None:
        """Toggle once, then require a confirmed release before re-arming."""
        arm = self.arms[side]
        release_grace = float(self.get_parameter("button_release_grace_sec").value)
        debounce = float(self.get_parameter("gripper_button_debounce_sec").value)
        if pressed:
            arm.trigger_release_started = None
            if not arm.trigger_was_pressed:
                arm.trigger_was_pressed = True
                if now - arm.last_gripper_toggle >= debounce:
                    arm.last_gripper_toggle = now
                    self.toggle_gripper(side)
            return
        if not arm.trigger_was_pressed:
            arm.trigger_release_started = None
            return
        if arm.trigger_release_started is None:
            arm.trigger_release_started = now
            return
        if now - arm.trigger_release_started >= release_grace:
            arm.trigger_was_pressed = False
            arm.trigger_release_started = None

    def toggle_gripper(self, side: str) -> None:
        name = f"{side}_gripper"
        closed = 0.32
        opened = -1.8030294104
        arm = self.arms[side]
        was_open_commanded = arm.gripper_open_commanded
        if was_open_commanded is None:
            was_open_commanded = self.positions.get(name, closed) <= -0.7
        open_commanded = not was_open_commanded
        target = opened if open_commanded else closed
        if not self.hardware_mode:
            self.positions[name] = target
            arm.gripper_open_commanded = open_commanded
            return
        client = self.gripper_clients[side]
        if not client.server_is_ready():
            return
        goal = GripperCommand.Goal()
        goal.command.position = target
        goal.command.max_effort = 0.0
        client.send_goal_async(goal)
        arm.gripper_open_commanded = open_commanded

    def update_base(
        self,
        left: dict | None,
        right: dict | None,
        lift_active: bool,
        dt: float,
    ) -> None:
        sticks = []
        turn = 0.0
        if right is not None:
            arm_clutch = bool(right["buttons"].get("sl") or right["buttons"].get("sr"))
            if arm_clutch:
                pass
            elif bool(right["buttons"]["shoulder"]):
                # R + right stick left/right: base turn only.
                turn += normalize_stick(right["stick"][0], self.deadzone)
            else:
                sticks.append(
                    (
                        normalize_stick(right["stick"][0], self.deadzone),
                        normalize_stick(right["stick"][1], self.deadzone),
                    )
                )
        if left is not None:
            arm_clutch = bool(left["buttons"].get("sl") or left["buttons"].get("sr"))
            if arm_clutch or lift_active or bool(left["buttons"]["shoulder"]):
                # L held: the left stick is reserved for the lift (vertical)
                # and contributes no base translation.
                pass
            else:
                sticks.append(
                    (
                        normalize_stick(left["stick"][0], self.deadzone),
                        normalize_stick(left["stick"][1], self.deadzone),
                    )
                )
        vx, vy, yaw = base_command(
            sticks,
            float(self.get_parameter("base_linear_speed_m_s").value),
            float(self.get_parameter("base_angular_speed_rad_s").value),
        )
        yaw = -max(-1.0, min(1.0, turn)) * float(
            self.get_parameter("base_angular_speed_rad_s").value
        )
        active = any(abs(value) > 1.0e-9 for value in (vx, vy, yaw))
        if active:
            if self.hardware_mode:
                message = Twist()
                message.linear.x, message.linear.y, message.angular.z = vx, vy, yaw
                self.cmd_vel_pub.publish(message)
            else:
                self.preview_base = integrate_base_preview(self.preview_base, vx, vy, yaw, dt)
                for name, value in zip(ROOT_JOINTS, self.preview_base, strict=True):
                    self.positions[name] = value
            self.base_was_active = True
        elif self.base_was_active:
            if self.hardware_mode:
                self.cmd_vel_pub.publish(Twist())
            self.base_was_active = False

    def publish_readiness(self) -> None:
        if not self.hardware_mode:
            return
        now = time.monotonic()
        if now - self.last_readiness_log < 10.0 and self.readiness_announced:
            return
        issues = []
        if self.last_measured_state is None or (
            now - self.last_measured_state > self.measured_state_timeout
        ):
            issues.append("no fresh measured joint feedback")
        if self.arm_control_mode == "moveit":
            for name, client in self.arm_clients.items():
                if not client.server_is_ready():
                    issues.append(f"{name} action server not ready")
        if issues:
            self.last_readiness_log = now
            self.get_logger().warning("hardware not ready yet: " + "; ".join(issues))
            return
        if not self.readiness_announced:
            self.readiness_announced = True
            self.get_logger().info("hardware ready: measured state fresh and arm action servers up")

    def update_lift(self, sample: dict | None, active: bool, now: float, dt: float) -> None:
        del now
        vertical = (
            normalize_stick(sample["stick"][1], self.deadzone)
            if active and sample is not None
            else 0.0
        )
        speed = float(self.get_parameter("lift_speed_m_s").value)
        if not self.hardware_mode:
            self.lift_target = max(
                -0.3,
                min(0.3, self.lift_target + vertical * speed * dt),
            )
            self.positions["vertical_move"] = self.lift_target
            self.lift_active = abs(vertical) > 1.0e-9
            return
        moving = abs(vertical) > 1.0e-9
        if not moving and not self.lift_active:
            return
        message = JointJog()
        message.joint_names = ["vertical_move"]
        message.header.stamp = self.get_clock().now().to_msg()
        message.velocities = [vertical * speed]
        self.lift_jog_pub.publish(message)
        self.lift_active = moving

    def publish_preview(self) -> None:
        if self.hardware_mode:
            return
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(self.positions)
        message.position = list(self.positions.values())
        self.preview_pub.publish(message)

    def publish_markers(self, side: str, pose: dict) -> None:
        marker = Marker()
        marker.header.frame_id = f"{side}_Base"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "joycon_tcp_target"
        marker.id = 0 if side == "left" else 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(
            x=float(pose["position"][0]),
            y=float(pose["position"][1]),
            z=float(pose["position"][2]),
        )
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.025
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (
            (0.2, 1.0, 0.2, 1.0) if side == "left" else (1.0, 0.7, 0.1, 1.0)
        )
        marker.frame_locked = True
        array = MarkerArray()
        array.markers = [marker]
        self.marker_pub.publish(array)

    def on_timer(self) -> None:
        now = time.monotonic()
        dt = min(0.1, max(0.0, now - self.last_loop))
        self.last_loop = now
        self.max_control_dt_ms = max(self.max_control_dt_ms, dt * 1.0e3)
        # Check the previous sample before a resumed stream can replace it.
        if any(now - item.received_at > self.input_timeout for item in self.samples.values()):
            self.invalidate_inputs()
        self.receive_inputs()
        if self.hardware_mode:
            ready = self.hardware_ready(now)
            if self.hardware_was_ready and not ready:
                self.invalidate_inputs()
            self.hardware_was_ready = ready
        left = self.sample("left", now)
        right = self.sample("right", now)
        if left is None and right is None and now - self.last_stale_warning > 5.0:
            self.get_logger().warning("No fresh Joy-Con input; no new robot targets")
            self.last_stale_warning = now
        measured_fresh = bool(not self.hardware_mode or self.hardware_was_ready)
        if not measured_fresh:
            left = None
            right = None
        if left is None:
            self.left_lift_stick_latched = False
        else:
            self.left_lift_stick_latched = next_lift_stick_latch(
                self.left_lift_stick_latched,
                left["buttons"],
                left["stick"][1],
                self.deadzone,
            )
        lift_active = self.left_lift_stick_latched
        self.update_lift(left, lift_active, now, dt)
        self.update_base(left, right, lift_active, dt)
        # Moving the lift moves both arm bases. Do not mix a cached Cartesian
        # target with that changing frame: cancel queued/in-flight arm motion
        # and re-latch FK at the new height on the next arm engagement.
        arm_left = None if lift_active else left
        arm_right = None if lift_active else right
        self.update_arm("left", arm_left, now, dt)
        self.update_arm("right", arm_right, now, dt)
        self.publish_preview()
        if now - self.last_timing_log >= 5.0:
            self.last_timing_log = now
            self.get_logger().info(
                "Joy-Con timing: "
                + ", ".join(
                    f"{side} dropped={self.input_dropped[side]} "
                    f"max_age={self.input_max_age_ms[side]:.1f}ms "
                    f"max_dls={self.max_dls_ms[side]:.1f}ms"
                    for side in ("left", "right")
                )
                + f", max_loop_dt={self.max_control_dt_ms:.1f}ms"
            )
            self.input_max_age_ms = dict.fromkeys(("left", "right"), 0.0)
            self.max_dls_ms = dict.fromkeys(("left", "right"), 0.0)
            self.max_control_dt_ms = 0.0


def main() -> None:
    rclpy.init()
    node = JoyConTeleop()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
