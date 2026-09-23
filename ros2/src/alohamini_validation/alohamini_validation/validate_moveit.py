# SPDX-License-Identifier: Apache-2.0
# Migrated from alohamini_ros2/alohamini_validation.
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetStateValidity
from rclpy.node import Node

from alohamini.model import get_robot_model

ARM_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw_joint",
    "wrist_roll",
)
DEFAULT_STATE = {
    "root_x_axis_joint": 0.0,
    "root_y_axis_joint": 0.0,
    "root_z_rotation_joint": 0.0,
    "vertical_move": 0.2,
    "wheel1_joint": 0.0,
    "wheel2_joint": 0.0,
    "wheel3_joint": 0.0,
    "left_shoulder_pan": 0.0,
    "left_shoulder_lift": -1.571,
    "left_elbow_flex": 1.571,
    "left_wrist_flex": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_roll": 0.0,
    "left_gripper": -1.8030294104,
    "right_shoulder_pan": 0.0,
    "right_shoulder_lift": -1.571,
    "right_elbow_flex": 1.571,
    "right_wrist_flex": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_roll": 0.0,
    "right_gripper": -1.8030294104,
}


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    rotation = matrix[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            x, y, z, w = (
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            x, y, z, w = (
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            )
        else:
            scale = math.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            x, y, z, w = (
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
    quaternion = np.asarray([x, y, z, w], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def pose_matrix(pose) -> np.ndarray:
    x, y, z, w = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def orientation_error(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    return math.acos(min(1.0, max(-1.0, (float(np.trace(relative)) - 1.0) / 2.0)))


class Validator(Node):
    def __init__(self) -> None:
        super().__init__("alohamini_moveit_validation", namespace="/alohamini_plan_only")
        self.fk_client = self.create_client(GetPositionFK, "compute_fk")
        self.ik_client = self.create_client(GetPositionIK, "compute_ik")
        self.validity_client = self.create_client(GetStateValidity, "check_state_validity")
        for client in (self.fk_client, self.ik_client, self.validity_client):
            if not client.wait_for_service(timeout_sec=15.0):
                name = client.srv_name
                self.destroy_node()
                raise RuntimeError(f"MoveIt service unavailable: {name}")

    def call(self, client, request, timeout: float = 5.0):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() is None:
            raise RuntimeError("MoveIt service call timed out")
        return future.result()

    @staticmethod
    def fill_state(robot_state, values: dict[str, float]) -> None:
        complete = dict(DEFAULT_STATE)
        complete.update(values)
        robot_state.joint_state.name = list(complete)
        robot_state.joint_state.position = list(complete.values())

    def fk_in_root(self, side: str, q_rad: list[float]) -> tuple[np.ndarray, np.ndarray]:
        request = GetPositionFK.Request()
        request.header.frame_id = "root"
        request.fk_link_names = [f"{side}_Base", f"{side}_tcp"]
        values = dict(zip((f"{side}_{name}" for name in ARM_SUFFIXES), q_rad, strict=True))
        self.fill_state(request.robot_state, values)
        response = self.call(self.fk_client, request)
        assert response.error_code.val == response.error_code.SUCCESS, response.error_code.val
        assert len(response.pose_stamped) == 2
        return tuple(pose_matrix(item.pose) for item in response.pose_stamped)

    def fk(self, side: str, q_rad: list[float]) -> np.ndarray:
        base, tcp = self.fk_in_root(side, q_rad)
        return np.linalg.inv(base) @ tcp

    def ik(self, side: str, target: np.ndarray, seed: list[float]) -> list[float]:
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = f"{side}_arm"
        ik.ik_link_name = f"{side}_tcp"
        ik.pose_stamped.header.frame_id = "root"
        base, _ = self.fk_in_root(side, seed)
        root_target = base @ target
        (
            ik.pose_stamped.pose.position.x,
            ik.pose_stamped.pose.position.y,
            ik.pose_stamped.pose.position.z,
        ) = (float(value) for value in root_target[:3, 3])
        (
            ik.pose_stamped.pose.orientation.x,
            ik.pose_stamped.pose.orientation.y,
            ik.pose_stamped.pose.orientation.z,
            ik.pose_stamped.pose.orientation.w,
        ) = matrix_to_quaternion(root_target)
        self.fill_state(
            ik.robot_state,
            dict(zip((f"{side}_{name}" for name in ARM_SUFFIXES), seed, strict=True)),
        )
        ik.avoid_collisions = False
        ik.timeout = Duration(sec=1)
        response = self.call(self.ik_client, request)
        assert response.error_code.val == response.error_code.SUCCESS, response.error_code.val
        solution = dict(
            zip(
                response.solution.joint_state.name,
                response.solution.joint_state.position,
                strict=True,
            )
        )
        return [float(solution[f"{side}_{name}"]) for name in ARM_SUFFIXES]

    def state_validity(self, values: dict[str, float]) -> tuple[bool, set[tuple[str, str]]]:
        request = GetStateValidity.Request()
        self.fill_state(request.robot_state, values)
        response = self.call(self.validity_client, request)
        pairs = {
            tuple(sorted((contact.contact_body_1, contact.contact_body_2)))
            for contact in response.contacts
        }
        return bool(response.valid), pairs


def main() -> None:
    validation = Path(get_package_share_directory("alohamini_validation"))
    semantic = ET.parse(get_robot_model("alohamini2pro").description_path("semantic")).getroot()
    golden = load_yaml(validation / "config/fk_golden.yaml")
    collision = load_yaml(validation / "config/collision_baseline.yaml")
    rclpy.init()
    node = None
    try:
        node = Validator()
        tolerance = float(golden["tolerance"])
        for sample_name, sample in golden["samples"].items():
            expected = np.asarray(sample["transform"], dtype=float)
            for side in ("left", "right"):
                actual = node.fk(side, sample["q_rad"])
                assert np.allclose(actual[:3, 3], expected[:3, 3], atol=tolerance), (
                    f"{sample_name}/{side} position"
                )
                assert orientation_error(actual, expected) <= 1e-7, (
                    f"{sample_name}/{side} orientation"
                )
        print("[PASS] MoveIt FK golden")

        reference = golden["samples"]["reference"]["q_rad"]
        target = np.asarray(golden["samples"]["offset_a"]["transform"], dtype=float)
        for side in ("left", "right"):
            solution = node.ik(side, target, reference)
            recovered = node.fk(side, solution)
            assert np.linalg.norm(recovered[:3, 3] - target[:3, 3]) < 1e-4
            assert orientation_error(recovered, target) < 1e-3
        print("[PASS] MoveIt IK reachable targets")

        for state_name, state in collision["state_validity"].items():
            values = state.get("joints")
            if values is None:
                values = {
                    joint.attrib["name"]: float(joint.attrib["value"])
                    for group in semantic.findall("group_state")
                    if group.attrib["name"] == state["named_state"]
                    for joint in group.findall("joint")
                }
                assert values, f"Missing SRDF state: {state['named_state']}"
            valid, pairs = node.state_validity({key: float(value) for key, value in values.items()})
            assert valid is bool(state["valid"]), (
                f"{state_name}: valid={valid}, contacts={sorted(pairs)}"
            )
            expected_pairs = {tuple(pair) for pair in state.get("contact_pairs", [])}
            assert pairs == expected_pairs, f"{state_name}: contacts={sorted(pairs)}"
        print("[PASS] MoveIt collision state baseline")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
