from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from alohamini.model import get_robot_model

ARM_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw_joint",
    "wrist_roll",
)


def _scale_to_max_abs(values: np.ndarray, limit: float) -> np.ndarray:
    """Bound a vector without changing its direction."""
    limit = max(0.0, float(limit))
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if peak <= limit or peak <= 1.0e-12:
        return values
    return values * (limit / peak)


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """Return the shortest rotation vector for a 3x3 rotation matrix."""
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1.0e-9:
        return 0.5 * np.array(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        )
    if math.pi - angle < 1.0e-5:
        diagonal = np.maximum(0.0, (np.diag(matrix) + 1.0) * 0.5)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] > 1.0e-8:
            if largest == 0:
                axis[1] = math.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
                axis[2] = math.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
            elif largest == 1:
                axis[0] = math.copysign(axis[0], matrix[0, 1] + matrix[1, 0])
                axis[2] = math.copysign(axis[2], matrix[1, 2] + matrix[2, 1])
            else:
                axis[0] = math.copysign(axis[0], matrix[0, 2] + matrix[2, 0])
                axis[1] = math.copysign(axis[1], matrix[1, 2] + matrix[2, 1])
        norm = np.linalg.norm(axis)
        return angle * axis / max(norm, 1.0e-12)
    scale = angle / (2.0 * math.sin(angle))
    return scale * np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )


def quaternion_to_matrix(quaternion_xyzw: list[float] | np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("quaternion must be finite and non-zero")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quaternion(matrix: np.ndarray) -> list[float]:
    matrix = np.asarray(matrix, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.tolist()


class AlohaMiniArmKinematics:
    """Exact standard-DH FK plus bounded differential IK for one arm.

    The packaged DH table is generated from the authoritative URDF. Both arms
    have the same chain in their own ``{side}_Base`` frame; only limits differ.
    """

    def __init__(
        self,
        dh_path: str | Path,
        tool_path: str | Path,
        joint_limits: list[tuple[float, float]] | None = None,
    ) -> None:
        with Path(dh_path).open(encoding="utf-8") as stream:
            dh = yaml.safe_load(stream)
        with Path(tool_path).open(encoding="utf-8") as stream:
            tool = yaml.safe_load(stream)["tool_frames"]
        rows = dh["standard_dh"]["rows"]
        self.rows = tuple(
            (
                float(row["a"]),
                float(row["alpha"]),
                float(row["d"]),
                float(row["theta_offset"]),
            )
            for row in rows
        )
        self.base_transform = np.asarray(dh["standard_dh"]["base_transform"], dtype=float)
        self.tool_transform = np.asarray(dh["standard_dh"]["tool_transform"], dtype=float)
        fixed_to_tcp = np.eye(4)
        fixed_to_tcp[:3, :3] = np.asarray(tool["delta_matrix"], dtype=float)
        fixed_to_tcp[:3, 3] = fixed_to_tcp[:3, :3] @ np.asarray(tool["tcp_tool_m"], dtype=float)
        self.tool_transform = self.tool_transform @ fixed_to_tcp
        self.joint_limits = joint_limits

    @classmethod
    def from_description(
        cls,
        side: str = "right",
        joint_limits: list[tuple[float, float]] | None = None,
    ) -> AlohaMiniArmKinematics:
        share = get_robot_model("alohamini2pro").directory
        directory = share / "config" / "kinematics"
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        if joint_limits is None:
            root = ET.parse(share / "urdf" / "alohamini2pro_kinematic.urdf").getroot()
            joints = {joint.get("name"): joint for joint in root.findall("joint")}
            joint_limits = []
            for suffix in ARM_SUFFIXES:
                limit = joints[f"{side}_{suffix}"].find("limit")
                joint_limits.append((float(limit.get("lower")), float(limit.get("upper"))))
        return cls(
            directory / "right_arm_kinematics.yaml",
            directory / "kinematics.yaml",
            joint_limits,
        )

    def forward(self, joints: list[float] | np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError("joints must contain six finite values")
        transform = self.base_transform.copy()
        for q, (a, alpha, d, offset) in zip(joints, self.rows, strict=True):
            theta = float(q) + offset
            ct, st = math.cos(theta), math.sin(theta)
            ca, sa = math.cos(alpha), math.sin(alpha)
            transform = transform @ np.array(
                [
                    [ct, -st * ca, st * sa, a * ct],
                    [st, ct * ca, -ct * sa, a * st],
                    [0.0, sa, ca, d],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        return transform @ self.tool_transform

    def link_transforms(self, joints: list[float] | np.ndarray) -> list[np.ndarray]:
        """Return base, six DH joint frames, and TCP for visualization."""
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError("joints must contain six finite values")
        transform = self.base_transform.copy()
        transforms = [transform.copy()]
        for q, (a, alpha, d, offset) in zip(joints, self.rows, strict=True):
            theta = float(q) + offset
            ct, st = math.cos(theta), math.sin(theta)
            ca, sa = math.cos(alpha), math.sin(alpha)
            transform = transform @ np.array(
                [
                    [ct, -st * ca, st * sa, a * ct],
                    [st, ct * ca, -ct * sa, a * st],
                    [0.0, sa, ca, d],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            transforms.append(transform.copy())
        transforms.append(transform @ self.tool_transform)
        return transforms

    def jacobian(self, joints: list[float] | np.ndarray, epsilon: float = 1.0e-6) -> np.ndarray:
        joints = np.asarray(joints, dtype=float)
        current = self.forward(joints)
        jacobian = np.empty((6, 6), dtype=float)
        for index in range(6):
            shifted = joints.copy()
            shifted[index] += epsilon
            moved = self.forward(shifted)
            jacobian[:3, index] = (moved[:3, 3] - current[:3, 3]) / epsilon
            jacobian[3:, index] = _rotation_vector(moved[:3, :3] @ current[:3, :3].T) / epsilon
        return jacobian

    def step(
        self,
        joints: list[float] | np.ndarray,
        target: np.ndarray,
        dt: float,
        *,
        position_gain: float = 8.0,
        orientation_gain: float = 5.0,
        orientation_weight: float = 0.35,
        damping: float = 0.04,
        max_joint_velocity: float = 1.5,
        max_joint_step: float = 0.08,
        joint_limit_margin: float = 0.03,
    ) -> tuple[np.ndarray, dict[str, float | list[int]]]:
        joints = np.asarray(joints, dtype=float)
        target = np.asarray(target, dtype=float)
        current = self.forward(joints)
        position_error = target[:3, 3] - current[:3, 3]
        orientation_error = _rotation_vector(target[:3, :3] @ current[:3, :3].T)
        jacobian = self.jacobian(joints)
        weighted_jacobian = jacobian.copy()
        weighted_jacobian[3:] *= orientation_weight
        twist = np.concatenate(
            (position_gain * position_error, orientation_gain * orientation_error)
        )
        twist[3:] *= orientation_weight
        singular_values = np.linalg.svd(weighted_jacobian, compute_uv=False)
        minimum_singular = float(singular_values[-1])
        adaptive_damping = damping * (1.0 + max(0.0, 0.08 - minimum_singular) / 0.08 * 4.0)
        regularizer = (adaptive_damping * adaptive_damping) * np.eye(6)
        velocity = weighted_jacobian.T @ np.linalg.solve(
            weighted_jacobian @ weighted_jacobian.T + regularizer,
            twist,
        )
        # Scale the complete DLS vector instead of clipping joints
        # independently. Component-wise saturation changes the joint-space
        # direction whenever a different joint reaches the bound, which shows
        # up as a small Cartesian kink during continuous teleoperation.
        velocity = _scale_to_max_abs(velocity, max_joint_velocity)
        delta = _scale_to_max_abs(velocity * max(0.0, float(dt)), max_joint_step)
        candidate = joints + delta
        joint_limit_hits: list[int] = []
        if self.joint_limits is not None:
            lower = np.asarray([limit[0] for limit in self.joint_limits])
            upper = np.asarray([limit[1] for limit in self.joint_limits])
            margin = max(0.0, float(joint_limit_margin))
            safe_lower = lower + margin
            safe_upper = upper - margin
            for index, (joint, proposed_delta) in enumerate(zip(joints, delta, strict=True)):
                if proposed_delta < 0.0:
                    if joint <= safe_lower[index]:
                        candidate[index] = joint
                        joint_limit_hits.append(index)
                    elif candidate[index] < safe_lower[index]:
                        candidate[index] = safe_lower[index]
                        joint_limit_hits.append(index)
                elif proposed_delta > 0.0:
                    if joint >= safe_upper[index]:
                        candidate[index] = joint
                        joint_limit_hits.append(index)
                    elif candidate[index] > safe_upper[index]:
                        candidate[index] = safe_upper[index]
                        joint_limit_hits.append(index)
        return candidate, {
            "position_error_m": float(np.linalg.norm(position_error)),
            "orientation_error_rad": float(np.linalg.norm(orientation_error)),
            "minimum_singular_value": minimum_singular,
            "damping": adaptive_damping,
            "max_joint_step_rad": float(np.max(np.abs(candidate - joints))),
            "joint_limit_hits": joint_limit_hits,
        }
