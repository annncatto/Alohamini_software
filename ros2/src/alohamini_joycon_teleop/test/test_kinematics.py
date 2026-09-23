from pathlib import Path

import numpy as np
import pytest
import yaml
from alohamini_joycon_teleop.kinematics import (
    AlohaMiniArmKinematics,
    _scale_to_max_abs,
)
from ament_index_python.packages import get_package_share_directory


@pytest.fixture(scope="module")
def kinematics():
    return AlohaMiniArmKinematics.from_description()


def test_dh_fk_matches_all_golden_tcp_transforms(kinematics):
    validation = Path(get_package_share_directory("alohamini_validation"))
    with (validation / "config/fk_golden.yaml").open(encoding="utf-8") as stream:
        golden = yaml.safe_load(stream)
    for sample in golden["samples"].values():
        assert np.allclose(
            kinematics.forward(sample["q_rad"]),
            np.asarray(sample["transform"]),
            atol=float(golden["tolerance"]),
        )


def test_differential_ik_reduces_xyz_error(kinematics):
    joints = np.array([-0.35, -1.2, 1.8, -0.25, 0.4, -0.6])
    target = kinematics.forward(joints)
    target[:3, 3] += np.array([0.005, -0.004, 0.003])
    initial_error = np.linalg.norm(target[:3, 3] - kinematics.forward(joints)[:3, 3])
    metrics = None
    for _ in range(30):
        joints, metrics = kinematics.step(joints, target, 1.0 / 30.0)
    final_error = np.linalg.norm(target[:3, 3] - kinematics.forward(joints)[:3, 3])
    assert final_error < initial_error * 0.1
    assert metrics["minimum_singular_value"] >= 0.0


def test_differential_ik_is_bounded(kinematics):
    joints = np.array([0.0, -1.571, 1.571, 0.0, 0.0, 0.0])
    target = kinematics.forward(joints)
    target[:3, 3] += 1.0
    candidate, _ = kinematics.step(joints, target, 1.0, max_joint_step=0.03)
    assert np.max(np.abs(candidate - joints)) <= 0.03 + 1.0e-12


def test_dls_vector_bound_preserves_joint_ratios():
    original = np.array([0.20, -0.10, 0.05, 0.0])
    bounded = _scale_to_max_abs(original, 0.08)
    assert bounded == pytest.approx([0.08, -0.04, 0.02, 0.0])


def test_joint_limit_margin_blocks_only_outward_motion(kinematics):
    joints = np.array([0.0, -1.571, 1.571, 0.0, 0.0, 0.0])
    lower, upper = kinematics.joint_limits[0]

    near_upper = joints.copy()
    near_upper[0] = upper - 0.01
    outward_target = kinematics.forward(near_upper)
    outward_target[:3, 3] += kinematics.jacobian(near_upper)[:3, 0] * 0.01
    outward, metrics = kinematics.step(
        near_upper,
        outward_target,
        1.0 / 30.0,
        orientation_gain=0.0,
        joint_limit_margin=0.03,
    )
    assert outward[0] == pytest.approx(near_upper[0])
    assert 0 in metrics["joint_limit_hits"]

    inward_target = kinematics.forward(near_upper)
    inward_target[:3, 3] -= kinematics.jacobian(near_upper)[:3, 0] * 0.01
    inward, _ = kinematics.step(
        near_upper,
        inward_target,
        1.0 / 30.0,
        orientation_gain=0.0,
        joint_limit_margin=0.03,
    )
    assert inward[0] < near_upper[0]
