import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
from alohamini_gazebo.sim_description import (
    ARM_JOINTS,
    SIM_BLACK_VISUAL_LINKS,
    make_sim_description,
)

from alohamini.model import get_robot_model

AUTHORITATIVE = get_robot_model("alohamini2pro").directory / "urdf/alohamini2pro.urdf"


def test_overlay_uses_native_meshes_and_isolated_control_namespace(tmp_path):
    original = AUTHORITATIVE.read_bytes()
    derived = ET.fromstring(make_sim_description(AUTHORITATIVE, tmp_path / "controllers.yaml"))
    for mesh in derived.findall(".//mesh"):
        uri = urlparse(mesh.get("filename"))
        assert uri.scheme == "file"
        path = Path(unquote(uri.path))
        assert path.is_file() and path.is_relative_to(AUTHORITATIVE.parent.parent)
    plugin = derived.find("./gazebo/plugin[@name='gz_ros2_control::GazeboSimROS2ControlPlugin']")
    assert plugin.findtext("ros/namespace") == "/alohamini_sim"
    assert plugin.findtext("robot_param_node") == "alohamini_gazebo_robot_state_publisher"
    assert {entry.text for entry in plugin.findall("ros/remapping")} == {
        "/clock:=/alohamini_sim/clock",
        "/tf:=/alohamini_sim/tf",
        "/tf_static:=/alohamini_sim/tf_static",
    }
    assert AUTHORITATIVE.read_bytes() == original


def test_overlay_rejects_missing_or_external_meshes(tmp_path):
    source = ET.parse(AUTHORITATIVE)
    source.getroot().find(".//mesh").set("filename", "/etc/passwd")
    modified = tmp_path / "urdf" / "model.urdf"
    modified.parent.mkdir()
    source.write(modified)
    with pytest.raises(ValueError, match="external model mesh"):
        make_sim_description(modified, tmp_path / "controllers.yaml")


def test_overlay_preserves_authoritative_joint_kinematics(tmp_path):
    source = ET.parse(AUTHORITATIVE).getroot()
    derived = ET.fromstring(make_sim_description(AUTHORITATIVE, tmp_path / "controllers.yaml"))

    def semantic_xml(element):
        return (
            element.tag,
            tuple(sorted(element.attrib.items())),
            (element.text or "").strip(),
            tuple(semantic_xml(child) for child in element),
        )

    source_joints = {joint.get("name"): semantic_xml(joint) for joint in source.findall("joint")}
    derived_joints = {
        joint.get("name"): semantic_xml(joint)
        for joint in derived.findall("joint")
        if joint.get("name") != "world_to_root"
    }
    assert derived_joints == source_joints
    assert derived.find("./joint[@name='world_to_root']") is not None
    assert derived.find("./ros2_control") is not None


def test_overlay_exports_only_expected_controlled_joints(tmp_path):
    derived = ET.fromstring(make_sim_description(AUTHORITATIVE, tmp_path / "controllers.yaml"))
    actual = {joint.get("name") for joint in derived.findall("./ros2_control/joint")}
    expected = {
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_rotation_joint",
        "vertical_move",
        "wheel1_joint",
        "wheel2_joint",
        "wheel3_joint",
        *ARM_JOINTS,
    }
    assert actual == expected


def test_overlay_makes_only_arms_and_cameras_matte_black(tmp_path):
    derived = ET.fromstring(make_sim_description(AUTHORITATIVE, tmp_path / "controllers.yaml"))
    for name in SIM_BLACK_VISUAL_LINKS:
        materials = derived.findall(f"./link[@name='{name}']/visual/material")
        assert materials
        assert all(material.get("name") == "sim_matte_black" for material in materials)
        assert all(
            material.find("color").get("rgba") == "0.012 0.015 0.020 1" for material in materials
        )

    body_material = derived.find("./link[@name='vertical_link']/visual/material")
    assert body_material is not None
    assert body_material.get("name") != "sim_matte_black"
