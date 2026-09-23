"""Derive a Gazebo-only overlay from the platform's authoritative URDF."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ARM_JOINTS = [
    *(
        f"left_{name}"
        for name in (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw_joint",
            "wrist_roll",
            "gripper",
        )
    ),
    *(
        f"right_{name}"
        for name in (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw_joint",
            "wrist_roll",
            "gripper",
        )
    ),
]

ARM_VISUAL_LINKS = {
    *(
        f"left_{name}"
        for name in (
            "Base",
            "Rotation_Pitch",
            "Upper_Arm",
            "Lower_Arm",
            "Wrist_Pitch_Roll",
            "wrist_yaw",
            "Fixed_Jaw",
            "Moving_Jaw",
        )
    ),
    *(
        f"right_{name}"
        for name in (
            "Base",
            "Rotation_Pitch",
            "Upper_Arm",
            "Lower_Arm",
            "Wrist_Pitch_Roll",
            "wrist_yaw",
            "Fixed_Jaw",
            "Moving_Jaw",
        )
    ),
}

CAMERA_VISUAL_LINKS = {
    "chest_camera",
    "front_camera",
    "back_camera",
    "left_camera",
    "right_camera",
}

SIM_BLACK_VISUAL_LINKS = ARM_VISUAL_LINKS | CAMERA_VISUAL_LINKS

INITIAL_POSITION = {
    "root_x_axis_joint": 0.0,
    "root_y_axis_joint": 0.0,
    "root_z_rotation_joint": 0.0,
    "vertical_move": -0.3,
    "left_wrist_flex": 1.435806017460960,
    "right_wrist_flex": 1.5,
    "left_gripper": 0.32,
    "right_gripper": 0.32,
}


def _sub(parent: ET.Element, tag: str, **attributes: str) -> ET.Element:
    return ET.SubElement(parent, tag, attributes)


def _add_tiny_inertial(link: ET.Element) -> None:
    if link.find("inertial") is not None:
        return
    inertial = _sub(link, "inertial")
    _sub(inertial, "mass", value="0.001")
    _sub(
        inertial,
        "inertia",
        ixx="1e-6",
        ixy="0",
        ixz="0",
        iyy="1e-6",
        iyz="0",
        izz="1e-6",
    )


def _add_joint_interface(
    control: ET.Element,
    name: str,
    command_interface: str,
) -> None:
    joint = _sub(control, "joint", name=name)
    command = _sub(joint, "command_interface", name=command_interface)
    if command_interface == "position":
        command.append(ET.Comment("Controller limits still come from the authoritative URDF."))
    for state_name in ("position", "velocity"):
        state = _sub(joint, "state_interface", name=state_name)
        if state_name == "position":
            _sub(state, "param", name="initial_value").text = str(INITIAL_POSITION.get(name, 0.0))


def _set_visual_material(link: ET.Element, name: str, rgba: str) -> None:
    for visual in link.findall("visual"):
        material = visual.find("material")
        if material is None:
            material = _sub(visual, "material")
        material.set("name", name)
        for child in list(material):
            material.remove(child)
        _sub(material, "color", rgba=rgba)


def make_sim_description(authoritative_urdf: Path, controllers_yaml: Path) -> str:
    """Return a simulation overlay without changing the authoritative file."""
    tree = ET.parse(authoritative_urdf)
    robot = tree.getroot()

    asset_root = authoritative_urdf.resolve().parent.parent
    for mesh in robot.findall(".//mesh"):
        path = (authoritative_urdf.parent / mesh.get("filename", "")).resolve()
        if not path.is_relative_to(asset_root) or not path.is_file():
            raise ValueError(f"Missing or external model mesh: {path}")
        mesh.set("filename", path.as_uri())

    existing_links = {link.get("name"): link for link in robot.findall("link")}
    if "world" in existing_links or robot.find("./ros2_control") is not None:
        raise ValueError("authoritative URDF already contains simulation-only elements")

    missing_black_links = SIM_BLACK_VISUAL_LINKS - existing_links.keys()
    if missing_black_links:
        raise ValueError(f"black visual link(s) missing: {sorted(missing_black_links)}")
    for name in SIM_BLACK_VISUAL_LINKS:
        _set_visual_material(existing_links[name], "sim_matte_black", "0.012 0.015 0.020 1")

    world = ET.Element("link", {"name": "world"})
    robot.insert(0, world)
    world_joint = ET.Element("joint", {"name": "world_to_root", "type": "fixed"})
    _sub(world_joint, "parent", link="world")
    _sub(world_joint, "child", link="root")
    robot.insert(1, world_joint)

    for name in ("root", "root_x_link", "root_y_link", "base_link"):
        _add_tiny_inertial(existing_links[name])

    control = _sub(robot, "ros2_control", name="AlohaMiniGazeboSystem", type="system")
    hardware = _sub(control, "hardware")
    _sub(hardware, "plugin").text = "gz_ros2_control/GazeboSimSystem"

    for name in ("root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"):
        _add_joint_interface(control, name, "position")
    _add_joint_interface(control, "vertical_move", "position")
    for name in ARM_JOINTS:
        _add_joint_interface(control, name, "position")
    for name in ("wheel1_joint", "wheel2_joint", "wheel3_joint"):
        _add_joint_interface(control, name, "velocity")

    # The first deterministic task demo uses ideal position actuators. Keep
    # gravity on the grasped object and world, but do not make unloaded robot
    # joints sag while there is no gravity-compensation controller yet.
    for link in robot.findall("link"):
        if link.get("name") == "world":
            continue
        reference = _sub(robot, "gazebo", reference=link.get("name"))
        _sub(reference, "gravity").text = "false"

    gazebo = _sub(robot, "gazebo")
    plugin = _sub(
        gazebo,
        "plugin",
        filename="libgz_ros2_control-system.so",
        name="gz_ros2_control::GazeboSimROS2ControlPlugin",
    )
    _sub(plugin, "robot_param").text = "robot_description"
    _sub(plugin, "robot_param_node").text = "alohamini_gazebo_robot_state_publisher"
    _sub(plugin, "parameters").text = str(controllers_yaml)
    ros = _sub(plugin, "ros")
    _sub(ros, "namespace").text = "/alohamini_sim"
    _sub(ros, "remapping").text = "/clock:=/alohamini_sim/clock"
    _sub(ros, "remapping").text = "/tf:=/alohamini_sim/tf"
    _sub(ros, "remapping").text = "/tf_static:=/alohamini_sim/tf_static"

    grasp_plugin = _sub(
        gazebo,
        "plugin",
        filename="ignition-gazebo-detachable-joint-system",
        name="ignition::gazebo::systems::DetachableJoint",
    )
    _sub(grasp_plugin, "parent_link").text = "right_Fixed_Jaw"
    _sub(grasp_plugin, "child_model").text = "demo_object"
    _sub(grasp_plugin, "child_link").text = "object"
    _sub(grasp_plugin, "detach_topic").text = "/demo_object/detach"
    _sub(grasp_plugin, "attach_topic").text = "/demo_object/attach"
    _sub(grasp_plugin, "output_topic").text = "/demo_object/attached"

    ET.indent(tree, space="  ")
    return ET.tostring(robot, encoding="unicode", xml_declaration=True)
