"""AlohaMini2Pro MoveIt launch that cannot execute hardware trajectories."""

import xml.etree.ElementTree as ET
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

from alohamini.model import get_robot_model


def generate_launch_description() -> LaunchDescription:
    model = get_robot_model("alohamini2pro")
    package = Path(get_package_share_directory("alohamini_moveit_config"))
    moveit_config = (
        MoveItConfigsBuilder("alohamini2pro", package_name="alohamini_moveit_config")
        .robot_description(file_path=str(model.description_path("collision")))
        .robot_description_semantic(file_path=str(model.description_path("semantic")))
        .robot_description_kinematics()
        .joint_limits()
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .trajectory_execution(
            file_path="config/moveit_controllers.yaml", moveit_manage_controllers=False
        )
        .planning_scene_monitor(
            publish_planning_scene=True,
            publish_geometry_updates=True,
            publish_state_updates=True,
            publish_transforms_updates=True,
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )
    moveit_config.robot_description = {"robot_description": model.description_xml("collision")}

    # Offline display poses are model data, never commands or installed-device homing.
    semantic = ET.fromstring(moveit_config.robot_description_semantic["robot_description_semantic"])
    home = {
        f"zeros.{joint.attrib['name']}": float(joint.attrib["value"])
        for group in semantic.findall("group_state")
        if group.attrib["name"] in ("home", "closed")
        for joint in group.findall("joint")
    }
    move_group_parameters = [
        moveit_config.to_dict(),
        {
            "allow_trajectory_execution": False,
            "publish_robot_description_semantic": True,
            "monitor_dynamics": False,
            # Never let an offline fake state race the real bridge on
            # /joint_states when both graphs are visible on the same domain.
            "planning_scene_monitor_options.joint_state_topic": "/alohamini_plan_only/joint_states",
        },
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument(
                "joycon_preview",
                default_value="false",
                description="Use Joy-Con joint states instead of the fixed Home publisher.",
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                name="alohamini_fake_joint_state_publisher",
                condition=UnlessCondition(LaunchConfiguration("joycon_preview")),
                parameters=[moveit_config.robot_description, home],
                remappings=[
                    ("joint_states", "/alohamini_plan_only/joint_states"),
                    ("robot_description", "/alohamini_plan_only/robot_description"),
                ],
                output="screen",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="alohamini_plan_only_robot_state_publisher",
                parameters=[moveit_config.robot_description],
                remappings=[
                    ("joint_states", "/alohamini_plan_only/joint_states"),
                    ("robot_description", "/alohamini_plan_only/robot_description"),
                    ("/tf", "/alohamini_plan_only/tf"),
                    ("/tf_static", "/alohamini_plan_only/tf_static"),
                ],
                output="screen",
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                namespace="alohamini_plan_only",
                output="screen",
                parameters=move_group_parameters,
                remappings=[
                    ("/joint_states", "/alohamini_plan_only/joint_states"),
                    ("/tf", "/alohamini_plan_only/tf"),
                    ("/tf_static", "/alohamini_plan_only/tf_static"),
                ],
            ),
            GroupAction(
                condition=IfCondition(LaunchConfiguration("use_rviz")),
                actions=[
                    Node(
                        package="rviz2",
                        executable="rviz2",
                        name="moveit_rviz",
                        condition=UnlessCondition(LaunchConfiguration("joycon_preview")),
                        namespace="alohamini_plan_only",
                        output="log",
                        arguments=["-d", str(package / "config/plan_only.rviz")],
                        parameters=[
                            moveit_config.robot_description,
                            moveit_config.robot_description_semantic,
                            moveit_config.robot_description_kinematics,
                            moveit_config.planning_pipelines,
                            moveit_config.joint_limits,
                        ],
                        remappings=[
                            ("/tf", "/alohamini_plan_only/tf"),
                            ("/tf_static", "/alohamini_plan_only/tf_static"),
                        ],
                    ),
                    Node(
                        package="rviz2",
                        executable="rviz2",
                        name="joycon_preview_rviz",
                        namespace="alohamini_plan_only",
                        output="log",
                        arguments=["-d", str(package / "config/joycon_preview.rviz")],
                        parameters=[moveit_config.robot_description],
                        remappings=[
                            ("/tf", "/alohamini_plan_only/tf"),
                            ("/tf_static", "/alohamini_plan_only/tf_static"),
                        ],
                        condition=IfCondition(LaunchConfiguration("joycon_preview")),
                    ),
                ],
            ),
        ]
    )
