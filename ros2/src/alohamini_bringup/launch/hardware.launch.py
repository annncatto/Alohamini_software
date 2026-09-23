"""Compose one AlohaMini hardware graph without duplicate resource owners."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from alohamini.paths import WorkspacePaths


def _launch(package: str, filename: str) -> PythonLaunchDescriptionSource:
    share = Path(get_package_share_directory(package))
    return PythonLaunchDescriptionSource(str(share / "launch" / filename))


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("host", default_value="127.0.0.1"),
            DeclareLaunchArgument(
                "arm_mapping_dir", default_value=str(WorkspacePaths().calibration / "hardware")
            ),
            DeclareLaunchArgument("observation_port", default_value="5556"),
            DeclareLaunchArgument("command_port", default_value="5555"),
            DeclareLaunchArgument("state_timestamp_mode", default_value="receipt"),
            DeclareLaunchArgument("max_state_response_age_sec", default_value="0.25"),
            DeclareLaunchArgument("enable_cameras", default_value="true"),
            DeclareLaunchArgument("enable_moveit", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("camera_stream_port", default_value="5557"),
            DeclareLaunchArgument("camera_publish_raw", default_value="true"),
            DeclareLaunchArgument("camera_timestamp_mode", default_value="receipt"),
            DeclareLaunchArgument("enable_camera_extrinsics", default_value="false"),
            DeclareLaunchArgument("camera_extrinsics", default_value=""),
            DeclareLaunchArgument("allow_candidate_camera_extrinsics", default_value="false"),
            # Exactly one authoritative description and robot_state_publisher.
            IncludeLaunchDescription(_launch("alohamini_description", "description.launch.py")),
            # Commands remain disabled until an explicit ROS service request.
            IncludeLaunchDescription(
                _launch("alohamini_bridge", "bridge.launch.py"),
                launch_arguments={
                    "host": LaunchConfiguration("host"),
                    "observation_port": LaunchConfiguration("observation_port"),
                    "command_port": LaunchConfiguration("command_port"),
                    "arm_mapping_dir": LaunchConfiguration("arm_mapping_dir"),
                    "state_timestamp_mode": LaunchConfiguration("state_timestamp_mode"),
                    "max_state_response_age_sec": LaunchConfiguration("max_state_response_age_sec"),
                }.items(),
            ),
            IncludeLaunchDescription(
                _launch("alohamini_camera", "camera.launch.py"),
                condition=IfCondition(LaunchConfiguration("enable_cameras")),
                launch_arguments={
                    "host": LaunchConfiguration("host"),
                    "port": LaunchConfiguration("camera_stream_port"),
                    "publish_raw": LaunchConfiguration("camera_publish_raw"),
                    "timestamp_mode": LaunchConfiguration("camera_timestamp_mode"),
                    "enable_extrinsics": LaunchConfiguration("enable_camera_extrinsics"),
                    "extrinsics": LaunchConfiguration("camera_extrinsics"),
                    "allow_candidate_extrinsics": LaunchConfiguration(
                        "allow_candidate_camera_extrinsics"
                    ),
                }.items(),
            ),
            IncludeLaunchDescription(
                _launch("alohamini_moveit_config", "move_group.launch.py"),
                condition=IfCondition(LaunchConfiguration("enable_moveit")),
                launch_arguments={"use_rviz": LaunchConfiguration("use_rviz")}.items(),
            ),
        ]
    )
