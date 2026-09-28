from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bridge_share = Path(get_package_share_directory("alohamini_bridge"))
    camera_share = Path(get_package_share_directory("alohamini_camera"))
    description_share = Path(get_package_share_directory("alohamini_description"))
    return LaunchDescription(
        [
            DeclareLaunchArgument("host", default_value="127.0.0.1"),
            DeclareLaunchArgument("arm_mapping_dir", default_value=""),
            DeclareLaunchArgument("enable_cameras", default_value="true"),
            DeclareLaunchArgument("camera_stream_port", default_value="5557"),
            DeclareLaunchArgument("camera_publish_raw", default_value="true"),
            DeclareLaunchArgument("enable_camera_extrinsics", default_value="false"),
            DeclareLaunchArgument("camera_extrinsics", default_value=""),
            DeclareLaunchArgument("allow_candidate_camera_extrinsics", default_value="false"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(description_share / "launch" / "description.launch.py")
                )
            ),
            Node(
                package="alohamini_bridge",
                executable="bridge_node",
                name="alohamini_lerobot_bridge",
                output="screen",
                parameters=[
                    str(bridge_share / "config" / "bridge.yaml"),
                    {
                        "host": LaunchConfiguration("host"),
                        "arm_mapping_dir": LaunchConfiguration("arm_mapping_dir"),
                    },
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(camera_share / "launch" / "camera.launch.py")),
                condition=IfCondition(LaunchConfiguration("enable_cameras")),
                launch_arguments={
                    "host": LaunchConfiguration("host"),
                    "port": LaunchConfiguration("camera_stream_port"),
                    "publish_raw": LaunchConfiguration("camera_publish_raw"),
                    "enable_extrinsics": LaunchConfiguration("enable_camera_extrinsics"),
                    "extrinsics": LaunchConfiguration("camera_extrinsics"),
                    "allow_candidate_extrinsics": LaunchConfiguration(
                        "allow_candidate_camera_extrinsics"
                    ),
                }.items(),
            ),
        ]
    )
