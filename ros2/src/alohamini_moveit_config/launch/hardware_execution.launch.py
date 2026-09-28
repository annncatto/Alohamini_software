"""Deprecated component alias; use alohamini_bringup for complete hardware."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    package = Path(get_package_share_directory("alohamini_moveit_config"))
    return LaunchDescription(
        [
            # Accepted for command-line compatibility only. Hardware transport
            # now belongs to alohamini_bringup.
            DeclareLaunchArgument("host", default_value="127.0.0.1"),
            DeclareLaunchArgument("arm_mapping_dir", default_value=""),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            LogInfo(
                msg=(
                    "hardware_execution.launch.py is now a MoveIt-only component; "
                    "use alohamini_bringup hardware.launch.py for physical hardware"
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(package / "launch" / "move_group.launch.py")),
                launch_arguments={
                    "use_rviz": LaunchConfiguration("use_rviz"),
                }.items(),
            ),
        ]
    )
