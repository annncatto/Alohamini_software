"""Deprecated component alias; use alohamini_bringup for complete hardware."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    package = Path(get_package_share_directory("alohamini_joycon_teleop"))
    return LaunchDescription(
        [
            # Accepted for command-line compatibility only. Hardware transport
            # now belongs to alohamini_bringup.
            DeclareLaunchArgument("host", default_value="127.0.0.1"),
            DeclareLaunchArgument("arm_mapping_dir", default_value=""),
            DeclareLaunchArgument("joycon_rviz", default_value="true"),
            DeclareLaunchArgument("start_native_reader", default_value="true"),
            DeclareLaunchArgument(
                "native_python",
                default_value="",
            ),
            LogInfo(
                msg=(
                    "alohamini_joycon_teleop hardware.launch.py is now a "
                    "Joy-Con-only component; use alohamini_bringup "
                    "hardware.launch.py for physical hardware"
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(package / "launch" / "teleop.launch.py")),
                launch_arguments={
                    "use_rviz": LaunchConfiguration("joycon_rviz"),
                    "start_native_reader": LaunchConfiguration("start_native_reader"),
                    "native_python": LaunchConfiguration("native_python"),
                }.items(),
            ),
        ]
    )
