"""Launch the first AlohaMini Gazebo pick-and-place acceptance scene."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _shutdown_after_demo(event, _context):
    if event.returncode != 0:
        raise RuntimeError(f"lift pick-and-place demo failed with exit code {event.returncode}")
    return [EmitEvent(event=Shutdown(reason="lift pick-and-place demo accepted"))]


def generate_launch_description() -> LaunchDescription:
    package = Path(get_package_share_directory("alohamini_gazebo"))
    demo = Node(
        package="alohamini_gazebo",
        executable="lift_pick_place_demo",
        namespace="/alohamini_sim",
        remappings=[("/clock", "/alohamini_sim/clock")],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="false"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(package / "launch" / "simulation.launch.py")),
                launch_arguments={"headless": LaunchConfiguration("headless")}.items(),
            ),
            TimerAction(
                period=12.0,
                actions=[demo],
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=demo,
                    on_exit=_shutdown_after_demo,
                )
            ),
        ]
    )
