from launch import LaunchDescription
from launch_ros.actions import Node

from alohamini.model import get_robot_model


def robot_description():
    return get_robot_model("alohamini2pro").description_xml("collision")


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="alohamini_robot_state_publisher",
                parameters=[
                    {
                        "robot_description": robot_description(),
                        "use_sim_time": False,
                    }
                ],
                remappings=[
                    ("robot_description", "/alohamini/robot_description"),
                ],
                output="screen",
            ),
        ]
    )
