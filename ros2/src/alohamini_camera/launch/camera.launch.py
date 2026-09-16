from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from alohamini.paths import WorkspacePaths


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("alohamini_camera"))
    return LaunchDescription(
        [
            DeclareLaunchArgument("host", default_value="127.0.0.1"),
            DeclareLaunchArgument("port", default_value="5557"),
            DeclareLaunchArgument(
                "cameras",
                default_value="[auto]",
                description="Follow Host streams or select camera names",
            ),
            DeclareLaunchArgument(
                "calibration_dir", default_value=str(WorkspacePaths().calibration / "cameras")
            ),
            DeclareLaunchArgument("publish_raw", default_value="true"),
            DeclareLaunchArgument("timestamp_mode", default_value="receipt"),
            DeclareLaunchArgument("enable_extrinsics", default_value="false"),
            DeclareLaunchArgument("extrinsics", default_value=""),
            DeclareLaunchArgument("allow_candidate_extrinsics", default_value="false"),
            Node(
                package="alohamini_camera",
                executable="camera_node",
                name="alohamini_camera",
                output="screen",
                parameters=[
                    str(share / "config/camera.yaml"),
                    {
                        "host": LaunchConfiguration("host"),
                        "port": ParameterValue(LaunchConfiguration("port"), value_type=int),
                        "camera_names": LaunchConfiguration("cameras"),
                        "calibration_dir": LaunchConfiguration("calibration_dir"),
                        "publish_raw": ParameterValue(
                            LaunchConfiguration("publish_raw"), value_type=bool
                        ),
                        "timestamp_mode": LaunchConfiguration("timestamp_mode"),
                    },
                ],
            ),
            Node(
                package="alohamini_camera",
                executable="extrinsics_node",
                name="alohamini_camera_extrinsics",
                output="screen",
                condition=IfCondition(LaunchConfiguration("enable_extrinsics")),
                parameters=[
                    {
                        "extrinsics_csv": LaunchConfiguration("extrinsics"),
                        "calibration_dir": LaunchConfiguration("calibration_dir"),
                        "allow_candidate": ParameterValue(
                            LaunchConfiguration("allow_candidate_extrinsics"), value_type=bool
                        ),
                    }
                ],
            ),
        ]
    )
