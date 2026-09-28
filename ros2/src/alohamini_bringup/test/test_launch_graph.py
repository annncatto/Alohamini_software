"""Expand the real launch graph without starting ROS, HID, or Host processes."""

import importlib.util
from pathlib import Path

import pytest
import yaml
from launch import LaunchContext, LaunchDescription
from launch.actions import ExecuteProcess
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters

SOURCE = Path(__file__).resolve().parents[2]


def launch_graph(package, filename, *, details=None, **arguments):
    path = SOURCE / package / "launch" / filename
    spec = importlib.util.spec_from_file_location("hardware_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations.update(arguments)
    nodes, processes = [], []

    def text(value):
        return perform_substitutions(context, normalize_to_list_of_substitutions(value))

    def walk(entity):
        condition = getattr(entity, "condition", None)
        if condition is not None and not condition.evaluate(context):
            return
        if isinstance(entity, Node):
            parameters = {}
            for item in evaluate_parameters(context, entity._Node__parameters or []):
                if isinstance(item, Path):
                    for config in yaml.safe_load(item.read_text()).values():
                        parameters.update(config.get("ros__parameters", {}))
                else:
                    parameters.update(item)
            nodes.append((text(entity.node_package), text(entity.node_executable), parameters))
            if details is not None:
                details.append(
                    dict(
                        package=text(entity.node_package),
                        namespace=text(entity._Node__node_namespace or ""),
                        remappings={text(k): text(v) for k, v in entity._Node__remappings or []},
                        arguments=[text(arg) for arg in entity._Node__arguments or []],
                    )
                )
        elif isinstance(entity, ExecuteProcess):
            processes.append([text(arg) for arg in entity.cmd])
        elif isinstance(entity, LaunchDescription):
            for child in entity.entities:
                walk(child)
        else:
            for child in entity.execute(context) or []:
                walk(child)

    walk(module.generate_launch_description())
    return nodes, processes


def hardware_graph(**arguments):
    return launch_graph("alohamini_bringup", "hardware.launch.py", **arguments)


@pytest.mark.parametrize("cameras", ["false", "true"])
@pytest.mark.parametrize("moveit", ["false", "true"])
@pytest.mark.parametrize("joycon", ["false", "true"])
@pytest.mark.parametrize("rviz", ["false", "true"])
def test_hardware_mode_matrix(cameras, moveit, joycon, rviz):
    nodes, processes = hardware_graph(
        enable_cameras=cameras,
        enable_moveit=moveit,
        enable_joycon=joycon,
        use_rviz=rviz,
        start_native_reader="false",
    )
    packages = [package for package, _, _ in nodes]
    assert packages.count("robot_state_publisher") == 1
    assert packages.count("alohamini_bridge") == 1
    assert packages.count("alohamini_camera") == int(cameras == "true")
    assert packages.count("moveit_ros_move_group") == int(moveit == "true")
    assert packages.count("alohamini_joycon_teleop") == int(joycon == "true")
    assert packages.count("rviz2") == int(rviz == "true" and "true" in (joycon, moveit))
    assert not processes
    if joycon == "true":
        params = next(p for package, _, p in nodes if package == "alohamini_joycon_teleop")
        assert params["hardware_mode"] is True
        assert params["input_endpoint"] == "tcp://127.0.0.1:5567"
        assert params["measured_joint_states_topic"] == (
            "/alohamini_lerobot_bridge/measured_joint_states"
        )


def test_hardware_defaults_match_original_modes():
    nodes, processes = hardware_graph()
    assert [package for package, _, _ in nodes] == [
        "robot_state_publisher",
        "alohamini_bridge",
        "alohamini_camera",
    ]
    assert not processes


def test_hardware_forwards_bridge_camera_and_extrinsics_parameters():
    nodes, _ = hardware_graph(
        host="192.0.2.10",
        arm_mapping_dir="/tmp/robot-profile",
        observation_port="6556",
        command_port="6555",
        state_timestamp_mode="host_wall",
        max_state_response_age_sec="0.4",
        camera_stream_port="6557",
        camera_publish_raw="false",
        camera_timestamp_mode="host_wall",
        enable_camera_extrinsics="true",
        camera_extrinsics="forward.yaml,wrist_right.yaml",
        allow_candidate_camera_extrinsics="true",
    )
    by_executable = {executable: params for _, executable, params in nodes}
    bridge = by_executable["bridge_node"]
    assert {
        name: bridge[name]
        for name in (
            "host",
            "arm_mapping_dir",
            "observation_port",
            "command_port",
            "state_timestamp_mode",
            "max_state_response_age_sec",
        )
    } == dict(
        host="192.0.2.10",
        arm_mapping_dir="/tmp/robot-profile",
        observation_port=6556,
        command_port=6555,
        state_timestamp_mode="host_wall",
        max_state_response_age_sec=0.4,
    )
    camera = by_executable["camera_node"]
    assert camera["host"] == "192.0.2.10"
    assert camera["port"] == 6557
    assert camera["publish_raw"] is False
    assert camera["timestamp_mode"] == "host_wall"
    extrinsics = by_executable["extrinsics_node"]
    assert extrinsics["extrinsics_csv"] == "forward.yaml,wrist_right.yaml"
    assert extrinsics["allow_candidate"] is True
    assert camera["calibration_dir"] == extrinsics["calibration_dir"]


@pytest.mark.parametrize("cameras", ["false", "true"])
def test_bridge_runtime_restores_description_bridge_and_optional_cameras(cameras):
    nodes, processes = launch_graph(
        "alohamini_bridge",
        "runtime.launch.py",
        host="192.0.2.11",
        enable_cameras=cameras,
        camera_stream_port="6557",
        camera_publish_raw="false",
    )
    assert not processes
    packages = [package for package, _, _ in nodes]
    assert packages == ["robot_state_publisher", "alohamini_bridge"] + (
        ["alohamini_camera"] if cameras == "true" else []
    )
    bridge = next(p for _, executable, p in nodes if executable == "bridge_node")
    assert bridge["host"] == "192.0.2.11"
    assert bridge["request_window"] == 3
    assert bridge["request_timeout_sec"] == 1.0
    if cameras == "true":
        camera = next(p for _, executable, p in nodes if executable == "camera_node")
        assert camera["host"] == "192.0.2.11"
        assert camera["port"] == 6557
        assert camera["publish_raw"] is False


@pytest.mark.parametrize("python", ["", "/opt/robot env/bin/python"])
def test_reader_default_environment_and_explicit_python(python):
    _, processes = hardware_graph(enable_joycon="true", native_python=python)
    assert len(processes) == 1
    command = processes[0]
    if python:
        assert command[0] == python
        assert len(command) == 4
    else:
        assert command[1:6] == ["run", "--no-capture-output", "-n", "alohamini", "python"]
    assert command[-3].endswith("/scripts/joycon_native_reader.py")
    assert command[-2:] == ["--endpoint", "tcp://127.0.0.1:5567"]


def test_bringup_declares_all_component_dependencies():
    import xml.etree.ElementTree as ET

    dependencies = {
        node.text
        for node in ET.parse(SOURCE / "alohamini_bringup/package.xml").findall("exec_depend")
    }
    assert {
        "alohamini_description",
        "alohamini_bridge",
        "alohamini_camera",
        "alohamini_moveit_config",
        "alohamini_joycon_teleop",
    } <= dependencies


@pytest.mark.parametrize("preview", ["false", "true"])
@pytest.mark.parametrize("rviz", ["false", "true"])
def test_plan_only_modes_keep_execution_and_topics_isolated(preview, rviz):
    details = []
    nodes, processes = launch_graph(
        "alohamini_moveit_config",
        "plan_only.launch.py",
        details=details,
        joycon_preview=preview,
        use_rviz=rviz,
    )
    packages = [package for package, _, _ in nodes]
    assert not processes
    assert "alohamini_bridge" not in packages
    assert packages.count("robot_state_publisher") == 1
    assert packages.count("joint_state_publisher") == int(preview == "false")
    assert packages.count("moveit_ros_move_group") == 1
    assert packages.count("rviz2") == int(rviz == "true")
    for package, _, params in nodes:
        if package == "moveit_ros_move_group":
            assert params["allow_trajectory_execution"] is False
            assert params["planning_scene_monitor_options.joint_state_topic"] == (
                "/alohamini_plan_only/joint_states"
            )
        if package == "joint_state_publisher":
            assert params["zeros.left_wrist_flex"] == 1.435806017460960
            assert params["zeros.right_wrist_flex"] == 1.5
            assert params["zeros.left_gripper"] == params["zeros.right_gripper"] == 0.32
    for node in details:
        if node["package"] in ("robot_state_publisher", "moveit_ros_move_group", "rviz2"):
            assert node["remappings"]["/tf"] == "/alohamini_plan_only/tf"
            assert node["remappings"]["/tf_static"] == "/alohamini_plan_only/tf_static"
        if node["package"] == "moveit_ros_move_group":
            assert node["namespace"] == "alohamini_plan_only"
        if node["package"] == "rviz2":
            config = "joycon_preview.rviz" if preview == "true" else "plan_only.rviz"
            assert node["arguments"][-1].endswith("/config/" + config)


def test_joycon_preview_has_moveit_without_a_second_state_publisher():
    nodes, processes = launch_graph(
        "alohamini_joycon_teleop",
        "preview.launch.py",
        use_rviz="false",
        native_python="/opt/robot env/bin/python",
    )
    packages = [package for package, _, _ in nodes]
    assert sorted(packages) == sorted(
        [
            "robot_state_publisher",
            "moveit_ros_move_group",
            "alohamini_joycon_teleop",
        ]
    )
    assert len(processes) == 1
    assert processes[0][0] == "/opt/robot env/bin/python"
    assert processes[0][-1] == "tcp://127.0.0.1:5568"
    teleop = next(params for package, _, params in nodes if package == "alohamini_joycon_teleop")
    assert teleop["hardware_mode"] is False
    assert teleop["input_endpoint"] == processes[0][-1]


@pytest.mark.parametrize("joycon", [False, True])
def test_old_hardware_aliases_remain_components_only(joycon):
    package = "alohamini_joycon_teleop" if joycon else "alohamini_moveit_config"
    filename = "hardware.launch.py" if joycon else "hardware_execution.launch.py"
    nodes, processes = launch_graph(
        package,
        filename,
        host="192.0.2.10",
        arm_mapping_dir="/unused",
        use_rviz="false",
        joycon_rviz="false",
        start_native_reader="false",
    )
    assert not processes
    assert [p for p, _, _ in nodes] == [
        "alohamini_joycon_teleop" if joycon else "moveit_ros_move_group"
    ]


def test_joycon_preview_rviz_uses_only_offline_topics():
    path = SOURCE / "alohamini_moveit_config/config/joycon_preview.rviz"
    config = yaml.safe_load(path.read_text())
    displays = {d["Class"]: d for d in config["Visualization Manager"]["Displays"]}
    assert displays["rviz_default_plugins/RobotModel"]["Description Topic"]["Value"] == (
        "/alohamini_plan_only/robot_description"
    )
    assert displays["rviz_default_plugins/MarkerArray"]["Topic"]["Value"] == (
        "/alohamini_plan_only/alohamini/joycon_tcp_markers"
    )
