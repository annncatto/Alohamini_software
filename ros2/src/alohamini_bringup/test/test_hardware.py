import importlib.util
import json
import math
import os
import signal
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import rclpy
import yaml
import zmq
from alohamini_bridge.bridge_node import AlohaMiniBridge, StateReceiver, load_mapper
from alohamini_bridge.commands import RobotCommands
from alohamini_bridge.mapping import ARM_JOINTS
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from tf2_msgs.msg import TFMessage

from alohamini.client import HostClient
from alohamini.errors import ResponseTimeoutError
from alohamini.model import get_robot_model
from alohamini.protocol import HostSnapshot
from alohamini.schema import BodyVelocity, CommandIdentity


def write_mapping(directory):
    model = get_robot_model("alohamini2pro")
    for side in ("left", "right"):
        document = dict(
            schema_version=1,
            robot_model=model.model_id,
            side=side,
            ticks_per_revolution=4096,
            joints={},
        )
        for motor in model.actuators:
            if motor.name.startswith(f"arm_{side}_"):
                document["joints"][motor.name.removeprefix(f"arm_{side}_")] = dict(
                    id=motor.motor_id,
                    model=motor.motor_model,
                    reference_tick=2048,
                    reference_q_rad=0.0,
                    sign=1,
                    safe_q_min_rad=-1.0,
                    safe_q_max_rad=1.0,
                )
        (directory / f"hardware_joint_map_{side}.yaml").write_text(yaml.safe_dump(document))
    lift = dict(
        schema_version=1,
        robot_model=model.model_id,
        mechanism=dict(physical_min_mm=0.0, physical_max_mm=600.0),
        urdf=dict(joint="vertical_move", q_at_physical_min_m=-0.3, q_at_physical_max_m=0.3),
    )
    (directory / "lift_axis.yaml").write_text(yaml.safe_dump(lift))


def observation(session="host-a", sample=10.0):
    motors = {}
    payload = {}
    for side in ("left", "right"):
        for joint in ARM_JOINTS:
            name = f"arm_{side}_{joint}"
            motors[name] = dict(range_min=0, range_max=4095, drive_mode=0, normalization="degrees")
            payload[f"{name}.pos"] = 0.5 * 360 / 4095
    payload.update(
        {
            "_robot_metadata": {
                "schema_version": 1,
                "robot_model": "alohamini2pro",
                "motors": motors,
                "lift_axis": {"soft_min_mm": 0.0, "soft_max_mm": 600.0},
            },
            "_images": [],
            "x.vel": 0.1,
            "y.vel": -0.02,
            "theta.vel": 90.0,
            "lift_axis.height_mm": 300.0,
            "_safety": {
                "version": 1,
                "feedback_valid": True,
                "host_session_id": session,
                "sampled_at_monotonic_s": sample + 0.002,
                "phase": "ready",
                "fault": None,
                "control_owner": None,
                "control_epoch": 0,
                "joint_hold_events": 0,
                "watchdog_events": 0,
                "command_watchdog_timeout_s": 1.0,
                "joint_holds": {},
                "lift_reference_valid": True,
                "command": {},
            },
            "_host_timing": {
                "state_sample_started_monotonic_s": sample,
                "state_sample_finished_monotonic_s": sample + 0.001,
                "state_sample_unix_ns": time.time_ns(),
            },
        }
    )
    now = time.monotonic()
    return HostSnapshot(payload, {}, now, now)


@pytest.fixture
def ros(monkeypatch, tmp_path):
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_DOMAIN_ID", "83")
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros_logs"))
    rclpy.init(args=[])
    yield
    rclpy.shutdown()


@pytest.fixture
def bridge(ros, tmp_path):
    write_mapping(tmp_path)
    with patch("alohamini_bridge.bridge_node.StateReceiver") as receiver:
        receiver.return_value.take.return_value = None
        node = AlohaMiniBridge(
            parameter_overrides=[Parameter("arm_mapping_dir", value=str(tmp_path))]
        )
    for name in (
        "joint_pub",
        "measured_joint_pub",
        "derived_wheel_pub",
        "base_velocity_pub",
        "raw_pub",
    ):
        setattr(node, name, Mock())
    yield node
    node.destroy_node()


def test_mapping_and_units_match_existing_host_and_ros_names(bridge):
    bridge.handle_observation(observation())
    joints = bridge.measured_joint_pub.publish.call_args.args[0]
    assert len(joints.name) == 15
    assert "left_wrist_yaw_joint" in joints.name
    assert "right_gripper" in joints.name
    assert joints.position == pytest.approx([0.0] * 15)
    assert not joints.effort  # Current is not joint torque.
    velocity = bridge.base_velocity_pub.publish.call_args.args[0]
    assert velocity.twist.angular.z == pytest.approx(math.pi / 2)
    assert velocity.twist.linear.x == 0.1
    assert velocity.header.stamp == joints.header.stamp
    assert velocity.header.frame_id == "base_link"
    derived = bridge.derived_wheel_pub.publish.call_args.args[0]
    assert list(derived.position[:3]) == [0.0] * 3
    assert derived.header.stamp == joints.header.stamp


def test_repeated_host_sample_does_not_republish_or_refresh_feedback(bridge):
    bridge.handle_observation(observation())
    receipt = bridge.last_observation_monotonic
    bridge.handle_observation(observation())
    assert bridge.measured_joint_pub.publish.call_count == 1
    assert bridge.last_observation_monotonic == receipt


@pytest.mark.parametrize("sample", [10.0, 9.0])
def test_reconnect_preserves_same_session_sample_watermark(bridge, sample):
    bridge.handle_observation(observation())
    bridge.reject("Host connection lost")
    bridge.receiver.take.return_value = observation(sample=sample), 1, ""
    bridge.on_timer()
    assert bridge.measured_joint_pub.publish.call_count == 1
    assert bridge.last_observation_monotonic is None
    bridge.handle_observation(observation(sample=10.02), generation=1)
    assert bridge.measured_joint_pub.publish.call_count == 2
    assert bridge.wheel_positions == [0.0] * 3


@pytest.mark.parametrize("host_time", [9.0, 11.0])
def test_recent_response_cannot_refresh_invalid_or_old_host_sample(bridge, host_time):
    snapshot = observation()
    snapshot.payload["_safety"]["sampled_at_monotonic_s"] = host_time
    bridge.receiver.take.return_value = snapshot, 0, ""
    bridge.on_timer()
    bridge.measured_joint_pub.publish.assert_not_called()
    assert bridge.last_observation_monotonic is None
    assert bridge.invalid_observations == 1


def test_restart_resets_encoder_continuity_and_wheel_integration(bridge):
    bridge.handle_observation(observation())
    bridge.handle_observation(observation(sample=10.02))
    assert any(bridge.wheel_positions)
    bridge.mapper.decoder._previous["left_wrist_roll"] = 2 * math.pi
    bridge.handle_observation(observation(session="host-b", sample=1.0))
    assert bridge.wheel_positions == [0.0] * 3
    joints = bridge.measured_joint_pub.publish.call_args.args[0]
    assert joints.position[joints.name.index("left_wrist_roll")] == pytest.approx(0.0)


def test_stale_replies_cannot_become_fresh_after_invalidation(bridge):
    bridge.handle_observation(observation())
    bridge.last_observation_monotonic -= 1
    bridge.handle_observation(observation())
    assert bridge.last_observation_monotonic is None
    bridge.handle_observation(observation())
    assert bridge.measured_joint_pub.publish.call_count == 1
    bridge.handle_observation(observation(sample=10.02))
    assert bridge.measured_joint_pub.publish.call_count == 2
    assert bridge.wheel_positions == [0.0] * 3


def test_sample_rollback_needs_a_new_host_session(bridge):
    bridge.handle_observation(observation())
    receipt = bridge.last_observation_monotonic
    bridge.receiver.take.return_value = observation(sample=1.0), 0, ""
    bridge.on_timer()
    assert bridge.last_observation_monotonic == receipt
    assert bridge.measured_joint_pub.publish.call_count == 1
    bridge.handle_observation(observation(session="host-b", sample=1.0))
    assert bridge.measured_joint_pub.publish.call_count == 2


@pytest.mark.parametrize("kind", ["duplicate", "rollback", "delayed", "aged"])
def test_single_discarded_feedback_preserves_publication_freshness(bridge, kind):
    bridge.handle_observation(observation())
    receipt = bridge.last_observation_monotonic
    bridge.commands = Mock()
    snapshot = observation(sample=10.02)
    if kind in ("duplicate", "rollback"):
        snapshot = observation(sample=10.0 if kind == "duplicate" else 9.0)
    elif kind == "delayed":
        snapshot.request_started_s -= 1
    else:
        snapshot.payload["_safety"]["sampled_at_monotonic_s"] += 1
    bridge.receiver.take.return_value = snapshot, 0, ""
    bridge.on_timer()
    bridge.commands.fail.assert_not_called()
    assert bridge.last_observation_monotonic == receipt
    assert bridge.measured_joint_pub.publish.call_count == 1
    bridge.last_observation_monotonic -= 1
    bridge.on_timer()
    bridge.commands.fail.assert_called_once()
    assert bridge.last_observation_monotonic is None


def test_request_timeout_does_not_immediately_invalidate_published_state(bridge):
    bridge.handle_observation(observation())
    bridge.commands = Mock()
    bridge.receiver.take.return_value = None, 0, ResponseTimeoutError("request timed out")
    bridge.on_timer()
    bridge.commands.fail.assert_not_called()
    bridge.last_observation_monotonic -= 1
    bridge.on_timer()
    bridge.commands.fail.assert_called_once()


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["_safety"].update(feedback_valid=False),
        lambda p: p["_safety"].update(version=2),
        lambda p: p["_safety"].update(host_session_id=""),
        lambda p: p.update({"x.vel": float("nan")}),
        lambda p: p.pop("arm_left_gripper.pos"),
        lambda p: p.update({"lift_axis.height_mm": None}),
        lambda p: p["_host_timing"].update(state_sample_finished_monotonic_s=0),
    ],
)
def test_invalid_observations_clear_freshness_and_never_publish(bridge, change):
    bridge.handle_observation(observation())
    snapshot = observation(sample=10.02)
    change(snapshot.payload)
    bridge.receiver.take.return_value = snapshot, 0, ""
    bridge.on_timer()
    assert bridge.measured_joint_pub.publish.call_count == 1
    assert bridge.last_observation_monotonic is None
    assert not bridge.mapper.decoder._previous


def test_delayed_response_and_calibration_changes_are_rejected(bridge):
    old = observation()
    old.request_started_s -= 1
    bridge.handle_observation(old)
    bridge.measured_joint_pub.publish.assert_not_called()
    assert bridge.last_observation_monotonic is None
    bridge.handle_observation(observation())
    changed = observation(sample=10.02)
    changed.payload["_robot_metadata"]["motors"]["arm_left_shoulder_pan"]["drive_mode"] = 1
    with pytest.raises(ValueError, match="calibration changed"):
        bridge.handle_observation(changed)


def test_lift_overrun_is_not_hidden_by_display_clamping(bridge):
    snapshot = observation()
    snapshot.payload["lift_axis.height_mm"] = 610.0
    bridge.handle_observation(snapshot)
    message = bridge.measured_joint_pub.publish.call_args.args[0]
    assert message.position[message.name.index("vertical_move")] == pytest.approx(0.31)


def test_explicit_host_wall_stamp_does_not_fall_back_to_receipt(bridge):
    bridge.state_timestamp_mode = "host_wall"
    snapshot = observation()
    snapshot.payload["_host_timing"]["state_sample_unix_ns"] = 1_200_000_003
    bridge.handle_observation(snapshot)
    stamp = bridge.measured_joint_pub.publish.call_args.args[0].header.stamp
    assert (stamp.sec, stamp.nanosec) == (1, 200_000_003)
    snapshot = observation(sample=10.02)
    snapshot.payload["_host_timing"].pop("state_sample_unix_ns")
    with pytest.raises(ValueError, match="host_wall"):
        bridge.handle_observation(snapshot)


def test_mapping_files_are_independent_and_validate_model(tmp_path):
    write_mapping(tmp_path)
    path = tmp_path / "hardware_joint_map_left.yaml"
    document = yaml.safe_load(path.read_text())
    document["side"] = "right"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="side"):
        load_mapper(tmp_path, get_robot_model("alohamini2pro"))


def test_description_uses_native_assets_without_changing_geometry():
    path = (
        Path(get_package_share_directory("alohamini_description")) / "launch/description.launch.py"
    )
    spec = importlib.util.spec_from_file_location("description_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = get_robot_model("alohamini2pro").description_path("collision")
    expected, actual = ET.parse(native).getroot(), ET.fromstring(module.robot_description())
    for original, rendered in zip(
        expected.findall(".//mesh"), actual.findall(".//mesh"), strict=True
    ):
        assert (
            rendered.attrib["filename"]
            == (native.parent / original.attrib["filename"]).resolve().as_uri()
        )
        rendered.attrib["filename"] = original.attrib["filename"]
    assert ET.tostring(actual) == ET.tostring(expected)


@pytest.fixture
def host():
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    port = router.bind_to_random_port("tcp://127.0.0.1")
    commands = context.socket(zmq.PULL)
    command_port = commands.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()
    state = {
        "respond": True,
        "count": 0,
        "tokens": [],
        "commands": [],
        "command_port": command_port,
        "safety": {},
        "ack": True,
        "follow": False,
        "targets": {},
    }

    def serve():
        while not stop.is_set():
            if commands.poll(0):
                command = commands.recv_json()
                state["commands"].append(command)
                if state["ack"]:
                    state["safety"].update(
                        command=command["_command"], control_owner=command["_command"]["client_id"]
                    )
                    if state["follow"]:
                        state["targets"].update(
                            {key: value for key, value in command.items() if key != "_command"}
                        )
            if not router.poll(10):
                continue
            address, token = router.recv_multipart()
            state["tokens"].append(token)
            if state["respond"]:
                state["count"] += 1
                payload = observation(sample=10 + state["count"] * 0.02).payload
                payload["_safety"].update(state["safety"])
                payload.update(state["targets"])
                router.send_multipart([address, token, json.dumps(payload).encode()])

    thread = threading.Thread(target=serve)
    thread.start()
    yield port, state
    stop.set()
    thread.join(2)
    assert not thread.is_alive()
    router.close(linger=0)
    commands.close(linger=0)
    context.term()


def wait_result(receiver, predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        result = receiver.take()
        if result is not None and predicate(result):
            return result
        time.sleep(0.005)
    pytest.fail("No matching receiver result")


def test_native_client_worker_is_read_only_bounded_and_recovers(host):
    port, state = host
    with patch.object(HostClient, "send_command", side_effect=AssertionError("read-only")):
        receiver = StateReceiver("127.0.0.1", port, "alohamini2pro", 0.05, 50)
        try:
            first = wait_result(receiver, lambda result: result[0] is not None)
            state["respond"] = False
            failed = wait_result(receiver, lambda result: bool(result[2]))
            assert failed[1] == first[1]  # A request timeout alone does not reset continuity.
            state["respond"] = True
            recovered = wait_result(receiver, lambda result: result[0] is not None)
            assert recovered[1] >= failed[1]
            assert all(token.endswith(b":state") for token in state["tokens"])
        finally:
            receiver.close()
        assert not receiver._thread.is_alive()


def test_worker_timeout_preserves_control_handshake_and_uses_feedback_expiry():
    receiver = StateReceiver.__new__(StateReceiver)
    receiver._stop = threading.Event()
    receiver._lock = threading.Lock()
    receiver._result = None
    receiver.commands = Mock()
    receiver.commands.status.return_value = True, False, "enabled"
    snapshot = observation()
    calls = 0

    def read():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ResponseTimeoutError("request timed out")
        receiver._stop.set()
        return snapshot

    with patch("alohamini_bridge.bridge_node.HostClient") as factory:
        client = factory.return_value.__enter__.return_value
        client.connect_control.return_value = snapshot
        client.read.side_effect = read
        receiver._run("localhost", 5556, "alohamini2pro", 0.05, 50, 5555)
        client.connect_control.assert_called_once()
        assert client.read.call_count == 2
    receiver.commands.discard_feedback.assert_called_once_with("request timed out")
    receiver.commands.fail.assert_not_called()
    assert receiver.take()[1] == 0


def test_hardware_launch_publishes_state_and_tf_without_commands(ros, host, tmp_path):
    port, state = host
    write_mapping(tmp_path)
    observer = Node("hardware_test_observer")
    names, frames = set(), set()
    observer.create_subscription(
        JointState, "/joint_states", lambda msg: names.update(msg.name), 100
    )
    observer.create_subscription(
        TFMessage, "/tf", lambda msg: frames.update(t.child_frame_id for t in msg.transforms), 100
    )
    process = subprocess.Popen(
        [
            "ros2",
            "launch",
            "alohamini_bringup",
            "hardware.launch.py",
            "host:=127.0.0.1",
            f"observation_port:={port}",
            f"arm_mapping_dir:={tmp_path}",
            "enable_cameras:=false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 8
        expected = {"left_wrist_yaw", "right_wrist_yaw", "base_link"}
        while time.monotonic() < deadline and process.poll() is None:
            rclpy.spin_once(observer, timeout_sec=0.05)
            if expected <= frames and len(names) == 21:
                break
        validation = subprocess.run(
            ["ros2", "run", "alohamini_validation", "validate_tf", "--ros-args", "-r", "__ns:=/"],
            capture_output=True,
            text=True,
            timeout=12,
        )
        assert validation.returncode == 0, validation.stdout + validation.stderr
        assert "[PASS] TF tree" in validation.stdout
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=2)
            pytest.fail(f"ROS launch did not stop: {output}")
        finally:
            observer.destroy_node()
    assert process.returncode == 0, output
    assert expected <= frames, output
    assert len(names) == 21, output
    assert "vertical_move" in names
    assert "left_gripper" in names and "right_gripper" in names
    assert state["tokens"] and all(token.endswith(b":state") for token in state["tokens"])
    assert not state["commands"]


@pytest.fixture
def command_channel():
    channel = RobotCommands(0.25)
    client = Mock(client_id="ros-test")
    sequence = 0

    def send(targets, *, based_on):
        nonlocal sequence
        sequence += 1
        safety = based_on.payload["_safety"]
        return CommandIdentity(
            client.client_id, sequence, safety["host_session_id"], safety["control_epoch"]
        )

    client.send_command.side_effect = send
    channel.step(client, observation())
    return channel, client


def command_reply(channel, sample=10.02, **changes):
    snapshot = observation(sample=sample)
    if channel._identity is not None:
        snapshot.payload["_safety"].update(
            control_owner=channel._identity.client_id, command=asdict(channel._identity)
        )
    snapshot.payload["_safety"].update(changes)
    return snapshot


def start_command(channel, client):
    assert channel.enable()[0]
    assert channel.accept(BodyVelocity(0.1, -0.02, 0.5), channel.input_epoch)
    channel.step(client, command_reply(channel))
    assert client.send_command.call_count == 1


def test_command_enable_requires_new_input_and_preserves_wire_units(command_channel):
    channel, client = command_channel
    assert not channel.accept(BodyVelocity(1, 0, 0), channel.input_epoch)
    assert channel.enable()[0]
    channel.step(client, observation(sample=10.02))
    client.send_command.assert_not_called()
    assert not channel.accept(BodyVelocity(1, 0, 0), channel.input_epoch - 1)
    assert channel.accept(BodyVelocity(0.1, -0.02, 0.5), channel.input_epoch)
    channel.step(client, observation(sample=10.04))
    targets = client.send_command.call_args.args[0]
    assert targets == {"x.vel": 0.1, "y.vel": -0.02, "theta.vel": math.degrees(0.5)}
    assert not channel.enable()[0]  # Re-enable must not replay an active setpoint.


def test_command_backlog_bounded_until_host_ack(command_channel):
    channel, client = command_channel
    start_command(channel, client)
    for index in range(5):
        channel.accept(BodyVelocity(index / 100, 0, 0), channel.input_epoch)
        channel.step(client, command_reply(channel, sample=10.04 + index * 0.02, command={}))
    assert client.send_command.call_count == 1
    channel.step(client, command_reply(channel, sample=10.14))
    assert client.send_command.call_count == 2
    assert client.send_command.call_args.args[0]["x.vel"] == 0.04


@pytest.mark.parametrize(
    "event", ["disable", "joint_hold_events", "watchdog_events", "calibration"]
)
def test_disable_and_protection_send_one_owned_stop(command_channel, event):
    channel, client = command_channel
    start_command(channel, client)
    snapshot = command_reply(channel, sample=10.04)
    if event == "disable":
        channel.disable()
    elif event == "calibration":
        snapshot.payload["_robot_metadata"]["motors"]["arm_left_gripper"]["drive_mode"] = 1
    else:
        snapshot.payload["_safety"][event] += 1
    channel.step(client, snapshot)
    assert not channel.status()[0]
    assert client.send_command.call_count == 2
    assert set(client.send_command.call_args.args[0].values()) == {0.0}
    assert not channel.enable()[0]
    channel.step(client, command_reply(channel, sample=10.06))
    channel.step(client, command_reply(channel, sample=10.08))
    assert client.send_command.call_count == 2
    assert not channel.status()[1]


@pytest.mark.parametrize(
    "changes",
    [
        {"host_session_id": "host-b"},
        {"control_epoch": 1},
        {"control_owner": "other-client"},
        {"feedback_valid": False},
        {"phase": "fault"},
        {"lift_reference_valid": False},
    ],
)
def test_no_stop_or_motion_into_invalid_or_foreign_lease(command_channel, changes):
    channel, client = command_channel
    start_command(channel, client)
    channel.step(client, command_reply(channel, sample=10.04, **changes))
    assert not channel.status()[0]
    assert client.send_command.call_count == 1


def test_repeated_or_old_feedback_cannot_renew_motion(command_channel):
    channel, client = command_channel
    start_command(channel, client)
    channel._sample_received -= 1
    channel.step(client, command_reply(channel))
    assert not channel.status()[0]
    assert client.send_command.call_count == 1


@pytest.mark.parametrize("kind", ["duplicate", "rollback", "delayed", "aged"])
def test_single_discarded_feedback_does_not_cancel_commands(command_channel, kind):
    channel, client = command_channel
    start_command(channel, client)
    previous, received, sample = channel._snapshot, channel._sample_received, channel._sample
    snapshot = command_reply(channel, sample=10.04)
    if kind in ("duplicate", "rollback"):
        snapshot = command_reply(channel, sample=10.02 if kind == "duplicate" else 9.0)
    elif kind == "delayed":
        snapshot.request_started_s -= 1
    else:
        snapshot.payload["_safety"]["sampled_at_monotonic_s"] += 1
    channel.step(client, snapshot)
    assert channel.status()[0]
    assert channel._snapshot is previous
    assert channel._sample_received == received
    assert channel._sample == sample
    assert client.send_command.call_count == 1
    channel.step(client, command_reply(channel, sample=10.06))
    assert channel.status()[0]
    assert client.send_command.call_count == 2
    channel._sample_received -= 1
    channel.step(client, snapshot)
    assert not channel.status()[0]
    assert client.send_command.call_count == 2
    channel.step(client, command_reply(channel, sample=10.08))
    assert not channel.status()[0]
    assert set(client.send_command.call_args.args[0].values()) == {0.0}


def test_request_timeout_keeps_commands_only_until_last_valid_feedback_expires(command_channel):
    channel, client = command_channel
    start_command(channel, client)
    received = channel._sample_received
    channel.discard_feedback("request timed out")
    assert channel.status()[0]
    assert channel._sample_received == received
    assert client.send_command.call_count == 1
    channel._sample_received -= 1
    channel.discard_feedback("request timed out")
    assert not channel.status()[0]
    assert client.send_command.call_count == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"feedback_valid": False},
        {"control_owner": "other"},
        {"control_epoch": 1},
        {"joint_hold_events": 1},
        {"watchdog_events": 1},
    ],
)
def test_delayed_feedback_still_invalidates_changed_safety_context(command_channel, changes):
    channel, client = command_channel
    start_command(channel, client)
    snapshot = command_reply(channel, sample=10.04, **changes)
    snapshot.request_started_s -= 1
    channel.step(client, snapshot)
    assert not channel.status()[0]
    assert client.send_command.call_count == 1


def test_missing_ack_disables_and_never_retries_stop_forever(command_channel):
    channel, client = command_channel
    start_command(channel, client)
    channel._sent_at -= 1
    channel.step(client, command_reply(channel, sample=10.04, command={}))
    assert not channel.status()[0]
    assert client.send_command.call_count == 2  # One zero command supersedes motion.
    channel._sent_at -= 1
    channel.step(client, command_reply(channel, sample=10.06, command={}))
    channel.step(client, command_reply(channel, sample=10.08, command={}))
    assert client.send_command.call_count == 2


def test_stop_supersedes_submitted_motion_even_before_host_claim_ack(command_channel):
    channel, client = command_channel
    start_command(channel, client)
    channel.disable()
    channel.step(client, command_reply(channel, sample=10.04, command={}, control_owner=None))
    assert client.send_command.call_count == 2
    assert set(client.send_command.call_args.args[0].values()) == {0.0}


@pytest.mark.parametrize(
    "changes",
    [
        {"joint_holds": {"arm_left_elbow_flex": "contact"}},
        {"control_owner": "teleoperator"},
        {"control_epoch": -1},
        {"watchdog_events": True},
        {"command_watchdog_timeout_s": 0.0},
    ],
)
def test_enable_rejects_blocked_or_invalid_host(command_channel, changes):
    channel, client = command_channel
    channel.step(client, command_reply(channel, **changes))
    assert not channel.enable()[0]
    client.send_command.assert_not_called()


def test_shutdown_stop_requires_same_lease(command_channel):
    channel, client = command_channel
    start_command(channel, client)
    client.read.side_effect = lambda: command_reply(
        channel, sample=10.02 + 0.02 * client.read.call_count
    )
    channel.finish(client, 0.2)
    assert client.send_command.call_count == 2
    assert not channel.status()[1]


def test_disable_does_not_wait_for_inflight_network_send(command_channel):
    channel, client = command_channel
    assert channel.enable()[0]
    channel.accept(BodyVelocity(0.1, 0, 0), channel.input_epoch)
    entered, release = threading.Event(), threading.Event()
    send = client.send_command.side_effect

    def blocked_send(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return send(*args, **kwargs)

    client.send_command.side_effect = blocked_send
    worker = threading.Thread(target=channel.step, args=(client, observation(sample=10.02)))
    worker.start()
    try:
        assert entered.wait(1)
        channel.disable()
        assert not channel.status()[0]
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    channel.step(client, command_reply(channel, sample=10.04))
    assert client.send_command.call_count == 2
    assert set(client.send_command.call_args.args[0].values()) == {0.0}


def test_ros_service_and_cmd_vel_use_native_client_and_stop_on_shutdown(ros, host, tmp_path):
    port, state = host
    write_mapping(tmp_path)
    node = AlohaMiniBridge(
        parameter_overrides=[
            Parameter("arm_mapping_dir", value=str(tmp_path)),
            Parameter("observation_port", value=port),
            Parameter("command_port", value=state["command_port"]),
        ]
    )
    operator = Node("base_command_test_operator")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(operator)
    service = operator.create_client(SetBool, "/alohamini_lerobot_bridge/command_enable")
    publisher = operator.create_publisher(Twist, "/cmd_vel", 1)

    def until(predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if predicate():
                return
            executor.spin_once(timeout_sec=0.02)
        pytest.fail(f"ROS condition timed out; command status: {node.commands.status()}")

    try:
        until(lambda: node.observation_count > 0 and service.service_is_ready())
        assert not state["commands"]
        enabled = service.call_async(SetBool.Request(data=True))
        until(enabled.done)
        assert enabled.result().success, enabled.result().message
        until(lambda: publisher.get_subscription_count() > 0)
        assert not state["commands"]
        message = Twist()
        message.linear.x = 0.1
        message.angular.z = 0.5
        publisher.publish(message)
        until(lambda: bool(state["commands"]))
        command = state["commands"][0]
        assert command["x.vel"] == 0.1
        assert command["theta.vel"] == math.degrees(0.5)
        assert set(command) == {"x.vel", "y.vel", "theta.vel", "_command"}
        disabled = service.call_async(SetBool.Request(data=False))
        until(disabled.done)
        assert disabled.result().success
        until(lambda: not node.commands.status()[1])
        assert state["commands"][-1]["x.vel"] == 0.0

        enabled = service.call_async(SetBool.Request(data=True))
        until(enabled.done)
        assert enabled.result().success, enabled.result().message
        until(lambda: publisher.get_subscription_count() > 0)
        publisher.publish(message)
        until(lambda: state["commands"][-1]["x.vel"] == 0.1)
    finally:
        executor.remove_node(node)
        executor.remove_node(operator)
        operator.destroy_node()
        node.destroy_node()
        executor.shutdown()
    assert state["commands"][-1]["x.vel"] == 0.0
    assert state["commands"][-1]["theta.vel"] == 0.0


def test_invalid_cmd_vel_requests_stop_instead_of_retaining_old_velocity(bridge):
    message = Twist()
    message.linear.x = math.nan
    bridge.commands = Mock(input_epoch=1)
    bridge.on_cmd_vel(message, 1)
    bridge.commands.accept.assert_not_called()
    bridge.commands.disable.assert_called_once()


def test_ros_velocity_limits_preserve_axes(bridge):
    message = Twist()
    message.linear.x, message.linear.y, message.angular.z = 2.0, -2.0, 3.0
    bridge.commands = Mock(input_epoch=1)
    bridge.on_cmd_vel(message, 1)
    assert bridge.commands.accept.call_args.args[0] == BodyVelocity(0.25, -0.25, 1.0)


def test_old_ros_callback_cannot_affect_a_new_enable_epoch(bridge):
    bridge.commands = Mock(input_epoch=2)
    message = Twist()
    message.linear.x = math.nan
    bridge.on_cmd_vel(message, 1)
    bridge.commands.disable.assert_not_called()
    bridge.commands.accept.assert_not_called()


def test_network_recovery_stops_old_motion_without_automatically_reenabling(host):
    port, state = host
    commands = RobotCommands(0.1)
    receiver = StateReceiver(
        "127.0.0.1",
        port,
        "alohamini2pro",
        0.1,
        50,
        commands=commands,
        command_port=state["command_port"],
    )
    try:
        wait_result(receiver, lambda result: result[0] is not None)
        assert commands.enable()[0]
        assert commands.accept(BodyVelocity(0.1, 0, 0), commands.input_epoch)
        wait_result(receiver, lambda result: bool(state["commands"]))
        state["respond"] = False
        wait_result(receiver, lambda result: bool(result[2]))
        assert not commands.status()[0]
        state["respond"] = True
        wait_result(receiver, lambda result: result[0] is not None and not commands.status()[1])
        assert state["commands"][-1]["x.vel"] == 0
        assert not commands.status()[0]
    finally:
        receiver.close()
