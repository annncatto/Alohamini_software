import json
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import rclpy
import zmq
from action_msgs.msg import GoalStatus
from alohamini_joycon_teleop.teleop_node import ArmControl, JoyConTeleop, arm_names
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from sensor_msgs.msg import JointState


@pytest.fixture
def real_node():
    rclpy.init(args=[])
    node = JoyConTeleop()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def input_payload(**changes):
    return {
        "schema_version": 2,
        "side": "left",
        "sequence": 1,
        "monotonic_ns": time.monotonic_ns(),
        "stick": [2000.0, 2000.0],
        "orientation_rpy": [0.0, 0.0, 0.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "buttons": {"trigger": False, "relatch": False, "shoulder": False, "sl": False},
        **changes,
    }


def inject(node, *payloads):
    socket = Mock()
    socket.recv_string.side_effect = [*(json.dumps(p) for p in payloads), zmq.Again()]
    with patch.object(node, "input_socket", socket):
        node.receive_inputs()


def test_preview_interfaces_do_not_use_real_robot_topics(real_node):
    for publisher in real_node.publishers:
        assert publisher.topic_name.startswith("/alohamini_plan_only/") or publisher.topic_name in (
            "/rosout",
            "/parameter_events",
        )
    assert not real_node.commands_enabled
    assert not hasattr(real_node, "enable_client")
    assert real_node.positions["left_wrist_flex"] == pytest.approx(1.435806017460960)


def test_hardware_subscribes_to_measured_not_mixed_joint_state_topic():
    rclpy.init(args=["--ros-args", "-p", "hardware_mode:=true"])
    node = JoyConTeleop()
    try:
        topics = {subscription.topic_name for subscription in node.subscriptions}
        assert "/alohamini_lerobot_bridge/measured_joint_states" in topics
        assert "/joint_states" not in topics
        assert "/alohamini_lerobot_bridge/derived_wheel_states" not in topics
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_input_validation_rejects_stale_invalid_and_replay_hardware_samples(real_node):
    real_node.hardware_mode = True
    inject(
        real_node,
        [],
        input_payload(side=[]),
        input_payload(stick=[float("nan"), 0.0]),
        input_payload(sequence="bad"),
        input_payload(monotonic_ns=1),
        input_payload(orientation_xyzw=[0.0] * 4),
        input_payload(replay=True),
        input_payload(buttons={"trigger": False, "relatch": False}),
    )
    assert not real_node.samples
    sample = input_payload()
    inject(real_node, sample)
    assert real_node.samples["left"].payload == sample
    inject(real_node, input_payload(monotonic_ns=sample["monotonic_ns"] - 1))
    assert real_node.samples["left"].payload == sample


def test_input_drain_is_bounded(real_node):
    socket = Mock()
    socket.recv_string.return_value = json.dumps(input_payload())
    with patch.object(real_node, "input_socket", socket):
        real_node.receive_inputs()
    assert socket.recv_string.call_count == 100


def test_hardware_gate_requires_enabled_bridge_and_full_fresh_feedback(real_node):
    real_node.hardware_mode = True
    now = time.monotonic()
    real_node.commands_enabled = True
    real_node.last_bridge_status = now
    assert not real_node.hardware_ready(now)
    names = (
        arm_names("left") + arm_names("right") + ["left_gripper", "right_gripper", "vertical_move"]
    )
    message = JointState(name=names, position=[0.0] * len(names))
    real_node.on_joint_state(message)
    assert real_node.hardware_ready(time.monotonic())
    assert not real_node.hardware_ready(now + 1)
    real_node.last_bridge_status = now - 3
    assert not real_node.hardware_ready(time.monotonic())
    real_node.last_bridge_status = now
    message.position[0] = float("nan")
    real_node.on_joint_state(message)
    assert not real_node.hardware_ready(time.monotonic())


def test_bridge_disable_clears_queues_and_old_async_results(real_node):
    real_node.hardware_mode = True
    real_node.commands_enabled = True
    real_node.samples["left"] = SimpleNamespace(received_at=time.monotonic(), payload={})
    arm = real_node.arms["left"]
    arm.queued_positions = [1.0] * 6
    arm.pending_ik = True
    old_generation = real_node.input_generation
    status = DiagnosticStatus(
        name="AlohaMini Host state bridge",
        hardware_id="alohamini2pro",
        values=[
            KeyValue(key="command_enabled", value="false"),
            KeyValue(key="stop_pending", value="true"),
        ],
    )
    real_node.on_bridge_status(DiagnosticArray(status=[status]))
    assert not real_node.commands_enabled and not real_node.samples
    assert arm.queued_positions is None and not arm.pending_ik
    assert all(real_node.require_neutral.values())
    future = Mock()
    real_node.on_fk("left", future, generation=old_generation)
    real_node.on_ik("left", {}, [], "kdl", False, future, generation=old_generation)
    future.result.assert_not_called()
    handle = Mock(accepted=True)
    future.result.return_value = handle
    real_node.on_arm_goal_response("left", future, generation=old_generation)
    handle.cancel_goal_async.assert_called_once()


def test_hardware_resume_requires_release_and_neutral_stick(real_node):
    real_node.hardware_mode = True
    real_node.invalidate_inputs()
    pressed = dict(input_payload()["buttons"], sl=True)
    inject(real_node, input_payload(buttons=pressed))
    assert real_node.sample("left", time.monotonic()) is None
    assert real_node.require_neutral["left"]
    inject(real_node, input_payload())
    assert real_node.sample("left", time.monotonic()) is None
    assert not real_node.require_neutral["left"]
    inject(real_node, input_payload(buttons=pressed))
    assert real_node.sample("left", time.monotonic())["buttons"]["sl"]


@pytest.mark.parametrize("failure", ["response", "cancel"])
def test_failed_obsolete_goal_callback_preserves_current_goal(real_node, failure):
    arm = real_node.arms["left"]
    current_handle = object()
    arm.goal_handle = current_handle
    arm.goal_busy = True
    arm.queued_positions = [0.1] * 6
    obsolete = Mock(accepted=True)
    future = Mock()
    future.result.return_value = obsolete
    if failure == "response":
        future.result.side_effect = RuntimeError("action server disconnected")
    else:
        obsolete.cancel_goal_async.side_effect = RuntimeError("cancel unavailable")
    real_node.on_arm_goal_response("left", future, generation=real_node.input_generation - 1)
    assert arm.goal_handle is current_handle
    assert arm.goal_busy
    assert arm.queued_positions == [0.1] * 6


def test_input_gap_is_not_hidden_by_a_new_held_sample(real_node):
    real_node.hardware_mode = True
    now = time.monotonic()
    real_node.commands_enabled = True
    real_node.hardware_was_ready = True
    real_node.last_bridge_status = now
    real_node.last_measured_state = now
    real_node.require_neutral = {"left": False, "right": False}
    real_node.samples["left"] = SimpleNamespace(
        received_at=now - 1.0, payload=input_payload(monotonic_ns=int((now - 1.0) * 1e9))
    )
    payload = input_payload(buttons=dict(input_payload()["buttons"], sl=True))
    socket = Mock()
    socket.recv_string.side_effect = [json.dumps(payload), zmq.Again()]
    with (
        patch.object(real_node, "input_socket", socket),
        patch.object(real_node, "update_base") as update_base,
        patch.object(real_node, "update_lift"),
        patch.object(real_node, "update_arm"),
    ):
        real_node.on_timer()
    assert real_node.require_neutral["left"]
    assert real_node.input_generation == 1
    assert update_base.call_args.args[:2] == (None, None)


@pytest.mark.parametrize("kind", ["fk", "ik"])
def test_cancelled_arm_ignores_late_kinematics_reply(real_node, kind):
    real_node.arm_control_mode = "moveit"
    arm = real_node.arms["left"]
    future = Mock()
    pose = {"position": [0.0] * 3, "orientation": [0.0, 0.0, 0.0, 1.0]}
    with patch.object(real_node, f"{kind}_client") as client:
        client.call_async.return_value = future
        if kind == "fk":
            real_node.request_fk("left")
        else:
            real_node.request_ik("left", pose, time.monotonic())
        callback = future.add_done_callback.call_args.args[0]
        real_node.deactivate_arm("left", cancel_goal=True)
        # A new engagement already has another request and target.
        setattr(arm, f"pending_{kind}", True)
        arm.target_pose = pose
        arm.cancel_after_accept = False
        callback(future)
    future.result.assert_not_called()
    assert getattr(arm, f"pending_{kind}")
    assert arm.target_pose is pose


def test_cancelling_other_arm_does_not_discard_valid_fk(real_node):
    from geometry_msgs.msg import PoseStamped

    real_node.arm_control_mode = "moveit"
    pose = PoseStamped()
    pose.pose.position.x = 0.1
    pose.pose.orientation.w = 1.0
    future = Mock()
    future.result.return_value = SimpleNamespace(
        error_code=SimpleNamespace(val=1, SUCCESS=1), pose_stamped=[pose]
    )
    with patch.object(real_node, "fk_client") as client:
        client.call_async.return_value = future
        real_node.request_fk("left")
        callback = future.add_done_callback.call_args.args[0]
        real_node.deactivate_arm("right", cancel_goal=True)
        callback(future)
    assert not real_node.arms["left"].pending_fk
    assert real_node.arms["left"].target_pose["position"] == [0.1, 0.0, 0.0]


def test_busy_arm_keeps_only_the_latest_goal():
    node = object.__new__(JoyConTeleop)
    arm = ArmControl(goal_busy=True)
    node.arms = {"left": arm}

    node.send_arm_goal("left", [1.0, 2.0])
    node.send_arm_goal("left", [3.0, 4.0])

    assert arm.queued_positions == [3.0, 4.0]


class _ResultFuture:
    def __init__(self):
        self.callback = None

    def add_done_callback(self, callback):
        self.callback = callback


class _AcceptedHandle:
    accepted = True

    def __init__(self):
        self.result_future = _ResultFuture()

    def get_result_async(self):
        return self.result_future


def test_accepted_goal_immediately_streams_latest_queued_target():
    node = object.__new__(JoyConTeleop)
    arm = ArmControl(
        active=True,
        goal_busy=True,
        queued_positions=[3.0, 4.0],
    )
    node.arms = {"left": arm}
    node.get_parameter = lambda _name: SimpleNamespace(value=True)
    streamed = []
    node.send_arm_goal = lambda side, positions: streamed.append((side, positions))
    handle = _AcceptedHandle()

    JoyConTeleop.on_arm_goal_response(node, "left", SimpleNamespace(result=lambda: handle))

    assert not arm.goal_busy
    assert arm.goal_handle is handle
    assert arm.queued_positions is None
    assert streamed == [("left", [3.0, 4.0])]
    assert handle.result_future.callback is not None


def test_expected_preemption_does_not_latch_arm_fault():
    node = object.__new__(JoyConTeleop)
    handle = object()
    arm = ArmControl(active=True, goal_busy=True, goal_handle=handle)
    node.arms = {"left": arm}
    response = SimpleNamespace(
        status=GoalStatus.STATUS_ABORTED,
        result=SimpleNamespace(error_code=-1, error_string=""),
    )

    JoyConTeleop.on_arm_result(node, "left", handle, SimpleNamespace(result=lambda: response))

    assert not arm.fault_latched
    assert arm.active


class _ArmParameter:
    value = 0.03


class _ArmHarness:
    hardware_mode = True
    ik_period = 0.1

    def __init__(self):
        self.arms = {"left": ArmControl(cancel_after_accept=True)}

    def get_parameter(self, _name):
        return _ArmParameter()

    def request_fk(self, side, _sample):
        self.arms[side].pending_fk = True

    def toggle_gripper(self, _side):
        raise AssertionError("gripper should remain inactive")


def test_fresh_arm_engagement_clears_startup_cancel_latch():
    harness = _ArmHarness()
    sample = {
        "buttons": {
            "up": True,
            "down": False,
            "left": False,
            "right": False,
            "shoulder": False,
            "sl": False,
            "sr": False,
            "relatch": False,
            "trigger": False,
        },
        "orientation_rpy": [0.0, 0.0, 0.0],
    }

    JoyConTeleop.update_arm(harness, "left", sample, 1.0, 0.05)

    arm = harness.arms["left"]
    assert arm.active
    assert not arm.cancel_after_accept
    assert arm.pending_fk


def test_arm_tcp_step_uses_elapsed_ik_time_not_timer_tick():
    harness = _ArmHarness()
    arm = ArmControl(
        active=True,
        target_pose={
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
        last_ik_request=0.89,
    )
    harness.arms["left"] = arm
    requested = {}
    harness.request_ik = lambda side, pose, now, **kwargs: requested.update(
        side=side, pose=pose, now=now, kwargs=kwargs
    )
    sample = {
        "buttons": {
            "up": True,
            "down": False,
            "left": False,
            "right": False,
            "shoulder": False,
            "sl": False,
            "sr": False,
            "relatch": False,
            "trigger": False,
        },
        "orientation_rpy": [0.0, 0.0, 0.0],
    }

    JoyConTeleop.update_arm(harness, "left", sample, 1.0, 1.0 / 30.0)

    # 30 mm/s over the 100 ms since the previous IK request = 3 mm.
    assert requested["pose"]["position"][1] == pytest.approx(-0.003)


class _Parameter:
    value = 0.02


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _LiftHarness:
    hardware_mode = True
    lift_active = False
    deadzone = 0.25

    def __init__(self):
        self.lift_jog_pub = _Publisher()

    def get_parameter(self, _name):
        return _Parameter()

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=10)))


def test_lift_joint_jog_streams_while_pressed_and_stops_once():
    harness = _LiftHarness()
    up = {"stick": (2000.0, 4000.0)}
    center = {"stick": (2000.0, 2000.0)}

    JoyConTeleop.update_lift(harness, up, True, 0.0, 0.05)
    assert list(harness.lift_jog_pub.messages[-1].velocities) == [0.02]
    assert harness.lift_jog_pub.messages[-1].header.stamp.sec == 10

    JoyConTeleop.update_lift(harness, center, True, 0.05, 0.05)
    assert list(harness.lift_jog_pub.messages[-1].velocities) == [0.0]

    count = len(harness.lift_jog_pub.messages)
    JoyConTeleop.update_lift(harness, center, True, 0.1, 0.05)
    assert len(harness.lift_jog_pub.messages) == count


def test_first_measured_lift_state_updates_removed_hold_state_safely():
    harness = object.__new__(JoyConTeleop)
    harness.hardware_mode = True
    harness.positions = {}
    harness.lift_active = False
    harness.lift_target = 0.0
    harness.measured_lift = 0.0
    harness.last_measured_state = None
    message = JointState()
    message.name = ["vertical_move"]
    message.position = [-0.12]

    JoyConTeleop.on_joint_state(harness, message)

    assert harness.last_measured_state is None
    assert harness.measured_lift == 0.0
    message.name += arm_names("left") + arm_names("right") + ["left_gripper", "right_gripper"]
    message.position = list(message.position) + [0.0] * 14
    JoyConTeleop.on_joint_state(harness, message)

    assert harness.measured_lift == -0.12
    assert harness.lift_target == -0.12


def test_gripper_trigger_is_edge_detected_and_debounced_on_both_sides():
    harness = object.__new__(JoyConTeleop)
    harness.arms = {"left": ArmControl(), "right": ArmControl()}
    parameters = {
        "button_release_grace_sec": 0.12,
        "gripper_button_debounce_sec": 0.30,
    }
    harness.get_parameter = lambda name: SimpleNamespace(value=parameters[name])
    toggles = []
    harness.toggle_gripper = lambda side: toggles.append(side)

    for side in ("left", "right"):
        JoyConTeleop.update_gripper_button(harness, side, True, 1.00)
        JoyConTeleop.update_gripper_button(harness, side, False, 1.05)
        # A short false report followed by true is contact/report bounce, not a
        # second physical press, even after the minimum press interval expires.
        JoyConTeleop.update_gripper_button(harness, side, True, 1.10)
        JoyConTeleop.update_gripper_button(harness, side, False, 1.35)
        JoyConTeleop.update_gripper_button(harness, side, False, 1.48)
        JoyConTeleop.update_gripper_button(harness, side, True, 1.50)

    assert toggles == ["left", "left", "right", "right"]


class _GripperClient:
    def __init__(self):
        self.goals = []

    def server_is_ready(self):
        return True

    def send_goal_async(self, goal):
        self.goals.append(goal)


def test_gripper_toggle_reverses_last_command_even_without_measured_motion():
    node = object.__new__(JoyConTeleop)
    node.hardware_mode = True
    node.positions = {"left_gripper": 0.32}
    node.arms = {"left": ArmControl()}
    client = _GripperClient()
    node.gripper_clients = {"left": client}

    JoyConTeleop.toggle_gripper(node, "left")
    # Keep measured feedback unchanged to model a stalled/loaded opening command.
    JoyConTeleop.toggle_gripper(node, "left")

    assert [goal.command.position for goal in client.goals] == pytest.approx([-1.8030294104, 0.32])


def test_command_buttons_ignore_short_false_reports_then_release():
    harness = object.__new__(JoyConTeleop)
    harness.input_timeout = 0.25
    harness.hardware_mode = False
    harness.button_last_true = {"left": {}, "right": {}}
    harness.get_parameter = lambda _name: SimpleNamespace(value=0.12)
    buttons = {
        "shoulder": True,
        "sl": False,
        "sr": False,
        "up": False,
        "down": False,
        "left": False,
        "right": False,
    }
    harness.samples = {"left": SimpleNamespace(received_at=1.0, payload={"buttons": buttons})}

    assert JoyConTeleop.sample(harness, "left", 1.00)["buttons"]["shoulder"]
    buttons["shoulder"] = False
    assert JoyConTeleop.sample(harness, "left", 1.05)["buttons"]["shoulder"]
    assert not JoyConTeleop.sample(harness, "left", 1.13)["buttons"]["shoulder"]
