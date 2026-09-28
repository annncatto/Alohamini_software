import math
import threading
import time
from unittest.mock import Mock

import pytest
from action_msgs.msg import GoalStatus
from alohamini_bridge.actions import ControllerActions
from alohamini_bridge.bridge_node import AlohaMiniBridge, load_mapper
from alohamini_bridge.commands import RobotCommands
from alohamini_bridge.trajectory import TerminalState, TrajectoryResource, TrajectorySample
from builtin_interfaces.msg import Duration, Time
from control_msgs.action import FollowJointTrajectory, GripperCommand
from control_msgs.msg import JointJog, JointTolerance
from rclpy.action import ActionClient, GoalResponse
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.task import Future
from std_srvs.srv import SetBool
from test_hardware import command_channel as command_channel
from test_hardware import command_reply, observation, write_mapping
from test_hardware import host as host
from test_hardware import ros as ros
from trajectory_msgs.msg import JointTrajectoryPoint

from alohamini.model import get_robot_model
from alohamini.schema import BodyVelocity


@pytest.fixture
def control(command_channel, tmp_path):
    _, client = command_channel
    write_mapping(tmp_path)
    commands = RobotCommands(0.25, mapper=load_mapper(tmp_path, get_robot_model("alohamini2pro")))
    commands.step(client, observation())
    assert commands.enable()[0]
    return commands, client


def trajectory(joint, target, duration=0.2):
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = [joint]
    goal.trajectory.points = [
        JointTrajectoryPoint(
            positions=[target],
            time_from_start=Duration(sec=int(duration), nanosec=round((duration % 1) * 1e9)),
        )
    ]
    return goal


def test_partial_goals_latch_other_joints_and_compose_once(control):
    commands, client = control
    for name, joint, target in (
        ("left_arm", "left_shoulder_pan", 0.1),
        ("right_arm", "right_elbow_flex", -0.1),
        ("lift", "vertical_move", 0.01),
    ):
        samples = (TrajectorySample(0.0, {joint: target}),)
        epoch = commands.validate_goal(name, [joint], samples)
        commands.start_goal(name, [joint], samples, epoch)
    commands.step(client, command_reply(commands))
    assert client.send_command.call_count == 1
    target = client.send_command.call_args.args[0]
    assert len([key for key in target if key.endswith(".pos")]) == 12
    assert target["lift_axis.height_mm"] == pytest.approx(310.0)
    assert target["arm_left_elbow_flex.pos"] == pytest.approx(0.5 * 360 / 4095)
    assert target["arm_left_shoulder_pan.pos"] > 0.0
    assert target["arm_right_elbow_flex.pos"] < 0.0
    assert target["x.vel"] == target["theta.vel"] == 0.0


def test_disable_stops_touched_arm_and_lift_without_zeroing_other_joints(control):
    commands, client = control
    for name, joint in (("left_arm", "left_shoulder_pan"), ("lift", "vertical_move")):
        commands.start_goal(
            name, [joint], (TrajectorySample(0.2, {joint: 0.01}),), commands.input_epoch
        )
    commands.step(client, command_reply(commands))
    commands.disable()
    commands.step(client, command_reply(commands, sample=10.04))
    target = client.send_command.call_args.args[0]
    assert target["lift_axis.stop"] == 1.0
    assert "lift_axis.height_mm" not in target
    assert "arm_right_shoulder_pan.pos" not in target
    assert target["arm_left_shoulder_pan.pos"] == pytest.approx(0.5 * 360 / 4095)
    assert "lift_axis.vel" not in target


def test_success_requires_submission_and_fresh_measured_goal(control):
    commands, client = control
    joint = "left_shoulder_pan"
    goal = commands.start_goal(
        "left_arm", [joint], (TrajectorySample(0, {joint: 0}),), commands.input_epoch
    )
    commands.step(client, command_reply(commands))
    assert commands.goal_status("left_arm", goal)[0] is None
    commands.step(client, command_reply(commands, sample=10.04, command={}))
    assert commands.goal_status("left_arm", goal)[0].state is TerminalState.SUCCEEDED


@pytest.mark.parametrize(
    "resource,joint",
    [
        ("left_arm", "left_shoulder_pan"),
        ("left_gripper", "left_gripper"),
    ],
)
def test_action_feedback_does_not_retimestamp_stale_pose(control, resource, joint):
    commands, _ = control
    goal_id = commands.start_goal(
        resource, [joint], (TrajectorySample(10.0, {joint: 0.1}),), commands.input_epoch
    )
    actions = ControllerActions.__new__(ControllerActions)
    actions.commands = commands
    actions._lock = threading.RLock()
    actions.node = Mock()
    actions.node.get_clock.return_value.now.return_value.to_msg.return_value = Time()
    handle = Mock(is_cancel_requested=False)
    future = Future()
    actions._active = {(resource, goal_id): (handle, future, (joint,))}
    actions.tick()
    assert handle.publish_feedback.call_count == 1
    commands._sample_received -= commands.observation_timeout + 0.1
    actions.tick()
    assert handle.publish_feedback.call_count == 1
    assert not future.done()
    commands.fail("Host observation stale")
    actions.tick()
    handle.abort.assert_called_once()
    assert future.done()
    assert not actions._active


def test_gripper_contact_is_not_reported_as_reached(control):
    commands, client = control
    goal = commands.start_goal(
        "left_gripper",
        ["left_gripper"],
        (TrajectorySample(1, {"left_gripper": -0.5}),),
        commands.input_epoch,
    )
    commands.step(client, command_reply(commands, gripper_holds={"arm_left_gripper": 0.0}))
    commands.resources["left_gripper"].start_time -= 2.1
    client.send_command.reset_mock()
    commands.step(
        client, command_reply(commands, sample=10.04, gripper_holds={"arm_left_gripper": 0.0})
    )
    event, _, measured = commands.goal_status("left_gripper", goal)
    result = ControllerActions.result("left_gripper", event, measured)
    assert result.stalled and not result.reached_goal
    assert math.isnan(result.effort)
    client.send_command.assert_not_called()  # Preserve the Host contact target.


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("target", [-0.1, 0.1])
def test_existing_gripper_hold_allows_retreat_target_to_reach_host(control, side, target):
    commands, client = control
    name = f"{side}_gripper"
    goal = commands.start_goal(
        name, [name], (TrajectorySample(1, {name: target}),), commands.input_epoch
    )
    commands.resources[name].start_time -= 0.5
    commands.step(client, command_reply(commands, gripper_holds={f"arm_{name}": 0.0}))
    assert commands.goal_status(name, goal)[0] is None
    key, expected = commands.mapper.urdf_to_host(
        name, target / 2, observation().payload["_robot_metadata"]
    )
    assert client.send_command.call_args.args[0][key] == pytest.approx(expected, abs=0.1)
    # Host releases the hold after accepting the retreat; execution can complete.
    commands.resources[name].start_time -= 0.6
    snapshot = command_reply(commands, sample=10.04, gripper_holds={})
    snapshot.payload[key] = commands.mapper.urdf_to_host(
        name, target, snapshot.payload["_robot_metadata"]
    )[1]
    commands.step(client, snapshot)
    assert commands.goal_status(name, goal)[0].state is TerminalState.SUCCEEDED


def test_base_timeout_preserves_arm_lift_and_accepts_new_base_input(control):
    commands, client = control
    goals = {}
    for name, joint in (("left_arm", "left_shoulder_pan"), ("lift", "vertical_move")):
        goals[name] = commands.start_goal(
            name, [joint], (TrajectorySample(5, {joint: 0.01}),), commands.input_epoch
        )
    epoch = commands.input_epoch
    commands.accept(BodyVelocity(0.1), epoch)
    commands.step(client, command_reply(commands))
    commands._command_at -= 1
    commands.step(client, command_reply(commands, sample=10.04))
    assert client.send_command.call_args.args[0]["x.vel"] == 0
    assert commands.status()[0] and commands.input_epoch == epoch
    assert all(commands.goal_status(name, goal)[0] is None for name, goal in goals.items())
    assert commands.accept(BodyVelocity(0.05), epoch)
    commands.step(client, command_reply(commands, sample=10.06))
    assert client.send_command.call_args.args[0]["x.vel"] == 0.05


@pytest.mark.parametrize(
    "resource,joint", [("left_arm", "left_shoulder_pan"), ("lift", "vertical_move")]
)
def test_cancel_supersedes_unacknowledged_motion_with_fresh_stop(control, resource, joint):
    commands, client = control
    goal = commands.start_goal(
        resource, [joint], (TrajectorySample(1.0, {joint: 0.01}),), commands.input_epoch
    )
    commands.step(client, command_reply(commands))
    motion = commands._identity
    assert commands.cancel_goal(resource, goal)
    commands.step(client, command_reply(commands, sample=10.04, command={}))
    assert client.send_command.call_count == 2
    assert commands._identity.sequence > motion.sequence
    targets = client.send_command.call_args.args[0]
    if resource == "lift":
        assert targets["lift_axis.stop"] == 1.0
        assert "lift_axis.height_mm" not in targets
    else:
        assert targets["arm_left_shoulder_pan.pos"] == pytest.approx(0.5 * 360 / 4095)
    assert commands.status()[0]


def test_lift_jog_release_supersedes_unacknowledged_motion(control):
    commands, client = control
    commands.jog_lift(0.01, commands.input_epoch)
    commands.step(client, command_reply(commands))
    commands.jog_lift(0.0, commands.input_epoch)
    commands.step(client, command_reply(commands, sample=10.04, command={}))
    assert client.send_command.call_count == 2
    assert client.send_command.call_args.args[0]["lift_axis.stop"] == 1.0


def test_lift_stop_survives_temporary_command_backpressure(control):
    commands, client = control
    commands.jog_lift(0.01, commands.input_epoch)
    commands.step(client, command_reply(commands))
    original_send = client.send_command.side_effect
    commands.jog_lift(0.0, commands.input_epoch)
    client.send_command.side_effect = lambda *args, **kwargs: None
    commands.step(client, command_reply(commands, sample=10.04))
    assert commands._lift_stop_requested
    assert commands.status()[0]
    client.send_command.side_effect = original_send
    commands.step(client, command_reply(commands, sample=10.06))
    assert client.send_command.call_args.args[0]["lift_axis.stop"] == 1.0
    assert not commands._lift_stop_requested


def test_cancel_hold_survives_temporary_command_backpressure(control):
    commands, client = control
    goal = commands.start_goal(
        "left_arm", ["left_shoulder_pan"],
        (TrajectorySample(1.0, {"left_shoulder_pan": 0.1}),), commands.input_epoch,
    )
    commands.step(client, command_reply(commands))
    original_send = client.send_command.side_effect
    commands.cancel_goal("left_arm", goal)
    client.send_command.side_effect = lambda *args, **kwargs: None
    commands.step(client, command_reply(commands, sample=10.04))
    assert "left_arm" in commands._cancel_stops
    client.send_command.side_effect = original_send
    commands.step(client, command_reply(commands, sample=10.06))
    assert "arm_left_shoulder_pan.pos" in client.send_command.call_args.args[0]
    assert not commands._cancel_stops


@pytest.mark.parametrize("change", [{"control_owner": "other"}, {"control_epoch": 1}])
def test_cancel_stop_cannot_cross_control_lease(control, change):
    commands, client = control
    goal = commands.start_goal(
        "left_arm",
        ["left_shoulder_pan"],
        (TrajectorySample(1.0, {"left_shoulder_pan": 0.1}),),
        commands.input_epoch,
    )
    commands.step(client, command_reply(commands))
    commands.cancel_goal("left_arm", goal)
    commands.step(client, command_reply(commands, sample=10.04, command={}, **change))
    assert client.send_command.call_count == 1
    assert not commands.status()[0]


def test_new_goal_replaces_queued_cancel_hold_without_waiting_for_ack(control):
    commands, client = control
    samples = (TrajectorySample(1.0, {"left_shoulder_pan": 0.1}),)
    first = commands.start_goal("left_arm", ["left_shoulder_pan"], samples, commands.input_epoch)
    commands.step(client, command_reply(commands))
    commands.cancel_goal("left_arm", first)
    second = commands.start_goal("left_arm", ["left_shoulder_pan"], samples, commands.input_epoch)
    commands.step(client, command_reply(commands, sample=10.04, command={}))
    assert client.send_command.call_count == 2
    assert commands.resources["left_arm"].active_goal_id == second


def test_cancel_preserves_other_resources_and_base(control):
    commands, client = control
    goals = {}
    for side in ("left", "right"):
        name, joint = f"{side}_arm", f"{side}_shoulder_pan"
        goals[name] = commands.start_goal(
            name, [joint], (TrajectorySample(2.0, {joint: 0.1}),), commands.input_epoch
        )
    commands.accept(BodyVelocity(0.05), commands.input_epoch)
    commands.step(client, command_reply(commands))
    commands.cancel_goal("left_arm", goals["left_arm"])
    commands.step(client, command_reply(commands, sample=10.04, command={}))
    assert client.send_command.call_count == 2
    assert client.send_command.call_args.args[0]["x.vel"] == 0.05
    assert commands.resources["right_arm"].active_goal_id == goals["right_arm"]


def test_cancel_before_first_submission_does_not_claim_host(control):
    commands, client = control
    goal = commands.start_goal(
        "left_arm",
        ["left_shoulder_pan"],
        (TrajectorySample(1.0, {"left_shoulder_pan": 0.1}),),
        commands.input_epoch,
    )
    assert commands.cancel_goal("left_arm", goal)
    commands.step(client, command_reply(commands))
    client.send_command.assert_not_called()


def test_lift_jog_stop_never_sends_cached_height(control):
    commands, client = control
    commands.jog_lift(0.01, commands.input_epoch)
    commands.step(client, command_reply(commands))
    assert client.send_command.call_args.args[0]["lift_axis.height_mm"] == pytest.approx(350.0)
    commands.jog_lift(0.0, commands.input_epoch)
    snapshot = command_reply(commands, sample=10.04)
    snapshot.payload["lift_axis.height_mm"] = 305.0
    commands.step(client, snapshot)
    assert client.send_command.call_args.args[0]["lift_axis.stop"] == 1.0
    assert "lift_axis.height_mm" not in client.send_command.call_args.args[0]
    commands.step(client, command_reply(commands, sample=10.06))
    assert "lift_axis.height_mm" not in client.send_command.call_args.args[0]
    assert "lift_axis.stop" not in client.send_command.call_args.args[0]


@pytest.mark.parametrize(
    "velocity,height,expected",
    [
        (0.01, 300.0, 350.0),
        (-0.01, 300.0, 250.0),
        (0.05, 590.0, 600.0),
        (-0.05, 10.0, 0.0),
    ],
)
def test_lift_jog_preserves_directional_lead_and_host_limits(control, velocity, height, expected):
    commands, client = control
    commands.jog_lift(velocity, commands.input_epoch)
    snapshot = command_reply(commands)
    snapshot.payload["lift_axis.height_mm"] = height
    commands.step(client, snapshot)
    assert commands.status()[0]
    assert client.send_command.call_args.args[0]["lift_axis.height_mm"] == pytest.approx(expected)


def test_lift_jog_timeout_keeps_other_resources_and_holds_feedback(control):
    commands, client = control
    commands.jog_lift(0.01, commands.input_epoch)
    commands.step(client, command_reply(commands))
    velocity, received = commands._lift_jog
    commands._lift_jog = velocity, received - 1
    commands.step(client, command_reply(commands, sample=10.04))
    assert client.send_command.call_args.args[0]["lift_axis.stop"] == 1.0
    assert "lift_axis.height_mm" not in client.send_command.call_args.args[0]
    assert commands._lift_jog is None
    assert commands.status()[0]


def test_lift_jog_stop_before_submission_does_not_claim_host(control):
    commands, client = control
    commands.jog_lift(0.01, commands.input_epoch)
    commands.jog_lift(0.0, commands.input_epoch)
    commands.step(client, command_reply(commands))
    client.send_command.assert_not_called()


def test_new_lift_goal_supersedes_queued_jog_stop(control):
    commands, client = control
    commands.jog_lift(0.01, commands.input_epoch)
    commands.step(client, command_reply(commands))
    commands.jog_lift(0, commands.input_epoch)
    commands.start_goal(
        "lift",
        ["vertical_move"],
        (TrajectorySample(1, {"vertical_move": 0.01}),),
        commands.input_epoch,
    )
    commands.step(client, command_reply(commands, sample=10.04))
    assert "lift_axis.stop" not in client.send_command.call_args.args[0]
    assert "lift_axis.height_mm" in client.send_command.call_args.args[0]


def test_cancel_lift_trajectory_uses_native_stop(control):
    commands, client = control
    goal = commands.start_goal(
        "lift",
        ["vertical_move"],
        (TrajectorySample(1, {"vertical_move": 0.01}),),
        commands.input_epoch,
    )
    commands.step(client, command_reply(commands))
    assert commands.cancel_goal("lift", goal)
    commands.step(client, command_reply(commands, sample=10.04))
    assert commands.goal_status("lift", goal)[0].state is TerminalState.CANCELED
    assert client.send_command.call_args.args[0]["lift_axis.stop"] == 1
    assert "lift_axis.height_mm" not in client.send_command.call_args.args[0]


def test_jog_expiration_latches_measured_pose_instead_of_dropping_target():
    resource = TrajectoryResource("arm", ("q",), 0.35, 0.03, 1, 0.25)
    resource.accept_stream_target({"q": 0.2}, {"q": 0.0}, 1.0, 0.1)
    assert resource.update({"q": 0.1}, True, 1.2) == {"q": 0.1}
    assert resource.update({"q": 0.11}, True, 1.3) == {"q": 0.1}


@pytest.fixture
def robot(ros, host, tmp_path):
    port, state = host
    state["follow"] = True
    write_mapping(tmp_path)
    node = AlohaMiniBridge(
        parameter_overrides=[
            Parameter("arm_mapping_dir", value=str(tmp_path)),
            Parameter("observation_port", value=port),
            Parameter("command_port", value=state["command_port"]),
        ]
    )
    operator = Node("trajectory_test_operator")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(operator)

    def until(predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            executor.spin_once(timeout_sec=0.02)
        pytest.fail(f"Condition timed out: {node.commands.status()}")

    clients = {}
    for name in node.commands.resources:
        gripper = name.endswith("gripper")
        clients[name] = ActionClient(
            operator,
            GripperCommand if gripper else FollowJointTrajectory,
            f"/{name}_controller/" + ("gripper_cmd" if gripper else "follow_joint_trajectory"),
        )
    service = operator.create_client(SetBool, "/alohamini_lerobot_bridge/command_enable")
    try:
        until(
            lambda: (
                node.observation_count > 0
                and service.service_is_ready()
                and all(client.server_is_ready() for client in clients.values())
            )
        )
        future = service.call_async(SetBool.Request(data=True))
        until(future.done)
        assert future.result().success, future.result().message
        yield node, clients, state, until
    finally:
        executor.remove_node(node)
        executor.remove_node(operator)
        for client in clients.values():
            client.destroy()
        operator.destroy_node()
        node.destroy_node()
        executor.shutdown()


def send(client, goal, until):
    request = client.send_goal_async(goal)
    until(request.done)
    handle = request.result()
    assert handle.accepted
    return handle, handle.get_result_async()


def test_two_arms_and_lift_run_concurrently_on_one_host_client(robot):
    node, clients, state, until = robot
    results = []
    for resource, joint, target in (
        ("left_arm", "left_shoulder_pan", 0.1),
        ("right_arm", "right_elbow_flex", -0.1),
        ("lift", "vertical_move", 0.01),
    ):
        _, result = send(clients[resource], trajectory(joint, target, 0.3), until)
        results.append(result)
    until(lambda: all(result.done() for result in results))
    assert all(result.result().status == GoalStatus.STATUS_SUCCEEDED for result in results)
    assert len({command["_command"]["client_id"] for command in state["commands"]}) == 1
    assert node.observation_count > 5
    assert node.commands.positions["vertical_move"] == pytest.approx(0.01, abs=0.003)


def test_cancel_and_preemption_do_not_block_state_or_other_goals(robot):
    node, clients, state, until = robot
    _, first = send(clients["left_arm"], trajectory("left_shoulder_pan", 0.3, 2), until)
    handle, second = send(clients["left_arm"], trajectory("left_shoulder_pan", 0.2, 2), until)
    until(first.done)
    assert first.result().status == GoalStatus.STATUS_ABORTED
    count = node.observation_count
    cancel = handle.cancel_goal_async()
    until(cancel.done)
    until(second.done)
    assert second.result().status == GoalStatus.STATUS_CANCELED
    until(lambda: node.observation_count > count + 3)
    assert node.commands.status()[0]


def test_protection_event_aborts_all_active_resources(robot):
    node, clients, state, until = robot
    _, left = send(clients["left_arm"], trajectory("left_shoulder_pan", 0.2, 2), until)
    _, right = send(clients["right_arm"], trajectory("right_shoulder_pan", 0.2, 2), until)
    state["safety"]["joint_hold_events"] = 1
    until(lambda: left.done() and right.done())
    assert left.result().status == right.result().status == GoalStatus.STATUS_ABORTED
    assert not node.commands.status()[0]


def test_requested_path_tolerance_is_enforced(robot):
    _, clients, _, until = robot
    goal = trajectory("left_shoulder_pan", 0.2, 0)
    goal.path_tolerance = [JointTolerance(name="left_shoulder_pan", position=0.01)]
    _, result = send(clients["left_arm"], goal, until)
    until(result.done)
    assert result.result().result.error_code == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
    assert "exceeds 0.010000" in result.result().result.error_string


def test_requested_goal_tolerance_and_grace_are_enforced(robot):
    _, clients, state, until = robot
    state["follow"] = False
    goal = trajectory("left_shoulder_pan", 0.1, 0.05)
    goal.goal_tolerance = [JointTolerance(name="left_shoulder_pan", position=0.001)]
    goal.goal_time_tolerance.nanosec = 50_000_000
    _, result = send(clients["left_arm"], goal, until)
    until(result.done)
    assert result.result().result.error_code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
    assert "exceeds 0.001000" in result.result().result.error_string


def test_gripper_action_reaches_position_without_force_claim(robot):
    _, clients, _, until = robot
    goal = GripperCommand.Goal()
    goal.command.position = -0.1
    _, result = send(clients["left_gripper"], goal, until)
    until(result.done)
    assert result.result().status == GoalStatus.STATUS_SUCCEEDED
    assert result.result().result.reached_goal
    assert math.isnan(result.result().result.effort)


def test_joint_jog_topics_reuse_epoch_and_discard_stale_stamps(robot):
    node, _, state, until = robot
    publisher = node.create_publisher(JointJog, "/left_arm_controller/joint_jog", 1)
    until(lambda: publisher.get_subscription_count() > 0)
    message = JointJog()
    message.joint_names = list(node.commands.resources["left_arm"].joints)
    message.displacements = [0.02] * 6
    node.on_jog(message, "left_arm", node.commands.input_epoch)
    assert not state["commands"]
    message.header.stamp = node.get_clock().now().to_msg()
    publisher.publish(message)
    until(lambda: bool(state["commands"]))
    assert "arm_left_shoulder_pan.pos" in state["commands"][-1]
    until(lambda: node.commands.resources["left_arm"].stream_positions is None)
    assert node.commands.resources["left_arm"].active_goal_id is None


@pytest.mark.parametrize(
    "change",
    [
        lambda goal: goal.trajectory.header.stamp.__setattr__("sec", 1),
        lambda goal: goal.trajectory.points[0].positions.__setitem__(0, 2.0),
        lambda goal: goal.trajectory.joint_names.append("left_shoulder_pan"),
        lambda goal: goal.trajectory.points[0].__setattr__("velocities", [math.nan]),
        lambda goal: goal.path_tolerance.append(
            JointTolerance(name="left_shoulder_pan", velocity=0.1)
        ),
    ],
)
def test_invalid_goals_never_enter_execution(robot, change):
    node, _, state, _ = robot
    goal = trajectory("left_shoulder_pan", 0.1)
    change(goal)
    assert node.actions.accept(goal, "left_arm") is GoalResponse.REJECT
    assert not state["commands"]
