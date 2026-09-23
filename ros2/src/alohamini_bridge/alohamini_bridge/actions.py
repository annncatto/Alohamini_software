# SPDX-License-Identifier: Apache-2.0
# Adapted from alohamini_lerobot_bridge/bridge_node.py action endpoints.
"""ROS action lifecycle without blocking execution callbacks or a second Host client."""

import math
import threading
from functools import partial

from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.task import Future

from .mapping import finite_number
from .trajectory import TerminalState, TrajectorySample


def seconds(duration):
    if duration.sec < 0 or not 0 <= duration.nanosec < 1_000_000_000:
        raise ValueError("Invalid ROS duration")
    return duration.sec + duration.nanosec / 1e9


def tolerances(entries, resource, names, default):
    result = dict.fromkeys(resource.joints, default)
    seen = set()
    for entry in entries:
        if entry.name not in names or entry.name in seen:
            raise ValueError("Tolerance joint is missing or duplicated")
        seen.add(entry.name)
        value = finite_number(entry.position, "position tolerance")
        if entry.velocity != 0.0 or entry.acceleration != 0.0:
            raise ValueError("Only position tolerances are supported")
        if value == -1:
            result[entry.name] = math.inf
        elif value > 0:
            result[entry.name] = value
        elif value != 0:
            raise ValueError("Position tolerance must be positive, zero (default), or -1")
    return result


class ControllerActions:
    def __init__(self, node):
        self.node, self.commands = node, node.commands
        self._lock = threading.RLock()
        self._accepted = {}
        self._active = {}
        self._closed = False
        self.servers = []
        group = ReentrantCallbackGroup()
        for resource in self.commands.resources:
            gripper = resource.endswith("gripper")
            action = GripperCommand if gripper else FollowJointTrajectory
            suffix = "gripper_cmd" if gripper else "follow_joint_trajectory"
            self.servers.append(
                ActionServer(
                    node,
                    action,
                    f"/{resource}_controller/{suffix}",
                    execute_callback=partial(self.execute, resource=resource),
                    goal_callback=partial(self.accept, resource=resource),
                    cancel_callback=lambda _: CancelResponse.ACCEPT,
                    callback_group=group,
                )
            )
        self.timer = node.create_timer(0.02, self.tick)

    def prepare(self, request, resource):
        controller = self.commands.resources[resource]
        if resource.endswith("gripper"):
            target = finite_number(request.command.position, "gripper position")
            if request.command.max_effort != 0.0:
                raise ValueError("Per-goal max_effort is unsupported; Host owns current protection")
            names = controller.joints
            samples = (TrajectorySample(1.0, {names[0]: target}),)
            options = {}
        else:
            trajectory = request.trajectory
            if trajectory.header.stamp.sec or trajectory.header.stamp.nanosec:
                raise ValueError("Trajectory header stamp must be zero (start on acceptance)")
            if request.multi_dof_trajectory.joint_names or request.multi_dof_trajectory.points:
                raise ValueError("Multi-DOF trajectories are not supported")
            for field in ("component_path_tolerance", "component_goal_tolerance"):
                if getattr(request, field, []):
                    raise ValueError("Component tolerances are not supported")
            names = tuple(trajectory.joint_names)
            if not 0 < len(trajectory.points) <= 10000:
                raise ValueError("Require 1..10000 trajectory points")
            samples = []
            for point in trajectory.points:
                if len(point.positions) != len(names) or point.effort:
                    raise ValueError("Require matching positions and no effort feedforward")
                for field in (point.velocities, point.accelerations):
                    if field and (
                        len(field) != len(names) or any(not math.isfinite(v) for v in field)
                    ):
                        raise ValueError("Invalid trajectory derivatives")
                samples.append(
                    TrajectorySample(
                        seconds(point.time_from_start),
                        dict(zip(names, point.positions, strict=True)),
                    )
                )
            samples = tuple(samples)
            options = dict(
                path=tolerances(
                    request.path_tolerance, controller, names, controller.tracking_error
                ),
                goal=tolerances(
                    request.goal_tolerance, controller, names, controller.goal_tolerance
                ),
                grace=seconds(request.goal_time_tolerance) or 1.0,
            )
        epoch = self.commands.validate_goal(resource, names, samples)
        return names, samples, epoch, options

    def accept(self, request, resource):
        with self._lock:
            if self._closed or len(self._accepted) + len(self._active) >= 32:
                return GoalResponse.REJECT
            try:
                prepared = self.prepare(request, resource)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                self.node.get_logger().warning(f"Rejected {resource} goal: {exc}")
                return GoalResponse.REJECT
            self._accepted[id(request)] = (request, prepared)
            return GoalResponse.ACCEPT

    async def execute(self, handle, resource):
        with self._lock:
            accepted = self._accepted.pop(id(handle.request), None)
            try:
                if self._closed or accepted is None or accepted[0] is not handle.request:
                    raise ValueError("Goal acceptance expired")
                names, samples, epoch, options = accepted[1]
                goal_id = self.commands.start_goal(resource, names, samples, epoch, **options)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                handle.abort()
                return self.result(resource, None, {}, message=str(exc))
            future = Future()
            self._active[resource, goal_id] = (handle, future, names)
        return await future

    @staticmethod
    def result(resource, event, measured, *, message=""):
        if resource.endswith("gripper"):
            result = GripperCommand.Result()
            result.position = measured.get(resource, math.nan)
            result.effort = math.nan  # Current is not calibrated joint force/torque.
            result.stalled = event is not None and "gripper contact" in event.message
            result.reached_goal = event is not None and event.state is TerminalState.SUCCEEDED
            return result
        result = FollowJointTrajectory.Result()
        result.error_string = message if event is None else event.message
        result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
        if event is not None:
            result.error_code = {
                TerminalState.SUCCEEDED: result.SUCCESSFUL,
                TerminalState.CANCELED: result.SUCCESSFUL,
                TerminalState.GOAL_TOLERANCE: result.GOAL_TOLERANCE_VIOLATED,
                TerminalState.ABORTED: result.PATH_TOLERANCE_VIOLATED,
                TerminalState.STALE: result.PATH_TOLERANCE_VIOLATED,
            }.get(event.state, result.INVALID_GOAL)
        return result

    def tick(self):
        with self._lock:
            for (resource, goal_id), (handle, future, names) in list(self._active.items()):
                if handle.is_cancel_requested:
                    self.commands.cancel_goal(resource, goal_id)
                event, desired, measured = self.commands.goal_status(resource, goal_id)
                if event is not None:
                    if event.state is TerminalState.SUCCEEDED:
                        handle.succeed()
                    elif event.state is TerminalState.CANCELED:
                        handle.canceled()
                    else:
                        handle.abort()
                    future.set_result(self.result(resource, event, measured))
                    del self._active[resource, goal_id]
                elif all(name in measured and name in desired for name in names):
                    if resource.endswith("gripper"):
                        feedback = GripperCommand.Feedback()
                        feedback.position, feedback.effort = measured[resource], math.nan
                    else:
                        feedback = FollowJointTrajectory.Feedback()
                        feedback.header.stamp = self.node.get_clock().now().to_msg()
                        feedback.joint_names = list(names)
                        feedback.desired.positions = [desired[name] for name in names]
                        feedback.actual.positions = [measured[name] for name in names]
                        feedback.error.positions = [
                            desired[name] - measured[name] for name in names
                        ]
                    handle.publish_feedback(feedback)

    def close(self):
        with self._lock:
            self._closed = True
            self._accepted.clear()
            self.commands.disable("ROS action servers shutting down")
            self.tick()
            self.node.destroy_timer(self.timer)
            for server in self.servers:
                server.destroy()
