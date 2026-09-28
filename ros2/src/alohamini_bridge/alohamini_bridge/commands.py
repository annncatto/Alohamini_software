# SPDX-License-Identifier: Apache-2.0
# Adapted from alohamini_lerobot_bridge.protocol.CommandGate and safety.HostSafety.
"""Compose ROS control resources behind one explicitly enabled Host lease."""

import math
import threading
import time
from dataclasses import asdict, replace

from alohamini.errors import AlohaMiniError
from alohamini.protocol import decode_command_context
from alohamini.schema import BodyVelocity

from .mapping import finite_number
from .trajectory import (
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    TerminalState,
    TrajectoryResource,
    TrajectorySample,
)


class RobotCommands:
    """ROS callbacks only change intent; one worker owns all Host I/O.

    Disable discards queued velocity immediately. An already submitting command
    cannot be recalled; the worker follows it with a lease-bound zero command.
    Host acknowledgement means acceptance, not a measured physical stop.
    """

    def __init__(
        self,
        max_age,
        command_timeout=0.5,
        *,
        mapper=None,
        observation_timeout=0.5,
        arm_tracking_error=0.35,
        lift_tracking_error=0.03,
        hold_duration=0.25,
        arm_goal_tolerance=0.03,
        lift_goal_tolerance=0.003,
        goal_time_tolerance=1.0,
        gripper_tracking_error=0.5,
        gripper_goal_tolerance=0.05,
        lift_jog_lookahead=0.05,
    ):
        self.max_age = max_age
        self.command_timeout = command_timeout
        self.observation_timeout = finite_number(observation_timeout, "observation timeout")
        self.lift_jog_lookahead = finite_number(lift_jog_lookahead, "lift jog lookahead")
        self.goal_time_tolerance = finite_number(goal_time_tolerance, "goal time tolerance")
        if (
            self.observation_timeout <= 0
            or self.lift_jog_lookahead <= 0
            or self.goal_time_tolerance < 0
        ):
            raise ValueError("Invalid observation timeout, lift lookahead or goal time tolerance")
        self._lock = threading.RLock()
        self._client_id = None
        self._snapshot = None
        self._sample = None
        self._sample_received = None
        self._context = self._metadata = None
        self._enabled = False
        self._input_epoch = 0
        self._command = None
        self._command_at = None
        self._identity = self._pending = None
        self._sent_at = 0.0
        self._inflight = self._stop_needed = self._stopping = False
        self._reason = "Commands disabled"
        self._fault = ""
        self._command_count = 0
        self.mapper = mapper
        self.positions = {}
        self.resources = {}
        if mapper is not None:
            for name, joints, path, goal, hold in (
                (
                    "left_arm",
                    LEFT_ARM_JOINTS,
                    arm_tracking_error,
                    arm_goal_tolerance,
                    hold_duration,
                ),
                (
                    "right_arm",
                    RIGHT_ARM_JOINTS,
                    arm_tracking_error,
                    arm_goal_tolerance,
                    hold_duration,
                ),
                (
                    "left_gripper",
                    ("left_gripper",),
                    gripper_tracking_error,
                    gripper_goal_tolerance,
                    hold_duration,
                ),
                (
                    "right_gripper",
                    ("right_gripper",),
                    gripper_tracking_error,
                    gripper_goal_tolerance,
                    hold_duration,
                ),
                ("lift", ("vertical_move",), lift_tracking_error, lift_goal_tolerance, math.inf),
            ):
                self.resources[name] = TrajectoryResource(
                    name, joints, path, goal, self.goal_time_tolerance, hold
                )
        self._touched = set()
        self._cancel_stops = set()
        self._sent_goals = {}
        self._lift_jog = None
        self._lift_stop_requested = self._lift_holding = False

    def status(self):
        with self._lock:
            return self._enabled, self._stop_needed or self._inflight, self._reason

    def diagnostics(self):
        """Copy control status atomically for the ROS diagnostic publisher."""
        with self._lock:
            return {
                "command_enabled": self._enabled,
                "stop_pending": self._stop_needed or self._inflight,
                "command_status": self._reason,
                "command_fault": self._fault,
                "command_count": self._command_count,
                "command_stream_started": self._identity is not None or self._inflight,
                **{
                    f"resource_{name}_active": resource.active
                    for name, resource in self.resources.items()
                },
            }

    @property
    def input_epoch(self):
        with self._lock:
            return self._input_epoch

    def enable(self):
        with self._lock:
            if self._enabled:
                return False, "Disable commands before enabling a new command epoch"
            if self._stop_needed or self._inflight or self._pending is not None:
                return False, "Previous command/stop is still pending"
            if self._snapshot is None:
                return False, "Fresh Host feedback is required"
            try:
                context = self._validate(self._snapshot, time.monotonic())
            except (KeyError, TypeError, ValueError) as exc:
                return False, str(exc)
            self._context = context
            self._metadata = self._snapshot.payload["_robot_metadata"]
            self._enabled = True
            self._input_epoch += 1
            self._command = self._command_at = None
            self._lift_jog = None
            self._lift_stop_requested = self._lift_holding = False
            self._touched.clear()
            self._cancel_stops.clear()
            self._reason = "Enabled; waiting for a new command"
            self._fault = ""
            return True, self._reason

    def accept(self, velocity: BodyVelocity, input_epoch: int):
        with self._lock:
            if not self._enabled or input_epoch != self._input_epoch:
                return False
            self._command = velocity
            self._command_at = time.monotonic()
            self._reason = "Enabled; base command received"
            return True

    def disable(
        self, reason="Commands disabled; stop queued if this client owns motion", *, fault=False
    ):
        with self._lock:
            if fault:
                self._fault = reason
            self._enabled = False
            self._command = self._command_at = None
            self._lift_jog = None
            self._lift_stop_requested = self._lift_holding = False
            self._stop_needed |= self._identity is not None or self._inflight
            self._cancel_stops.clear()
            self._reason = reason
            for resource in self.resources.values():
                resource._finish(TerminalState.ABORTED, reason, None, time.monotonic(), hold=False)

    def fail(self, reason):
        with self._lock:
            self.disable(reason, fault=True)
            self._snapshot = None

    def discard_feedback(self, reason):
        """Skip one unusable reply without extending the last valid sample's lifetime."""
        with self._lock:
            if (
                self._sample_received is None
                or time.monotonic() - self._sample_received > self.observation_timeout
            ):
                self.fail(reason)

    @staticmethod
    def _context_of(status):
        counters = tuple(
            status[key] for key in ("control_epoch", "joint_hold_events", "watchdog_events")
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("Invalid Host protection counters")
        return status["host_session_id"], *counters

    def _validate(self, snapshot, now, *, stopping=False):
        status, timing = snapshot.payload["_safety"], snapshot.payload["_host_timing"]
        sampled = finite_number(timing["state_sample_finished_monotonic_s"], "state sample")
        host_now = finite_number(status["sampled_at_monotonic_s"], "Host status time")
        if (
            host_now < sampled
            or now - snapshot.request_started_s + host_now - sampled > self.max_age
            or self._sample_received is None
            or now - self._sample_received > self.observation_timeout
        ):
            raise ValueError("Host feedback is stale")
        return self._validate_status(status, stopping=stopping)

    def _validate_status(self, status, *, stopping=False):
        if (
            status.get("feedback_valid") is not True
            or status.get("lift_reference_valid") is not True
            or status.get("phase") not in ("ready", "active")
            or status.get("fault")
            or decode_command_context({"_safety": status}, client_id=self._client_id) is None
        ):
            raise ValueError("Host is not ready or another client owns control")
        watchdog = finite_number(status["command_watchdog_timeout_s"], "Host watchdog")
        if watchdog <= 0:
            raise ValueError("Invalid Host watchdog timeout")
        if not isinstance(status.get("joint_holds"), dict):
            raise ValueError("Invalid Host joint protection metadata")
        if status["joint_holds"] and not stopping:
            raise ValueError("Host joint protection active")
        return self._context_of(status)

    def _prepare(self, client, snapshot, now):
        self._client_id = client.client_id
        status = snapshot.payload["_safety"]
        sample = (
            status["host_session_id"],
            finite_number(
                snapshot.payload["_host_timing"]["state_sample_finished_monotonic_s"], "sample"
            ),
        )
        new_sample = (
            self._sample is None or sample[0] != self._sample[0] or sample[1] > self._sample[1]
        )
        context = self._context_of(status)
        if self._identity is not None and (
            status.get("control_owner") not in (None, client.client_id)
            or context[:2] != (self._identity.host_session_id, self._identity.control_epoch)
        ):
            self.disable("Host control lease changed; enable again", fault=True)
            self._identity = self._pending = None
            self._stop_needed = self._stopping = False
            self._touched.clear()
        # Protection and lease changes remain fatal even in a delayed reply.
        context = self._validate_status(status, stopping=True)
        if self._enabled and (
            context != self._context
            or snapshot.payload["_robot_metadata"] != self._metadata
            or status["joint_holds"]
        ):
            self.disable(
                "Host session, ownership epoch, calibration or protection changed; enable again",
                fault=True,
            )
        timing = snapshot.payload["_host_timing"]
        started = finite_number(timing["state_sample_started_monotonic_s"], "state start")
        host_now = finite_number(status["sampled_at_monotonic_s"], "Host status time")
        if not 0 <= started <= sample[1] <= host_now:
            raise ValueError("Invalid Host sample interval")
        if not new_sample or now - snapshot.request_started_s + host_now - sample[1] > self.max_age:
            self.discard_feedback("Host feedback is stale")
            return None
        if (
            self._sample_received is not None
            and now - self._sample_received > self.observation_timeout
        ):
            self.disable("Host feedback gap; enable again", fault=True)
            self._snapshot = None
        self._sample, self._sample_received = sample, snapshot.received_s
        if self.mapper is not None:
            if (
                self._snapshot is None
                or context[:2] != self._context_of(self._snapshot.payload["_safety"])[:2]
            ):
                self.mapper.reset()
            self.positions = self.mapper.observation_to_joint_positions(
                snapshot.payload, snapshot.payload["_robot_metadata"]
            )
        self._snapshot = snapshot
        if self._pending is not None:
            acknowledged = status.get("command") == asdict(self._pending)
            if acknowledged:
                self._pending = None
                if self._stopping:
                    self._identity = None
                    self._stop_needed = self._stopping = False
                    self._touched.clear()
                    self._reason += "; stop targets accepted by Host"
            elif now - self._sent_at >= min(0.5, status["command_watchdog_timeout_s"] / 2):
                was_stop = self._stopping
                self.disable(
                    "Host did not acknowledge command; Host watchdog remains active", fault=True
                )
                self._pending = None
                if was_stop:
                    self._identity = None
                    self._stop_needed = self._stopping = False
            else:
                return None  # Only an explicit stop waits for acknowledgement.

        if (
            self._enabled
            and self._command_at is not None
            and now - self._command_at > self.command_timeout
        ):
            # Base input expiry is local to the base, not a Host safety event.
            self._command = BodyVelocity()
            self._command_at = None
            self._reason = "/cmd_vel timed out; base stopped"
        if self._stop_needed and not self._stopping:
            if self._identity is None:
                return None
            # A submitted claim may still be in flight while the Host says idle.
            # A higher-sequence zero in the same session/epoch supersedes it.
            # No stop is generated if this client never submitted motion.
            self._inflight = True
            targets = self._base_targets(BodyVelocity())
            for name in self._touched:
                if name == "lift":
                    targets["lift_axis.stop"] = 1.0
                    continue
                for joint in self.resources[name].joints:
                    targets.update(self._convert(joint, self.positions[joint], snapshot))
            return targets, True, {}
        if not self._enabled or self._pending is not None:
            return None
        self._validate(snapshot, now)
        targets = {}
        if self._lift_jog is not None:
            velocity, received = self._lift_jog
            if now - received > self.command_timeout:
                self._stop_lift("Lift JointJog timed out", now)
            else:
                # Preserve CommandComposer.compose(): direction selects a fixed
                # lead from fresh measured position, not a velocity integrator.
                measured = self.positions["vertical_move"]
                target = measured + math.copysign(self.lift_jog_lookahead, velocity)
                limits = snapshot.payload["_robot_metadata"]["lift_axis"]
                lower = max(
                    self.mapper.lift.position_min_m,
                    self.mapper.lift_height_to_urdf(limits["soft_min_mm"]),
                )
                upper = min(
                    self.mapper.lift.position_max_m,
                    self.mapper.lift_height_to_urdf(limits["soft_max_mm"]),
                )
                target = max(lower, min(upper, target))
                targets.update(self._convert("vertical_move", target, snapshot))
                self._touched.add("lift")
        if self._lift_stop_requested:
            targets["lift_axis.stop"] = 1.0
            self._lift_stop_requested = False
        if self._lift_holding:
            # Maintain the lease without replacing the Host-local settled height.
            targets.update(self._base_targets(BodyVelocity()))
        if self._command is not None:
            targets.update(self._base_targets(self._command))
        for name, resource in self.resources.items():
            if name == "lift" and self._lift_jog is not None:
                continue
            if name in self._cancel_stops:
                resource.hold_positions = {
                    joint: self.positions[joint] for joint in resource.joints
                }
                resource.hold_until = now + resource.hold_duration
            contact = name.endswith("gripper") and (
                f"arm_{name}" in status.get("gripper_holds", {})
            )
            goal_id = resource.active_goal_id
            positions = resource.update(
                self.positions,
                True,
                now,
                allow_success=self._sent_goals.get(name) == goal_id and not contact,
            )
            if contact and goal_id is not None and resource.active_goal_id is None:
                # Let retreat targets reach the Host guard. If it keeps holding
                # until path/goal failure, report contact without overwriting the
                # guard's grasp target with a cached measured position.
                resource.terminals[goal_id] = replace(
                    resource.terminals[goal_id], message="Host gripper contact"
                )
                resource.hold_positions = None
                positions = None
            if positions is not None:
                self._touched.add(name)
                for joint, position in positions.items():
                    targets.update(self._convert(joint, position, snapshot))
        if not targets:
            return None
        for key, value in self._base_targets(BodyVelocity()).items():
            targets.setdefault(key, value)
        self._inflight = True
        self._cancel_stops.clear()
        return (
            targets,
            False,
            {name: resource.active_goal_id for name, resource in self.resources.items()},
        )

    @staticmethod
    def _base_targets(velocity):
        return {
            "x.vel": velocity.x_m_s,
            "y.vel": velocity.y_m_s,
            "theta.vel": math.degrees(velocity.yaw_rad_s),
        }

    def _convert(self, joint, position, snapshot):
        metadata = snapshot.payload["_robot_metadata"]
        if joint == "vertical_move":
            height = self.mapper.lift_urdf_to_height(position)
            limits = metadata["lift_axis"]
            if not limits["soft_min_mm"] <= height <= limits["soft_max_mm"]:
                raise ValueError("Lift target exceeds Host soft limits")
            return {"lift_axis.height_mm": height}
        key, value = self.mapper.urdf_to_host(joint, position, metadata)
        return {key: value}

    def validate_goal(self, name, names, samples):
        with self._lock:
            if not self._enabled or self._snapshot is None:
                raise ValueError("ROS commands are disabled or feedback is unavailable")
            context = self._validate(self._snapshot, time.monotonic())
            if context != self._context:
                raise ValueError("Host control context changed")
            resource = self.resources[name]
            resource.validate(names, samples)
            for point in samples:
                for joint, position in point.positions.items():
                    self._convert(joint, position, self._snapshot)
            for joint in resource.joints:
                self._convert(joint, self.positions[joint], self._snapshot)
            return self._input_epoch

    def start_goal(self, name, names, samples, epoch, *, path=None, goal=None, grace=None):
        with self._lock:
            if epoch != self.validate_goal(name, names, samples):
                raise ValueError("Goal belongs to an expired command epoch")
            resource = self.resources[name]
            self._cancel_stops.discard(name)
            if name == "lift":
                self._lift_jog = None
                self._lift_stop_requested = self._lift_holding = False
            goal_id = resource.activate(names, samples, self.positions, time.monotonic())
            resource.path_limits = path or dict.fromkeys(resource.joints, resource.tracking_error)
            resource.goal_limits = goal or dict.fromkeys(resource.joints, resource.goal_tolerance)
            resource.goal_time_tolerance = self.goal_time_tolerance if grace is None else grace
            return goal_id

    def cancel_goal(self, name, goal_id):
        with self._lock:
            fresh = self._snapshot is not None
            if fresh:
                try:
                    self._validate(self._snapshot, time.monotonic(), stopping=True)
                except (KeyError, TypeError, ValueError):
                    fresh = False
            canceled = self.resources[name].cancel(
                goal_id, self.positions, fresh and name in self._touched, time.monotonic()
            )
            if canceled and name == "lift" and fresh:
                self._stop_lift("Lift trajectory canceled", time.monotonic())
            elif canceled and fresh and name in self._touched:
                self._cancel_stops.add(name)
            return canceled

    def goal_status(self, name, goal_id):
        with self._lock:
            resource = self.resources[name]
            event = resource.terminal(goal_id)
            desired = dict(resource.desired or self.positions)
            # Terminal results may report the last known pose. Periodic feedback
            # must not stamp an expired cache as a fresh measured position.
            fresh = (
                self._snapshot is not None
                and self._sample_received is not None
                and time.monotonic() - self._sample_received <= self.observation_timeout
            )
            measured = dict(self.positions) if fresh or event is not None else {}
            return event, desired, measured

    def jog(self, name, names, displacements, epoch, *, timeout=0.2):
        with self._lock:
            resource = self.resources[name]
            if tuple(names) != resource.joints or len(displacements) != len(names):
                raise ValueError("JointJog requires the resource's canonical joint order")
            targets = {
                joint: self.positions[joint] + finite_number(value, "JointJog displacement")
                for joint, value in zip(names, displacements, strict=True)
            }
            samples = (TrajectorySample(0.0, targets),)
            if epoch != self.validate_goal(name, names, samples):
                raise ValueError("JointJog belongs to an expired command epoch")
            self._cancel_stops.discard(name)
            resource.accept_stream_target(targets, self.positions, time.monotonic(), timeout)

    def jog_lift(self, velocity, epoch):
        with self._lock:
            velocity = finite_number(velocity, "lift jog velocity")
            measured = self.positions["vertical_move"]
            if epoch != self.validate_goal(
                "lift", ["vertical_move"], (TrajectorySample(0, {"vertical_move": measured}),)
            ):
                raise ValueError("JointJog belongs to an expired command epoch")
            now = time.monotonic()
            resource = self.resources["lift"]
            if abs(velocity) <= 1e-9:
                if self._lift_jog is not None:
                    self._stop_lift("Lift JointJog stopped", now)
            else:
                self._lift_stop_requested = self._lift_holding = False
                if resource.active_goal_id is not None:
                    resource._finish(
                        TerminalState.PREEMPTED, "Preempted by lift JointJog", None, now, hold=False
                    )
                resource.hold_positions = resource.stream_positions = None
                self._lift_jog = velocity, now

    def _stop_lift(self, reason, now):
        self.resources["lift"]._finish(TerminalState.CANCELED, reason, None, now, hold=False)
        self._lift_jog = None
        self._lift_stop_requested = self._lift_holding = "lift" in self._touched

    def step(self, client, snapshot):
        """Called only by the client-owning worker, at most once per fresh sample."""
        try:
            with self._lock:
                cancel_stops = self._cancel_stops.copy()
                prepared = self._prepare(client, snapshot, time.monotonic())
            if prepared is None:
                return
            targets, stopping, sent_goals = prepared
            identity = client.send_command(targets, based_on=snapshot)
            with self._lock:
                if identity is None:
                    self._inflight = False
                    self._cancel_stops.update(
                        name
                        for name in cancel_stops
                        if self.resources[name].active_goal_id == sent_goals.get(name)
                    )
                    if (
                        "lift_axis.stop" in targets
                        and self._lift_holding
                        and self._lift_jog is None
                    ):
                        self._lift_stop_requested = True
                    return
                self._command_count += 1
                self._identity = identity
                # Continuous targets remain latest-only in HostClient's PUSH
                # socket. Do not serialize motion on one ACK per target.
                self._pending = identity if stopping else None
                self._sent_goals = sent_goals
                self._sent_at = time.monotonic()
                self._stopping = stopping
                self._inflight = False
                if not self._enabled:
                    self._stop_needed = True
        except (AlohaMiniError, KeyError, TypeError, ValueError, OverflowError) as exc:
            with self._lock:
                self._inflight = False
                self.fail(str(exc))

    def finish(self, client, timeout):
        """Bounded shutdown stop; never reconnect/claim a different control lease."""
        self.disable("Bridge shutting down")
        deadline = time.monotonic() + timeout
        while self.status()[1] and time.monotonic() < deadline:
            try:
                self.step(client, client.read())
            except AlohaMiniError:
                break
