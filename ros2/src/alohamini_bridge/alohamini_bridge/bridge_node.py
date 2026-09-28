# SPDX-License-Identifier: Apache-2.0
# State publication adapted from alohamini_lerobot_bridge/bridge_node.py.
"""Host state and explicitly enabled base commands through the native client."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import rclpy
import yaml
from builtin_interfaces.msg import Time as TimeMsg
from control_msgs.msg import JointJog
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool

from alohamini.client import HostClient
from alohamini.errors import AlohaMiniError, ResponseTimeoutError
from alohamini.kinematics import OmniBaseKinematics
from alohamini.model import get_robot_model
from alohamini.paths import WorkspacePaths
from alohamini.schema import BodyVelocity

from .actions import ControllerActions
from .commands import RobotCommands
from .mapping import ARM_JOINTS, JointMapper, finite_number


class StateReceiver:
    """One client-owning thread and one latest-result slot; callbacks never wait on I/O."""

    def __init__(
        self,
        host,
        port,
        model,
        timeout,
        rate,
        *,
        commands=None,
        command_port=5555,
        request_window=3,
    ):
        # Validate connection parameters before starting a background thread.
        HostClient(
            host,
            port=port,
            command_port=command_port,
            expected_model=model,
            timeout_s=timeout,
            request_window=request_window,
        ).close()
        self.commands = commands
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._result = None
        self._thread = threading.Thread(
            target=self._run,
            args=(host, port, model, timeout, rate, command_port, request_window),
            name="AlohaMiniRosState",
            daemon=True,
        )
        self._timeout = timeout
        self._thread.start()

    def _run(self, host, port, model, timeout, rate, command_port, request_window=3):
        generation = 0
        try:
            with HostClient(
                host,
                port=port,
                command_port=command_port,
                expected_model=model,
                timeout_s=timeout,
                request_window=request_window,
            ) as client:
                control_connected = False
                while not self._stop.is_set():
                    started = time.monotonic()
                    try:
                        enabled = self.commands is not None and self.commands.status()[0]
                        if enabled and not control_connected:
                            snapshot = client.connect_control()
                            control_connected = True
                        else:
                            snapshot = client.read()
                        if not enabled:
                            control_connected = False
                        result = snapshot, generation, ""
                        if self.commands is not None:
                            if self._stop.is_set():
                                self.commands.disable("Bridge shutting down")
                            self.commands.step(client, result[0])
                    except ResponseTimeoutError as exc:
                        result = None, generation, exc
                        if self.commands is not None:
                            self.commands.discard_feedback(str(exc))
                    except AlohaMiniError as exc:
                        control_connected = False
                        generation += 1
                        result = None, generation, str(exc)
                        if self.commands is not None:
                            self.commands.fail(str(exc))
                    with self._lock:
                        self._result = result
                    self._stop.wait(max(0, 1 / rate - (time.monotonic() - started)))
                if self.commands is not None:
                    self.commands.finish(client, 2 * timeout)
        except Exception as exc:
            if self.commands is not None:
                self.commands.fail(f"Host worker stopped: {exc}")
            with self._lock:
                self._result = None, generation + 1, f"State receiver stopped: {exc}"

    def take(self):
        with self._lock:
            result, self._result = self._result, None
        return result

    def close(self):
        if self.commands is not None:
            self.commands.disable("Bridge shutting down")
        self._stop.set()
        # Include the bounded 5 s control handshake when closing during enable.
        self._thread.join(timeout=max(6, 4 * self._timeout + 1))
        if self._thread.is_alive():
            raise RuntimeError("Host state receiver did not stop")


def load_mapper(directory: Path, model):
    def read(name):
        path = directory / name
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if (
            not isinstance(document, dict)
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1
            or document.get("robot_model") != model.model_id
        ):
            raise ValueError(f"Invalid model/schema in {path}")
        return document

    arms = {}
    motors = {motor.name: motor for motor in model.actuators}
    for side in ("left", "right"):
        arm = read(f"hardware_joint_map_{side}.yaml")
        if arm.get("side") != side:
            raise ValueError(f"Wrong arm mapping side: {side}")
        for joint in ARM_JOINTS:
            entry = arm["joints"][joint]
            motor = motors[f"arm_{side}_{joint}"]
            if entry["id"] != motor.motor_id or entry["model"] != motor.motor_model:
                raise ValueError(f"Mapping motor differs from model: {motor.name}")
        arms[side] = arm
    lift = read("lift_axis.yaml")
    if lift["urdf"].get("joint") != "vertical_move":
        raise ValueError("Lift mapping must target vertical_move")
    return JointMapper(arms, lift)


class AlohaMiniBridge(Node):
    def __init__(self, **kwargs):
        # Retain existing private ROS topic paths for existing ROS clients.
        super().__init__("alohamini_lerobot_bridge", **kwargs)
        defaults = {
            "host": "127.0.0.1",
            "observation_port": 5556,
            "command_port": 5555,
            "request_window": 3,
            "request_timeout_sec": 1.0,
            "rate_hz": 50.0,
            "observation_timeout_sec": 0.5,
            "max_state_response_age_sec": 0.25,
            "expected_robot_model": "alohamini2pro",
            "state_timestamp_mode": "receipt",
            "arm_mapping_dir": str(WorkspacePaths().calibration / "hardware"),
            "joint_states_topic": "/joint_states",
            "base_velocity_topic": "/alohamini/base_velocity",
            "base_frame": "base_link",
            "max_clock_offset_ms": 250.0,
            "command_timeout_sec": 0.5,
            "cmd_vel_topic": "/cmd_vel",
            "max_linear_speed": 0.25,
            "max_lateral_speed": 0.25,
            "max_angular_speed": 1.0,
            "max_arm_jog_displacement_rad": 0.1,
            "arm_jog_timeout_sec": 0.15,
            "max_lift_jog_speed_m_s": 0.05,
            "lift_jog_lookahead_m": 0.05,
            "trajectory_hold_sec": 0.25,
            "arm_path_tolerance_rad": 0.35,
            "gripper_path_tolerance_rad": 0.5,
            "lift_path_tolerance_m": 0.03,
            "arm_goal_tolerance_rad": 0.03,
            "gripper_goal_tolerance_rad": 0.05,
            "lift_goal_tolerance_m": 0.003,
            "goal_time_tolerance_sec": 1.0,
            "gripper_command_duration_sec": 1.0,
            "left_arm_jog_topic": "/left_arm_controller/joint_jog",
            "right_arm_jog_topic": "/right_arm_controller/joint_jog",
            "lift_jog_topic": "/lift_controller/joint_jog",
            "linear_x_scale": 1.0,
            "linear_y_scale": 1.0,
            "angular_z_scale": 1.0,
            "swap_xy": False,
            "wheel_radius": 0.063,
            "base_radius": 0.195,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        def parameter(name):
            return self.get_parameter(name).value

        self.model = get_robot_model(parameter("expected_robot_model"))
        directory = Path(
            parameter("arm_mapping_dir") or WorkspacePaths().calibration / "hardware"
        ).expanduser()
        if not directory.is_absolute():
            raise ValueError("arm_mapping_dir must be absolute")
        self.mapper = load_mapper(directory, self.model)
        self.obs_timeout = finite_number(
            parameter("observation_timeout_sec"), "observation timeout"
        )
        self.max_age = finite_number(parameter("max_state_response_age_sec"), "state response age")
        request_timeout = finite_number(parameter("request_timeout_sec"), "request timeout")
        if not self.max_age <= request_timeout <= 60:
            raise ValueError("request_timeout_sec must be >= max_state_response_age_sec and <= 60")
        rate = finite_number(parameter("rate_hz"), "rate_hz")
        self.max_clock_offset_ms = finite_number(parameter("max_clock_offset_ms"), "clock limit")
        if not 0 < self.max_age <= 1 or not 0 < self.obs_timeout <= 2 or not 0 < rate <= 100:
            raise ValueError(
                "Require response age in (0,1], observation timeout in (0,2], rate in (0,100]"
            )
        if self.max_clock_offset_ms <= 0:
            raise ValueError("max_clock_offset_ms must be positive")
        self.state_timestamp_mode = parameter("state_timestamp_mode")
        if self.state_timestamp_mode not in ("receipt", "host_wall"):
            raise ValueError("state_timestamp_mode must be receipt or host_wall")
        self.base_frame = parameter("base_frame")
        command_timeout = finite_number(parameter("command_timeout_sec"), "command timeout")
        if not 0 < command_timeout <= 1:
            raise ValueError("command_timeout_sec must be in (0,1]")
        self.velocity_limits = tuple(
            finite_number(parameter(name), name)
            for name in ("max_linear_speed", "max_lateral_speed", "max_angular_speed")
        )
        if any(limit <= 0 for limit in self.velocity_limits):
            raise ValueError("Base velocity limits must be positive")
        # Publication and execution decode separate sequential feedback streams.
        self.commands = RobotCommands(
            self.max_age,
            command_timeout,
            mapper=load_mapper(directory, self.model),
            observation_timeout=self.obs_timeout,
            arm_tracking_error=parameter("arm_path_tolerance_rad"),
            lift_tracking_error=parameter("lift_path_tolerance_m"),
            hold_duration=parameter("trajectory_hold_sec"),
            arm_goal_tolerance=parameter("arm_goal_tolerance_rad"),
            lift_goal_tolerance=parameter("lift_goal_tolerance_m"),
            goal_time_tolerance=parameter("goal_time_tolerance_sec"),
            gripper_tracking_error=parameter("gripper_path_tolerance_rad"),
            gripper_goal_tolerance=parameter("gripper_goal_tolerance_rad"),
            lift_jog_lookahead=parameter("lift_jog_lookahead_m"),
        )
        self.gripper_command_duration = finite_number(
            parameter("gripper_command_duration_sec"), "gripper command duration"
        )
        if self.gripper_command_duration <= 0:
            raise ValueError("gripper_command_duration_sec must be positive")
        self.base_scales = tuple(
            finite_number(parameter(name), name)
            for name in ("linear_x_scale", "linear_y_scale", "angular_z_scale")
        )
        if any(value == 0 for value in self.base_scales):
            raise ValueError("Base coordinate scales must be nonzero")
        self.swap_xy = parameter("swap_xy")
        self.jog_topics = {
            name: parameter(f"{name}_jog_topic") for name in ("left_arm", "right_arm", "lift")
        }
        self.arm_jog_limit = finite_number(
            parameter("max_arm_jog_displacement_rad"), "arm jog limit"
        )
        self.arm_jog_timeout = finite_number(parameter("arm_jog_timeout_sec"), "arm jog timeout")
        self.lift_jog_limit = finite_number(parameter("max_lift_jog_speed_m_s"), "lift jog limit")
        if self.arm_jog_limit <= 0 or self.lift_jog_limit <= 0 or not 0 < self.arm_jog_timeout <= 1:
            raise ValueError("Jog limits must be positive; arm timeout must be in (0,1]")
        self.command_enabled_at_ns = 0
        self.jog_subscriptions = []
        self.kinematics = OmniBaseKinematics(parameter("wheel_radius"), parameter("base_radius"))
        self.last_observation_monotonic = None
        self.last_state_clock_offset_ms = math.nan
        self.last_observation_error = ""
        self.observation_count = self.invalid_observations = 0
        self.late_state_responses = 0
        self.last_state_response_age_ms = math.nan
        self._host_model = "unknown"
        self._host_joint_holds = ()
        self._lift_command_ready = False
        self._stream = self._motor_metadata = self._last_host_sample = None
        self.wheel_positions = [0.0, 0.0, 0.0]
        self.joint_pub = self.create_publisher(JointState, parameter("joint_states_topic"), 10)
        self.measured_joint_pub = self.create_publisher(JointState, "~/measured_joint_states", 10)
        self.derived_wheel_pub = self.create_publisher(JointState, "~/derived_wheel_states", 10)
        self.base_velocity_pub = self.create_publisher(
            TwistStamped, parameter("base_velocity_topic"), 10
        )
        self.raw_pub = self.create_publisher(String, "~/state_json", 10)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.cmd_vel_topic = parameter("cmd_vel_topic")
        self.cmd_subscription = None
        self.create_service(SetBool, "~/command_enable", self.on_command_enable)
        self.actions = ControllerActions(self)
        self.create_timer(1 / rate, self.on_timer)
        self.create_timer(1.0, self.publish_diagnostics)
        self.receiver = StateReceiver(
            parameter("host"),
            parameter("observation_port"),
            self.model.model_id,
            request_timeout,
            rate,
            commands=self.commands,
            command_port=parameter("command_port"),
            request_window=parameter("request_window"),
        )
        self.get_logger().info(f"Host bridge; commands disabled; arm mappings: {directory}")

    def on_cmd_vel(self, message, input_epoch):
        if input_epoch != self.commands.input_epoch:
            return
        try:
            values = tuple(
                finite_number(value, "/cmd_vel")
                for value in (message.linear.x, message.linear.y, message.angular.z)
            )
            x, y, yaw = (
                max(-limit, min(limit, value)) * scale
                for value, limit, scale in zip(
                    values, self.velocity_limits, self.base_scales, strict=True
                )
            )
            if self.swap_xy:
                x, y = y, x
            velocity = BodyVelocity(x, y, yaw)
            self.commands.accept(velocity, input_epoch)
        except (TypeError, ValueError) as exc:
            self.commands.disable(f"Invalid /cmd_vel: {exc}", fault=True)
            self.get_logger().warning(f"Rejected /cmd_vel: {exc}")

    def on_command_enable(self, request, response):
        if request.data:
            if (
                self.last_observation_monotonic is None
                or time.monotonic() - self.last_observation_monotonic > self.obs_timeout
            ):
                response.success, response.message = (
                    False,
                    "Fresh valid ROS joint feedback is required",
                )
            else:
                response.success, response.message = self.commands.enable()
                if response.success:
                    if self.cmd_subscription is not None:
                        self.destroy_subscription(self.cmd_subscription)
                    # Humble callbacks receive no MessageInfo. A new volatile
                    # subscription discards the pre-enable queue; the captured
                    # epoch also rejects callbacks already taken by the executor.
                    epoch = self.commands.input_epoch
                    self.command_enabled_at_ns = self.get_clock().now().nanoseconds
                    self.cmd_subscription = self.create_subscription(
                        Twist,
                        self.cmd_vel_topic,
                        lambda message: self.on_cmd_vel(message, epoch),
                        1,
                    )
                    for subscription in self.jog_subscriptions:
                        self.destroy_subscription(subscription)
                    self.jog_subscriptions = [
                        self.create_subscription(
                            JointJog,
                            self.jog_topics[name],
                            lambda message, resource=name: self.on_jog(message, resource, epoch),
                            1,
                        )
                        for name in ("left_arm", "right_arm", "lift")
                    ]
        else:
            self.commands.disable()
            if self.cmd_subscription is not None:
                self.destroy_subscription(self.cmd_subscription)
                self.cmd_subscription = None
            for subscription in self.jog_subscriptions:
                self.destroy_subscription(subscription)
            self.jog_subscriptions.clear()
            response.success = True
            response.message = (
                "Commands disabled; any in-flight motion is followed by a stop request"
            )
        return response

    def on_jog(self, message, resource, epoch):
        if epoch != self.commands.input_epoch or not self.commands.status()[0]:
            return
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        age = self.get_clock().now().nanoseconds - stamp
        if stamp < self.command_enabled_at_ns or not 0 <= age <= int(
            self.commands.command_timeout * 1e9
        ):
            return
        try:
            if resource == "lift":
                if (
                    list(message.joint_names) != ["vertical_move"]
                    or len(message.velocities) != 1
                    or message.displacements
                ):
                    raise ValueError("Lift JointJog requires vertical_move and one velocity")
                velocity = finite_number(message.velocities[0], "lift jog velocity")
                self.commands.jog_lift(
                    max(-self.lift_jog_limit, min(self.lift_jog_limit, velocity)), epoch
                )
            else:
                if message.velocities:
                    raise ValueError("Arm JointJog uses displacements, not velocities")
                values = [
                    finite_number(value, "arm displacement") for value in message.displacements
                ]
                if any(abs(value) > self.arm_jog_limit + 1e-9 for value in values):
                    raise ValueError("Arm JointJog displacement exceeds the configured limit")
                values = [
                    max(-self.arm_jog_limit, min(self.arm_jog_limit, value)) for value in values
                ]
                self.commands.jog(
                    resource, message.joint_names, values, epoch, timeout=self.arm_jog_timeout
                )
        except (KeyError, TypeError, ValueError) as exc:
            self.get_logger().warning(f"Rejected {resource} JointJog: {exc}")

    def reject(self, reason):
        self.commands.fail(reason)
        self.mapper.reset()
        # Keep the sample watermark so stale replies cannot become fresh after rejection.
        self.last_observation_monotonic = None
        self.invalid_observations += 1
        if reason != self.last_observation_error:
            self.get_logger().warning(f"Host observation rejected: {reason}")
        self.last_observation_error = reason

    def discard_observation(self, reason):
        """Do not refresh state or cancel execution for an isolated stale reply."""
        if (
            self.last_observation_monotonic is None
            or time.monotonic() - self.last_observation_monotonic > self.obs_timeout
        ):
            self.reject(reason)

    def on_timer(self):
        result = self.receiver.take()
        if result is None:
            if (
                self.last_observation_monotonic is not None
                and time.monotonic() - self.last_observation_monotonic > self.obs_timeout
            ):
                self.reject("Host observation stale")
            return
        snapshot, generation, error = result
        if isinstance(error, ResponseTimeoutError):
            self.discard_observation(str(error))
            return
        if error:
            self.reject(error)
            return
        try:
            self.handle_observation(snapshot, generation)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self.reject(str(exc))

    def handle_observation(self, snapshot, generation=0):
        observation = snapshot.payload
        now = time.monotonic()
        metadata, safety, timing = (
            observation[key] for key in ("_robot_metadata", "_safety", "_host_timing")
        )
        if (
            metadata["robot_model"] != self.model.model_id
            or type(safety.get("version")) is not int
            or safety["version"] != 1
            or safety.get("feedback_valid") is not True
        ):
            raise ValueError("Host model mismatch or invalid feedback")
        session = safety.get("host_session_id")
        if not isinstance(session, str) or not session:
            raise ValueError("Host session is missing")
        started = finite_number(timing["state_sample_started_monotonic_s"], "state start")
        finished = finite_number(timing["state_sample_finished_monotonic_s"], "state finish")
        if not 0 <= started <= finished:
            raise ValueError("Invalid Host sample interval")
        host_now = finite_number(safety["sampled_at_monotonic_s"], "Host status time")
        if host_now < finished:
            raise ValueError("Invalid Host status time")
        if self._motor_metadata is not None and metadata["motors"] != self._motor_metadata:
            raise ValueError("Host motor calibration changed; verify mappings and restart bridge")
        if now - snapshot.request_started_s + host_now - finished > self.max_age:
            self.late_state_responses += 1
            self.discard_observation("Host feedback is stale")
            return
        stream = session, generation
        if stream != self._stream:
            self.mapper.reset()
            self.wheel_positions = [0.0, 0.0, 0.0]
            # Reconnection breaks continuity, not the Host's monotonic clock.
            if self._stream is None or session != self._stream[0]:
                self._last_host_sample = None
            self.last_observation_monotonic = None
            self._stream = stream
        sample = (started + finished) / 2
        if self._last_host_sample is not None:
            if sample <= self._last_host_sample:
                self.discard_observation("Host observation stale")
                return  # Repeated replies must not refresh old feedback's ROS stamp.
        if (
            self._last_host_sample is not None
            and sample - self._last_host_sample > self.obs_timeout
        ):
            self.mapper.reset()
        stamp = self.observation_stamp(observation)
        x = finite_number(observation["x.vel"], "x.vel")
        y = finite_number(observation["y.vel"], "y.vel")
        if self.swap_xy:
            x, y = y, x
        velocity = BodyVelocity(
            x / self.base_scales[0],
            y / self.base_scales[1],
            math.radians(finite_number(observation["theta.vel"], "theta.vel"))
            / self.base_scales[2],
        )
        positions = self.mapper.observation_to_joint_positions(observation, metadata)
        wheels = self.kinematics.body_to_wheels(velocity)
        dt = (
            0
            if self._last_host_sample is None or self.last_observation_monotonic is None
            else sample - self._last_host_sample
        )
        if 0 < dt <= self.obs_timeout:
            self.wheel_positions = [
                p + v * dt for p, v in zip(self.wheel_positions, wheels, strict=True)
            ]
        self._last_host_sample = sample
        self._motor_metadata = metadata["motors"]
        self.last_observation_monotonic = snapshot.received_s
        self.last_state_response_age_ms = (snapshot.received_s - snapshot.request_started_s) * 1000
        self._host_model = metadata["robot_model"]
        self._host_joint_holds = tuple(safety.get("joint_holds", {}))
        self._lift_command_ready = safety.get("lift_reference_valid") is True and safety.get(
            "phase"
        ) in ("ready", "active")
        self.last_observation_error = ""
        self.observation_count += 1
        measured = TwistStamped()
        measured.header.stamp, measured.header.frame_id = stamp, self.base_frame
        measured.twist.linear.x, measured.twist.linear.y = velocity.x_m_s, velocity.y_m_s
        measured.twist.angular.z = velocity.yaw_rad_s
        self.base_velocity_pub.publish(measured)
        self.raw_pub.publish(String(data=json.dumps(observation, separators=(",", ":"))))
        message = JointState()
        message.header.stamp = stamp
        message.name, message.position = list(positions), list(positions.values())
        self.joint_pub.publish(message)
        self.measured_joint_pub.publish(message)
        derived = JointState()
        derived.header.stamp = stamp
        # Virtual root and integrated wheel angles are not measured odometry.
        derived.name = [
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_rotation_joint",
            "wheel1_joint",
            "wheel2_joint",
            "wheel3_joint",
        ]
        derived.position = [0.0, 0.0, 0.0, *self.wheel_positions]
        derived.velocity = [0.0, 0.0, 0.0, *wheels]
        self.joint_pub.publish(derived)
        self.derived_wheel_pub.publish(derived)

    def observation_stamp(self, observation):
        if self.state_timestamp_mode == "receipt":
            self.last_state_clock_offset_ms = math.nan
            return self.get_clock().now().to_msg()
        unix_ns = observation["_host_timing"].get("state_sample_unix_ns")
        if type(unix_ns) is not int or not 0 < unix_ns < 2**31 * 1_000_000_000:
            raise ValueError("host_wall requires a valid state_sample_unix_ns")
        self.last_state_clock_offset_ms = (time.time_ns() - unix_ns) / 1e6
        return TimeMsg(sec=unix_ns // 1_000_000_000, nanosec=unix_ns % 1_000_000_000)

    def publish_diagnostics(self):
        age = (
            math.inf
            if self.last_observation_monotonic is None
            else time.monotonic() - self.last_observation_monotonic
        )
        status = DiagnosticStatus(
            name="AlohaMini Host state bridge", hardware_id=self.model.model_id
        )
        status.level = DiagnosticStatus.OK
        control = self.commands.diagnostics()
        status.message = "State bridge healthy; commands disabled"
        if age > self.obs_timeout:
            status.level = DiagnosticStatus.ERROR
            status.message = self.last_observation_error or "Host observation stale or unavailable"
        elif control["command_fault"]:
            status.level = DiagnosticStatus.ERROR
            status.message = control["command_fault"]
        elif self._host_joint_holds:
            status.level = DiagnosticStatus.ERROR
            status.message = "Host joint protection: " + ", ".join(self._host_joint_holds)
        elif abs(self.last_state_clock_offset_ms) > self.max_clock_offset_ms:
            status.level = DiagnosticStatus.WARN
            status.message = "Host/ROS clock offset or transport latency too large"
        elif not self._lift_command_ready:
            status.level = DiagnosticStatus.WARN
            status.message = "Lift is not command-ready"
        elif control["command_enabled"] or control["stop_pending"]:
            status.level = DiagnosticStatus.WARN
            status.message = (
                "ROS command channel enabled"
                if control["command_enabled"]
                else "Stop request pending"
            )
        status.values = [
            KeyValue(key="observation_age_sec", value=f"{age:.3f}"),
            KeyValue(key="observation_count", value=str(self.observation_count)),
            KeyValue(key="invalid_observations", value=str(self.invalid_observations)),
            KeyValue(key="host_robot_model", value=self._host_model),
            KeyValue(key="host_joint_holds", value=",".join(self._host_joint_holds)),
            KeyValue(
                key="lift_command_ready",
                value=str(self._lift_command_ready and age <= self.obs_timeout).lower(),
            ),
            KeyValue(key="last_observation_error", value=self.last_observation_error),
            KeyValue(key="state_timestamp_mode", value=self.state_timestamp_mode),
            KeyValue(key="state_clock_offset_ms", value=f"{self.last_state_clock_offset_ms:.3f}"),
            KeyValue(
                key="last_state_response_age_ms", value=f"{self.last_state_response_age_ms:.3f}"
            ),
            KeyValue(key="late_state_responses", value=str(self.late_state_responses)),
            *[
                KeyValue(
                    key=key, value=str(value).lower() if isinstance(value, bool) else str(value)
                )
                for key, value in control.items()
            ],
        ]
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [status]
        self.diagnostics_pub.publish(message)

    def destroy_node(self):
        if getattr(self, "actions", None) is not None:
            self.actions.close()
            self.actions = None
        if getattr(self, "receiver", None) is not None:
            self.receiver.close()
            self.receiver = None
        return super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = AlohaMiniBridge()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
