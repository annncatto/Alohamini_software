# SPDX-License-Identifier: Apache-2.0
# Adapted from AlohaMini alohamini_host.py and camera_buffer.py.
"""50 Hz Host with bounded ZMQ work and pre-encoded camera caches."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict

from alohamini.calibration.encoder import HostPositionUnits
from alohamini.calibration.servo import MotorCalibration
from alohamini.errors import ProtocolError
from alohamini.kinematics import OmniBaseKinematics
from alohamini.model import RobotModel
from alohamini.protocol import command_target_keys, decode_command, encode_reply
from alohamini.runtime.arm_contact import ArmJointSpec
from alohamini.runtime.arm_control import ArmController
from alohamini.runtime.base_lift_control import BaseLiftController
from alohamini.runtime.camera_stream import CameraStreamPublisher
from alohamini.runtime.lifecycle import CycleResult, HostPhase, HostSupervisor
from alohamini.runtime.lift_control import LiftAxisSpec
from alohamini.runtime.robot_control import RobotController
from alohamini.schema import BodyVelocity, RobotCommand

logger = logging.getLogger(__name__)


class RecordingCameraBuffer:
    """Deployed per-recording pairing over bounded, nonblocking JPEG histories.

    Each camera supplies read_frame_history(): (Host monotonic timestamp, JPEG).
    Camera capture and JPEG encoding must happen outside the control thread.
    Timestamps describe camera read completion, not a hardware exposure clock.
    """

    MAX_SESSIONS = 8
    MAX_AGE_S = 0.1
    MAX_SKEW_S = 1 / 30

    def __init__(self) -> None:
        self._cursors: OrderedDict[tuple[bytes, bytes], dict[str, float]] = OrderedDict()

    def select(self, histories, identity: bytes, token: bytes, *, now: float) -> dict:
        parts = token.split(b":")
        if len(parts) != 3 or parts[-1] != b"record" or not 0 < len(parts[1]) <= 64:
            raise ProtocolError("Invalid recording request token")
        key = (identity, parts[1])
        if key not in self._cursors:
            self._cursors[key] = dict.fromkeys(histories, now)
        self._cursors.move_to_end(key)
        while len(self._cursors) > self.MAX_SESSIONS:
            self._cursors.popitem(last=False)
        cursor = self._cursors[key]
        queues = {
            name: deque(
                (stamp, jpeg)
                for stamp, jpeg in history
                if cursor.get(name, now) < stamp <= now and now - stamp <= self.MAX_AGE_S
            )
            for name, history in histories.items()
        }
        while queues and all(queues.values()):
            anchor_name = next(iter(queues))
            anchor = queues[anchor_name][0][0]
            if any(queue[-1][0] < anchor for queue in queues.values()):
                break
            candidates = {
                name: min(queue, key=lambda item: abs(item[0] - anchor))
                for name, queue in queues.items()
            }
            stamps = [item[0] for item in candidates.values()]
            if max(stamps) - min(stamps) <= self.MAX_SKEW_S:
                cursor.update({name: pair[0] for name, pair in candidates.items()})
                return candidates
            queues[anchor_name].popleft()
        return {}


class NativeHost:
    """Own configured devices and serve the deployed command/state protocol.

    Low-level start() opens and inspects existing hardware settings, without calibration
    writes, torque enable or homing. A verified lift reference is a separate local
    task; unknown height is never reported as zero. This is not a hard-real-time
    scheduler or an authentication boundary: bind only on a trusted robot network.
    runtime.startup.open_host supplies the complete calibrated startup sequence.

    Cameras are owned cache producers with start/close/read_frame_history methods.
    read_frame_history must return at most eight timestamp/JPEG pairs immediately,
    without device I/O, waiting for frames, or encoding work.
    """

    PERIOD_S = 1 / 50

    def __init__(
        self,
        model: RobotModel,
        devices: Mapping,
        joints: Sequence[ArmJointSpec],
        lift: LiftAxisSpec,
        position_units: Mapping[str, HostPositionUnits],
        *,
        cameras: Mapping | None = None,
        bind_host: str = "127.0.0.1",
        command_port: int = 5555,
        state_port: int = 5556,
        camera_port: int = 5557,
        motor_calibrations: Mapping[str, MotorCalibration] | None = None,
    ) -> None:
        if not isinstance(model, RobotModel):
            raise TypeError("Expected RobotModel")
        if (
            not isinstance(bind_host, str)
            or not bind_host
            or any(c.isspace() or c in "/:[]" for c in bind_host)
        ):
            raise ValueError("Provide a bind IPv4 address or hostname")
        for port in (command_port, state_port, camera_port):
            if type(port) is not int or not 1 <= port <= 65535:
                raise ValueError("Ports must be integers in [1, 65535]")
        if command_port == state_port:
            raise ValueError("Command and state ports must differ")
        if cameras and camera_port in (command_port, state_port):
            raise ValueError("Camera, command and state ports must differ")
        actual = [motor for device in devices.values() for motor in device.actuators]
        if len(actual) != len(model.actuators) or set(actual) != set(model.actuators):
            raise ValueError("Devices must match the complete configured robot model")
        self._joints = {spec.actuator.name: spec for spec in joints}
        arm_names = {m.name for m in model.actuators if m.name.startswith("arm_")}
        if set(self._joints) != arm_names or set(position_units) != arm_names:
            raise ValueError("Provide physical calibration and wire units for every arm joint")
        for name, units in position_units.items():
            if not isinstance(units, HostPositionUnits):
                raise TypeError("Expected HostPositionUnits")
            calibration = self._joints[name].calibration
            # Both representations must describe targets accepted by this device.
            for tick in (units.range_min, units.range_max):
                calibration.position_to_tick(calibration.position_from_tick(tick))
        by_name = {motor.name: motor for motor in model.actuators}
        if (
            lift.actuator != by_name["lift_axis"]
            or lift.lead_m_per_revolution != model.lift_lead_m_per_rev
        ):
            raise ValueError("Lift specification does not match the robot model")
        self.model = model
        self._units = dict(position_units)
        self._keys = command_target_keys(model.model_id)
        self.control = RobotController(
            ArmController(devices, joints),
            BaseLiftController(
                devices,
                wheels=tuple(by_name[f"base_{side}_wheel"] for side in ("left", "back", "right")),
                kinematics=OmniBaseKinematics(model.wheel_radius_m, model.base_radius_m),
                lift=lift,
            ),
        )
        self.supervisor = HostSupervisor(devices, model.actuators, control=self.control)
        self._cameras = dict(cameras or {})
        if len(self._cameras) > 16 or any(
            not isinstance(name, str) or not name or len(name) > 64 or name.startswith("_")
            for name in self._cameras
        ):
            raise ValueError("Provide at most 16 named cameras")
        self._camera_buffer = RecordingCameraBuffer()
        self._camera_stream = (
            CameraStreamPublisher(
                self._cameras,
                port=camera_port,
                bind_host=bind_host,
                host_session_id=self.supervisor.status.host_session_id,
            )
            if self._cameras
            else None
        )
        self._attempted_cameras = []
        self._context = self._commands = self._states = None
        self._endpoints = (f"tcp://{bind_host}:{command_port}", f"tcp://{bind_host}:{state_port}")
        self._used = self._started = self._closed = False
        self._thread = threading.get_ident()
        self._last_identity = None
        self._accepted_at = None
        self._requested = {}
        self._last_error = None
        self._rejected = self._dropped_responses = self._overruns = 0
        self._pending_responses = deque()
        self._sequence = 0
        self.timing_ms: dict[str, float] = {}
        self._metadata = {
            "schema_version": 1,
            "robot_model": model.model_id,
            "cameras": list(self._cameras),
            "arm_profile": {"arm_goal_velocity": 2000, "arm_acceleration": 100},
            "lift_axis": {
                "soft_min_mm": 0.0,
                "soft_max_mm": 600.0,
                "descent_floor_mm": 5.0,
                "ticks_per_revolution": 4096,
                "lead_mm_per_revolution": lift.lead_m_per_revolution * 1000,
                "direction_sign": lift.direction,
            },
            "motors": {
                name: {
                    "id": self._joints[name].actuator.motor_id,
                    "model": self._joints[name].actuator.motor_model,
                    **asdict(units),
                }
                for name, units in self._units.items()
            },
        }
        if motor_calibrations is not None:
            if motor_calibrations.keys() != by_name.keys():
                raise ValueError("Metadata calibration must cover the complete robot")
            for name, motor in by_name.items():
                calibration = motor_calibrations[name]
                if (
                    not isinstance(calibration, MotorCalibration)
                    or calibration.id != motor.motor_id
                ):
                    raise ValueError(f"Metadata calibration identity mismatch: {name}")
                normalization = (
                    self._units[name].normalization
                    if name in self._units
                    else "degrees"
                    if name == "lift_axis"
                    else "range_m100_100"
                )
                units = calibration.position_units(normalization)
                if name in self._units and units != self._units[name]:
                    raise ValueError(f"Metadata calibration range/direction mismatch: {name}")
                self._metadata["motors"][name] = {
                    "id": motor.motor_id,
                    "model": motor.motor_model,
                    **asdict(units),
                    "homing_offset": calibration.homing_offset,
                }
        self._target_source = "startup"

    def _check_thread(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("Host must run on its owning thread")

    def begin_lift_homing(self) -> None:
        """Schedule local startup homing; run/step keeps servicing feedback and ZMQ.

        Devices must already be configured and enabled by the local startup flow.
        This is not exposed as a remote command. Clear the mechanical descent path
        before calling; current contact cannot identify a physical limit switch.
        """
        self._check_thread()
        status = self.supervisor.status
        if (
            not self._started
            or self._closed
            or status.phase not in (HostPhase.STARTING, HostPhase.READY)
            or status.control_owner is not None
        ):
            raise RuntimeError("Lift homing requires an idle local Host startup")
        self.control.begin_lift_homing()

    def __enter__(self) -> NativeHost:
        self.start()
        return self

    def __exit__(self, _exc_type, exc, _traceback) -> None:
        if exc is None:
            self.close()
        else:
            self._close_after_error(exc)

    def _close_after_error(self, primary: BaseException) -> None:
        """Report cleanup failures without replacing the operation that failed first."""
        try:
            self.close()
        except BaseException as cleanup:
            if not isinstance(primary, Exception):
                primary.add_note(f"Host cleanup failed: {cleanup}")
                logger.error("Host cleanup failed: %s", cleanup)
            elif not isinstance(cleanup, Exception):
                cleanup.add_note(f"Host failed: {type(primary).__name__}: {primary}")
                raise
            else:
                raise RuntimeError(
                    f"Host failed: {type(primary).__name__}: {primary}; cleanup failed: {cleanup}"
                ) from primary

    def start(self) -> None:
        self._check_thread()
        if self._used or self._closed:
            raise RuntimeError("Create a new Host instance for each session")
        self._used = True
        import zmq

        try:
            self._context = zmq.Context()
            self._commands = self._context.socket(zmq.PULL)
            self._states = self._context.socket(zmq.ROUTER)
            for socket in (self._commands, self._states):
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.MAXMSGSIZE, 8192)
            self._commands.setsockopt(zmq.CONFLATE, 1)
            self._states.setsockopt(zmq.SNDHWM, 3)
            self._states.setsockopt(zmq.RCVHWM, 3)
            self._states.setsockopt(zmq.ROUTER_MANDATORY, 1)
            self._commands.bind(self._endpoints[0])
            self._states.bind(self._endpoints[1])
            # Bind before opening hardware, so occupied ports cannot leave motors running.
            if self._camera_stream is not None:
                self._camera_stream.start()
            for camera in self._cameras.values():
                self._attempted_cameras.append(camera)
                camera.start()
            self.supervisor.start()
            self._started = True
        except BaseException as exc:
            self._close_after_error(exc)
            raise

    def close(self) -> None:
        self._check_thread()
        if self._closed:
            return
        self._closed = True
        errors = []
        try:
            # Mechanical cleanup precedes camera or network cleanup.
            status = self.supervisor.close()
            if status.cleanup_failures:
                errors.append(
                    RuntimeError(
                        "; ".join(
                            f"{failure.source_id}.{failure.operation}: {failure.error}"
                            for failure in status.cleanup_failures
                        )
                    )
                )
        except BaseException as exc:
            errors.append(exc)
        if self._camera_stream is not None:
            try:
                self._camera_stream.close()
            except BaseException as exc:
                errors.append(exc)
        for camera in self._attempted_cameras:
            try:
                camera.close()
            except BaseException as exc:
                errors.append(exc)
        for socket in (self._commands, self._states):
            if socket is not None:
                try:
                    socket.close(linger=0)
                except BaseException as exc:
                    errors.append(exc)
        if self._context is not None:
            try:
                self._context.term()
            except BaseException as exc:
                errors.append(exc)
        self._pending_responses.clear()
        # Preserve interrupts, but only after every resource has been attempted.
        for error in errors:
            if not isinstance(error, Exception):
                raise error
        if errors:
            raise RuntimeError("Host cleanup failed: " + "; ".join(map(str, errors))) from errors[0]

    @staticmethod
    def _receive(socket, *, max_frames: int) -> list[bytes] | None:
        import zmq

        try:
            first = socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        parts = [first]
        while socket.getsockopt(zmq.RCVMORE):
            if len(parts) >= max_frames:
                # Cannot drain an unbounded multipart message in the control thread.
                # Fail the session and stop devices instead of blocking supervision.
                raise ProtocolError("Inbound multipart frame count exceeds limit")
            parts.append(socket.recv(flags=zmq.NOBLOCK))
        return parts

    def _submission(self, data: bytes):
        identity, targets = decode_command(data, allowed_targets=self._keys)
        positions = {}
        for name, spec in self._joints.items():
            key = f"{name}.pos"
            if key in targets:
                tick = self._units[name].to_tick(targets[key])
                positions[name] = spec.calibration.position_from_tick(tick)
        # Deployed send_action zeros omitted base axes on every new wire command.
        # This differs intentionally from RobotCommand's explicit partial-target API.
        command = RobotCommand(
            positions,
            BodyVelocity(
                targets.get("x.vel", 0),
                targets.get("y.vel", 0),
                math.radians(targets.get("theta.vel", 0)),
            ),
            targets["lift_axis.height_mm"] / 1000 if "lift_axis.height_mm" in targets else None,
            lift_stop="lift_axis.stop" in targets,
        )
        return self.control.submission(identity, command), targets

    def _wire_positions(self, values: Mapping[str, float]) -> dict[str, float]:
        return {
            f"{name}.pos": self._units[name].from_tick(
                self._joints[name].calibration.position_to_tick(value)
            )
            for name, value in values.items()
        }

    def _payload(self, result: CycleResult) -> dict:
        status = result.status
        feedback_valid = status.phase in (HostPhase.READY, HostPhase.ACTIVE)
        holds = {
            name.removesuffix(".pos"): value
            for name, value in self._wire_positions(self.control.arms.contact_holds).items()
        }
        motors, positions, currents = {}, {}, {}
        for batch in result.feedback:
            for name, sample in batch.samples.items():
                motors[name] = {
                    **sample.registers,
                    "packet_error": sample.packet_error,
                    "sample_started_s": batch.request_started_s,
                    "sample_finished_s": batch.received_s,
                    "field_errors": dict(sample.field_errors),
                }
                if sample.current_a is not None:
                    currents[name] = sample.current_a * 1000
                    motors[name]["current_ma"] = currents[name]
                if name in self._units and "position_raw" in sample.registers:
                    tick = sample.registers["position_raw"]
                    if 0 <= tick <= 4095:
                        positions[f"{name}.pos"] = self._units[name].from_tick(tick)
        body = self.control.base_lift.measured_base_velocity
        if feedback_valid and body is not None:
            positions.update(
                {
                    "x.vel": body.x_m_s,
                    "y.vel": body.y_m_s,
                    "theta.vel": math.degrees(body.yaw_rad_s),
                }
            )
        height = self.control.base_lift.lift_height_m if feedback_valid else None
        if height is not None:
            positions["lift_axis.height_mm"] = height * 1000
        # Serial replies may arrive in any order; the public state order must not.
        state_keys = [f"{m.name}.pos" for m in self.model.actuators if m.name.startswith("arm_")]
        state_keys += ["x.vel", "y.vel", "theta.vel", "lift_axis.height_mm"]
        positions = {key: positions[key] for key in state_keys if key in positions}
        motors = {m.name: motors[m.name] for m in self.model.actuators if m.name in motors}
        started = min((b.request_started_s for b in result.feedback), default=None)
        finished = max((b.received_s for b in result.feedback), default=None)
        clock_reference = time.monotonic()
        unix_reference = time.time_ns()
        return {
            **positions,
            **{
                f"lift_axis.{key}": value
                for key, value in (
                    self.control.base_lift.lift_calibration_feedback.items()
                    if feedback_valid
                    else {"homed": False}.items()
                )
            },
            "_robot_metadata": self._metadata,
            "_motor_feedback": {"version": 1, "motors": motors},
            "_host_timing": {
                "state_sequence": self._sequence,
                "state_sample_monotonic_s": finished,
                "state_sample_started_monotonic_s": started,
                "state_sample_finished_monotonic_s": finished,
                "state_sample_unix_ns": (
                    unix_reference - round((clock_reference - (started + finished) / 2) * 1e9)
                    if started is not None
                    else None
                ),
                "host_clock_reference": {"monotonic_s": clock_reference, "unix_ns": unix_reference},
                "camera_capture_monotonic_s": {},
                "clock_id": status.host_session_id,
            },
            "_safety": {
                "version": 1,
                "sampled_at_monotonic_s": time.monotonic(),
                "phase": status.phase.value,
                "fault": status.fault,
                "feedback_valid": feedback_valid,
                "host_session_id": status.host_session_id,
                "control_epoch": status.control_epoch,
                "control_owner": status.control_owner,
                "watchdog_events": status.watchdog_events,
                "watchdog_active": status.watchdog_events > 0 and status.control_owner is None,
                "command_watchdog_timeout_s": self.supervisor.COMMAND_WATCHDOG_TIMEOUT_S,
                "command": asdict(self._last_identity) if self._last_identity else {},
                "target_source": self._target_source,
                "requested_targets": dict(self._requested),
                "accepted_targets": self._wire_positions(self.control.arms.sent_targets),
                "accepted_at_monotonic_s": self._accepted_at,
                "joint_holds": {
                    name: value for name, value in holds.items() if not name.endswith("gripper")
                },
                "gripper_holds": {
                    name: value for name, value in holds.items() if name.endswith("gripper")
                },
                "joint_hold_events": self.control.arms.joint_hold_events,
                "joint_stall_currents_a": self.control.arms.joint_stall_currents_a,
                "currents_ma": currents,
                "lift_reference_valid": height is not None,
                "lift_homing_phase": self.control.base_lift.lift_homing_phase,
                "lift_reference_source": (
                    (
                        "current_contact"
                        if self.control.base_lift.lift_homing_phase == "complete"
                        else "local_reference"
                    )
                    if height is not None
                    else None
                ),
                "rejected_commands": self._rejected,
                "last_command_error": self._last_error,
            },
        }

    def _camera_images(self, payload: dict, identity: bytes, token: bytes) -> dict:
        if token.endswith(b":state"):
            return {}
        histories, errors = {}, {}
        now = time.monotonic()
        for name, camera in self._cameras.items():
            camera_started = time.perf_counter()
            try:
                history = tuple(camera.read_frame_history())
                if len(history) > 8:
                    raise ValueError("Camera history exceeds eight frames")
                previous = -math.inf
                for stamp, jpeg in history:
                    if (
                        isinstance(stamp, bool)
                        or not isinstance(stamp, (int, float))
                        or not math.isfinite(stamp)
                        or not previous < stamp <= now
                        or not isinstance(jpeg, bytes)
                        or not 0 < len(jpeg) <= 8 * 1024 * 1024
                    ):
                        raise ValueError("Invalid timestamp/JPEG history")
                    previous = stamp
                histories[name] = history
            except Exception as exc:
                histories[name] = ()
                errors[name] = str(exc)
            finally:
                self.timing_ms[f"camera_cache_{name}"] = (
                    time.perf_counter() - camera_started
                ) * 1e3
        if token.endswith(b":record"):
            selected = self._camera_buffer.select(histories, identity, token, now=now)
            payload["_camera_buffer"] = {"version": 1, "pending": bool(histories) and not selected}
        else:
            selected = {
                name: history[-1]
                for name, history in histories.items()
                if history and now - history[-1][0] <= 0.5
            }
        payload["_host_timing"]["camera_capture_monotonic_s"] = {
            name: pair[0] for name, pair in selected.items()
        }
        payload["_camera_status"] = {
            "unavailable": [
                name
                for name in histories
                if not histories[name] or now - histories[name][-1][0] > 0.5
            ],
            "errors": errors,
        }
        return {name: pair[1] for name, pair in selected.items()}

    def step(self) -> CycleResult:
        """One control cycle: at most one command and one observation request."""
        self._check_thread()
        if not self._started or self._closed:
            raise RuntimeError("Host must be started")

        self.timing_ms = {}
        try:
            request_started = time.perf_counter()
            request = (
                self._receive(self._states, max_frames=2)
                if len(self._pending_responses) < 8
                else None
            )
            request_done = time.perf_counter()
            submission, targets = None, {}

            def poll_command():
                nonlocal submission, targets
                command_started = time.perf_counter()
                message = self._receive(self._commands, max_frames=1)
                if message is not None:
                    try:
                        submission, targets = self._submission(message[0])
                    except (ProtocolError, ValueError) as exc:
                        self._rejected += 1
                        self._last_error = str(exc)
                self.timing_ms["command"] = (time.perf_counter() - command_started) * 1e3
                return submission

            self.timing_ms["request_poll"] = (request_done - request_started) * 1e3
            before = self.control.arms.sent_targets
            watchdog_events = self.supervisor.status.watchdog_events
            result = self.supervisor.cycle(poll_command=poll_command)
            if result.status.watchdog_events > watchdog_events:
                logger.warning("Host command watchdog expired; stopping motion")
            self.timing_ms.update(self.supervisor.timing_ms)
            if self.control.arms.sent_targets and (
                result.command_applied or before != self.control.arms.sent_targets
            ):
                self._accepted_at = time.monotonic()
                self._target_source = "command" if result.command_applied else "protection"
            self._sequence += 1
            if (
                self._last_identity
                and self._last_identity.control_epoch != result.status.control_epoch
            ):
                self._last_identity = None
                self._requested.clear()
                self._accepted_at = None
                self._target_source = "watchdog"
            if result.command_applied:
                self._target_source = "command"
                self._last_identity = submission.identity
                self._requested.update(targets)
                self._requested.update(
                    {key: targets.get(key, 0.0) for key in ("x.vel", "y.vel", "theta.vel")}
                )
                self._last_error = None
            elif submission is not None:
                self._rejected += 1
                self._last_error = result.command_error or "Command identity or lease rejected"
            if request is not None:
                if len(request) == 2 and 0 < len(request[1]) <= 256:
                    identity, token = request
                    payload = self._payload(result)
                    try:
                        images = self._camera_images(payload, identity, token)
                    except ProtocolError:
                        images = {}
                        payload["_camera_status"] = {"error": "Invalid recording request"}
                    pack_started = time.perf_counter()
                    try:
                        reply = encode_reply(payload, images)
                    except ProtocolError as exc:
                        if not images:
                            raise
                        # A large image group must not interrupt motor supervision.
                        # State serialization errors remain fatal if this also fails.
                        payload["_host_timing"]["camera_capture_monotonic_s"] = {}
                        payload["_camera_status"] = {
                            "unavailable": list(self._cameras),
                            "error": str(exc),
                        }
                        if token.endswith(b":record"):
                            payload["_camera_buffer"] = {"version": 1, "pending": True}
                        reply = encode_reply(payload, {})
                    self.timing_ms["response_pack"] = (time.perf_counter() - pack_started) * 1e3
                    self._pending_responses.append([identity, token, *reply])
            send_started = time.perf_counter()
            self._flush_responses()
            self.timing_ms["response_send"] = (time.perf_counter() - send_started) * 1e3
            return result
        except BaseException as exc:
            self._close_after_error(exc)
            raise

    def _flush_responses(self) -> None:
        """Retry bounded replies without blocking control or other clients."""
        import zmq

        for _ in range(len(self._pending_responses)):
            parts = self._pending_responses.popleft()
            try:
                self._states.send_multipart(parts, flags=zmq.NOBLOCK)
            except zmq.ZMQError as exc:
                if exc.errno == zmq.EAGAIN:
                    self._pending_responses.append(parts)
                elif exc.errno == zmq.EHOSTUNREACH:
                    self._dropped_responses += 1
                else:
                    raise

    def run(
        self, stop_event: threading.Event | None = None, *, profile_timing: bool = False
    ) -> None:
        """Run at 50 Hz without catch-up bursts; faults and exceptions end the session."""
        stop = stop_event if stop_event is not None else threading.Event()
        try:
            if not self._started:
                self.start()
            deadline = time.monotonic()
            timing_report_start_t = time.perf_counter()
            timing_loop_count = timing_command_count = 0
            timing_totals_ms: dict[str, float] = {}
            action_timing_totals_ms: dict[str, float] = {}
            camera_report_stats = {}
            homing_phase = self.control.base_lift.lift_homing_phase
            if homing_phase is None:
                logger.info("Waiting for commands...")
            while not stop.is_set():
                loop_start_t = time.perf_counter()
                result = self.step()
                if result.status.phase is HostPhase.FAULT:
                    raise RuntimeError(result.status.fault)
                phase = self.control.base_lift.lift_homing_phase
                if phase != homing_phase:
                    if phase == "complete":
                        logger.info("Lift axis homed to 0mm.")
                        logger.info("Waiting for commands...")
                    homing_phase = phase
                now = time.monotonic()
                deadline += self.PERIOD_S
                if now > deadline:
                    missed = math.floor((now - deadline) / self.PERIOD_S) + 1
                    self._overruns += missed
                    deadline += missed * self.PERIOD_S
                sleep_started = time.perf_counter()
                stop.wait(max(0.0, deadline - time.monotonic()))
                loop_done_t = time.perf_counter()
                if not profile_timing:
                    continue
                # Accumulation/report cadence retained from alohamini_host.py.
                loop_timings_ms = {
                    **self.timing_ms,
                    "sleep": (loop_done_t - sleep_started) * 1e3,
                    "loop": (loop_done_t - loop_start_t) * 1e3,
                }
                for name, value_ms in loop_timings_ms.items():
                    timing_totals_ms[name] = timing_totals_ms.get(name, 0.0) + value_ms
                timing_loop_count += 1
                if result.command_applied:
                    action_timings = {
                        **self.control.timing_ms,
                        "total": self.timing_ms["robot_action"],
                    }
                    for name, value_ms in action_timings.items():
                        action_timing_totals_ms[name] = (
                            action_timing_totals_ms.get(name, 0.0) + value_ms
                        )
                    timing_command_count += 1
                timing_elapsed_s = loop_done_t - timing_report_start_t
                if timing_elapsed_s >= 1.0:
                    averages = {
                        name: total / timing_loop_count for name, total in timing_totals_ms.items()
                    }
                    image_text = " ".join(
                        f"{name}={value:.1f}"
                        for name, value in averages.items()
                        if name.startswith("camera_cache_")
                    )
                    # A bus read now includes positions, base/lift and currents;
                    # reporting separate old read costs would double-count it.
                    print(
                        f"[HOST TIMING avg ms/loop] Hz={timing_loop_count / timing_elapsed_s:.1f} "
                        f"cmd={averages.get('command', 0.0):.1f} "
                        f"robot_obs={averages.get('robot_observation', 0.0):.1f} "
                        f"robot_action={averages.get('robot_action', 0.0):.1f} "
                        f"left_bus={averages.get('left_bus', 0.0):.1f} "
                        f"right_bus={averages.get('right_bus', 0.0):.1f} {image_text} "
                        f"pack={averages.get('response_pack', 0.0):.1f} "
                        f"send={averages.get('response_send', 0.0):.1f} "
                        f"sleep={averages['sleep']:.1f} loop={averages['loop']:.1f} "
                        f"overruns={self._overruns} dropped={self._dropped_responses} "
                        f"rejected={self._rejected}",
                        flush=True,
                    )
                    if timing_command_count:
                        action_averages = {
                            name: total / timing_command_count
                            for name, total in action_timing_totals_ms.items()
                        }
                        action_text = " ".join(
                            f"{name}={value:.1f}" for name, value in action_averages.items()
                        )
                        print(
                            f"[HOST ACTION avg ms/control-cycle] n={timing_command_count} "
                            f"{action_text}",
                            flush=True,
                        )
                    # JPEG work is asynchronous: report per-frame worker cost,
                    # never charge it to the Host thread's loop latency.
                    for name, camera in self._cameras.items():
                        if not hasattr(camera, "timing_stats"):
                            continue
                        stats = camera.timing_stats()
                        previous = camera_report_stats.get(name, dict.fromkeys(stats, 0))
                        count = stats["frames"] - previous["frames"]
                        if count > 0:
                            capture = (stats["capture_ms"] - previous["capture_ms"]) / count
                            encode = (stats["encode_ms"] - previous["encode_ms"]) / count
                            print(
                                f"[HOST CAMERA avg ms/frame][{name}] n={count} "
                                f"capture={capture:.1f} encode={encode:.1f}",
                                flush=True,
                            )
                        stream_count = stats.get("stream_frames", 0) - previous.get(
                            "stream_frames", 0
                        )
                        if stream_count > 0:
                            stream_encode = (
                                stats["stream_encode_ms"] - previous.get("stream_encode_ms", 0)
                            ) / stream_count
                            print(
                                f"[HOST CAMERA STREAM avg ms/frame][{name}] n={stream_count} "
                                f"encode={stream_encode:.1f}",
                                flush=True,
                            )
                        camera_report_stats[name] = stats
                    timing_report_start_t = loop_done_t
                    timing_loop_count = timing_command_count = 0
                    timing_totals_ms.clear()
                    action_timing_totals_ms.clear()
        except BaseException as exc:
            self._close_after_error(exc)
            raise
        else:
            self.close()
