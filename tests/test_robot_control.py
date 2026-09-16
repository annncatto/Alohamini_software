import unittest
from unittest.mock import patch

from test_arm_control import joint_spec

from alohamini.hardware.feedback import FeedbackBatch, MotorFeedback
from alohamini.kinematics import OmniBaseKinematics
from alohamini.model import ActuatorSpec
from alohamini.runtime.arm_control import ArmController
from alohamini.runtime.base_lift_control import BaseLiftController
from alohamini.runtime.lifecycle import CommandSubmission, HostPhase, HostSupervisor
from alohamini.runtime.lift_control import LiftAxisSpec
from alohamini.runtime.robot_control import RobotController
from alohamini.schema import BodyVelocity, CommandIdentity, RobotCommand


class MemoryDevice:
    def __init__(self, fixture, source, actuators, positions):
        self.fixture, self.source = fixture, source
        self.actuators = tuple(actuators)
        self.position_calibrations = positions
        self.velocity_limits = {
            m.name: 1300 if m.name == "lift" else 3000 for m in actuators if m.name not in positions
        }
        self.sequence = 0
        self.fail_write = False
        self.fail_feedback = False
        self.current = 0.1

    def connect(self, session):
        self.session = session

    def read_feedback(self):
        started = self.fixture.now
        self.fixture.now += 0.001
        self.sequence += 1
        samples = {
            motor.name: MotorFeedback(
                {"position_raw": 2048, "velocity_raw": 0, "moving": 0},
                current_a=self.current,
            )
            for motor in self.actuators
        }
        return FeedbackBatch(
            self.source,
            self.session,
            self.sequence,
            started,
            self.fixture.now,
            samples,
            {"lift": "unavailable"} if self.fail_feedback else {},
        )

    def write_targets(self, *, positions_rad, velocities_raw):
        if self.fail_write:
            raise OSError("write failed")
        self.fixture.events.append(
            (self.source, "write", dict(positions_rad), dict(velocities_raw))
        )

    def stop(self):
        self.fixture.events.append((self.source, "stop"))

    def disable_torque(self):
        self.fixture.events.append((self.source, "disable"))

    def close(self):
        self.fixture.events.append((self.source, "close"))


class RobotControlTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.events = []
        timer = patch("alohamini.runtime.lifecycle.time.monotonic", side_effect=lambda: self.now)
        timer.start()
        self.addCleanup(timer.stop)
        joints = [joint_spec(f"{side}_joint", side) for side in ("left", "right")]
        wheels = tuple(ActuatorSpec(f"wheel_{i}", "left", i, "sts3250") for i in (2, 3, 4))
        lift = LiftAxisSpec(ActuatorSpec("lift", "left", 5, "sts3095"), 0.131, -1, 4096)
        actuators = (joints[0].actuator, *wheels, lift.actuator, joints[1].actuator)
        self.devices = {
            side: MemoryDevice(
                self,
                side,
                [m for m in actuators if m.bus == side],
                {s.actuator.name: s.calibration for s in joints if s.actuator.bus == side},
            )
            for side in ("left", "right")
        }
        arms = ArmController(self.devices, joints)
        base_lift = BaseLiftController(
            self.devices, wheels=wheels, kinematics=OmniBaseKinematics(0.063, 0.195), lift=lift
        )
        self.control = RobotController(arms, base_lift)
        self.host = HostSupervisor(self.devices, actuators, control=self.control)
        self.host.start()
        self.addCleanup(self.host.close)
        self.sequence = 0

    def identity(self):
        self.sequence += 1
        status = self.host.status
        return CommandIdentity("pc", self.sequence, status.host_session_id, status.control_epoch)

    def cycle(self, command):
        self.now += 0.02
        return self.host.cycle(self.control.submission(self.identity(), command))

    def reference(self):
        self.host.cycle(
            CommandSubmission(
                self.identity(), lambda: self.control.base_lift.establish_lift_reference(0.1)
            )
        )
        self.events.clear()

    def test_homing_rejects_all_remote_targets_without_claiming_control(self):
        self.control.begin_lift_homing()
        for command in (
            RobotCommand({"left_joint": 0.2}),
            RobotCommand(base_velocity=BodyVelocity(0.1)),
            RobotCommand(lift_height_m=0.1),
        ):
            result = self.cycle(command)
            self.assertFalse(result.command_applied)
            self.assertIn("homing", result.command_error)
            self.assertEqual(result.status.phase, HostPhase.READY)
            self.assertIsNone(result.status.control_owner)
        self.assertTrue(all(not event[2] for event in self.events if event[1] == "write"))

    def test_whole_robot_uses_one_feedback_cycle_and_one_owner(self):
        self.reference()
        before = {source: device.sequence for source, device in self.devices.items()}
        result = self.cycle(
            RobotCommand({"left_joint": 0.2, "right_joint": -0.2}, BodyVelocity(0.1), 0.2)
        )
        self.assertTrue(result.command_applied)
        self.assertEqual(result.status.control_owner, "pc")
        self.assertEqual(len(self.events), 3)
        for source, device in self.devices.items():
            self.assertEqual(device.sequence, before[source] + 1)
        self.assertEqual(self.control.arms.measured_positions, {"left_joint": 0, "right_joint": 0})
        self.assertEqual(self.control.base_lift.lift_output.velocity_raw, -1300)

    def test_unknown_lift_height_rejects_whole_command_without_claiming_owner(self):
        result = self.cycle(RobotCommand({"left_joint": 0.2}, BodyVelocity(0.1), 0.2))
        self.assertFalse(result.command_applied)
        self.assertIn("height reference", result.command_error)
        self.assertEqual(result.status.phase, HostPhase.READY)
        self.assertIsNone(result.status.control_owner)
        self.assertFalse(self.events)
        self.assertFalse(self.control.arms.requested_targets)
        self.assertIsNone(self.control.base_lift.base_target)
        self.assertTrue(self.cycle(RobotCommand({"left_joint": 0.1})).command_applied)

    def test_invalid_arm_does_not_stage_base_or_change_old_targets(self):
        self.cycle(RobotCommand({"left_joint": 0.1}, BodyVelocity(0.1)))
        previous = self.control.base_lift.base_target
        self.events.clear()
        result = self.cycle(RobotCommand({"right_joint": 100}, BodyVelocity(-0.1)))
        self.assertFalse(result.command_applied)
        self.assertIsNotNone(result.command_error)
        self.assertEqual(self.control.arms.requested_targets, {"left_joint": 0.1})
        self.assertEqual(self.control.base_lift.base_target, previous)
        self.assertFalse(self.events)

    def test_invalid_commands_do_not_renew_watchdog(self):
        self.cycle(RobotCommand(base_velocity=BodyVelocity(0.1)))
        self.now += 0.7
        self.cycle(RobotCommand({"unknown": 1}))
        self.now += 0.3
        result = self.cycle(RobotCommand(base_velocity=BodyVelocity(0.1)))
        self.assertFalse(result.command_applied)
        self.assertEqual(result.status.watchdog_events, 1)
        self.assertIsNone(result.status.control_owner)
        self.assertIsNone(self.control.base_lift.base_target)

    def test_partial_write_failure_stops_and_disables_both_buses(self):
        self.devices["right"].fail_write = True
        result = self.cycle(
            RobotCommand({"left_joint": 0.1, "right_joint": 0.1}, BodyVelocity(0.1))
        )
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertFalse(result.command_applied)
        self.assertEqual(self.events[0][:2], ("left", "write"))
        for source in self.devices:
            for operation in ("stop", "disable", "close"):
                self.assertIn((source, operation), self.events)
        self.assertFalse(self.control.arms.requested_targets)
        self.assertIsNone(self.control.base_lift.base_target)

    def test_feedback_failure_prevents_all_new_commands(self):
        self.devices["left"].fail_feedback = True
        result = self.cycle(RobotCommand({"left_joint": 0.1}, BodyVelocity(0.1)))
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertFalse(any(event[1] == "write" for event in self.events))

    def test_idle_cycle_does_not_repeat_unchanged_arm_targets(self):
        self.cycle(RobotCommand({"left_joint": 0.1}, BodyVelocity(0.1)))
        self.events.clear()
        self.now += 0.02
        self.host.cycle()
        self.assertFalse(self.events)

    def test_command_owns_immutable_copy_of_targets(self):
        values = {"left_joint": 0.1}
        command = RobotCommand(values)
        values["left_joint"] = 100
        self.assertEqual(command.positions_rad["left_joint"], 0.1)
        with self.assertRaises(TypeError):
            command.positions_rad["left_joint"] = 100

    def test_watchdog_clears_all_subsystem_targets(self):
        self.reference()
        self.cycle(RobotCommand({"left_joint": 0.1}, BodyVelocity(0.1), 0.2))
        # The control loop keeps sampling while the client is absent. A 1 s
        # feedback gap would separately invalidate the single-turn lift reference.
        for _ in range(51):
            self.now += 0.02
            result = self.host.cycle()
        self.assertEqual(result.status.watchdog_events, 1)
        self.assertEqual(result.status.phase, HostPhase.READY)
        self.assertFalse(self.control.arms.requested_targets)
        self.assertIsNone(self.control.base_lift.base_target)
        self.assertIsNone(self.control.base_lift.lift_target_height_m)
        self.events.clear()
        self.host.cycle()
        self.assertFalse(self.events)

    def test_partial_command_retains_omitted_subsystems(self):
        self.cycle(RobotCommand(base_velocity=BodyVelocity(0.1)))
        previous = self.control.base_lift.base_target
        self.cycle(RobotCommand({"left_joint": 0.1}))
        self.assertEqual(self.control.base_lift.base_target, previous)

    def test_internal_programming_error_still_faults(self):
        def bad_validation():
            raise RuntimeError("internal error")

        result = self.host.cycle(CommandSubmission(self.identity(), lambda: None, bad_validation))
        self.assertEqual(result.status.phase, HostPhase.FAULT)

    def test_empty_or_nonfinite_robot_command_rejected(self):
        for fields in (
            {},
            {"positions_rad": {"joint": float("nan")}},
            {"lift_height_m": True},
            {"base_velocity": 1},
        ):
            with self.subTest(fields=fields), self.assertRaises((TypeError, ValueError)):
                RobotCommand(**fields)
