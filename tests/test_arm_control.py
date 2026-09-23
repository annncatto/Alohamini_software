import importlib.util
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

from alohamini.calibration import EncoderCalibration
from alohamini.hardware.feedback import FeedbackBatch, MotorFeedback
from alohamini.model import ActuatorSpec
from alohamini.runtime.arm_contact import (
    ArmContactGuard,
    ArmJointSpec,
    GripperContactCalibration,
    JointContactCalibration,
)
from alohamini.runtime.arm_control import ArmController
from alohamini.runtime.lifecycle import CommandSubmission, HostPhase, HostSupervisor
from alohamini.schema import CommandIdentity


def joint_spec(name="joint", source="left", *, contact=None, ratio=1):
    return ArmJointSpec(
        ActuatorSpec(name, source, 1, "sts3250"),
        EncoderCalibration(4096, 2048, 0, 1, ratio, -1, 1),
        contact or JointContactCalibration(math.radians(1)),
    )


class ArmContactTests(unittest.TestCase):
    def setUp(self):
        self.spec = joint_spec()
        self.guard = ArmContactGuard({"joint": self.spec})

    def limit(self, now, *, goal=0.5, position=0, current=2.2):
        return self.guard.limit({"joint": goal}, {"joint": position}, {"joint": current}, now=now)[
            "joint"
        ]

    def test_low_current_high_error_passes_without_smoothing(self):
        self.assertEqual(self.limit(0, current=0), 0.5)
        self.assertEqual(self.limit(1, current=0), 0.5)
        self.assertFalse(self.guard.holds)

    def test_high_current_small_error_does_not_trip(self):
        self.assertEqual(self.limit(0, goal=0.01), 0.01)
        self.assertEqual(self.limit(1, goal=0.01), 0.01)

    def test_stall_holds_after_150ms_and_persists_after_current_drops(self):
        self.assertEqual(self.limit(0), 0.5)
        self.assertEqual(self.limit(0.149), 0.5)
        self.assertEqual(self.limit(0.150), 0)
        self.assertEqual(self.limit(0.200, current=0), 0)
        self.assertEqual(self.guard.hold_events, 1)

    def test_progress_and_direction_change_restart_candidate(self):
        self.limit(0)
        self.assertEqual(self.limit(0.1, position=0.004), 0.5)
        self.assertEqual(self.limit(0.2, position=0.004), 0.5)
        self.assertEqual(self.limit(0.23, goal=-0.5, position=0.004), -0.5)
        self.assertEqual(self.limit(0.3, goal=-0.5, position=0.004), -0.5)
        self.assertEqual(self.limit(0.4, goal=-0.5, position=0.004), 0.004)

    def test_current_drop_restarts_candidate(self):
        self.limit(0)
        self.limit(0.1, current=0)
        self.assertEqual(self.limit(0.2), 0.5)
        self.assertEqual(self.limit(0.3), 0.5)

    def test_moving_away_from_target_does_not_count_as_progress(self):
        self.limit(0)
        self.assertEqual(self.limit(0.2, position=-0.01), -0.01)

    def test_retreat_uses_explicit_release_margin(self):
        self.limit(0)
        self.limit(0.2)
        self.assertEqual(self.limit(0.3, goal=-math.radians(0.9)), 0)
        goal = -math.radians(1)
        self.assertEqual(self.limit(0.4, goal=goal), goal)
        self.assertFalse(self.guard.holds)

    def test_encoder_progress_threshold_preserved_with_gear_ratio(self):
        self.guard = ArmContactGuard({"joint": joint_spec(ratio=0.5)})
        self.limit(0)
        self.assertEqual(self.limit(0.1, position=0.002), 0.5)
        self.assertEqual(self.limit(0.2, position=0.002), 0.5)

    def test_one_encoder_tick_below_and_above_old_two_degree_boundary(self):
        # Old driver uses 360/4095 degrees per tick: 22 ticks < 2deg, 23 ticks > 2deg.
        for ticks, expect_hold in ((22, False), (23, True)):
            self.guard = ArmContactGuard({"joint": self.spec})
            goal = ticks * math.tau / 4096
            self.limit(0, goal=goal)
            result = self.limit(0.2, goal=goal)
            self.assertEqual(result == 0, expect_hold)

    def test_missing_or_nonfinite_feedback_is_not_an_unprotected_target(self):
        self.limit(0)
        for positions in ({}, {"joint": float("nan")}):
            with self.assertRaises((KeyError, ValueError)):
                self.guard.limit({"joint": 0.5}, positions, {"joint": 2.2}, now=0.1)
        self.assertEqual(self.limit(0.2), 0.5)
        self.assertEqual(self.limit(0.3), 0.5)

    def test_stop_cancels_candidates_but_does_not_release_holds(self):
        self.limit(0)
        self.guard.cancel_candidates()
        self.assertEqual(self.limit(0.2), 0.5)
        self.assertEqual(self.limit(0.4), 0)
        self.guard.cancel_candidates()
        self.assertEqual(self.limit(0.5, current=0), 0)

    def test_duplicate_time_and_invalid_targets_rejected(self):
        self.limit(0)
        with self.assertRaises(ValueError):
            self.limit(0)
        with self.assertRaises(ValueError):
            self.limit(0.1, goal=5)

    def test_gripper_closing_hold_and_retreat_for_both_installation_directions(self):
        for closed, opened in ((0, 1), (1, 0)):
            spec = joint_spec(contact=GripperContactCalibration(closed, opened))
            guard = ArmContactGuard({"joint": spec})
            hold = 0.5 - 0.03 * (opened - closed)
            result = guard.limit({"joint": closed}, {"joint": 0.5}, {"joint": 0.5}, now=0)
            self.assertAlmostEqual(result["joint"], hold)
            result = guard.limit({"joint": closed}, {"joint": 0.5}, {"joint": 0}, now=0.1)
            self.assertAlmostEqual(result["joint"], hold)
            retreat = hold + 0.02 * (opened - closed)
            result = guard.limit({"joint": retreat}, {"joint": 0.5}, {"joint": 0}, now=0.2)
            self.assertEqual(result["joint"], retreat)
            self.assertFalse(guard.holds)

    def test_gripper_open_endpoint_contact_allows_closing_retreat(self):
        spec = joint_spec(contact=GripperContactCalibration(0, 1))
        guard = ArmContactGuard({"joint": spec})
        result = guard.limit({"joint": 1}, {"joint": 0.9}, {"joint": 0.6}, now=0)
        self.assertEqual(result["joint"], 0.9)
        result = guard.limit({"joint": 0.8}, {"joint": 0.9}, {"joint": 0.1}, now=0.1)
        self.assertEqual(result["joint"], 0.8)

    def test_gripper_nudge_is_clipped_to_calibrated_stroke(self):
        spec = joint_spec(contact=GripperContactCalibration(0, 1))
        guard = ArmContactGuard({"joint": spec})
        result = guard.limit({"joint": 0}, {"joint": 0.01}, {"joint": 0.6}, now=0)
        self.assertEqual(result["joint"], 0)

    def test_gripper_below_limit_remains_unfiltered(self):
        spec = joint_spec(contact=GripperContactCalibration(0, 1))
        guard = ArmContactGuard({"joint": spec})
        result = guard.limit({"joint": 0}, {"joint": 0.5}, {"joint": 0.499}, now=0)
        self.assertEqual(result["joint"], 0)
        self.assertFalse(guard.holds)

    def test_contact_calibration_validation(self):
        for factory in (
            lambda: JointContactCalibration(0),
            lambda: GripperContactCalibration(1, 1),
            lambda: joint_spec(contact=GripperContactCalibration(0, 5)),
        ):
            with self.assertRaises(ValueError):
                factory()


class ArmDeviceMemory:
    def __init__(self, fixture, spec):
        self.fixture = fixture
        self.spec = spec
        self.actuators = (spec.actuator,)
        self.position_calibrations = {spec.actuator.name: spec.calibration}
        self.sequence = 0
        self.position = 2048
        self.current = 2.2

    def connect(self, session):
        self.session = session

    def read_feedback(self):
        started = self.fixture.now
        self.fixture.now += 0.001
        result = FeedbackBatch(
            self.spec.actuator.bus,
            self.session,
            self.sequence,
            started,
            self.fixture.now,
            {
                self.spec.actuator.name: MotorFeedback(
                    {"position_raw": self.position, "velocity_raw": 0}, current_a=self.current
                )
            },
            {},
        )
        self.sequence += 1
        return result

    def write_targets(self, *, positions_rad, velocities_raw):
        self.fixture.writes.append((self.spec.actuator.bus, dict(positions_rad)))
        assert not velocities_raw

    def stop(self):
        self.fixture.stops.append(self.spec.actuator.bus)

    def stop_velocity(self):
        pass  # This arm-only fixture has no velocity axes.

    def stop_motion(self, feedback):
        self.stop()

    def disable_torque(self):
        pass

    def close(self):
        pass


class ArmControllerTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.writes, self.stops = [], []
        clock = patch("alohamini.runtime.arm_control.time.monotonic", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.specs = [joint_spec(f"{source}_joint", source) for source in ("left", "right")]
        self.devices = {spec.actuator.bus: ArmDeviceMemory(self, spec) for spec in self.specs}
        self.controller = ArmController(self.devices, self.specs)
        self.host = HostSupervisor(
            self.devices, [spec.actuator for spec in self.specs], control=self.controller
        )
        self.host.start()

    def command(self, targets, sequence=0):
        status = self.host.status
        return CommandSubmission(
            CommandIdentity("pc", sequence, status.host_session_id, status.control_epoch),
            lambda: self.controller.set_targets(targets),
        )

    def test_idle_cycles_do_not_send_targets(self):
        self.host.cycle()
        self.host.cycle()
        self.assertFalse(self.writes)

    def test_wrist_wrap_keeps_feedback_and_targets_on_the_same_single_turn_branch(self):
        calibration = EncoderCalibration(4096, 2048, 0, 1, 1, -math.pi, math.pi - math.tau / 4096)
        spec = replace(joint_spec("wrist"), calibration=calibration)
        device = ArmDeviceMemory(self, spec)
        controller = ArmController({"left": device}, [spec])
        controller.bind_session("single-turn")
        device.connect("single-turn")
        for now, tick in ((0, 4090), (0.02, 5), (0.2, 5)):
            self.now, device.position = now, tick
            controller.observe((device.read_feedback(),))
            goal = calibration.position_from_tick(tick)
            controller.set_targets({"wrist": goal})
            controller.supervise()
            self.assertEqual(controller.measured_positions["wrist"], goal)
            self.assertFalse(controller.contact_holds)

    def test_new_command_writes_both_arms_and_idle_only_writes_contact_corrections(self):
        command = self.command({"left_joint": 0.5, "right_joint": 0.5})
        self.assertTrue(self.host.cycle(command).command_applied)
        self.assertEqual(len(self.writes), 2)
        self.now += 0.1
        self.host.cycle()
        self.assertEqual(len(self.writes), 2)
        self.now += 0.06
        self.host.cycle()
        self.assertEqual(
            self.writes[-2:], [("left", {"left_joint": 0}), ("right", {"right_joint": 0})]
        )
        self.now += 0.1
        self.host.cycle()
        self.assertEqual(len(self.writes), 4)
        self.assertEqual(self.controller.requested_targets["left_joint"], 0.5)
        self.assertEqual(self.controller.sent_targets["left_joint"], 0)

    def test_all_arm_targets_validated_before_any_bus_write(self):
        result = self.host.cycle(self.command({"left_joint": 0.5, "right_joint": 5}))
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertFalse(result.command_applied)
        self.assertFalse(self.writes)
        self.assertEqual(set(self.stops), {"left", "right"})

    def test_missing_position_faults_instead_of_sending_unguarded_target(self):
        self.devices["right"].position = -1
        result = self.host.cycle(self.command({"left_joint": 0.5}))
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertFalse(self.writes)

    def test_partial_command_preserves_other_arm_goal(self):
        self.host.cycle(self.command({"left_joint": 0.5, "right_joint": 0.5}))
        self.now += 0.02
        self.host.cycle(self.command({"left_joint": 0.2}, sequence=1))
        self.assertEqual(self.controller.requested_targets, {"left_joint": 0.2, "right_joint": 0.5})

    def test_identical_new_target_keeps_existing_new_command_write_semantics(self):
        self.host.cycle(self.command({"left_joint": 0.5}))
        self.host.cycle(self.command({"left_joint": 0.5}, sequence=1))
        self.assertEqual(len(self.writes), 2)

    def test_invalid_hold_on_second_arm_is_checked_before_first_arm_correction(self):
        self.devices["right"].position = 100
        self.host.cycle(self.command({"left_joint": 0.5, "right_joint": 0.5}))
        self.writes.clear()
        self.now += 0.16
        result = self.host.cycle()
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertFalse(self.writes)

    def test_watchdog_clears_targets_and_does_not_resend_on_next_cycle(self):
        self.host.cycle(self.command({"left_joint": 0.5}))
        self.writes.clear()
        self.now += 1.01
        result = self.host.cycle()
        self.assertEqual(result.status.control_epoch, 1)
        self.assertFalse(self.controller.requested_targets)
        self.host.cycle()
        self.assertFalse(self.writes)

    def test_contact_hold_survives_watchdog_until_retreat(self):
        self.host.cycle(self.command({"left_joint": 0.5}))
        self.now += 0.16
        self.host.cycle()
        self.now += 1.01
        self.host.cycle()
        self.assertEqual(self.controller.contact_holds, {"left_joint": 0})
        self.host.cycle(self.command({"left_joint": 0.5}, sequence=1))
        self.assertEqual(self.controller.sent_targets["left_joint"], 0)
        self.host.cycle(self.command({"left_joint": -0.1}, sequence=2))
        self.assertFalse(self.controller.contact_holds)

    def test_projection_cannot_reuse_old_feedback_to_write_again(self):
        self.host.cycle(self.command({"left_joint": 0.5}))
        with self.assertRaises(RuntimeError):
            self.controller.supervise()

    def test_overcurrent_checked_before_motion_controller(self):
        self.devices["left"].current = 4
        self.host.cycle(self.command({"left_joint": 0.5}))
        self.writes.clear()
        self.now += 0.081
        result = self.host.cycle(self.command({"left_joint": 0.6}, sequence=1))
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertFalse(self.writes)

    def test_supervision_write_failure_is_not_reported_as_applied(self):
        with patch.object(self.devices["right"], "write_targets", side_effect=OSError("lost bus")):
            result = self.host.cycle(self.command({"left_joint": 0.5, "right_joint": 0.5}))
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertFalse(result.command_applied)
        self.assertFalse(self.controller.requested_targets)

    def test_clear_target_failure_still_stops_hardware(self):
        with patch.object(
            self.controller, "clear_targets", side_effect=RuntimeError("clear failed")
        ):
            status = self.host.close()
        self.assertEqual(status.phase, HostPhase.FAULT)
        self.assertEqual(set(self.stops), {"left", "right"})

    def test_joint_device_calibration_mismatch_rejected(self):
        changed = replace(
            self.specs[0], calibration=replace(self.specs[0].calibration, reference_tick=1000)
        )
        with self.assertRaises(ValueError):
            ArmController(self.devices, [changed, self.specs[1]])

    def test_controller_cannot_be_rebound_to_another_session(self):
        with self.assertRaises(RuntimeError):
            self.controller.bind_session("another-session")

    def test_failed_rebinding_does_not_clear_another_hosts_targets(self):
        self.host.cycle(self.command({"left_joint": 0.5}))
        other_host = HostSupervisor(
            self.devices, [spec.actuator for spec in self.specs], control=self.controller
        )
        with self.assertRaises(RuntimeError):
            other_host.start()
        self.assertEqual(self.host.status.phase, HostPhase.ACTIVE)
        self.assertEqual(self.controller.requested_targets, {"left_joint": 0.5})
        self.assertFalse(self.stops)

    def test_returned_target_maps_are_copies(self):
        self.host.cycle(self.command({"left_joint": 0.5}))
        targets = self.controller.requested_targets
        targets["left_joint"] = 10
        self.assertEqual(self.controller.requested_targets["left_joint"], 0.5)


@unittest.skipUnless(importlib.util.find_spec("scservo_sdk"), "optional Feetech SDK not installed")
class ArmBackendIntegrationTests(unittest.TestCase):
    def test_real_packets_contact_hold_and_reverse_release(self):
        from test_feetech_device import RegisterSerial

        from alohamini.hardware.feetech_device import FeetechBusDevice

        serial = RegisterSerial()
        serial.set(1, 56, 2, 2048)
        serial.set(1, 69, 2, 80)  # 520 mA: gripper contact, below motor overload.
        spec = joint_spec(contact=GripperContactCalibration(0, 1))
        wheel = ActuatorSpec("wheel", "left", 2, "sts3250")
        device = FeetechBusDevice(
            "/dev/test-only",
            (spec.actuator, wheel),
            position_calibrations={"joint": spec.calibration},
            velocity_limits={"wheel": 3000},
        )
        controller = ArmController({"left": device}, [spec])
        host = HostSupervisor({"left": device}, (spec.actuator, wheel), control=controller)
        with patch("serial.Serial", return_value=serial), host:
            # The measured physical position is zero. Closing stroke direction is negative.
            serial.set(1, 56, 2, spec.calibration.position_to_tick(0.5))
            identity = CommandIdentity("pc", 0, host.status.host_session_id, 0)
            result = host.cycle(
                CommandSubmission(identity, lambda: controller.set_targets({"joint": 0}))
            )
            self.assertTrue(result.command_applied)
            held = controller.contact_holds["joint"]
            self.assertAlmostEqual(held, 0.47, delta=math.tau / 4096)
            self.assertEqual(serial.get(1, 42, 2), spec.calibration.position_to_tick(held))
            result = host.cycle(
                CommandSubmission(
                    replace(identity, sequence=1), lambda: controller.set_targets({"joint": 0.7})
                )
            )
            self.assertTrue(result.command_applied)
            self.assertFalse(controller.contact_holds)
            self.assertEqual(serial.get(1, 42, 2), spec.calibration.position_to_tick(0.7))


if __name__ == "__main__":
    unittest.main()
