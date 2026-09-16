import importlib.util
import unittest
from dataclasses import replace
from unittest.mock import patch

from alohamini.hardware.feedback import FeedbackBatch, MotorFeedback
from alohamini.kinematics import OmniBaseKinematics
from alohamini.model import ActuatorSpec
from alohamini.runtime.base_control import BaseDrive
from alohamini.runtime.base_lift_control import BaseLiftController
from alohamini.runtime.lifecycle import CommandSubmission, HostPhase, HostSupervisor
from alohamini.runtime.lift_control import (
    LiftAxisSpec,
    LiftHeightTracker,
    LiftHoming,
    lift_height_target,
)
from alohamini.schema import BodyVelocity, CommandIdentity


def layout():
    wheels = tuple(ActuatorSpec(f"wheel_{i}", "left", i, "sts3250") for i in (1, 2, 3))
    lift = LiftAxisSpec(ActuatorSpec("lift", "left", 4, "sts3095"), 0.131, -1, 4096)
    return wheels, lift


def lift_batch(sequence, tick, *, speed=0, moving=0, current=0.1, time_s=None):
    started = sequence * 0.02 if time_s is None else time_s
    return FeedbackBatch(
        "left",
        "session",
        sequence,
        started,
        started + 0.001,
        {
            "lift": MotorFeedback(
                {"position_raw": tick, "velocity_raw": speed, "moving": moving}, current_a=current
            )
        },
        {},
    )


class BaseDriveTests(unittest.TestCase):
    def setUp(self):
        self.drive = BaseDrive(OmniBaseKinematics(0.063, 0.195))

    def test_zero_target(self):
        target = self.drive.target(BodyVelocity())
        self.assertEqual(target.wheel_ticks_s, (0, 0, 0))
        self.assertEqual(target.body_velocity, BodyVelocity())

    def test_wheel_order_and_axes_round_trip(self):
        for velocity in (BodyVelocity(0.1, 0, 0), BodyVelocity(0, 0.1, 0), BodyVelocity(0, 0, 0.3)):
            accepted = self.drive.target(velocity).body_velocity
            for field in ("x_m_s", "y_m_s", "yaw_rad_s"):
                self.assertAlmostEqual(
                    getattr(velocity, field), getattr(accepted, field), delta=0.0003
                )
        wheels = self.drive.target(BodyVelocity(0.1)).wheel_ticks_s
        self.assertGreater(wheels[0], 0)
        self.assertEqual(wheels[1], 0)
        self.assertLess(wheels[2], 0)

    def test_proportional_saturation_preserves_combined_direction(self):
        requested = BodyVelocity(4, -2, 3)
        target = self.drive.target(requested)
        self.assertEqual(max(abs(v) for v in target.wheel_ticks_s), 3000)
        ratios = [
            getattr(target.body_velocity, f) / getattr(requested, f)
            for f in ("x_m_s", "y_m_s", "yaw_rad_s")
        ]
        self.assertLess(max(ratios) - min(ratios), 0.0002)

    def test_feedback_not_clamped_to_command_limit(self):
        actual = self.drive.measured_velocity((6000, 6000, 6000))
        limited = self.drive.target(actual)
        self.assertLess(limited.body_velocity.yaw_rad_s, actual.yaw_rad_s)

    def test_missing_or_encoded_feedback_rejected(self):
        for speeds in ((0, 0), (0, 0, None), (0, True, 0), (0, 0, 0x8001)):
            with self.assertRaises(ValueError):
                self.drive.measured_velocity(speeds)

    def test_large_finite_target_is_scaled_without_tick_overflow(self):
        result = self.drive.target(BodyVelocity(1e300, 0, 0))
        self.assertLessEqual(max(abs(v) for v in result.wheel_ticks_s), 3000)


class LiftLawTests(unittest.TestCase):
    def test_gain_and_speed_cap_with_installation_direction(self):
        self.assertEqual(lift_height_target(0.102, 0.1, direction=1).velocity_raw, 600)
        self.assertEqual(lift_height_target(0.2, 0.1, direction=-1).velocity_raw, -1300)
        self.assertEqual(lift_height_target(0.1, 0.2, direction=-1).velocity_raw, 1300)

    def test_deadband_including_floating_point_unit_conversion(self):
        for target, measured in ((0.110, 0.109), (0.109, 0.110), (0.2, 0.2005)):
            output = lift_height_target(target, measured, direction=-1)
            self.assertEqual(output.velocity_raw, 0)
            self.assertEqual(output.reason, "at_target")

    def test_floor_blocks_only_downward_motion(self):
        self.assertEqual(lift_height_target(0, 0.005, direction=-1).reason, "descent_floor")
        self.assertEqual(lift_height_target(0, 0.005, direction=-1).velocity_raw, 0)
        self.assertLess(lift_height_target(0.1, 0.005, direction=-1).velocity_raw, 0)

    def test_target_clamping_does_not_clamp_measured_height(self):
        self.assertEqual(lift_height_target(1, 0.1, direction=1).target_height_m, 0.6)
        self.assertEqual(lift_height_target(-1, 0.1, direction=1).target_height_m, 0)
        output = lift_height_target(0.6, 0.61, direction=1)
        self.assertEqual(output.velocity_raw, -1300)

    def test_nonfinite_height_and_invalid_direction_rejected(self):
        for values in ((float("nan"), 0, 1), (0, float("inf"), 1), (0, 0, True)):
            with self.assertRaises(ValueError):
                lift_height_target(values[0], values[1], direction=values[2])


class LiftHomingTests(unittest.TestCase):
    def setUp(self):
        self.home = LiftHoming(layout()[1])

    def test_contact_requires_stop_then_new_stationary_feedback(self):
        for sequence in range(7):
            self.home.update(lift_batch(sequence, 1000, current=0.31))
        self.assertEqual(self.home.phase, "settling")
        self.assertEqual(self.home.velocity_raw, 0)
        for sequence in range(7, 14):
            self.home.update(lift_batch(sequence, 1000, current=0.05))
            if self.home.phase == "complete":
                break
        self.assertEqual(self.home.phase, "complete")

    def test_stationary_without_current_is_failure_not_home(self):
        with self.assertRaisesRegex(RuntimeError, "without contact current"):
            for sequence in range(30):
                self.home.update(lift_batch(sequence, 1000, current=0.1))
        self.assertEqual(self.home.phase, "failed")
        self.assertEqual(self.home.velocity_raw, 0)

    def test_current_while_moving_does_not_set_zero(self):
        for sequence in range(100):
            self.home.update(
                lift_batch(
                    sequence, (1000 + sequence * 26) % 4096, speed=1300, moving=1, current=0.4
                )
            )
        self.assertEqual(self.home.phase, "seeking")
        self.assertEqual(self.home.velocity_raw, 1300)

    def test_missing_current_and_position_fail_instead_of_falling_back_to_zero(self):
        for sample in (
            MotorFeedback({"position_raw": 1000, "velocity_raw": 0, "moving": 0}),
            MotorFeedback({"velocity_raw": 0, "moving": 0}, current_a=0.4),
        ):
            home = LiftHoming(layout()[1])
            with self.assertRaises((TypeError, ValueError)):
                home.update(replace(lift_batch(0, 1000), samples={"lift": sample}))
            self.assertEqual(home.velocity_raw, 0)
            self.assertEqual(home.phase, "failed")

    def test_moving_until_deadline_does_not_become_success(self):
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            for sequence in range(1502):
                self.home.update(lift_batch(sequence, (1000 + sequence) % 4096, speed=50, moving=1))
        self.assertEqual(self.home.phase, "failed")

    def test_feedback_gap_and_replay_fail(self):
        for invalid in (lift_batch(0, 1000), lift_batch(1, 1000, time_s=1)):
            home = LiftHoming(layout()[1])
            home.update(lift_batch(0, 1000))
            with self.assertRaises(ValueError):
                home.update(invalid)
            self.assertEqual(home.velocity_raw, 0)

    def test_contact_does_not_complete_if_motion_continues_after_stop(self):
        for sequence in range(7):
            self.home.update(lift_batch(sequence, 1000, current=0.31))
        self.assertEqual(self.home.phase, "settling")
        with self.assertRaisesRegex(RuntimeError, "standstill"):
            for sequence in range(7, 65):
                self.home.update(lift_batch(sequence, 1000 + sequence, speed=50, moving=1))
        self.assertEqual(self.home.phase, "failed")
        self.assertEqual(self.home.velocity_raw, 0)


class LiftTrackerTests(unittest.TestCase):
    def setUp(self):
        self.spec = layout()[1]
        self.tracker = LiftHeightTracker(self.spec)
        self.tracker.bind_session("session")

    def reference(self, *, tick=1000, height=0.1):
        self.tracker.observe(lift_batch(0, tick))
        self.tracker.establish_reference(height)

    def test_unknown_height_is_not_zero(self):
        self.tracker.observe(lift_batch(0, 1000))
        self.assertIsNone(self.tracker.height_m)
        with self.assertRaises(RuntimeError):
            LiftHeightTracker(self.spec).establish_reference(0)

    def test_reference_requires_stationary_valid_feedback(self):
        self.tracker.observe(lift_batch(0, 1000, speed=1, moving=1))
        with self.assertRaises(RuntimeError):
            self.tracker.establish_reference(0)

    def test_wraparound_and_direction(self):
        self.reference(tick=20)
        self.tracker.observe(lift_batch(1, 4080))
        self.assertAlmostEqual(self.tracker.height_m, 0.1 + 36 * 0.131 / 4096)
        self.tracker.observe(lift_batch(2, 20))
        self.assertAlmostEqual(self.tracker.height_m, 0.1)

    def test_multiple_revolutions_accumulate_integer_displacement(self):
        self.reference(tick=1000)
        for sequence in range(1, 150):
            self.tracker.observe(lift_batch(sequence, (1000 - sequence * 64) % 4096))
        self.assertAlmostEqual(self.tracker.height_m, 0.1 + 149 * 64 * 0.131 / 4096)

    def test_missing_reply_invalidates_reference(self):
        self.reference()
        with self.assertRaises(ValueError):
            self.tracker.observe(
                replace(lift_batch(1, 1000), samples={}, failures={"lift": "missing"})
            )
        self.assertIsNone(self.tracker.height_m)

    def test_sampling_gap_cannot_infer_turn_even_if_tick_repeats(self):
        self.reference()
        with self.assertRaises(ValueError):
            self.tracker.observe(lift_batch(1, 1000, time_s=0.5))
        self.assertIsNone(self.tracker.height_m)

    def test_short_gap_with_skipped_sequence_can_remain_unambiguous(self):
        self.reference()
        self.tracker.observe(lift_batch(3, 980, time_s=0.04))
        self.assertAlmostEqual(self.tracker.height_m, 0.1 + 20 * 0.131 / 4096)

    def test_impossible_jump_invalidates_reference(self):
        self.reference()
        with self.assertRaises(ValueError):
            self.tracker.observe(lift_batch(1, 1200))
        self.assertIsNone(self.tracker.height_m)

    def test_half_turn_is_ambiguous(self):
        self.reference()
        with self.assertRaises(ValueError):
            self.tracker.observe(lift_batch(1, 3048, time_s=0.3))

    def test_wrong_session_or_replayed_feedback_invalidates_reference(self):
        for update in (lambda b: replace(b, clock_id="old"), lambda b: replace(b, sequence=0)):
            tracker = LiftHeightTracker(self.spec)
            tracker.bind_session("session")
            tracker.observe(lift_batch(0, 1000))
            tracker.establish_reference(0.1)
            with self.assertRaises(ValueError):
                tracker.observe(update(lift_batch(1, 1000)))
            self.assertIsNone(tracker.height_m)

    def test_height_overrun_is_visible_not_clipped(self):
        self.reference(height=0.599)
        self.tracker.observe(lift_batch(1, 940))
        self.assertGreater(self.tracker.height_m, 0.6)

    def test_reference_cannot_survive_new_host_session(self):
        self.reference()
        with self.assertRaises(RuntimeError):
            self.tracker.bind_session("new")

    def test_bad_registers_or_exceeded_physical_bound_rejected(self):
        for batch in (
            lift_batch(1, -1),
            lift_batch(1, 1000, speed=4097),
            lift_batch(1, 1000, moving=2),
        ):
            tracker = LiftHeightTracker(self.spec)
            tracker.bind_session("session")
            tracker.observe(lift_batch(0, 1000))
            tracker.establish_reference(0.1)
            with self.assertRaises(ValueError):
                tracker.observe(batch)
            self.assertIsNone(tracker.height_m)


@unittest.skipUnless(importlib.util.find_spec("scservo_sdk"), "optional SDK not installed")
class BaseLiftIntegrationTests(unittest.TestCase):
    def setUp(self):
        from test_feetech_device import RegisterSerial

        from alohamini.hardware.feetech_device import FeetechBusDevice

        self.serial = RegisterSerial()
        self.wheels, self.lift = layout()
        for motor_id in (3, 4):
            self.serial.registers[motor_id] = self.serial.registers[2].copy()
        for motor_id in (1, 2, 3, 4):
            self.serial.set(motor_id, 33, 1, 1)
            self.serial.set(motor_id, 58, 2, 0)
            self.serial.set(motor_id, 66, 1, 0)
        self.serial.set(4, 3, 2, 2569)
        self.serial.set(4, 56, 2, 1000)
        self.time_s = 0.0

        def clock():
            self.time_s += 0.00001
            return self.time_s

        patch("alohamini.runtime.lifecycle.time.monotonic", side_effect=clock).start()
        patch("serial.Serial", return_value=self.serial).start()
        self.addCleanup(patch.stopall)
        self.actuators = (*self.wheels, self.lift.actuator)
        self.device = FeetechBusDevice(
            "/dev/test-only",
            self.actuators,
            position_calibrations={},
            velocity_limits={**{w.name: 3000 for w in self.wheels}, "lift": 1300},
        )
        self.control = BaseLiftController(
            {"left": self.device},
            wheels=self.wheels,
            kinematics=OmniBaseKinematics(0.063, 0.195),
            lift=self.lift,
        )
        self.host = HostSupervisor({"left": self.device}, self.actuators, control=self.control)
        self.host.start()
        self.sequence = 0

    def command(self, write):
        self.sequence += 1
        state = self.host.status
        return CommandSubmission(
            CommandIdentity("pc", self.sequence, state.host_session_id, state.control_epoch), write
        )

    def reference(self):
        result = self.host.cycle(self.command(lambda: self.control.establish_lift_reference(0.1)))
        self.assertTrue(result.command_applied)
        self.serial.requests.clear()

    def test_base_only_works_without_fabricating_lift_height(self):
        result = self.host.cycle(self.command(lambda: self.control.set_targets(BodyVelocity(0.1))))
        self.assertTrue(result.command_applied)
        self.assertIsNone(self.control.lift_height_m)
        self.assertEqual(self.control.measured_base_velocity, BodyVelocity())
        self.assertGreater(self.control.base_target.body_velocity.x_m_s, 0)

    def test_height_target_without_reference_rejected_before_base_write(self):
        self.serial.requests.clear()
        result = self.host.cycle(
            self.command(lambda: self.control.set_targets(BodyVelocity(0.1), lift_height_m=0.2))
        )
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertFalse(any(packet[4] == 0x83 for packet in self.serial.requests))

    def test_reference_stages_only_zero_velocities(self):
        self.reference()
        self.assertEqual(self.control.lift_height_m, 0.1)
        self.assertTrue(all(self.serial.get(i, 46, 2) == 0 for i in (1, 2, 3, 4)))

    def test_fixed_height_goal_stops_at_target_without_new_command(self):
        self.reference()
        self.time_s += 0.02
        result = self.host.cycle(
            self.command(lambda: self.control.set_targets(BodyVelocity(), lift_height_m=0.11))
        )
        self.assertTrue(result.command_applied)
        self.assertEqual(self.serial.get(4, 46, 2), 0x8000 | 1300)
        self.serial.set(4, 56, 2, 687)
        self.time_s += 0.1
        self.host.cycle()
        self.assertEqual(self.serial.get(4, 46, 2), 0)
        self.assertEqual(self.control.lift_output.reason, "at_target")
        self.assertAlmostEqual(self.control.lift_height_m, 0.11, delta=0.0001)

    def test_stable_feedback_does_not_resend_unchanged_velocity(self):
        self.reference()
        self.time_s += 0.02
        self.host.cycle(
            self.command(lambda: self.control.set_targets(BodyVelocity(0.1), lift_height_m=0.2))
        )
        self.serial.requests.clear()
        self.time_s += 0.02
        self.host.cycle()
        self.assertFalse(any(packet[4] == 0x83 for packet in self.serial.requests))

    def test_reference_cannot_be_established_outside_current_cycle(self):
        self.host.cycle()
        with self.assertRaises(RuntimeError):
            self.control.establish_lift_reference(0.1)

    def test_feedback_gap_invalidates_reference_and_cancels_target(self):
        self.reference()
        self.time_s += 0.02
        self.host.cycle(
            self.command(lambda: self.control.set_targets(BodyVelocity(), lift_height_m=0.2))
        )
        self.time_s += 0.5
        result = self.host.cycle()
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertIsNone(self.control.lift_height_m)
        self.assertIsNone(self.control.lift_target_height_m)
        self.assertEqual(self.serial.get(4, 46, 2), 0)

    def test_watchdog_with_continuous_feedback_keeps_reference_but_clears_goal(self):
        self.reference()
        self.time_s += 0.02
        self.host.cycle(
            self.command(lambda: self.control.set_targets(BodyVelocity(0.1), lift_height_m=0.2))
        )
        for _ in range(60):
            self.time_s += 0.02
            result = self.host.cycle()
        self.assertEqual(result.status.phase, HostPhase.READY)
        self.assertEqual(result.status.control_epoch, 1)
        self.assertEqual(self.control.lift_height_m, 0.1)
        self.assertIsNone(self.control.lift_target_height_m)
        self.assertIsNone(self.control.base_target)
        self.assertTrue(all(self.serial.get(i, 46, 2) == 0 for i in (1, 2, 3, 4)))

    def test_homing_runs_without_remote_commands_and_stops_before_setting_zero(self):
        self.control.begin_lift_homing()
        self.serial.set(4, 69, 2, 50)  # 0.325 A, above homing contact but below overload.
        for _ in range(20):
            self.time_s += 0.02
            result = self.host.cycle()
            self.assertEqual(result.status.phase, HostPhase.READY)
            self.assertIsNone(result.status.control_owner)
            if self.control.lift_homing_phase == "settling":
                self.assertIsNone(self.control.lift_height_m)
                self.assertEqual(self.serial.get(4, 46, 2), 0)
            if self.control.lift_homing_phase == "complete":
                break
        self.assertEqual(self.control.lift_homing_phase, "complete")
        self.assertEqual(self.control.lift_height_m, 0)
        self.assertEqual(self.serial.get(4, 46, 2), 0)

    def test_homing_stop_readback_failure_latches_fault_without_zero(self):
        self.control.begin_lift_homing()
        self.serial.set(4, 69, 2, 50)
        self.host.cycle()
        self.serial.ignore_writes.add((4, 46))
        for _ in range(20):
            self.time_s += 0.02
            result = self.host.cycle()
            if result.status.phase is HostPhase.FAULT:
                break
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertIsNone(self.control.lift_height_m)
        self.assertTrue(all(self.serial.get(i, 40, 1) == 0 for i in (1, 2, 3, 4)))

    def test_global_overcurrent_still_preempts_homing_contact(self):
        self.control.begin_lift_homing()
        self.serial.set(4, 69, 2, 1300)  # 8.45 A, above STS3095 near-stall threshold.
        for _ in range(10):
            self.time_s += 0.02
            result = self.host.cycle()
            if result.status.phase is HostPhase.FAULT:
                break
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertIsNotNone(result.status.current_trip)
        self.assertIsNone(self.control.lift_height_m)
        self.assertEqual(self.serial.get(4, 46, 2), 0)

    def test_missing_feedback_during_homing_stops_and_disables(self):
        self.control.begin_lift_homing()
        self.host.cycle()
        self.serial.drop.add((4, 0x82, 56))
        self.time_s += 0.02
        result = self.host.cycle()
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertIsNone(self.control.lift_height_m)
        self.assertEqual(self.serial.get(4, 46, 2), 0)
        self.assertTrue(all(self.serial.get(i, 40, 1) == 0 for i in (1, 2, 3, 4)))

    def test_close_cancels_homing_without_establishing_reference(self):
        self.control.begin_lift_homing()
        self.host.cycle()
        self.host.close()
        self.assertEqual(self.control.lift_homing_phase, "failed")
        self.assertIsNone(self.control.lift_height_m)
        self.assertEqual(self.serial.get(4, 46, 2), 0)


if __name__ == "__main__":
    unittest.main()
