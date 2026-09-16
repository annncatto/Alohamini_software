import importlib.util
import threading
import unittest
from dataclasses import replace
from unittest.mock import patch

from alohamini.hardware.feedback import FeedbackBatch, MotorFeedback
from alohamini.model import ActuatorSpec
from alohamini.runtime.current_protection import CurrentProtection
from alohamini.runtime.lifecycle import CommandSubmission, HostPhase, HostSupervisor
from alohamini.schema import CommandIdentity


class CurrentProtectionTests(unittest.TestCase):
    def test_all_models_keep_deployed_near_stall_thresholds(self):
        for model, limit in (("sts3215", 2.16), ("sts3250", 3.36), ("sts3095", 7.84)):
            with self.subTest(model=model):
                guard = CurrentProtection({"motor": model})
                self.assertIsNone(guard.update({"motor": limit}, now=0))
                self.assertIsNone(guard.update({"motor": limit}, now=0.079))
                trip = guard.update({"motor": limit}, now=0.080)
                self.assertEqual(trip.cause, "near_stall_current")
                self.assertAlmostEqual(trip.limit_a, limit)

    def test_all_models_keep_deployed_sustained_thresholds(self):
        for model, limit in (("sts3215", 1.8), ("sts3250", 2.8), ("sts3095", 4.4)):
            with self.subTest(model=model):
                guard = CurrentProtection({"motor": model})
                self.assertIsNone(guard.update({"motor": limit}, now=0))
                self.assertIsNone(guard.update({"motor": limit}, now=0.649))
                trip = guard.update({"motor": limit}, now=0.650)
                self.assertEqual(trip.cause, "sustained_overload")
                self.assertAlmostEqual(trip.limit_a, limit)

    def test_short_spike_resets_timer(self):
        guard = CurrentProtection({"motor": "sts3250"})
        for now, current in ((0, 4), (0.06, 0), (0.07, 4), (0.1, 4)):
            self.assertIsNone(guard.update({"motor": current}, now=now))
        self.assertIsNotNone(guard.update({"motor": 4}, now=0.16))

    def test_frequency_independent_trip_time(self):
        for hz in (30, 50, 60, 100):
            guard = CurrentProtection({"motor": "sts3250"})
            for index in range(hz):
                trip = guard.update({"motor": 3}, now=index / hz)
                if trip:
                    self.assertGreaterEqual(index / hz, 0.650)
                    self.assertLess(index / hz, 0.650 + 1 / hz)
                    break
            else:
                self.fail("Sustained current did not trip")

    def test_motor_timers_are_independent_and_signed_current_uses_magnitude(self):
        guard = CurrentProtection({"a": "sts3250", "b": "sts3095"})
        self.assertIsNone(guard.update({"a": -4, "b": 0}, now=0))
        self.assertIsNone(guard.update({"a": 0, "b": -8}, now=0.05))
        trip = guard.update({"a": 0, "b": -8}, now=0.14)
        self.assertEqual(trip.motor, "b")

    def test_missing_nonfinite_and_replayed_samples_rejected(self):
        guard = CurrentProtection({"motor": "sts3250"})
        guard.update({"motor": 0}, now=0)
        for currents in ({}, {"other": 0}, {"motor": float("nan")}, {"motor": True}):
            with self.subTest(currents=currents), self.assertRaises(ValueError):
                guard.update(currents, now=0.02)
        with self.assertRaises(ValueError):
            guard.update({"motor": 0}, now=0)

    def test_unknown_model_is_not_assigned_a_guessed_limit(self):
        with self.assertRaises(ValueError):
            CurrentProtection({"motor": "unknown"})


class MemoryDevice:
    def __init__(self, fixture, source):
        self.fixture, self.source = fixture, source
        self.motor = f"{source}_motor"
        self.session = None
        self.sequence = 0
        self.current = 0.1
        self.errors = {}
        self.transform = lambda batch: batch

    def operation(self, name):
        self.fixture.events.append((self.source, name))
        if name in self.errors:
            raise self.errors[name]

    def connect(self, session):
        self.session = session
        self.operation("connect")

    def read_feedback(self):
        self.operation("read")
        start = self.fixture.now
        self.fixture.now += 0.001
        batch = FeedbackBatch(
            self.source,
            self.session,
            self.sequence,
            start,
            self.fixture.now,
            {self.motor: MotorFeedback({"position_raw": 100}, current_a=self.current)},
            {},
        )
        self.sequence += 1
        return self.transform(batch)

    def stop(self):
        self.operation("stop")

    def disable_torque(self):
        self.operation("disable_torque")

    def close(self):
        self.operation("close")


class HostLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.events = []
        self.clock = patch(
            "alohamini.runtime.lifecycle.time.monotonic", side_effect=lambda: self.now
        )
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.devices = {source: MemoryDevice(self, source) for source in ("left", "right")}
        self.actuators = [
            ActuatorSpec(f"{source}_motor", source, 1, "sts3250") for source in self.devices
        ]
        self.host = HostSupervisor(self.devices, self.actuators)

    def command(self, sequence=0, *, client="pc", epoch=None, write=None):
        status = self.host.status
        identity = CommandIdentity(
            client,
            sequence,
            status.host_session_id,
            status.control_epoch if epoch is None else epoch,
        )
        return CommandSubmission(identity, write or (lambda: self.events.append((client, "write"))))

    def start(self):
        self.host.start()
        self.events.clear()

    def assert_cleanup(self):
        cleanup = [(s, op) for s, op in self.events if op in ("stop", "disable_torque", "close")]
        self.assertEqual(
            cleanup, [(s, op) for op in ("stop", "disable_torque", "close") for s in self.devices]
        )

    def test_start_opens_only_and_first_cycle_reads_before_write(self):
        status = self.host.start()
        self.assertEqual(status.phase, HostPhase.STARTING)
        self.assertEqual(self.events, [("left", "connect"), ("right", "connect")])
        self.events.clear()
        result = self.host.cycle(self.command())
        self.assertEqual(self.events, [("left", "read"), ("right", "read"), ("pc", "write")])
        self.assertTrue(result.command_applied)
        self.assertEqual(result.status.phase, HostPhase.ACTIVE)

    def test_idle_host_does_not_trigger_command_watchdog(self):
        self.start()
        self.now = 100
        result = self.host.cycle()
        self.assertEqual(result.status.phase, HostPhase.READY)
        self.assertEqual(result.status.watchdog_events, 0)

    def test_context_manager_cleans_up_on_application_interrupt(self):
        with self.assertRaises(KeyboardInterrupt), self.host:
            self.events.clear()
            self.host.cycle(self.command())
            raise KeyboardInterrupt()
        self.assert_cleanup()
        self.assertEqual(self.host.status.phase, HostPhase.CLOSED)

    def test_interrupt_during_feedback_cleans_up_and_propagates(self):
        self.start()
        self.devices["left"].errors["read"] = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.host.cycle(self.command())
        self.assert_cleanup()
        self.assertEqual(self.host.status.phase, HostPhase.FAULT)

    @unittest.skipUnless(importlib.util.find_spec("scservo_sdk"), "optional SDK not installed")
    def test_real_feedback_reader_to_supervisor_fault_path(self):
        from test_feedback import MemorySerial, packet

        from alohamini.hardware.feetech import FeetechFeedbackReader

        self.start()
        for source, device in self.devices.items():
            response = packet(1, error=1 if source == "right" else 0)
            reader = FeetechFeedbackReader(
                MemorySerial([response]),
                [motor for motor in self.actuators if motor.bus == source],
                clock_id=self.host.status.host_session_id,
            )
            device.read_feedback = reader.read
        result = self.host.cycle(self.command())
        self.assertEqual(len(result.feedback), 2)
        self.assertEqual(result.feedback[1].samples["right_motor"].packet_error, 1)
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertFalse(result.command_applied)
        self.assert_cleanup()

    def test_no_new_command_does_not_repeat_target_writes(self):
        self.start()
        self.host.cycle(self.command())
        self.events.clear()
        self.now += 0.02
        self.host.cycle()
        self.assertEqual(self.events, [("left", "read"), ("right", "read")])

    def test_watchdog_precedes_queued_command_and_revokes_epoch(self):
        self.start()
        self.host.cycle(self.command())
        queued = self.command(1)
        self.events.clear()
        self.now += 1.01
        result = self.host.cycle(queued)
        self.assertFalse(result.command_applied)
        self.assertEqual(self.events[:2], [("left", "stop"), ("right", "stop")])
        self.assertEqual(result.status.control_epoch, 1)
        self.assertIsNone(result.status.control_owner)
        self.assertEqual(result.status.watchdog_events, 1)
        self.assertEqual(result.status.phase, HostPhase.READY)
        self.assertFalse(self.host.cycle(queued).command_applied)
        self.assertTrue(self.host.cycle(self.command(2)).command_applied)

    def test_watchdog_is_rechecked_after_bus_reads(self):
        self.start()
        self.host.cycle(self.command())
        self.events.clear()
        self.now += 0.999
        result = self.host.cycle(self.command(1))
        self.assertFalse(result.command_applied)
        self.assertEqual(
            self.events,
            [
                ("left", "read"),
                ("right", "read"),
                ("left", "stop"),
                ("right", "stop"),
            ],
        )

    def test_rejected_client_cannot_renew_lease(self):
        self.start()
        self.host.cycle(self.command())
        self.now += 0.7
        self.assertFalse(self.host.cycle(self.command(client="ros")).command_applied)
        self.now += 0.31
        self.assertEqual(self.host.cycle().status.watchdog_events, 1)

    def test_replayed_command_cannot_renew_lease(self):
        self.start()
        command = self.command()
        self.host.cycle(command)
        self.now += 0.7
        self.assertFalse(self.host.cycle(command).command_applied)
        self.now += 0.31
        self.assertEqual(self.host.cycle().status.watchdog_events, 1)

    def test_stop_failure_does_not_release_owner_and_cleans_other_bus(self):
        self.start()
        self.host.cycle(self.command())
        self.devices["left"].errors["stop"] = OSError("disconnected")
        self.events.clear()
        self.now += 1.01
        result = self.host.cycle(self.command(1))
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertEqual(result.status.control_owner, "pc")
        self.assertEqual(result.status.control_epoch, 0)
        self.assertEqual(result.status.cleanup_failures[0].source_id, "left")
        self.assert_cleanup()

    def test_partial_connection_failure_cleans_every_attempted_device(self):
        self.devices["right"].errors["connect"] = OSError("open failed")
        with self.assertRaises(OSError):
            self.host.start()
        self.assertEqual(self.host.status.phase, HostPhase.FAULT)
        self.assert_cleanup()

    def test_cleanup_failures_do_not_skip_other_operations(self):
        self.start()
        self.devices["left"].errors.update(
            {operation: OSError(operation) for operation in ("stop", "disable_torque", "close")}
        )
        status = self.host.close()
        self.assertEqual(status.phase, HostPhase.FAULT)
        self.assertEqual(len(status.cleanup_failures), 3)
        self.assert_cleanup()

    def test_interrupt_during_cleanup_still_closes_other_devices(self):
        self.start()
        self.devices["left"].errors["stop"] = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.host.close()
        self.assert_cleanup()
        self.assertEqual(self.host.status.phase, HostPhase.FAULT)

    def test_feedback_exception_faults_before_command(self):
        self.start()
        self.devices["right"].errors["read"] = OSError("timeout")
        result = self.host.cycle(self.command())
        self.assertEqual(len(result.feedback), 1)
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assertFalse(result.command_applied)
        self.assert_cleanup()

    def test_partial_feedback_preserved_but_not_authorized_for_motion(self):
        self.start()
        self.devices["left"].transform = lambda b: replace(b, failures={"left_motor": "fault"})
        result = self.host.cycle(self.command())
        self.assertIn("left_motor", result.feedback[0].samples)
        self.assertFalse(result.command_applied)
        self.assertEqual(result.status.phase, HostPhase.FAULT)

    def test_bad_bus_session_time_sequence_or_current_faults(self):
        transforms = [
            lambda b: replace(b, source_id="right"),
            lambda b: replace(b, clock_id="old-session"),
            lambda b: replace(b, request_started_s=0),
            lambda b: replace(b, received_s=b.received_s + 1),
            lambda b: replace(b, sequence=0),
            lambda b: replace(b, samples={}),
            lambda b: replace(b, samples={"left_motor": MotorFeedback({})}),
            lambda b: replace(
                b, samples={"left_motor": MotorFeedback({}, packet_error=1, current_a=0)}
            ),
        ]
        for transform in transforms:
            with self.subTest(transform=transform):
                self.host = HostSupervisor(self.devices, self.actuators)
                self.start()
                self.devices["left"].transform = lambda b: b
                self.host.cycle()
                self.devices["left"].transform = transform
                result = self.host.cycle(self.command())
                self.assertEqual(result.status.phase, HostPhase.FAULT)
                self.assertFalse(result.command_applied)

    def test_overload_trips_without_waiting_for_new_command(self):
        self.start()
        self.devices["right"].current = 4
        self.host.cycle(self.command())
        self.events.clear()
        self.now += 0.080
        result = self.host.cycle()
        self.assertEqual(result.status.current_trip.motor, "right_motor")
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assert_cleanup()

    def test_command_write_failure_is_treated_as_possible_partial_write(self):
        self.start()

        def failing_write():
            self.events.append(("pc", "partial_write"))
            raise OSError("second bus failed")

        result = self.host.cycle(self.command(write=failing_write))
        self.assertFalse(result.command_applied)
        self.assertEqual(result.status.phase, HostPhase.FAULT)
        self.assert_cleanup()

    def test_fault_is_latched_close_is_idempotent_and_new_session_is_required(self):
        self.start()
        self.devices["left"].errors["read"] = OSError("lost")
        self.host.cycle()
        self.events.clear()
        self.assertEqual(self.host.close().phase, HostPhase.FAULT)
        with self.assertRaises(RuntimeError):
            self.host.cycle(self.command())
        with self.assertRaises(RuntimeError):
            self.host.start()
        self.assertFalse(self.events)
        restarted = HostSupervisor(self.devices, self.actuators)
        self.assertNotEqual(restarted.status.host_session_id, self.host.status.host_session_id)

    def test_normal_close_and_close_before_start(self):
        self.start()
        self.host.cycle(self.command())
        self.events.clear()
        self.assertEqual(self.host.close().phase, HostPhase.CLOSED)
        self.assertIsNone(self.host.status.control_owner)
        self.assert_cleanup()
        self.events.clear()
        self.host.close()
        self.assertFalse(self.events)
        unopened = HostSupervisor(self.devices, self.actuators)
        self.assertEqual(unopened.close().phase, HostPhase.CLOSED)
        self.assertFalse(self.events)

    def test_cross_thread_and_reentrant_calls_rejected(self):
        self.start()
        errors = []

        def other_thread():
            try:
                self.host.cycle()
            except RuntimeError as exc:
                errors.append(exc)

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(timeout=1)
        self.assertEqual(len(errors), 1)
        result = self.host.cycle(self.command(write=lambda: self.host.cycle()))
        self.assertEqual(result.status.phase, HostPhase.FAULT)

    def test_invalid_submission_faults_an_active_session(self):
        self.start()
        self.host.cycle(self.command())
        result = self.host.cycle({})
        self.assertEqual(result.status.phase, HostPhase.FAULT)

    def test_layout_validation_precedes_device_access(self):
        for devices, actuators in (
            ({}, []),
            (self.devices, []),
            (self.devices, [self.actuators[0]]),
            (self.devices, [*self.actuators, self.actuators[0]]),
        ):
            with self.assertRaises(ValueError):
                HostSupervisor(devices, actuators)
        self.assertFalse(self.events)


if __name__ == "__main__":
    unittest.main()
