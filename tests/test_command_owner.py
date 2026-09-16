import unittest
from dataclasses import FrozenInstanceError, replace

from alohamini.runtime.command_owner import CommandOwner
from alohamini.schema import CommandIdentity


class CommandOwnerTests(unittest.TestCase):
    def setUp(self):
        self.owner = CommandOwner("host")
        self.command = CommandIdentity("pc", 1, "host", 0)

    def test_single_writer_until_explicit_release(self):
        self.assertTrue(self.owner.accept(self.command))
        other = replace(self.command, client_id="ros")
        self.assertFalse(self.owner.accept(other))
        self.assertTrue(self.owner.accept(replace(self.command, sequence=2)))
        self.owner.release()
        self.assertIsNone(self.owner.owner)
        self.assertEqual(self.owner.epoch, 1)
        self.assertFalse(self.owner.accept(replace(self.command, sequence=3)))
        self.assertTrue(self.owner.accept(replace(other, control_epoch=1)))
        self.assertEqual(self.owner.owner, "ros")

    def test_replay_remains_rejected_after_release(self):
        self.assertTrue(self.owner.accept(self.command))
        self.assertFalse(self.owner.accept(self.command))
        self.owner.release()
        self.assertFalse(self.owner.accept(replace(self.command, control_epoch=1)))
        self.assertTrue(self.owner.accept(replace(self.command, sequence=2, control_epoch=1)))

    def test_rejected_frames_do_not_consume_sequences_or_ownership(self):
        for invalid in (
            replace(self.command, host_session_id="old-host", sequence=900),
            replace(self.command, control_epoch=5, sequence=900),
        ):
            self.assertFalse(self.owner.accept(invalid))
            self.assertIsNone(self.owner.owner)
        self.assertTrue(self.owner.accept(self.command))

    def test_restart_requires_new_session(self):
        restarted = CommandOwner("new-host")
        self.assertFalse(restarted.accept(self.command))
        self.assertTrue(restarted.accept(replace(self.command, host_session_id="new-host")))

    def test_no_anonymous_command_fallback(self):
        with self.assertRaises(TypeError):
            self.owner.accept({})
        self.assertIsNone(self.owner.owner)

    def test_identity_is_validated_and_immutable(self):
        for field in ("client_id", "host_session_id"):
            for invalid in ("", " " * 3, "x" * 65, None, [], 1):
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    replace(self.command, **{field: invalid})
        for field in ("sequence", "control_epoch"):
            for invalid in (-1, True, 0.5, "1", None):
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    replace(self.command, **{field: invalid})
        with self.assertRaises(FrozenInstanceError):
            self.command.sequence = 99
        with self.assertRaises(AttributeError):
            self.owner.epoch = 0

    def test_bounded_history_does_not_accept_previous_epoch_after_eviction(self):
        for index in range(257):
            self.assertTrue(self.owner.accept(CommandIdentity(str(index), 0, "host", index)))
            self.owner.release()
        self.assertEqual(len(self.owner._sequences), 256)
        self.assertFalse(self.owner.accept(CommandIdentity("0", 10, "host", 0)))
