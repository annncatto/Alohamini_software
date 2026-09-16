import threading
import unittest
from queue import Queue

from support import state_frames, state_payload

from alohamini.client import HostClient
from alohamini.errors import (
    CommandRejectedError,
    ConnectionError,
    ProtocolError,
    ResponseTimeoutError,
)
from alohamini.protocol import HostSnapshot
from alohamini.runtime.command_owner import CommandOwner
from alohamini.schema import CommandIdentity

try:
    import zmq
except ImportError:
    zmq = None


class CommandConfigurationTests(unittest.TestCase):
    def test_invalid_command_port_and_window(self):
        for options in (
            {"command_port": 0},
            {"command_port": 65536},
            {"command_port": True},
            {"request_window": 0},
            {"request_window": 17},
            {"request_window": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                HostClient("127.0.0.1", **options)

    def test_no_command_connection_without_state(self):
        with HostClient("127.0.0.1") as client:
            snapshot = HostSnapshot(state_payload(), {}, 0, 0)
            with self.assertRaises(CommandRejectedError):
                client.send_command({"x.vel": 0.1}, based_on=snapshot)
            self.assertIsNone(client._context)
            self.assertIsNone(client._command_socket)


@unittest.skipIf(zmq is None, "pyzmq is not installed")
class CommandClientTests(unittest.TestCase):
    def setUp(self):
        self.ports = Queue()
        self.requests = Queue()
        self.commands = Queue()
        self.errors = Queue()
        self.stop = threading.Event()
        self.status_patch = {}
        self.request_count = 0
        self.handler = self.reply

        def serve():
            context = zmq.Context()
            router = context.socket(zmq.ROUTER)
            pull = context.socket(zmq.PULL)
            router.linger = pull.linger = 0
            owner = CommandOwner("host-session")
            try:
                state_port = router.bind_to_random_port("tcp://127.0.0.1")
                command_port = pull.bind_to_random_port("tcp://127.0.0.1")
                self.ports.put((state_port, command_port))
                poller = zmq.Poller()
                poller.register(router, zmq.POLLIN)
                poller.register(pull, zmq.POLLIN)
                while not self.stop.is_set():
                    ready = dict(poller.poll(10))
                    if pull in ready:
                        command = pull.recv_json()
                        accepted = owner.accept(CommandIdentity(**command["_command"]))
                        self.commands.put((command, accepted))
                    if router in ready:
                        request = router.recv_multipart()
                        self.requests.put(request)
                        self.request_count += 1
                        status = {
                            "version": 1,
                            "host_session_id": owner.host_session_id,
                            "control_epoch": owner.epoch,
                            "control_owner": owner.owner,
                            **self.status_patch,
                        }
                        response = self.handler(request, status)
                        if response is not None:
                            router.send_multipart(response)
            except Exception as exc:
                self.errors.put(exc)
            finally:
                router.close()
                pull.close()
                context.term()

        self.server = threading.Thread(target=serve, daemon=True)
        self.server.start()
        self.state_port, self.command_port = self.ports.get(timeout=2)
        self.client = self.make_client()

    def make_client(self):
        return HostClient(
            "127.0.0.1",
            port=self.state_port,
            command_port=self.command_port,
            expected_model="alohamini2pro",
            timeout_s=0.5,
            request_window=1,
        )

    def reply(self, request, status):
        payload = state_payload()
        payload["_safety"] = status
        payload["sample_number"] = self.request_count
        return [request[0], *state_frames(request[1], payload)]

    def tearDown(self):
        self.client.close()
        self.stop.set()
        self.server.join(timeout=2)
        self.assertFalse(self.server.is_alive())
        if not self.errors.empty():
            raise self.errors.get()

    def test_read_does_not_create_command_socket(self):
        self.client.read()
        self.assertIsNone(self.client._command_socket)
        self.assertTrue(self.commands.empty())

    def test_invalid_target_does_not_consume_command_sequence(self):
        state = self.client.read()
        with self.assertRaises(ValueError):
            self.client.send_command({"x.vel": True}, based_on=state)
        identity = self.client.send_command({"x.vel": 0.0}, based_on=state)
        self.assertEqual(identity.sequence, 1)
        self.assertTrue(self.commands.get(timeout=1)[1])

    def test_writing_requires_explicit_robot_model(self):
        with HostClient("127.0.0.1", port=self.state_port) as reader:
            state = reader.read()
            with self.assertRaisesRegex(CommandRejectedError, "expected_model"):
                reader.send_command({"x.vel": 0.0}, based_on=state)
            self.assertIsNone(reader._command_socket)

    def test_transport_failure_clears_context_without_retrying_command(self):
        from unittest.mock import Mock

        state = self.client.read()
        socket = Mock()
        socket.poll.return_value = True
        socket.send.side_effect = zmq.Again()
        self.client._command_socket = socket
        with self.assertRaises(ConnectionError):
            self.client.send_command({"x.vel": 0.1}, based_on=state)
        socket.send.assert_called_once()
        socket.close.assert_called_once_with(linger=0)
        with self.assertRaises(CommandRejectedError):
            self.client.send_command({"x.vel": 0.1}, based_on=state)

    def test_unavailable_command_channel_never_queues_offline_target(self):
        from unittest.mock import Mock

        state = self.client.read()
        socket = Mock()
        socket.poll.return_value = False
        self.client._command_socket = socket
        with self.assertRaises(ConnectionError):
            self.client.send_command({"x.vel": 0.1}, based_on=state)
        socket.send.assert_not_called()
        self.assertIsNone(self.client._command_context)

    def test_model_manifest_is_not_loaded_every_read(self):
        from unittest.mock import patch

        from alohamini.model import get_robot_model

        with patch("alohamini.protocol.get_robot_model", wraps=get_robot_model) as loader:
            self.client.read()
            self.client.read()
            loader.assert_called_once_with("alohamini2pro")

    def test_targets_keep_wire_units_and_host_accepts_identity(self):
        state = self.client.read()
        targets = {"arm_left_shoulder_pan.pos": -12.5, "lift_axis.height_mm": 123.0, "x.vel": 0.1}
        identity = self.client.send_command(targets, based_on=state)
        actual, accepted = self.commands.get(timeout=1)
        self.assertTrue(accepted)
        self.assertEqual(CommandIdentity(**actual.pop("_command")), identity)
        self.assertEqual(actual, targets)
        self.assertEqual(self.client._command_socket.getsockopt(zmq.CONFLATE), 1)
        self.assertEqual(self.client._command_socket.getsockopt(zmq.IMMEDIATE), 1)

    def test_command_sequence_is_monotonic_and_does_not_mutate_targets(self):
        state = self.client.read()
        targets = {"x.vel": 0.0}
        first = self.client.send_command(targets, based_on=state)
        self.commands.get(timeout=1)
        second = self.client.send_command(targets, based_on=state)
        self.commands.get(timeout=1)
        self.assertGreater(second.sequence, first.sequence)
        self.assertEqual(targets, {"x.vel": 0.0})

    def test_inference_duration_is_not_a_250_ms_send_gate(self):
        state = self.client.read()
        state.request_started_s -= 5
        state.received_s -= 5
        self.client.send_command({"x.vel": 0.0}, based_on=state)
        self.assertTrue(self.commands.get(timeout=1)[1])

    def test_old_work_not_relabelled_after_epoch_change(self):
        old = self.client.read()
        self.status_patch = {"control_epoch": 1}
        new = self.client.read()
        self.assertEqual(new._command_context.control_epoch, 1)
        with self.assertRaises(CommandRejectedError):
            self.client.send_command({"x.vel": 0.1}, based_on=old)
        self.assertTrue(self.commands.empty())

    def test_host_restart_invalidates_old_work(self):
        old = self.client.read()
        self.status_patch = {"host_session_id": "restarted"}
        self.client.read()
        with self.assertRaises(CommandRejectedError):
            self.client.send_command({"x.vel": 0.1}, based_on=old)

    def test_newer_state_in_same_epoch_does_not_invalidate_inference(self):
        old = self.client.read()
        self.client.read()
        self.client.send_command({"x.vel": 0.0}, based_on=old)
        self.assertTrue(self.commands.get(timeout=1)[1])

    def test_another_clients_snapshot_cannot_supply_context(self):
        own = self.client.read()
        with self.make_client() as other:
            other.read()
            with self.assertRaises(CommandRejectedError):
                other.send_command({"x.vel": 0.0}, based_on=own)

    def test_owner_change_revokes_current_context(self):
        old = self.client.read()
        self.status_patch = {"control_owner": "another-client"}
        self.client.read()
        with self.assertRaises(CommandRejectedError):
            self.client.send_command({"x.vel": 0.0}, based_on=old)

    def test_invalid_safety_metadata_is_readable_but_not_writable(self):
        for patch in (
            {"version": True},
            {"version": 0},
            {"control_epoch": True},
            {"control_epoch": -1},
            {"host_session_id": None},
        ):
            self.status_patch = patch
            state = self.client.read()
            with self.subTest(patch=patch), self.assertRaises(CommandRejectedError):
                self.client.send_command({"x.vel": 0.0}, based_on=state)

    def test_mutating_public_metadata_does_not_rebind_command(self):
        state = self.client.read()
        state.payload["_safety"]["host_session_id"] = "invented"
        state.payload["_safety"]["control_epoch"] = 123
        identity = self.client.send_command({"x.vel": 0.0}, based_on=state)
        self.assertEqual(identity.host_session_id, "host-session")
        self.assertEqual(identity.control_epoch, 0)
        self.assertTrue(self.commands.get(timeout=1)[1])

    def test_invalid_targets_fail_before_command_connection(self):
        state = self.client.read()
        for targets in (
            {},
            {"x.vel": True},
            {"x.vel": float("inf")},
            {"x.vel": "1"},
            {"unknown.pos": 1},
            {"_command": 1},
            {"lift_axis.vel": 100},
        ):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.client.send_command(targets, based_on=state)
        self.assertIsNone(self.client._command_socket)

    def test_joint_hold_does_not_prevent_teleoperation_retreat(self):
        self.status_patch = {"joint_holds": {"arm_left_shoulder_pan": 0.2}}
        state = self.client.read()
        self.client.send_command({"arm_left_shoulder_pan.pos": -0.1}, based_on=state)
        self.assertTrue(self.commands.get(timeout=1)[1])

    def test_read_timeout_clears_command_context(self):
        state = self.client.read()
        self.handler = lambda *_: None
        self.client._timeout_s = 0.05
        with self.assertRaises(ResponseTimeoutError):
            self.client.read()
        with self.assertRaises(CommandRejectedError):
            self.client.send_command({"x.vel": 0.0}, based_on=state)

    def test_malformed_response_clears_command_context(self):
        state = self.client.read()
        self.handler = lambda request, _: [request[0], request[1], b"invalid"]
        with self.assertRaises(ProtocolError):
            self.client.read()
        with self.assertRaises(CommandRejectedError):
            self.client.send_command({"x.vel": 0.0}, based_on=state)

    def test_request_window_retains_bounded_matching_samples(self):
        self.client._request_window = 3
        first = self.client.read()
        self.assertEqual(len(self.client._pending), 2)
        second = self.client.read()
        self.assertGreater(second.payload["sample_number"], first.payload["sample_number"])
        self.assertEqual(len(self.client._pending), 2)
        for _ in range(3):
            self.assertTrue(self.requests.get(timeout=1)[1].endswith(b":state"))

    def test_newer_response_supersedes_unanswered_earlier_request(self):
        self.client._request_window = 3
        self.handler = lambda request, status: (
            None if self.request_count == 1 else self.reply(request, status)
        )
        state = self.client.read()
        self.assertEqual(state.payload["sample_number"], 2)
        self.assertEqual(len(self.client._pending), 1)

    def test_camera_mode_switch_discards_old_window(self):
        self.client._request_window = 3
        self.client.read()
        old_identity = self.requests.get(timeout=1)[0]

        def images(request, status):
            if request[1].endswith(b":state"):
                return self.reply(request, status)
            payload = state_payload()
            payload["_safety"] = status
            payload["_images"] = ["forward"]
            return [request[0], *state_frames(request[1], payload), b"forward", b"jpeg"]

        self.handler = images
        state = self.client.read(include_images=True)
        self.assertEqual(state.images, {"forward": b"jpeg"})
        while True:
            request = self.requests.get(timeout=1)
            if request[1].endswith(b":full"):
                self.assertEqual(request[0], old_identity)
                break

    def test_close_is_idempotent_and_forbids_further_commands(self):
        state = self.client.read()
        self.client.close()
        self.client.close()
        with self.assertRaises(RuntimeError):
            self.client.send_command({"x.vel": 0.0}, based_on=state)


if __name__ == "__main__":
    unittest.main()
