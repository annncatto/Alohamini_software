import threading
import time
import unittest
from queue import Queue
from unittest.mock import Mock, patch

from support import state_frames, state_payload

from alohamini.client import HostClient
from alohamini.errors import (
    ConnectionError,
    ModelMismatchError,
    ProtocolError,
    ResponseTimeoutError,
)

try:
    import zmq
except ImportError:
    zmq = None


class ClientConfigurationTests(unittest.TestCase):
    def test_read_accepts_matching_reply_when_receiving_finishes_after_poll_deadline(self):
        if zmq is None:
            self.skipTest("pyzmq is not installed")
        with HostClient("127.0.0.1", expected_model="alohamini2pro", timeout_s=0.2) as client:
            client._socket = Mock()
            client._pending[b"token:state"] = 1.0
            client._image_mode = False
            client._socket.recv.side_effect = state_frames(b"token:state")
            client._socket.getsockopt.side_effect = [True, False]
            with (
                patch.object(client, "_connect"),
                patch.object(client, "_fill_requests"),
                patch("alohamini.client.time.monotonic", side_effect=[1.0, 1.1, 1.21]),
            ):
                result = client.read()
            self.assertEqual(result.received_s, 1.21)
            self.assertEqual(result.request_started_s, 1.0)

    def test_prefetch_never_connects_or_consumes_recording_camera_groups(self):
        with HostClient("127.0.0.1") as client:
            client.prefetch()
            self.assertIsNone(client._context)
            client._socket = Mock()
            client.set_recording_cameras(True)
            client.prefetch(include_images=True)
            client._socket.send.assert_not_called()
            self.assertFalse(client._pending)

    def test_only_recording_image_requests_use_a_single_request_window(self):
        with HostClient("127.0.0.1", request_window=3) as client:
            client._socket = Mock()
            transport = Mock(NOBLOCK=1, Again=RuntimeError)
            for enabled, images, expected in (
                (False, False, 3),
                (True, False, 3),
                (True, True, 1),
                (False, True, 3),
            ):
                with self.subTest(recording=enabled, images=images):
                    client.set_recording_cameras(enabled)
                    client._image_mode = images
                    client._socket.reset_mock()
                    client._fill_requests(transport)
                    self.assertEqual(client._socket.send.call_count, expected)
                    self.assertEqual(len(client._pending), expected)

    def test_rejects_invalid_connection_parameters(self):
        for kwargs in (
            {"host": ""},
            {"host": "tcp://localhost"},
            {"host": "bad host"},
            {"port": 0},
            {"port": True},
            {"port": 65536},
            {"timeout_s": 0},
            {"timeout_s": float("nan")},
            {"timeout_s": float("inf")},
            {"timeout_s": True},
            {"timeout_s": "1"},
            {"expected_model": 1},
            {"prefetch_before_decode": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HostClient(**{"host": "127.0.0.1", **kwargs})

    @unittest.skipIf(zmq is None, "pyzmq is not installed")
    def test_control_handshake_waits_for_command_channel_without_sending(self):
        with HostClient("127.0.0.1", expected_model="alohamini2pro", timeout_s=0.2) as client:
            client._context = Mock()
            command = Mock()
            client._command_socket = command
            with patch.object(client, "_connect"), patch.object(client, "_read") as read:
                state = client.connect_control()
            self.assertIs(state, read.return_value)
            self.assertGreater(read.call_args.kwargs["timeout_s"], 4.0)
            self.assertGreater(command.poll.call_args.args[0], 4000)
            command.send.assert_not_called()
            self.assertEqual(client._timeout_s, 0.2)

    @unittest.skipIf(zmq is None, "pyzmq is not installed")
    def test_control_handshake_failure_closes_both_channels(self):
        with HostClient("127.0.0.1", expected_model="alohamini2pro") as client:
            client._context, client._socket, client._command_socket = Mock(), Mock(), Mock()
            state_socket, command = client._socket, client._command_socket
            command.poll.return_value = 0
            with patch.object(client, "_connect"), patch.object(client, "_read"):
                with self.assertRaisesRegex(ConnectionError, "5555; no command sent"):
                    client.connect_control()
            command.send.assert_not_called()
            command.close.assert_called_once()
            state_socket.close.assert_called_once()

    def test_construction_and_close_do_not_open_sockets(self):
        client = HostClient("127.0.0.1")
        self.assertIsNone(client._context)
        self.assertIsNone(client._socket)
        client.close()
        client.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            client.read()

    def test_client_rejects_cross_thread_use(self):
        client = HostClient("127.0.0.1")
        client.__enter__()
        errors = []

        def read_from_another_thread():
            try:
                client.read()
            except RuntimeError as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=read_from_another_thread)
        thread.start()
        thread.join(timeout=1)
        self.assertEqual(errors, ["Use a separate client for each thread"])
        client.close()


@unittest.skipIf(zmq is None, "pyzmq is not installed")
class ClientTransportTests(unittest.TestCase):
    def setUp(self):
        self.port_queue = Queue()
        self.requests = Queue()
        self.errors = Queue()
        self.stop = threading.Event()
        self.handler = lambda request: [request[0], *state_frames(request[1])]

        def serve():
            context = zmq.Context()
            socket = context.socket(zmq.ROUTER)
            socket.setsockopt(zmq.LINGER, 0)
            try:
                port = socket.bind_to_random_port("tcp://127.0.0.1")
                self.port_queue.put(port)
                while not self.stop.is_set():
                    if not socket.poll(20):
                        continue
                    request = socket.recv_multipart()
                    self.requests.put(request)
                    response = self.handler(request)
                    if response is not None:
                        socket.send_multipart(response)
            except Exception as exc:
                self.errors.put(exc)
            finally:
                socket.close(linger=0)
                context.term()

        self.server = threading.Thread(target=serve, daemon=True)
        self.server.start()
        self.port = self.port_queue.get(timeout=2)
        self.client = HostClient(
            "127.0.0.1",
            port=self.port,
            timeout_s=0.5,
            expected_model="alohamini2pro",
            request_window=1,
        )

    def tearDown(self):
        self.client.close()
        self.stop.set()
        self.server.join(timeout=2)
        self.assertFalse(self.server.is_alive(), "Local test server did not stop")
        if not self.errors.empty():
            raise self.errors.get()

    def test_state_read_uses_only_dealer_and_one_token(self):
        snapshot = self.client.read()
        request = self.requests.get(timeout=1)
        self.assertEqual(len(request), 2)
        self.assertTrue(request[1].endswith(b":state"))
        self.assertEqual(self.client._socket.getsockopt(zmq.TYPE), zmq.DEALER)
        self.assertEqual(snapshot.robot_model, "alohamini2pro")
        self.assertGreaterEqual(snapshot.round_trip_s, 0)
        self.assertLess(snapshot.round_trip_s, 0.5)

    def test_image_read_preserves_jpeg_bytes(self):
        def images(request):
            payload = state_payload()
            payload["_images"] = ["forward"]
            return [request[0], *state_frames(request[1], payload), b"forward", b"jpeg"]

        self.handler = images
        snapshot = self.client.read(include_images=True)
        self.assertTrue(self.requests.get(timeout=1)[1].endswith(b":full"))
        self.assertEqual(snapshot.images, {"forward": b"jpeg"})

    def test_control_startup_accepts_response_slower_than_runtime_timeout(self):
        context = zmq.Context()
        commands = context.socket(zmq.PULL)
        commands.setsockopt(zmq.LINGER, 0)
        port = commands.bind_to_random_port("tcp://127.0.0.1")
        self.client._command_endpoint = f"tcp://127.0.0.1:{port}"
        self.client._timeout_s = 0.05

        def delayed_reply(request):
            time.sleep(0.15)
            return [request[0], *state_frames(request[1])]

        self.handler = delayed_reply
        try:
            self.assertEqual(self.client.connect_control().robot_model, "alohamini2pro")
            self.assertFalse(commands.poll(0))  # Handshake never moves the robot.
            self.assertEqual(self.client._timeout_s, 0.05)
            with self.assertRaises(ResponseTimeoutError):
                self.client.read()
        finally:
            commands.close()
            context.term()

    def test_timeout_never_returns_cached_state_and_next_read_recovers(self):
        self.client.read()
        connection = self.client._socket
        old_identity = self.requests.get(timeout=1)[0]
        self.handler = lambda _: None
        self.client._timeout_s = 0.08
        started = time.monotonic()
        with self.assertRaises(ResponseTimeoutError):
            self.client.read()
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertIs(self.client._socket, connection)
        self.assertFalse(self.client._pending)
        self.assertIsNone(self.client._command_context)
        self.requests.get(timeout=1)
        self.handler = lambda request: [request[0], *state_frames(request[1])]
        self.client._timeout_s = 0.5
        self.assertEqual(self.client.read().robot_model, "alohamini2pro")
        self.assertEqual(old_identity, self.requests.get(timeout=1)[0])

    def test_wrong_tokens_do_not_reset_deadline(self):
        self.handler = lambda request: [request[0], *state_frames(b"old:state")]
        self.client._timeout_s = 0.1
        with self.assertRaises(ResponseTimeoutError):
            self.client.read()

    def test_prefetched_reply_is_not_expired_before_receive_for_teleop_or_recording(self):
        self.client.read()  # Complete the initial asynchronous TCP connection.
        for ordered in (False, True):
            with self.subTest(ordered=ordered):
                self.client._invalidate_requests()
                self.client.prefetch()
                token = next(iter(self.client._pending))
                started = time.monotonic() - 2.0
                self.client._pending[token] = started
                read = self.client.read_recording if ordered else self.client.read
                result = read()
                self.assertEqual(result.request_started_s, started)
                self.assertEqual(result.robot_model, "alohamini2pro")
                self.assertNotIn(token, self.client._pending)

    def test_refresh_discards_prefetched_replies_without_reconnecting(self):
        self.client._request_window = 3
        self.client.read()
        connection = self.client._socket
        old_tokens = set(self.client._pending)
        result = self.client.refresh()
        self.assertIs(self.client._socket, connection)
        self.assertEqual(result.robot_model, "alohamini2pro")
        self.assertFalse(old_tokens & set(self.client._pending))

    def test_late_old_reply_is_ignored_after_timeout_on_same_connection(self):
        delayed = []
        self.client._request_window = 3
        self.handler = lambda request: delayed.append(request) or None
        self.client._timeout_s = 0.05
        with self.assertRaises(ResponseTimeoutError):
            self.client.read()
        connection = self.client._socket
        calls = 0

        def reply(request):
            nonlocal calls
            calls += 1
            token = delayed[0][1] if calls == 1 else request[1]
            return [request[0], *state_frames(token)]

        self.handler = reply
        self.client._request_window = 3
        self.client._timeout_s = 0.5
        result = self.client.read()
        self.assertEqual(result.robot_model, "alohamini2pro")
        self.assertIs(self.client._socket, connection)

    def test_brief_timeout_keeps_command_channel_and_context(self):
        def reply(request):
            payload = state_payload()
            payload["_safety"]["control_epoch"] = 0
            return [request[0], *state_frames(request[1], payload)]

        self.handler = reply
        old = self.client.read()
        command = Mock()
        self.client._command_socket = command
        self.handler = lambda _: None
        self.client._timeout_s = 0.08
        with self.assertRaises(ResponseTimeoutError):
            self.client.read()
        self.assertIs(self.client._command_socket, command)
        command.close.assert_not_called()
        self.assertIsNotNone(self.client.send_command({"x.vel": 0}, based_on=old))
        command.send.assert_called_once()
        command.reset_mock()
        self.handler = reply
        self.client._timeout_s = 0.5
        fresh = self.client.read()
        self.client.send_command({"x.vel": 0}, based_on=fresh)
        command.send.assert_called_once()

    def test_prefetch_submits_next_request_before_the_next_read(self):
        self.client.read()
        first = self.requests.get(timeout=1)
        self.client.prefetch()
        second = self.requests.get(timeout=1)
        self.assertEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])
        self.assertEqual(self.client.read().robot_model, "alohamini2pro")

    def test_teleop_replenishes_three_requests_before_decoding(self):
        from alohamini.protocol import decode_reply

        self.client._request_window = 3
        self.client._prefetch_before_decode = True

        def decode(*args, **kwargs):
            # Three initial requests plus the replenishment are already sent,
            # even if the decoder/input/preview hasn't done any work yet.
            requests = [self.requests.get(timeout=1) for _ in range(4)]
            self.assertEqual(len({request[1] for request in requests}), 4)
            self.assertEqual(len(self.client._pending), 3)
            return decode_reply(*args, **kwargs)

        with patch("alohamini.client.decode_reply", side_effect=decode):
            self.assertEqual(self.client.read().robot_model, "alohamini2pro")

    def test_prefetched_protocol_failure_invalidates_context_and_pending_requests(self):
        self.client._request_window = 3
        self.client._prefetch_before_decode = True
        self.handler = lambda request: [request[0], request[1], b"invalid"]
        with self.assertRaises(ProtocolError):
            self.client.read()
        self.assertFalse(self.client._pending)
        self.assertIsNone(self.client._command_context)

    def test_recording_preserves_mixed_request_order_and_refills_before_decode(self):
        from alohamini.protocol import decode_reply

        self.client._request_window = 3
        self.client.set_recording_cameras(True)
        tokens, decoded = [], []

        def reply(request):
            payload = state_payload()
            payload["_images"] = [] if request[1].endswith(b":state") else ["forward"]
            images = [] if not payload["_images"] else [b"forward", request[1]]
            return [request[0], *state_frames(request[1], payload), *images]

        def decode(*args, **kwargs):
            count = 4 if not decoded else 1
            tokens.extend(self.requests.get(timeout=1)[1] for _ in range(count))
            decoded.append(kwargs["token"])
            self.assertEqual(decoded[-1], tokens[len(decoded) - 1])
            self.assertEqual(len(self.client._pending), 3)
            self.assertEqual(kwargs["include_images"], not decoded[-1].endswith(b":state"))
            return decode_reply(*args, **kwargs)

        self.handler = reply
        with patch("alohamini.client.decode_reply", side_effect=decode):
            for images in (True, False, True, False, True, False, True):
                snapshot = self.client.read_recording(include_images=images)
                expected = {} if decoded[-1].endswith(b":state") else {"forward": decoded[-1]}
                self.assertEqual(snapshot.images, expected)
        self.assertEqual(len(set(tokens)), len(tokens))
        self.assertTrue(tokens[4].endswith(b":state"))
        self.assertTrue(tokens[5].endswith(b":record"))

    def test_recording_new_episode_does_not_consume_old_prefetched_images(self):
        self.client._request_window = 3
        self.client.set_recording_cameras(True)
        self.client.read_recording(include_images=True)
        old_tokens = set(self.client._pending)
        old_episode = self.client._camera_recording_id
        self.client.set_recording_cameras(True)
        self.assertFalse(self.client._pending)
        self.assertNotEqual(self.client._camera_recording_id, old_episode)
        self.client.read_recording(include_images=True)
        self.assertFalse(old_tokens & set(self.client._pending))

    def test_recording_timeout_invalidates_old_queue_and_recovers(self):
        self.client._request_window = 3
        self.client.set_recording_cameras(True)
        self.handler = lambda _: None
        self.client._timeout_s = 0.05
        with self.assertRaises(ResponseTimeoutError):
            self.client.read_recording(include_images=True)
        self.assertFalse(self.client._pending)
        self.assertIsNone(self.client._command_context)
        self.handler = lambda request: [request[0], *state_frames(request[1])]
        self.assertEqual(self.client.read_recording().robot_model, "alohamini2pro")

    def test_protocol_failure_still_closes_the_command_channel(self):
        command = Mock()
        self.client._command_socket = command
        self.handler = lambda request: [request[0], request[1], b"invalid"]
        with self.assertRaises(ProtocolError):
            self.client.read()
        command.close.assert_called_once()
        self.assertIsNone(self.client._command_socket)

    def test_protocol_failure_can_be_followed_by_a_good_read(self):
        self.handler = lambda request: [request[0], request[1], b"invalid"]
        with self.assertRaises(ProtocolError):
            self.client.read()
        self.assertIsNone(self.client._socket)
        self.handler = lambda request: [request[0], *state_frames(request[1])]
        self.assertEqual(self.client.read().robot_model, "alohamini2pro")

    def test_model_mismatch_discards_connection(self):
        def wrong_model(request):
            payload = state_payload()
            payload["_robot_metadata"]["robot_model"] = "alohamini1"
            return [request[0], *state_frames(request[1], payload)]

        self.handler = wrong_model
        with self.assertRaises(ModelMismatchError):
            self.client.read()
        self.assertIsNone(self.client._socket)

    def test_excessive_multipart_response_discards_connection(self):
        self.handler = lambda request: [request[0], request[1], *([b"x"] * 40)]
        with self.assertRaises(ProtocolError):
            self.client.read()
        self.assertIsNone(self.client._socket)

    def test_repeated_reads_use_distinct_tokens(self):
        self.client.read()
        self.client.read()
        first, second = self.requests.get(timeout=1), self.requests.get(timeout=1)
        self.assertEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])

    def test_recording_episode_and_mode_changes_preserve_transport_identity(self):
        self.client.set_recording_cameras(True)
        self.client.read(include_images=True)
        first = self.requests.get(timeout=1)
        self.assertTrue(first[1].endswith(b":record"))
        episode = first[1].split(b":")[1]
        self.client.read()
        state = self.requests.get(timeout=1)
        self.assertTrue(state[1].endswith(b":state"))
        self.client.read(include_images=True)
        second = self.requests.get(timeout=1)
        self.assertEqual(second[0], first[0])
        self.assertEqual(second[1].split(b":")[1], episode)
        self.client.set_recording_cameras(True)
        self.client.read(include_images=True)
        third = self.requests.get(timeout=1)
        self.assertEqual(third[0], first[0])
        self.assertNotEqual(third[1].split(b":")[1], episode)
        self.client.set_recording_cameras(False)
        self.client.read(include_images=True)
        self.assertTrue(self.requests.get(timeout=1)[1].endswith(b":full"))

    def test_recording_does_not_prefetch_frames_discarded_by_state_reads(self):
        self.client._request_window = 3
        self.client._prefetch_before_decode = True
        consumed = []

        def recording(request):
            payload = state_payload()
            if request[1].endswith(b":record"):
                consumed.append(len(consumed) + 1)
                payload["capture_index"] = consumed[-1]
            return [request[0], *state_frames(request[1], payload)]

        self.handler = recording
        self.client.set_recording_cameras(True)
        for capture in range(1, 4):
            result = self.client.read(include_images=True)
            self.assertEqual(result.payload["capture_index"], capture)
            self.assertFalse(self.client._pending)
            # Multirate clients alternate camera reads with control-state reads.
            self.client.read()
        self.assertEqual(consumed, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
