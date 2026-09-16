import threading
import time
import unittest
from queue import Queue

from support import state_frames, state_payload

from alohamini.client import HostClient
from alohamini.errors import ModelMismatchError, ProtocolError, ResponseTimeoutError

try:
    import zmq
except ImportError:
    zmq = None


class ClientConfigurationTests(unittest.TestCase):
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
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HostClient(**{"host": "127.0.0.1", **kwargs})

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

    def test_timeout_never_returns_cached_state_and_next_read_recovers(self):
        self.client.read()
        old_identity = self.requests.get(timeout=1)[0]
        self.handler = lambda _: None
        self.client._timeout_s = 0.08
        started = time.monotonic()
        with self.assertRaises(ResponseTimeoutError):
            self.client.read()
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertIsNone(self.client._socket)
        self.requests.get(timeout=1)
        self.handler = lambda request: [request[0], *state_frames(request[1])]
        self.client._timeout_s = 0.5
        self.assertEqual(self.client.read().robot_model, "alohamini2pro")
        self.assertNotEqual(old_identity, self.requests.get(timeout=1)[0])

    def test_wrong_tokens_do_not_reset_deadline(self):
        self.handler = lambda request: [request[0], *state_frames(b"old:state")]
        self.client._timeout_s = 0.1
        with self.assertRaises(ResponseTimeoutError):
            self.client.read()

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


if __name__ == "__main__":
    unittest.main()
