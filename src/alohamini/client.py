# SPDX-License-Identifier: Apache-2.0
"""Client for the deployed AlohaMini Host, independent of LeRobot and ROS."""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from uuid import uuid4

from alohamini.errors import (
    CommandRejectedError,
    ConnectionError,
    ProtocolError,
    ResponseTimeoutError,
)
from alohamini.protocol import (
    MAX_FRAME_BYTES,
    MAX_MESSAGE_BYTES,
    MAX_MESSAGE_FRAMES,
    HostSnapshot,
    command_target_keys,
    decode_command_context,
    decode_reply,
    encode_command,
    encode_request,
)
from alohamini.schema import CommandIdentity


def control_feedback_valid(snapshot: HostSnapshot | None) -> bool:
    """Bound blind control by the Host watchdog, independently of dataset timing."""
    if snapshot is None:
        return False
    timeout = snapshot.payload["_safety"].get("command_watchdog_timeout_s", 1.0)
    return time.monotonic() - snapshot.request_started_s < timeout


class HostClient:
    """Single-threaded client with a bounded observation request window.

    Use a context manager or call close(). Ordinary read timeouts retain pending
    requests until their lifetime expires. Brief response gaps retain control
    context; prolonged feedback loss prevents new commands.
    Reading never opens a command socket. Commands use deployed Host units, not
    implicit SI conversions. Closing drops queued messages; the Host watchdog,
    not this method, is responsible for stopping a disconnected controller.
    Teleoperation opts into prefetch_before_decode; other workflows retain
    explicit prefetch so their sampling/refresh boundaries do not change.
    """

    def __init__(
        self,
        host: str,
        *,
        port: int = 5556,
        command_port: int = 5555,
        expected_model: str | None = None,
        timeout_s: float = 1.0,
        request_window: int = 3,
        prefetch_before_decode: bool = False,
    ) -> None:
        if not isinstance(host, str) or not host or any(c.isspace() or c in "/:[]" for c in host):
            raise ValueError("host must be an IPv4 address or hostname")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("port must be an integer in [1, 65535]")
        if type(command_port) is not int or not 1 <= command_port <= 65535:
            raise ValueError("command_port must be an integer in [1, 65535]")
        if type(request_window) is not int or not 1 <= request_window <= 16:
            raise ValueError("request_window must be an integer in [1, 16]")
        if type(prefetch_before_decode) is not bool:
            raise ValueError("prefetch_before_decode must be a bool")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or not 0 < timeout_s <= 60
        ):
            raise ValueError("timeout_s must be finite and in (0, 60]")
        if expected_model is not None and (
            not isinstance(expected_model, str) or not expected_model
        ):
            raise ValueError("expected_model must be a nonempty string or None")
        self._endpoint = f"tcp://{host}:{port}"
        self._command_endpoint = f"tcp://{host}:{command_port}"
        self._expected_model = expected_model
        self._timeout_s = float(timeout_s)
        self._context = None
        self._socket = None
        self._command_socket = None
        self._request_window = request_window
        self._prefetch_before_decode = prefetch_before_decode
        self._pending: OrderedDict[bytes, float] = OrderedDict()
        self._image_mode: bool | None = None
        self._camera_recording_id: str | None = None
        self._client_id = uuid4().hex
        self._command_sequence = 0
        self._command_context: CommandIdentity | None = None
        self._last_snapshot: HostSnapshot | None = None
        self._command_keys: frozenset[str] = frozenset()
        self._model_keys: dict[str, frozenset[str]] = {}
        self._closed = False
        self._thread_id: int | None = None

    def __enter__(self) -> HostClient:
        self._check_thread()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def client_id(self) -> str:
        return self._client_id

    def _check_thread(self) -> None:
        if self._closed:
            raise RuntimeError("Client is closed")
        current = threading.get_ident()
        if self._thread_id is not None and self._thread_id != current:
            raise RuntimeError("Use a separate client for each thread")
        self._thread_id = current

    def _connect(self, zmq) -> None:
        if self._context is None:
            self._context = zmq.Context()
        if self._socket is None:
            self._socket = self._context.socket(zmq.DEALER)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt(zmq.IMMEDIATE, 1)
            self._socket.setsockopt(zmq.SNDHWM, self._request_window)
            self._socket.setsockopt(zmq.RCVHWM, self._request_window)
            self._socket.setsockopt(zmq.MAXMSGSIZE, MAX_FRAME_BYTES)
            self._socket.connect(self._endpoint)

    def _invalidate_requests(self) -> None:
        self._pending.clear()
        self._command_context = None
        self._command_keys = frozenset()
        self._last_snapshot = None

    def _discard_socket(self, *, discard_commands: bool = True) -> None:
        self._invalidate_requests()
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if discard_commands and self._command_socket is not None:
            self._command_socket.close(linger=0)
            self._command_socket = None

    def _connect_commands(self, zmq) -> None:
        if self._command_socket is None:
            self._command_socket = self._context.socket(zmq.PUSH)
            self._command_socket.setsockopt(zmq.CONFLATE, 1)
            self._command_socket.setsockopt(zmq.LINGER, 0)
            self._command_socket.setsockopt(zmq.IMMEDIATE, 1)
            self._command_socket.connect(self._command_endpoint)

    def connect_control(self) -> HostSnapshot:
        """Source 5 s startup handshake; connect both channels without sending motion.

        Normal reads retain timeout_s. State-only consumers need not call this
        method and never open a command socket.
        """
        self._check_thread()
        if self._expected_model is None:
            raise CommandRejectedError("Set expected_model before connecting for control")
        import zmq

        deadline = time.monotonic() + 5.0
        try:
            self._connect(zmq)
            self._connect_commands(zmq)
            snapshot = self._read(
                include_images=False, timeout_s=max(0.0, deadline - time.monotonic())
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._command_socket.poll(
                max(1, math.ceil(remaining * 1000)), zmq.POLLOUT
            ):
                raise ConnectionError(
                    f"Command channel unavailable: {self._command_endpoint}; no command sent"
                )
            return snapshot
        except zmq.ZMQError as exc:
            self._discard_socket()
            raise ConnectionError("Host connection failed; no command sent") from exc
        except (ProtocolError, ResponseTimeoutError, ConnectionError):
            self._discard_socket()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._check_thread()
        self._discard_socket()
        if self._context is not None:
            self._context.term()
            self._context = None
        self._closed = True

    @staticmethod
    def _remaining_ms(deadline: float) -> int:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ResponseTimeoutError("Host response deadline exceeded")
        return max(1, math.ceil(remaining * 1000))

    def _fill_requests(self, zmq, *, ordered: bool = False) -> None:
        # Ordinary read() changes mode by discarding pending tokens; only the
        # ordered recorder can retain a full window of consumable camera groups.
        window = (
            1
            if self._image_mode and self._camera_recording_id and not ordered
            else self._request_window
        )
        while len(self._pending) < window:
            token = encode_request(uuid4().hex, include_images=self._image_mode)
            if self._image_mode and self._camera_recording_id is not None:
                token = token.split(b":")[0] + f":{self._camera_recording_id}:record".encode(
                    "ascii"
                )
            started = time.monotonic()
            try:
                self._socket.send(token, flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            self._pending[token] = started

    def set_recording_cameras(self, enabled: bool) -> None:
        """Start a new episode cursor, or return to realtime camera requests.

        As in AlohaMiniClient, reset only request bookkeeping, not the DEALER
        identity. Each enable call starts a new episode, even if already enabled.
        """
        self._check_thread()
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        self._camera_recording_id = uuid4().hex if enabled else None
        self._pending.clear()

    def prefetch(self, *, include_images: bool = False) -> None:
        """Overlap the next request with application work without waiting for I/O.

        This latest-mode helper never prefetches recording camera groups.
        The recorder uses read_recording() to preserve mixed request ordering.
        """
        self._check_thread()
        if type(include_images) is not bool:
            raise ValueError("include_images must be a bool")
        if self._socket is None or (include_images and self._camera_recording_id):
            return
        import zmq

        if self._image_mode != include_images:
            self._pending.clear()
        self._image_mode = include_images
        try:
            self._fill_requests(zmq)
        except zmq.ZMQError as exc:
            self._discard_socket()
            raise ConnectionError(f"Host transport failed: {exc}") from exc

    def _bind_command_context(self, snapshot: HostSnapshot) -> None:
        self._last_snapshot = snapshot
        self._command_context = None
        self._command_keys = frozenset()
        context = decode_command_context(snapshot.payload, client_id=self._client_id)
        if context is None:
            return
        try:
            if snapshot.robot_model not in self._model_keys:
                self._model_keys[snapshot.robot_model] = command_target_keys(snapshot.robot_model)
        except (TypeError, ValueError, KeyError):
            return
        self._command_keys = self._model_keys[snapshot.robot_model]
        self._command_context = context
        snapshot._command_context = context

    def read(self, *, include_images: bool = False) -> HostSnapshot:
        """Return the newest matching reply in a bounded ready batch, or raise.

        The deadline bounds this receive call, not time spent prefetched or sensor age. Host clocks
        and telemetry validity remain explicitly available in the returned payload.
        No cached state is returned here. Recording cursors retain ordered reads.
        """
        return self._read(include_images=include_images, timeout_s=self._timeout_s)

    def read_recording(self, *, include_images: bool = False) -> HostSnapshot:
        """Consume the oldest reply and refill before decoding, as in AlohaMiniClient.

        include_images selects newly queued requests, not the reply being consumed.
        Mixed state/image requests remain ordered; returned images always belong
        to the matched token. Episode boundaries use set_recording_cameras().
        """
        return self._read(include_images=include_images, timeout_s=self._timeout_s, ordered=True)

    def refresh(self, *, include_images: bool = False) -> HostSnapshot:
        """Discard prefetched replies, retaining bounded command context on timeout.

        Only a decoded response updates feedback age and ownership. A failed
        refresh does not renew either; send_command still enforces their validity.
        """
        self._check_thread()
        self._pending.clear()
        return self.read(include_images=include_images)

    def _read(
        self, *, include_images: bool, timeout_s: float, ordered: bool = False
    ) -> HostSnapshot:
        self._check_thread()
        if type(include_images) is not bool:
            raise ValueError("include_images must be a bool")
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("The Host client requires the SDK's 'zmq' extra.") from exc
        now = time.monotonic()
        deadline = now + timeout_s
        request_lifetime_s = (
            self._last_snapshot.payload["_safety"].get("command_watchdog_timeout_s", 1.0)
            if self._last_snapshot is not None
            else max(1.0, timeout_s)
        )
        if not ordered:
            # A poll timeout is not request expiry. Keep delayed replies usable,
            # but release slots occupied by lost or already stale requests.
            self._pending = OrderedDict(
                (token, started)
                for token, started in self._pending.items()
                if now - started < request_lifetime_s
            )
        if self._image_mode != include_images and not ordered:
            # Keep the Host's (DEALER identity, episode UUID) camera cursor alive
            # across multirate state/image requests. Old tokens are discarded below.
            self._pending.clear()
        self._image_mode = include_images
        try:
            self._connect(zmq)
            self._fill_requests(zmq, ordered=ordered)
            while not self._pending:
                if not self._socket.poll(self._remaining_ms(deadline), zmq.POLLOUT):
                    raise ResponseTimeoutError("Could not send request before deadline")
                self._fill_requests(zmq, ordered=ordered)
            selected = None
            extra_reads = 0
            while True:
                if selected is None:
                    if not self._socket.poll(self._remaining_ms(deadline), zmq.POLLIN):
                        raise ResponseTimeoutError("Host did not respond before deadline")
                else:
                    # Do not wait for future frames or drain indefinitely. Defer
                    # replenishment until this batch ends so instant replies to
                    # new requests cannot keep this read busy forever.
                    if (
                        ordered
                        or self._camera_recording_id is not None
                        or extra_reads >= self._request_window - 1
                        or not self._socket.poll(0, zmq.POLLIN)
                    ):
                        break
                    extra_reads += 1
                parts = []
                size = 0
                while True:
                    part = self._socket.recv(flags=zmq.NOBLOCK)
                    parts.append(part)
                    size += len(part)
                    if len(parts) > MAX_MESSAGE_FRAMES or size > MAX_MESSAGE_BYTES:
                        raise ProtocolError("Multipart response exceeds the size limit")
                    if not self._socket.getsockopt(zmq.RCVMORE):
                        break
                received = time.monotonic()
                token = parts[0]
                if token not in self._pending:
                    continue
                if ordered and token != next(iter(self._pending)):
                    continue
                started = self._pending[token]
                # A newer response supersedes earlier unanswered requests, as in the ROS client.
                while self._pending:
                    candidate, _ = self._pending.popitem(last=False)
                    if candidate == token:
                        break
                if not ordered and received - started >= request_lifetime_s:
                    if selected is None:
                        self._fill_requests(zmq)
                    continue
                # Popping through token above means an older/out-of-order reply
                # cannot replace this selection, even if request times are equal.
                selected = parts, token, started, received
            if ordered or (
                self._prefetch_before_decode and not (include_images and self._camera_recording_id)
            ):
                self._fill_requests(zmq, ordered=ordered)
            parts, token, started, received = selected
            payload, images = decode_reply(
                parts,
                token=token,
                expected_model=self._expected_model,
                include_images=not token.endswith(b":state") if ordered else include_images,
            )
            snapshot = HostSnapshot(payload, images, started, received)
            self._bind_command_context(snapshot)
            return snapshot
        except zmq.ZMQError as exc:
            self._discard_socket()
            raise ConnectionError(f"Host transport failed: {exc}") from exc
        except ResponseTimeoutError:
            # Ordered recording keeps its existing cursor recovery semantics.
            # Realtime readers can consume a delayed reply on their next poll.
            if ordered:
                self._pending.clear()
            raise
        except ProtocolError:
            self._discard_socket()
            raise

    def send_command(
        self, targets: Mapping[str, float], *, based_on: HostSnapshot
    ) -> CommandIdentity | None:
        """Queue one command in deployed units, without a motion acknowledgement.

        based_on must retain this client's current Host session/epoch. The latest
        validated feedback, not the policy input's age, bounds blind control.
        Applications must validate inference results against the current context
        before sending them. Contact telemetry does not prohibit submission;
        gripper contact limits and sustained overcurrent stops remain on Host.
        Returns None on prolonged feedback loss or temporary backpressure without
        closing the connection or retrying the target. Call connect_control()
        before starting a control loop.
        """
        self._check_thread()
        if self._expected_model is None:
            raise CommandRejectedError("Set expected_model before sending motor commands")
        if (
            not isinstance(based_on, HostSnapshot)
            or self._command_context is None
            or based_on._command_context != self._command_context
        ):
            raise CommandRejectedError("Command requires this client's current Host session/epoch")
        identity = CommandIdentity(
            self._client_id,
            self._command_sequence + 1,
            self._command_context.host_session_id,
            self._command_context.control_epoch,
        )
        if not control_feedback_valid(self._last_snapshot):
            return None
        encoded = encode_command(targets, identity, allowed_targets=self._command_keys)
        self._command_sequence = identity.sequence
        import zmq

        try:
            self._connect_commands(zmq)
            self._command_socket.send(encoded, flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        except zmq.ZMQError as exc:
            self._discard_socket()
            raise ConnectionError("Command transport failed; command was not queued") from exc
        except ConnectionError:
            self._discard_socket()
            raise
        return identity
