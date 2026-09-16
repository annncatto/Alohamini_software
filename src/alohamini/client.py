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


class HostClient:
    """Single-threaded client with a bounded observation request window.

    Use a context manager or call close(). Failed reads discard their connection;
    the next read creates a new DEALER identity and cannot consume the old reply.
    Reading never opens a command socket. Commands use deployed Host units, not
    implicit SI conversions. Closing drops queued messages; the Host watchdog,
    not this method, is responsible for stopping a disconnected controller.
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
    ) -> None:
        if not isinstance(host, str) or not host or any(c.isspace() or c in "/:[]" for c in host):
            raise ValueError("host must be an IPv4 address or hostname")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("port must be an integer in [1, 65535]")
        if type(command_port) is not int or not 1 <= command_port <= 65535:
            raise ValueError("command_port must be an integer in [1, 65535]")
        if type(request_window) is not int or not 1 <= request_window <= 16:
            raise ValueError("request_window must be an integer in [1, 16]")
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
        self._pending: OrderedDict[bytes, float] = OrderedDict()
        self._image_mode: bool | None = None
        self._camera_recording_id: str | None = None
        self._client_id = uuid4().hex
        self._command_sequence = 0
        self._command_context: CommandIdentity | None = None
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

    def _discard_socket(self, *, discard_commands: bool = True) -> None:
        self._pending.clear()
        self._command_context = None
        self._command_keys = frozenset()
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if discard_commands and self._command_socket is not None:
            self._command_socket.close(linger=0)
            self._command_socket = None

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

    def _fill_requests(self, zmq) -> None:
        while len(self._pending) < self._request_window:
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

    def _bind_command_context(self, snapshot: HostSnapshot) -> None:
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
        """Return one matching response, or raise; never return cached state.

        The deadline bounds request/response time, not physical sensor age. Host clocks
        and telemetry validity remain explicitly available in the returned payload.
        """
        self._check_thread()
        if type(include_images) is not bool:
            raise ValueError("include_images must be a bool")
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("The Host client requires the SDK's 'zmq' extra.") from exc
        now = time.monotonic()
        deadline = now + self._timeout_s
        if self._image_mode != include_images:
            # Keep the Host's (DEALER identity, episode UUID) camera cursor alive
            # across multirate state/image requests. Old tokens are discarded below.
            self._pending.clear()
        elif self._pending and now - next(iter(self._pending.values())) >= self._timeout_s:
            self._discard_socket(discard_commands=False)
        self._image_mode = include_images
        try:
            self._connect(zmq)
            self._fill_requests(zmq)
            while not self._pending:
                if not self._socket.poll(self._remaining_ms(deadline), zmq.POLLOUT):
                    raise ResponseTimeoutError("Could not send request before deadline")
                self._fill_requests(zmq)
            while True:
                if not self._socket.poll(self._remaining_ms(deadline), zmq.POLLIN):
                    raise ResponseTimeoutError("Host did not respond before deadline")
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
                if received > deadline:
                    raise ResponseTimeoutError("Host response arrived after deadline")
                token = parts[0]
                if token not in self._pending:
                    continue
                started = self._pending[token]
                if received - started > self._timeout_s:
                    raise ResponseTimeoutError("Pending Host response expired")
                # A newer response supersedes earlier unanswered requests, as in the ROS client.
                while self._pending:
                    candidate, _ = self._pending.popitem(last=False)
                    if candidate == token:
                        break
                payload, images = decode_reply(
                    parts,
                    token=token,
                    expected_model=self._expected_model,
                    include_images=include_images,
                )
                snapshot = HostSnapshot(payload, images, started, received)
                self._bind_command_context(snapshot)
                return snapshot
        except zmq.ZMQError as exc:
            self._discard_socket()
            raise ConnectionError(f"Host transport failed: {exc}") from exc
        except (ProtocolError, ResponseTimeoutError):
            self._discard_socket()
            raise

    def send_command(
        self, targets: Mapping[str, float], *, based_on: HostSnapshot
    ) -> CommandIdentity:
        """Queue one command in deployed units, without a motion acknowledgement.

        based_on must retain this client's observed Host session/epoch. Inference
        duration is not a feedback-age gate; Host feedback supervision remains
        authoritative. Never relabel old work with a new epoch or retry a failed
        write. Joint holds are exposed to applications, not an unconditional send
        prohibition: teleoperation must still be able to retreat out of contact.
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
        encoded = encode_command(targets, identity, allowed_targets=self._command_keys)
        self._command_sequence = identity.sequence
        import zmq

        try:
            if self._command_socket is None:
                self._command_socket = self._context.socket(zmq.PUSH)
                self._command_socket.setsockopt(zmq.CONFLATE, 1)
                self._command_socket.setsockopt(zmq.LINGER, 0)
                self._command_socket.setsockopt(zmq.IMMEDIATE, 1)
                self._command_socket.connect(self._command_endpoint)
            if not self._command_socket.poll(
                max(1, math.ceil(self._timeout_s * 1000)), zmq.POLLOUT
            ):
                raise ConnectionError("Command channel unavailable; command was not queued")
            self._command_socket.send(encoded, flags=zmq.NOBLOCK)
        except zmq.ZMQError as exc:
            self._discard_socket()
            raise ConnectionError("Command transport failed; command was not acknowledged") from exc
        except ConnectionError:
            self._discard_socket()
            raise
        return identity
