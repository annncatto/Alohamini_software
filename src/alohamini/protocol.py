"""Encode and decode the deployed Host protocol, independently of transport.

This module preserves legacy values and metadata. It does not interpret motor
normalization, turn telemetry into training vectors, or authorize commands.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from alohamini._validation import finite_number, identifier
from alohamini.errors import ModelMismatchError, ProtocolError
from alohamini.model import get_robot_model
from alohamini.schema import CommandIdentity

MAX_JSON_BYTES = 1024 * 1024
MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_MESSAGE_BYTES = 32 * 1024 * 1024
MAX_MESSAGE_FRAMES = 34  # token, JSON, and at most 16 camera/name pairs


def encode_request(request_id: str, *, include_images: bool = False) -> bytes:
    """Build a state/full request token; the caller supplies its unique ID."""
    identifier(request_id, "request_id")
    if not request_id.isascii() or any(c.isspace() or c == ":" for c in request_id):
        raise ValueError("request_id must be ASCII without whitespace or colons")
    if type(include_images) is not bool:
        raise ValueError("include_images must be a bool")
    return f"{request_id}:{'full' if include_images else 'state'}".encode("ascii")


def decode_command_context(payload: Mapping[str, Any], *, client_id: str) -> CommandIdentity | None:
    """Read session/epoch for this owner, or None if unavailable or incompatible.

    Sequence zero is a context marker, not a sequence allocated for transmission.
    This parses advertised state; it does not grant control or authenticate a peer.
    """
    status = payload.get("_safety", {})
    if not isinstance(status, Mapping):
        return None
    if type(status.get("version")) is not int or status["version"] != 1:
        return None
    if status.get("feedback_valid", True) is not True:
        return None
    if "control_owner" not in status or status["control_owner"] not in (None, client_id):
        return None
    try:
        return CommandIdentity(
            client_id, 0, status.get("host_session_id"), status.get("control_epoch")
        )
    except (TypeError, ValueError):
        return None


def command_target_keys(robot_model: str) -> frozenset[str]:
    """Resolve the deployed wire target names from an authoritative model."""
    model = get_robot_model(robot_model)
    return frozenset(
        [f"{m.name}.pos" for m in model.actuators if m.name.startswith("arm_")]
        + ["x.vel", "y.vel", "theta.vel", "lift_axis.height_mm"]
    )


def encode_command(
    targets: Mapping[str, float],
    identity: CommandIdentity,
    *,
    allowed_targets: frozenset[str],
) -> bytes:
    """Validate targets and encode one command without changing deployed units.

    The caller resolves allowed_targets once with command_target_keys(), maintains
    sequence/session state, and checks that its source snapshot remains applicable.
    Hardware limits and execution authority remain Host responsibilities.
    """
    if not isinstance(identity, CommandIdentity):
        raise ValueError("identity must be a CommandIdentity")
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("Command targets must be a nonempty mapping")
    payload = dict(targets)
    for name, value in payload.items():
        if name == "_command" or name not in allowed_targets:
            raise ValueError(f"Unknown deployed Host target: {name}")
        finite_number(value, name)
    payload["_command"] = {
        "client_id": identity.client_id,
        "sequence": identity.sequence,
        "host_session_id": identity.host_session_id,
        "control_epoch": identity.control_epoch,
    }
    return json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()


def decode_command(
    data: bytes, *, allowed_targets: frozenset[str]
) -> tuple[CommandIdentity, dict[str, float]]:
    """Validate inbound commands before they enter the Host command gate."""
    if not isinstance(data, bytes) or len(data) > MAX_JSON_BYTES:
        raise ProtocolError("Command exceeds the JSON size limit")
    try:
        payload = json.loads(
            data, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        if not isinstance(payload, dict):
            raise ValueError("Command JSON must be an object")
        _check_numbers(payload)
        metadata = payload.pop("_command", None)
        if not isinstance(metadata, dict):
            raise ValueError("Command identity is required")
        identity = CommandIdentity(**metadata)
        # Share field/value validation with clients; never trust client-side checks.
        encode_command(payload, identity, allowed_targets=allowed_targets)
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise ProtocolError(f"Invalid command: {exc}") from exc
    return identity, payload


def encode_reply(payload: Mapping[str, Any], images: Mapping[str, bytes]) -> list[bytes]:
    """Build JSON/camera frames; the ROUTER adds routing identity and request token."""
    state = {**payload, "_images": list(images), "_image_encoding": "jpeg"}
    parts = [json.dumps(state, allow_nan=False, separators=(",", ":")).encode()]
    for name, jpeg in images.items():
        if not isinstance(name, str) or not name or not isinstance(jpeg, bytes) or not jpeg:
            raise ProtocolError("Images require nonempty names and encoded bytes")
        parts.extend((name.encode(), jpeg))
    if (
        len(parts) + 1 > MAX_MESSAGE_FRAMES
        or len(parts[0]) > MAX_JSON_BYTES
        or any(len(part) > MAX_FRAME_BYTES for part in parts)
        or sum(map(len, parts)) + 256 > MAX_MESSAGE_BYTES
    ):
        raise ProtocolError("Response exceeds the multipart size limit")
    return parts


@dataclass
class HostSnapshot:
    """One uncached response, with client-clock request and receipt timestamps.

    ``payload`` retains the Host's units, clock domains and validity metadata.
    ``images`` contains encoded JPEG bytes, not decoded pixel arrays. Each response
    owns its dictionaries; consumers may modify them without altering another read.
    """

    payload: dict[str, Any]
    images: dict[str, bytes]
    request_started_s: float
    received_s: float
    _command_context: CommandIdentity | None = field(default=None, init=False, repr=False)

    @property
    def robot_model(self) -> str:
        return self.payload["_robot_metadata"]["robot_model"]

    @property
    def round_trip_s(self) -> float:
        """Client-side elapsed time; this is not sensor age or clock offset."""
        return self.received_s - self.request_started_s


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProtocolError(f"Non-finite JSON number: {value}")


def _check_numbers(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise ProtocolError("JSON nesting exceeds 64 levels")
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolError("Non-finite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _check_numbers(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_numbers(item, depth + 1)


def decode_reply(
    parts: list[bytes],
    *,
    token: bytes,
    expected_model: str | None = None,
    include_images: bool = False,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Validate an envelope and preserve unrecognized metadata fields."""
    if not 2 <= len(parts) <= MAX_MESSAGE_FRAMES or len(parts) % 2:
        raise ProtocolError("Expected token, JSON and optional camera/JPEG pairs")
    if any(not isinstance(part, bytes) for part in parts):
        raise ProtocolError("Multipart frames must be bytes")
    if parts[0] != token:
        raise ProtocolError("Response token does not match the request")
    if len(parts[1]) > MAX_JSON_BYTES:
        raise ProtocolError("JSON frame exceeds the size limit")
    if any(len(part) > MAX_FRAME_BYTES for part in parts):
        raise ProtocolError("Multipart frame exceeds the size limit")
    if sum(map(len, parts)) > MAX_MESSAGE_BYTES:
        raise ProtocolError("Multipart message exceeds the size limit")
    try:
        payload = json.loads(
            parts[1].decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ProtocolError("Invalid JSON response") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("State JSON must be an object")
    _check_numbers(payload)
    metadata = payload.get("_robot_metadata")
    if not isinstance(metadata, dict):
        raise ProtocolError("Host response has no robot metadata")
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise ProtocolError("Unsupported robot metadata schema_version")
    model = metadata.get("robot_model")
    if not isinstance(model, str) or not model:
        raise ProtocolError("Host robot_model must be a nonempty string")
    if expected_model is not None and model != expected_model:
        raise ModelMismatchError(f"Expected {expected_model}, received {model}")
    for key in ("_host_timing", "_safety"):
        if key in payload and not isinstance(payload[key], dict):
            raise ProtocolError(f"{key} must be an object")
    names = payload.get("_images")
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ProtocolError("_images must list camera names")
    if len(set(names)) != len(names):
        raise ProtocolError("Duplicate camera names")
    if len(parts) != 2 + len(names) * 2:
        raise ProtocolError("Camera metadata and multipart frames disagree")
    if not include_images and names:
        raise ProtocolError("State-only request received camera frames")
    if names and payload.get("_image_encoding") != "jpeg":
        raise ProtocolError("Unsupported image encoding")
    images = {}
    for offset, name in enumerate(names):
        try:
            actual_name = parts[2 + offset * 2].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("Invalid camera name encoding") from exc
        if actual_name != name:
            raise ProtocolError("Camera order does not match _images")
        image = parts[3 + offset * 2]
        if not image:
            raise ProtocolError(f"Empty JPEG frame for {name}")
        images[name] = image
    return payload, images
