from __future__ import annotations

import json
import math
from dataclasses import dataclass

CAMERA_STREAM_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CameraFrame:
    camera_name: str
    sequence: int
    width: int
    height: int
    capture_monotonic_s: float
    capture_unix_ns: int
    jpeg: bytes
    host_session_id: str | None = None


def parse_camera_message(parts: list[bytes]) -> CameraFrame:
    """Validate one Host ``[topic, metadata, JPEG]`` message."""
    if len(parts) != 3:
        raise ValueError("camera stream message must have exactly three parts")
    topic, metadata_bytes, jpeg = parts
    if (
        not all(isinstance(part, bytes) for part in parts)
        or len(topic) > 256
        or len(metadata_bytes) > 8192
        or len(jpeg) > 8 * 1024 * 1024
    ):
        raise ValueError("camera stream message has invalid types or exceeds size limits")
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("camera metadata is not valid UTF-8 JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("camera metadata must be an object")
    if (
        type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != CAMERA_STREAM_SCHEMA_VERSION
    ):
        raise ValueError("unsupported camera stream schema_version")
    camera_name = metadata.get("camera_name")
    if not isinstance(camera_name, str) or not camera_name.strip() or len(camera_name) > 64:
        raise ValueError("camera_name must be a non-empty string")
    if topic != f"camera/{camera_name}".encode():
        raise ValueError("camera topic does not match metadata camera_name")
    if metadata.get("encoding") != "jpeg" or not jpeg.startswith(b"\xff\xd8"):
        raise ValueError("camera payload must be a JPEG image")
    for key in ("sequence", "width", "height", "capture_unix_ns"):
        if type(metadata.get(key)) is not int:
            raise ValueError(f"camera metadata {key} must be an integer")
    sequence, width, height = (metadata[key] for key in ("sequence", "width", "height"))
    capture_monotonic_s = metadata.get("capture_monotonic_s")
    capture_unix_ns = metadata["capture_unix_ns"]
    if sequence < 1 or not 1 <= width <= 4096 or not 1 <= height <= 4096:
        raise ValueError("camera sequence and dimensions must be positive")
    if (
        type(capture_monotonic_s) not in (int, float)
        or not math.isfinite(capture_monotonic_s)
        or capture_monotonic_s < 0
        or capture_unix_ns <= 0
    ):
        raise ValueError("camera capture timestamps must be finite and positive")
    session = metadata.get("host_session_id")
    if session is not None and (
        not isinstance(session, str) or not session.strip() or len(session) > 64
    ):
        raise ValueError("invalid camera Host session ID")
    return CameraFrame(
        camera_name=camera_name,
        sequence=sequence,
        width=width,
        height=height,
        capture_monotonic_s=capture_monotonic_s,
        capture_unix_ns=capture_unix_ns,
        jpeg=jpeg,
        host_session_id=session,
    )
