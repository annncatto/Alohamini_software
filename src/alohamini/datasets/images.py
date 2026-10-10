"""Host JPEG files/shards, legacy PNG and frame-indexed RGB video reads."""

from __future__ import annotations

import hashlib
import io
import os
import re
import tarfile
from pathlib import Path

import numpy as np

IMAGE_FORMAT = "host-jpeg-tar"
JPEG_FORMAT = "host-jpeg"
VIDEO_FORMAT = "rgb-mp4"
IMAGE_COLOR = "opencv_imdecode_color_is_rgb"
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 32 * 1024 * 1024


def image_type(image_format: str):
    import pyarrow as pa

    if image_format in ("png", JPEG_FORMAT):
        return pa.string()
    if image_format == VIDEO_FORMAT:
        return pa.struct(
            [
                ("path", pa.string()),
                ("sha256", pa.string()),
                ("frame_index", pa.int64()),
            ]
        )
    if image_format != IMAGE_FORMAT:
        raise ValueError(f"Unsupported image format: {image_format}")
    return pa.struct(
        [
            ("path", pa.string()),
            ("member", pa.string()),
            ("offset", pa.int64()),
            ("size", pa.int64()),
            ("sha256", pa.string()),
        ]
    )


def _shape(data: bytes, expected: str, *, decode=False) -> tuple[int, int, int]:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            if (
                image.format != expected
                or image.mode != "RGB"
                or image.width * image.height > MAX_IMAGE_PIXELS
            ):
                raise ValueError("Invalid image format, mode or dimensions")
            if decode:
                image.load()
            return image.height, image.width, 3
    except Image.DecompressionBombError as exc:
        raise ValueError("Image dimensions exceed the pixel limit") from exc


def _host_image_shape(jpeg: bytes, *, decode=False) -> tuple[int, int, int]:
    if (
        not 4 <= len(jpeg) <= MAX_IMAGE_BYTES
        or not jpeg.startswith(b"\xff\xd8")
        or not jpeg.endswith(b"\xff\xd9")
    ):
        raise ValueError("Invalid or oversized Host JPEG")
    try:
        return _shape(jpeg, "JPEG", decode=decode)
    except OSError as exc:
        raise ValueError("Invalid Host JPEG") from exc


def decode_host_image(jpeg: bytes) -> np.ndarray:
    """Decode a 5556 Host snapshot JPEG to RGB (not the standard 5557 stream)."""
    import cv2

    shape = _host_image_shape(jpeg, decode=True)
    rgb = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if rgb is None or rgb.shape != shape:
        raise ValueError("Invalid Host JPEG")
    return rgb


def validate_wire_image(jpeg: bytes) -> tuple[int, int, int]:
    """Validate a complete Host JPEG without changing its encoded bytes."""
    return decode_host_image(jpeg).shape


def prepare_recording_image(jpeg: bytes) -> tuple[bytes, tuple[int, int, int]]:
    """Retain wire bytes; inspect headers without decoding or re-encoding pixels.

    Full decoding happens during statistics/video encoding or explicit checking.
    """
    return jpeg, _host_image_shape(jpeg)


class ImageShards:
    """Append-only USTAR files; indexes address individual JPEG payloads directly."""

    MAX_SHARD_BYTES = 256 * 1024 * 1024

    def __init__(self, episode: Path):
        self.episode = episode
        self.archive = self.stream = None
        self.index = -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.stream is not None:
            try:
                if self.archive is not None:
                    self.archive.close()
                self.stream.flush()
                os.fsync(self.stream.fileno())
            finally:
                self.stream.close()
                self.archive = self.stream = None

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def append(self, camera: str, capture: int, jpeg: bytes) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_]+", camera) or type(capture) is not int or capture < 0:
            raise ValueError("Invalid camera or capture index")
        if not 0 < len(jpeg) <= MAX_IMAGE_BYTES:
            raise ValueError("Invalid image size")
        member = tarfile.TarInfo(f"{camera}/frame_{capture:06d}.jpg")
        member.size, member.mode, member.mtime = len(jpeg), 0o644, 0
        # Include the final TAR markers and record padding in the shard limit.
        entry_size = 512 + ((len(jpeg) + 511) // 512) * 512

        def closed_size(offset):
            return ((offset + 1024 + 10239) // 10240) * 10240

        if closed_size(entry_size) > self.MAX_SHARD_BYTES:
            raise ValueError("Image exceeds shard size limit")
        if (
            self.archive is not None
            and closed_size(self.archive.offset + entry_size) > self.MAX_SHARD_BYTES
        ):
            self.close()
        if self.archive is None:
            self.index += 1
            self.path = f"images/chunk-{self.index:06d}.tar"
            (self.episode / "images").mkdir(exist_ok=True)
            self.stream = (self.episode / self.path).open("xb")
            self.archive = tarfile.open(fileobj=self.stream, mode="w", format=tarfile.USTAR_FORMAT)
        offset = self.archive.offset + 512
        self.archive.addfile(member, io.BytesIO(jpeg))
        # tarfile otherwise retains an in-memory list of every image in the shard.
        self.archive.members.clear()
        return {
            "path": self.path,
            "member": member.name,
            "offset": offset,
            "size": len(jpeg),
            "sha256": hashlib.sha256(jpeg).hexdigest(),
        }


def image_path(episode: Path, camera: str, reference) -> Path:
    """Resolve temporary JPEG/PNG paths or bounded camera-specific references."""
    if isinstance(reference, str):
        value = reference
        valid = re.fullmatch(rf"images/{re.escape(camera)}/frame_[0-9]{{6,}}\.(?:png|jpg)", value)
    elif isinstance(reference, dict):
        value = reference.get("path")
        if "frame_index" in reference:
            valid = (
                set(reference) == {"path", "sha256", "frame_index"}
                and value == f"videos/{camera}.mp4"
                and re.fullmatch(r"[A-Za-z0-9_]+", camera)
                and type(reference["frame_index"]) is int
                and reference["frame_index"] >= 0
                and isinstance(reference["sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])
            )
            if not valid:
                raise ValueError(f"Invalid video reference: {reference!r}")
            path = episode / value
            if not path.resolve().is_relative_to(episode.resolve()):
                raise ValueError("Video reference escapes episode")
            return path
        valid = (
            set(reference) == {"path", "member", "offset", "size", "sha256"}
            and isinstance(value, str)
            and re.fullmatch(r"images/chunk-[0-9]{6,}\.tar", value)
            and isinstance(reference["member"], str)
            and re.fullmatch(rf"{re.escape(camera)}/frame_[0-9]{{6,}}\.jpg", reference["member"])
            and type(reference["offset"]) is int
            and reference["offset"] >= 512
            and reference["offset"] % 512 == 0
            and type(reference["size"]) is int
            and 0 < reference["size"] <= MAX_IMAGE_BYTES
            and isinstance(reference["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])
        )
    else:
        valid = False
    if not valid:
        raise ValueError(f"Invalid image reference: {reference!r}")
    path = episode / value
    if not path.resolve().is_relative_to(episode.resolve()):
        raise ValueError("Image reference escapes episode")
    return path


def image_bytes(episode: Path, camera: str, reference) -> bytes:
    path = image_path(episode, camera, reference)
    if isinstance(reference, dict) and "frame_index" in reference:
        from PIL import Image

        output = io.BytesIO()
        Image.fromarray(image_rgb(episode, camera, reference)).save(output, format="PNG")
        return output.getvalue()
    if isinstance(reference, str):
        return path.read_bytes()
    with path.open("rb") as stream:
        stream.seek(reference["offset"] - 512)
        try:
            header = tarfile.TarInfo.frombuf(stream.read(512), "utf-8", "strict")
        except (tarfile.HeaderError, UnicodeError) as exc:
            raise ValueError("Invalid image TAR header") from exc
        if (
            not header.isfile()
            or header.name != reference["member"]
            or header.size != reference["size"]
        ):
            raise ValueError("Image TAR header does not match its index")
        data = stream.read(reference["size"])
    if len(data) != reference["size"] or hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("Image payload is truncated or checksum does not match")
    return data


def image_shape(episode: Path, camera: str, reference, *, decode=False) -> tuple[int, int, int]:
    if isinstance(reference, dict) and "frame_index" in reference:
        if decode:
            return image_rgb(episode, camera, reference).shape
        from alohamini.datasets.video import inspect_video

        return tuple(inspect_video(image_path(episode, camera, reference))["shape"])
    data = image_bytes(episode, camera, reference)
    if isinstance(reference, str) and reference.endswith(".png"):
        return _shape(data, "PNG", decode=decode)
    if decode:
        return validate_wire_image(data)
    return _host_image_shape(data) if isinstance(reference, str) else _shape(data, "JPEG")


def video_rgb(path: Path, target: int) -> np.ndarray:
    """Read an indexed video frame in RGB, shared by platform and v3 datasets."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate
        if not rate or not stream.time_base:
            raise ValueError("Video must have a fixed frame rate and timestamps")
        container.seek(int(target / rate / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Video frame has no timestamp")
            position = frame.pts * frame.time_base * rate
            index = round(position)
            if abs(position - index) > 0.01:
                raise ValueError("Video frame timestamp is off the dataset timeline")
            if index == target:
                return frame.to_ndarray(format="rgb24")
            if index > target:
                break
    raise ValueError(f"Missing video frame {target}: {path}")


def image_rgb(episode: Path, camera: str, reference) -> np.ndarray:
    if isinstance(reference, dict) and "frame_index" in reference:
        return video_rgb(image_path(episode, camera, reference), reference["frame_index"])
    data = image_bytes(episode, camera, reference)
    if isinstance(reference, str) and reference.endswith(".png"):
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image).copy()
    # Preserve Host/client color semantics: this decoded array is already RGB.
    return decode_host_image(data)


def image_png(episode: Path, camera: str, reference) -> bytes:
    """Standard RGB PNG for downstream tools; never re-encode during capture."""
    if (isinstance(reference, str) and reference.endswith(".png")) or (
        isinstance(reference, dict) and "frame_index" in reference
    ):
        return image_bytes(episode, camera, reference)
    from PIL import Image

    output = io.BytesIO()
    Image.fromarray(image_rgb(episode, camera, reference)).save(
        output, format="PNG", compress_level=6
    )
    return output.getvalue()
