"""Workspace paths and recoverable capture manifests shared by calibration tools."""

import hashlib
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from alohamini.paths import WorkspacePaths


def camera_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ValueError("Camera name must contain only letters, digits, underscores or hyphens")
    return value


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def capture_path(camera: str, purpose: str, requested: Path | None) -> Path:
    camera_name(camera)
    path = (
        requested
        or WorkspacePaths().calibration
        / "cameras"
        / "captures"
        / f"{camera}_{purpose}_{timestamp()}"
    )
    path = path.expanduser()
    path.mkdir(parents=True, exist_ok=False)
    return path


def result_path(camera: str, category: str, requested: Path | None) -> Path:
    camera_name(camera)
    path = (
        requested
        or WorkspacePaths().calibration
        / "cameras"
        / category
        / f"{camera}_candidate_{timestamp()}.yaml"
    )
    path = path.expanduser()
    if path.exists():
        raise FileExistsError(path)
    return path


def write_result(path: Path, document: dict) -> None:
    text = yaml.safe_dump(document, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def save_manifest(directory: Path, document: dict, *, filename: str = "manifest.yaml") -> None:
    if Path(filename).name != filename or filename in ("", ".", ".."):
        raise ValueError("Manifest filename must be a single path component")
    text = yaml.safe_dump(document, sort_keys=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix=".manifest-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / filename)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sample_path(directory: Path, filename: str, sha256: str | None = None) -> Path:
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"Capture image is missing or outside capture directory: {filename}")
    if sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"Capture image checksum mismatch: {filename}")
    return path
