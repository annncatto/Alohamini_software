"""Workspace paths and recoverable capture manifests shared by calibration tools."""

import argparse
import hashlib
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from alohamini.paths import WorkspacePaths


def parse_hand_eye_args(parser, fields):
    """Apply an explicit YAML preset as defaults; CLI values take precedence."""
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--preset")
    selected, _ = selector.parse_known_args()
    parser.add_argument("--preset", help="Hand-eye camera preset name or YAML path")
    document = None
    if selected.preset:
        path = Path(selected.preset).expanduser()
        if path.suffix not in (".yaml", ".yml"):
            from ament_index_python.packages import get_package_share_directory

            path = Path(get_package_share_directory("alohamini_calibration")) / (
                f"config/cameras/hand_eye/{camera_name(selected.preset)}.yaml"
            )
        path = path.resolve()
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise ValueError("Hand-eye preset must use schema_version 1")
        if document.get("calibration_type") not in ("eye_in_hand", "eye_to_hand"):
            raise ValueError("Invalid preset calibration_type")
        for key in (
            "camera_name",
            "image_topic",
            "base_frame",
            "gripper_frame",
            "mount_link",
            "optical_frame",
            "board",
        ):
            if not isinstance(document.get(key), str) or not document[key].strip():
                raise ValueError(f"Hand-eye preset lacks {key}")
        camera_name(document["camera_name"])
        document["board"] = str((path.parent / document["board"]).resolve())
        defaults = {dest: document[key] for dest, key in fields.items() if key in document}
        for action in parser._actions:
            if action.dest in defaults:
                action.required = False
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    args.preset_document = document
    return args


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
