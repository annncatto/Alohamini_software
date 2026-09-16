from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


def load_camera_info_document(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("camera info YAML must contain an object")
    for field in (
        "image_width",
        "image_height",
        "camera_name",
        "camera_matrix",
        "distortion_model",
        "distortion_coefficients",
        "rectification_matrix",
        "projection_matrix",
        "frame_id",
    ):
        if field not in document:
            raise ValueError(f"camera info YAML lacks {field}")
    expected_sizes = {
        "camera_matrix": 9,
        "rectification_matrix": 9,
        "projection_matrix": 12,
    }
    for field, size in expected_sizes.items():
        data = document[field].get("data") if isinstance(document[field], dict) else None
        if not isinstance(data, list) or len(data) != size:
            raise ValueError(f"{field}.data must contain {size} values")
    distortion = document["distortion_coefficients"]
    if not isinstance(distortion, dict) or not isinstance(distortion.get("data"), list):
        raise ValueError("distortion_coefficients.data must be a list")
    if any(
        type(document[key]) is not int or not 1 <= document[key] <= 4096
        for key in ("image_width", "image_height")
    ):
        raise ValueError("camera info image dimensions must be positive")
    for key in ("camera_name", "frame_id", "distortion_model"):
        if not isinstance(document[key], str) or not document[key].strip():
            raise ValueError(f"camera info {key} must be a nonempty string")
    for key in (*expected_sizes, "distortion_coefficients"):
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in document[key]["data"]
        ):
            raise ValueError(f"{key}.data must contain finite numbers")
    matrix = document["camera_matrix"]["data"]
    if matrix[0] <= 0 or matrix[4] <= 0 or not math.isclose(matrix[8], 1.0):
        raise ValueError("camera matrix must have positive focal lengths and K[2,2]=1")
    return document


def camera_info_is_accepted(document: dict[str, Any]) -> bool:
    """Unmarked ROS CameraInfo YAMLs are installed inputs, not bundled candidates."""
    status = document.get("status")
    return status is None or (isinstance(status, str) and status.startswith("accepted_"))


def validate_frame_against_camera_info(
    document: dict[str, Any], camera_name: str, width: int, height: int
) -> None:
    if document["camera_name"] != camera_name:
        raise ValueError(
            f"camera info name {document['camera_name']!r} does not match {camera_name!r}"
        )
    expected = (int(document["image_width"]), int(document["image_height"]))
    if expected != (width, height):
        raise ValueError(
            f"camera frame is {width}x{height}, calibration expects {expected[0]}x{expected[1]}"
        )
