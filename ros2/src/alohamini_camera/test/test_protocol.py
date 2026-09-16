import json

import pytest
import yaml
from alohamini_camera.camera_info import (
    camera_info_is_accepted,
    load_camera_info_document,
    validate_frame_against_camera_info,
)
from alohamini_camera.protocol import parse_camera_message


def message(camera="forward", sequence=1):
    metadata = {
        "schema_version": 1,
        "camera_name": camera,
        "sequence": sequence,
        "encoding": "jpeg",
        "width": 640,
        "height": 480,
        "capture_monotonic_s": 1.5,
        "capture_unix_ns": 2_000_000_000,
    }
    return [
        f"camera/{camera}".encode(),
        json.dumps(metadata).encode(),
        b"\xff\xd8jpeg",
    ]


def test_camera_protocol_parses_timestamped_jpeg():
    frame = parse_camera_message(message(sequence=4))

    assert frame.camera_name == "forward"
    assert frame.sequence == 4
    assert frame.capture_unix_ns == 2_000_000_000
    assert frame.jpeg == b"\xff\xd8jpeg"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda parts: parts[:2],
        lambda parts: [b"camera/other", *parts[1:]],
        lambda parts: [parts[0], b"{}", parts[2]],
        lambda parts: [parts[0], parts[1], b"not-jpeg"],
    ],
)
def test_camera_protocol_rejects_malformed_messages(mutate):
    with pytest.raises(ValueError):
        parse_camera_message(mutate(message()))


def camera_info():
    return {
        "image_width": 16,
        "image_height": 8,
        "camera_name": "forward",
        "frame_id": "forward_camera_optical",
        "distortion_model": "plumb_bob",
        "camera_matrix": {"data": [10.0, 0, 8, 0, 10.0, 4, 0, 0, 1]},
        "distortion_coefficients": {"data": [0.0] * 5},
        "rectification_matrix": {"data": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
        "projection_matrix": {"data": [10.0, 0, 8, 0, 0, 10.0, 4, 0, 0, 0, 1, 0]},
    }


def test_installed_camera_info_matches_stream(tmp_path):
    path = tmp_path / "forward.yaml"
    path.write_text(yaml.safe_dump(camera_info()))
    document = load_camera_info_document(path)
    validate_frame_against_camera_info(document, "forward", 16, 8)
    with pytest.raises(ValueError):
        validate_frame_against_camera_info(document, "forward", 1280, 720)


@pytest.mark.parametrize(
    "status,accepted",
    [
        (None, True),
        ("accepted_intrinsics", True),
        ("candidate_intrinsics_requires_review", False),
        ("unavailable", False),
    ],
)
def test_candidate_intrinsics_are_not_implicitly_accepted(status, accepted):
    assert camera_info_is_accepted({"status": status}) is accepted


@pytest.mark.parametrize(
    "key,value",
    [("image_width", True), ("frame_id", ""), ("camera_matrix", {"data": [float("nan")] * 9})],
)
def test_invalid_camera_info_is_rejected(tmp_path, key, value):
    document = camera_info()
    document[key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):
        load_camera_info_document(path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("sequence", 1.5),
        ("width", 5000),
        ("capture_monotonic_s", float("nan")),
        ("host_session_id", ""),
    ],
)
def test_invalid_wire_fields_are_not_coerced(key, value):
    parts = message()
    metadata = json.loads(parts[1])
    metadata[key] = value
    parts[1] = json.dumps(metadata).encode()
    with pytest.raises(ValueError):
        parse_camera_message(parts)
