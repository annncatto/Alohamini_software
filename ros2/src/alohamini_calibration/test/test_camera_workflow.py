import hashlib
import json
import runpy
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
import zmq

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "scripts"))

from alohamini_camera.camera_info import (  # noqa: E402
    camera_info_is_accepted,
    load_camera_info_document,
)
from alohamini_camera.extrinsics_node import load_extrinsic  # noqa: E402
from calibration_io import (  # noqa: E402
    capture_path,
    result_path,
    sample_path,
    save_manifest,
    write_result,
)
from camera_board import BoardDetection, load_board  # noqa: E402

BOARD = PACKAGE / "config/cameras/boards/checkerboard_9x6_25mm.yaml"


def script(name):
    return runpy.run_path(str(PACKAGE / "scripts" / name))["main"]


def invoke(monkeypatch, main, *arguments):
    monkeypatch.setattr(sys, "argv", ["calibration", *map(str, arguments)])
    return main()


def test_workspace_paths_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path))
    directory = capture_path("forward", "intrinsics", None)
    assert directory.parent == tmp_path / "calibration/cameras/captures"
    with pytest.raises(FileExistsError):
        capture_path("forward", "intrinsics", directory)
    with pytest.raises(ValueError):
        capture_path("../forward", "intrinsics", None)
    output = result_path("forward", "intrinsics", None)
    write_result(output, {"status": "candidate_intrinsics_requires_review"})
    with pytest.raises(FileExistsError):
        write_result(output, {})
    assert yaml.safe_load(output.read_text())["status"].startswith("candidate_")


def test_capture_manifest_and_contained_checksums(tmp_path):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"jpeg")
    assert sample_path(tmp_path, image.name, hashlib.sha256(b"jpeg").hexdigest()) == image
    with pytest.raises(ValueError, match="checksum"):
        sample_path(tmp_path, image.name, "wrong")
    with pytest.raises(ValueError, match="outside"):
        sample_path(tmp_path, "../missing.jpg")
    save_manifest(tmp_path, {"samples": []})
    save_manifest(tmp_path, {"samples": [{"file": image.name}]})
    assert len(yaml.safe_load((tmp_path / "manifest.yaml").read_text())["samples"]) == 1
    assert not list(tmp_path.glob(".manifest-*"))


@pytest.mark.parametrize("ending", ["timeout", "restart", "dimensions", "interrupt"])
def test_camera_capture_retains_unique_samples_on_failure(tmp_path, monkeypatch, ending):
    main = script("capture_camera_calibration")
    env = main.__globals__
    jpeg = cv2.imencode(".jpg", np.zeros((24, 32, 3), dtype=np.uint8))[1].tobytes()
    metadata = dict(
        schema_version=1,
        camera_name="forward",
        encoding="jpeg",
        sequence=1,
        width=32,
        height=24,
        capture_monotonic_s=1.0,
        capture_unix_ns=10**18,
        host_session_id="session1",
    )
    frames = [metadata, metadata.copy()]
    if ending in ("restart", "dimensions"):
        last = dict(metadata, sequence=2)
        last.update(
            host_session_id="session2" if ending == "restart" else "session1",
            width=64 if ending == "dimensions" else 32,
        )
        frames.append(last)

    class Socket:
        closed = False

        def setsockopt(self, *args):
            pass

        def connect(self, *args):
            pass

        def poll(self, *args):
            if not frames and ending == "interrupt":
                raise KeyboardInterrupt
            return bool(frames)

        def recv_multipart(self):
            return [b"camera/forward", json.dumps(frames.pop(0)).encode(), jpeg]

        def close(self, **kwargs):
            self.closed = True

    socket = Socket()

    class Context:
        terminated = False

        def socket(self, *args):
            return socket

        def term(self):
            self.terminated = True

    context = Context()
    monkeypatch.setattr(env["zmq"], "Context", lambda: context)
    ticks = iter(np.arange(0.0, 30.0, 0.1))
    monkeypatch.setattr(env["time"], "monotonic", lambda: float(next(ticks)))
    args = (
        "--host",
        "127.0.0.1",
        "--camera",
        "forward",
        "--output",
        tmp_path / "capture",
        "--count",
        "3",
        "--min-interval-sec",
        "0",
        "--timeout-sec",
        "2",
        "--no-preview",
    )
    if ending == "interrupt":
        assert invoke(monkeypatch, main, *args) == 0
    else:
        with pytest.raises(TimeoutError if ending == "timeout" else ValueError):
            invoke(monkeypatch, main, *args)
    manifest = yaml.safe_load((tmp_path / "capture/manifest.yaml").read_text())
    assert manifest["captured_samples"] == 1
    assert len(manifest["samples"]) == 1
    assert manifest["samples"][0]["host_session_id"] == "session1"
    assert (tmp_path / "capture/forward_0000.jpg").read_bytes() == jpeg
    assert socket.closed and context.terminated


def test_intrinsic_solver_outputs_readable_candidate(tmp_path, monkeypatch):
    main = script("calibrate_camera_intrinsics")
    rng = np.random.default_rng(23)
    obj = np.zeros((54, 1, 3), np.float32)
    obj[:, 0, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025
    matrix = np.array([[600.0, 0, 320], [0, 605.0, 240], [0, 0, 1]])
    detections = []
    for index in range(20):
        points, _ = cv2.projectPoints(
            obj,
            rng.uniform(-0.5, 0.5, 3),
            rng.uniform([-0.12, -0.10, 0.5], [0.0, 0.0, 0.9]),
            matrix,
            np.zeros(5),
        )
        detections.append((tmp_path / f"{index}.jpg", BoardDetection(obj, points, 54)))
    monkeypatch.setitem(
        main.__globals__, "detect_points", lambda *args: ((640, 480), detections, [])
    )
    output = tmp_path / "candidate.yaml"
    assert (
        invoke(
            monkeypatch,
            main,
            "--capture-dir",
            tmp_path,
            "--board",
            BOARD,
            "--camera",
            "forward",
            "--frame-id",
            "camera_forward_optical_frame",
            "--output",
            output,
        )
        == 0
    )
    document = load_camera_info_document(output)
    assert not camera_info_is_accepted(document)
    np.testing.assert_allclose(
        np.array(document["camera_matrix"]["data"]).reshape(3, 3), matrix, atol=0.02
    )
    assert document["calibration_report"]["rms_px"] < 0.001


@pytest.mark.parametrize("calibration_type", ["eye_in_hand", "eye_to_hand"])
@pytest.mark.parametrize("failure", [None, "degenerate", "invalid_solver"])
def test_hand_eye_solver_preserves_transform_direction(
    tmp_path, monkeypatch, calibration_type, failure
):
    main = script("calibrate_hand_eye")
    env = main.__globals__
    rng = np.random.default_rng(91)

    def transform(rotation, translation):
        value = np.eye(4)
        value[:3, :3] = cv2.Rodrigues(np.asarray(rotation, dtype=float))[0]
        value[:3, 3] = translation
        return value

    def document(value):
        return {
            "translation_m": value[:3, 3].tolist(),
            "quaternion_xyzw": env["matrix_quaternion"](value[:3, :3]),
        }

    desired = transform([0.2, -0.15, 0.1], [0.04, 0.05, 0.1])
    fixed = transform([-0.1, 0.3, -0.2], [0.5, -0.1, 0.8])
    mount_offset = transform([0.02, 0.01, -0.1], [0.01, 0.02, 0.03])
    samples, targets = [], {}
    for index in range(20):
        base_gripper = transform(rng.uniform(-0.8, 0.8, 3), rng.uniform(-0.2, 0.2, 3))
        if failure == "degenerate":
            base_gripper[:3, :3] = np.eye(3)
        if calibration_type == "eye_in_hand":
            camera_target = np.linalg.inv(desired) @ np.linalg.inv(base_gripper) @ fixed
            base_mount = base_gripper @ mount_offset
        else:
            camera_target = np.linalg.inv(desired) @ base_gripper @ fixed
            base_mount = mount_offset
        filename = f"{index}.jpg"
        (tmp_path / filename).write_bytes(b"image")
        targets[filename] = camera_target
        samples.append(
            {
                "file": filename,
                "T_base_from_gripper": document(base_gripper),
                "T_base_from_mount": document(base_mount),
            }
        )
    save_manifest(
        tmp_path,
        {
            "calibration_type": calibration_type,
            "camera_name": "forward",
            "base_frame": "base_link",
            "gripper_frame": "tool_link",
            "mount_link": "camera_mount",
            "samples": samples,
        },
    )
    monkeypatch.setitem(env, "target_to_camera", lambda path, *args: targets[path.name])
    monkeypatch.setitem(
        env,
        "load_camera_info_document",
        lambda path: {
            "camera_name": "forward",
            "frame_id": "camera_optical",
            "distortion_model": "plumb_bob",
        },
    )
    output = tmp_path / "extrinsics.yaml"
    if failure == "invalid_solver":
        monkeypatch.setattr(
            cv2,
            "calibrateHandEye",
            lambda *args, **kwargs: (np.full((3, 3), np.nan), np.zeros((3, 1))),
        )
    if failure is not None:
        with pytest.raises(ValueError, match="independent axes|no valid hand-eye solution"):
            invoke(
                monkeypatch,
                main,
                "--capture-dir",
                tmp_path,
                "--intrinsics",
                "unused.yaml",
                "--board",
                BOARD,
                "--optical-frame",
                "camera_optical",
                "--output",
                output,
            )
        assert not output.exists()
        return
    assert (
        invoke(
            monkeypatch,
            main,
            "--capture-dir",
            tmp_path,
            "--intrinsics",
            "unused.yaml",
            "--board",
            BOARD,
            "--optical-frame",
            "camera_optical",
            "--output",
            output,
        )
        == 0
    )
    result = yaml.safe_load(output.read_text())
    with pytest.raises(ValueError, match="non-accepted"):
        load_extrinsic(output)
    assert load_extrinsic(output, allow_candidate=True)[:2] == ("camera_mount", "camera_optical")
    assert result["status"] == "candidate_hand_eye_requires_review"
    assert result["parent_frame"] == (
        "tool_link" if calibration_type == "eye_in_hand" else "base_link"
    )
    np.testing.assert_allclose(result["T_parent_from_camera_optical"]["matrix"], desired, atol=1e-7)
    np.testing.assert_allclose(
        result["T_mount_link_from_camera_optical"]["matrix"],
        np.linalg.inv(mount_offset) @ desired,
        atol=1e-7,
    )


def test_generate_board_in_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path))
    main = script("generate_charuco_board")
    board = PACKAGE / "config/cameras/boards/charuco_9x7_26mm_18p7_ids300_330.yaml"
    assert invoke(monkeypatch, main, "--board", board) == 0
    config = load_board(board)["print"]
    directory = tmp_path / "calibration/cameras/boards"
    assert (directory / config["pdf"]).read_bytes().startswith(b"%PDF")
    assert cv2.imread(str(directory / config["png"])).shape[:2] == (2100, 2970)
    with pytest.raises(ValueError, match="new file"):
        invoke(monkeypatch, main, "--board", board)


def test_camera_capture_over_local_zmq(tmp_path):
    context = zmq.Context()
    publisher = context.socket(zmq.XPUB)
    publisher.setsockopt(zmq.LINGER, 0)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    process = subprocess.Popen(
        [
            sys.executable,
            str(PACKAGE / "scripts/capture_camera_calibration"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--camera",
            "forward",
            "--output",
            str(tmp_path / "capture"),
            "--count",
            "3",
            "--min-interval-sec",
            "0",
            "--timeout-sec",
            "5",
            "--no-preview",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert publisher.poll(5000), "capture did not subscribe"
        assert publisher.recv() == b"\x01camera/forward"
        jpeg = cv2.imencode(".jpg", np.zeros((24, 32, 3), dtype=np.uint8))[1].tobytes()
        for sequence in range(1, 4):
            metadata = dict(
                schema_version=1,
                camera_name="forward",
                encoding="jpeg",
                sequence=sequence,
                width=32,
                height=24,
                host_session_id="local-test",
                capture_monotonic_s=time.monotonic(),
                capture_unix_ns=time.time_ns(),
            )
            publisher.send_multipart([b"camera/forward", json.dumps(metadata).encode(), jpeg])
            time.sleep(0.05)
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 0, stdout + stderr
        manifest = yaml.safe_load((tmp_path / "capture/manifest.yaml").read_text())
        assert [sample["sequence"] for sample in manifest["samples"]] == [1, 2, 3]
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=3)
        publisher.close()
        context.term()


@pytest.mark.parametrize("interruption", [None, "zero_timestamp", "tool", "camera_mount"])
def test_hand_eye_capture_saves_timestamp_matched_stationary_pose(tmp_path, interruption):
    env = runpy.run_path(str(PACKAGE / "scripts/capture_hand_eye_samples"))
    cls = env["HandEyeCapture"]
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import CompressedImage

    args = SimpleNamespace(
        camera="wrist_right",
        count=3,
        output=tmp_path,
        calibration_type="eye_in_hand",
        image_topic="/image",
        base_frame="base_link",
        gripper_frame="tool",
        mount_link="camera_mount",
        stationary_translation_mm=1.5,
        stationary_rotation_deg=0.75,
        stationary_dwell_sec=0.4,
        fixed_mount_translation_mm=2.0,
        fixed_mount_rotation_deg=1.0,
        min_translation_m=0.015,
        min_rotation_deg=8.0,
    )
    timestamps = []
    missing_frame = None

    def lookup(parent, child, stamp, **kwargs):
        if child == missing_frame:
            raise LookupError("TF unavailable at capture time")
        timestamps.append((parent, child, stamp.nanoseconds))
        transform = TransformStamped()
        transform.transform.rotation.w = 1.0
        return transform

    node = SimpleNamespace(
        args=args,
        samples=[],
        last_capture_ns=0,
        image_frame=None,
        stop_requested=False,
        stable_reference=None,
        stable_since_ns=None,
        show_preview=lambda jpeg: None,
        buffer=SimpleNamespace(lookup_transform=lookup),
        get_logger=lambda: SimpleNamespace(info=lambda text: None, warning=lambda text: None),
    )
    node.save_capture = lambda: cls.save_capture(node)
    message = CompressedImage()
    message.header.frame_id = "right_camera_optical"
    message.header.stamp.sec = 10
    message.data = cv2.imencode(".jpg", np.zeros((24, 32, 3), np.uint8))[1].tobytes()
    cls.on_image(node, message)
    assert not node.samples
    if interruption is not None:
        message.header.stamp.nanosec = 300_000_000
        if interruption == "zero_timestamp":
            message.header.stamp.sec = 0
            message.header.stamp.nanosec = 0
        else:
            missing_frame = interruption
        cls.on_image(node, message)
        missing_frame = None
        message.header.stamp.sec = 10
    message.header.stamp.nanosec = 500_000_000
    cls.on_image(node, message)
    expected_capture_ns = 10_500_000_000
    if interruption is not None:
        assert not node.samples  # Unknown motion cannot count toward the stationary dwell.
        message.header.stamp.sec = 11
        message.header.stamp.nanosec = 0
        cls.on_image(node, message)
        expected_capture_ns = 11_000_000_000
    cls.on_image(node, message)  # repeated timestamp must not produce another sample
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert len(manifest["samples"]) == 1
    assert manifest["samples"][0]["capture_unix_ns"] == expected_capture_ns
    assert manifest["optical_frame"] == "right_camera_optical"
    assert timestamps[-2:] == [
        ("base_link", "tool", expected_capture_ns),
        ("base_link", "camera_mount", expected_capture_ns),
    ]
