import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "scripts"))

from calibration_io import parse_hand_eye_args  # noqa: E402


@pytest.mark.parametrize("camera", ["forward", "backward", "chest", "wrist_left", "wrist_right"])
def test_packaged_presets_resolve_board_and_explicit_overrides(camera, monkeypatch):
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", required=True)
    parser.add_argument("--mount-link", required=True)
    parser.add_argument("--count", type=int, default=25)
    monkeypatch.setattr(sys, "argv", ["capture", "--preset", camera, "--count", "32"])
    args = parse_hand_eye_args(
        parser,
        {
            "camera": "camera_name",
            "mount_link": "mount_link",
            "count": "minimum_samples",
        },
    )
    assert args.camera == camera
    assert args.count == 32
    assert Path(args.preset_document["board"]).is_file()
    assert args.preset_document["calibration_type"] == (
        "eye_in_hand" if camera.startswith("wrist") else "eye_to_hand"
    )


def test_capture_uses_preset_and_saves_effective_configuration(tmp_path, monkeypatch):
    env = runpy.run_path(str(PACKAGE / "scripts/capture_hand_eye_samples"))
    captured = {}

    class Node:
        def __init__(self, args):
            self.args = args
            self.samples = []
            self.image_frame = None
            self.stop_requested = True
            self.listener = None
            captured["args"] = args

        def save_capture(self):
            env["HandEyeCapture"].save_capture(self)

        def close_preview(self):
            pass

        def destroy_node(self):
            pass

    main = env["main"]
    monkeypatch.setitem(main.__globals__, "HandEyeCapture", Node)
    # Save the original class method; runpy functions use a distinct globals dict.
    save = env["HandEyeCapture"].save_capture
    Node.save_capture = lambda self: save(self)
    monkeypatch.setattr(env["rclpy"], "init", lambda: None)
    monkeypatch.setattr(env["rclpy"], "ok", lambda: False)
    monkeypatch.setitem(
        main.__globals__,
        "MultiThreadedExecutor",
        lambda **kw: SimpleNamespace(
            add_node=lambda n: None,
            remove_node=lambda n: None,
            shutdown=lambda **kw: None,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture",
            "--preset",
            "wrist_right",
            "--gripper-frame",
            "custom_tool",
            "--count",
            "31",
            "--no-preview",
            "--output",
            str(tmp_path / "capture"),
        ],
    )
    assert main() == 0
    args = captured["args"]
    assert args.image_topic == "/alohamini/cameras/wrist_right/image_raw/compressed"
    assert args.optical_frame == "right_camera_optical"
    manifest = yaml.safe_load((args.output / "manifest.yaml").read_text())
    assert manifest["hand_eye_preset"]["gripper_frame"] == "custom_tool"
    assert manifest["hand_eye_preset"]["minimum_samples"] == 20
    assert manifest["hand_eye_preset"]["capture_count"] == 31
    assert manifest["hand_eye_preset"]["board"] == "board.yaml"
    assert (args.output / "board.yaml").is_file()


def test_import_preserves_bytes_status_and_never_overwrites(tmp_path):
    importer = runpy.run_path(str(PACKAGE / "scripts/import_calibration"))["import_profile"]
    source = tmp_path / "old"
    (source / "cameras/intrinsics").mkdir(parents=True)
    (source / "lerobot").mkdir()
    original = b"status: candidate_intrinsics_requires_review\ncamera_name: forward\n"
    (source / "cameras/intrinsics/forward.yaml").write_bytes(original)
    (source / "lerobot/AlohaMiniRobot.json").write_text("{}")
    output = tmp_path / "new"
    manifest = importer(source, output)
    assert manifest["hardware_verified"] is False
    assert manifest["status"] == "imported_requires_review"
    assert (output / "cameras/intrinsics/forward.yaml").read_bytes() == original
    assert (output / "robots/AlohaMiniRobot.json").read_text() == "{}"
    assert len(manifest["consumer_issues"]) == 3
    assert any(f["sha256"] == hashlib.sha256(original).hexdigest() for f in manifest["files"])
    assert json.loads((output / "import_manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        importer(source, output)
    with pytest.raises(ValueError, match="outside"):
        importer(source, source / "nested")
    assert (source / "cameras/intrinsics/forward.yaml").read_bytes() == original


def test_import_rejects_symlinks_without_creating_output(tmp_path):
    importer = runpy.run_path(str(PACKAGE / "scripts/import_calibration"))["import_profile"]
    source = tmp_path / "source"
    source.mkdir()
    (source / "unsafe.yaml").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="symbolic link"):
        importer(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
