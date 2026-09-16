# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Migrated from lerobot_find_cameras.py and OpenCVCamera.find_cameras().
"""Local V4L2 discovery and snapshots, independent of LeRobot and desktop OpenCV."""

import logging
import math
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

from alohamini.hardware.camera import CameraConfig, OpenCVCamera
from alohamini.paths import WorkspacePaths

logger = logging.getLogger(__name__)


def _probe_opencv(target, connection):
    """Keep camera-driver open/get/release outside the discovery process."""
    camera = None
    try:
        import cv2

        camera = cv2.VideoCapture(target, cv2.CAP_V4L2)
        if not camera.isOpened():
            connection.send(None)
            return
        code = int(camera.get(cv2.CAP_PROP_FOURCC))
        connection.send(
            {
                "name": f"OpenCV Camera @ {target}",
                "type": "OpenCV",
                "id": target,
                "backend_api": camera.getBackendName(),
                "default_stream_profile": {
                    "format": camera.get(cv2.CAP_PROP_FORMAT),
                    "fourcc": "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)),
                    "width": int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": camera.get(cv2.CAP_PROP_FPS),
                },
            }
        )
    except Exception as exc:
        connection.send({"error": str(exc)})
    finally:
        connection.close()
        if camera is not None:
            camera.release()


def _probe_one(target):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_probe_opencv, args=(target, sender), daemon=True)
    try:
        process.start()
        sender.close()
        if not receiver.poll(3.0):
            logger.warning("Camera discovery timed out: %s", target)
            return None
        result = receiver.recv()
        if result and "error" in result:
            logger.warning("Camera discovery failed for %s: %s", target, result["error"])
            return None
        return result
    except EOFError:
        logger.warning("Camera discovery worker stopped: %s", target)
        return None
    finally:
        receiver.close()
        sender.close()
        if process.pid is not None:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.3)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.3)
            if process.is_alive():
                raise RuntimeError(f"Camera discovery worker did not exit: {target}")
            process.close()


def find_all_opencv_cameras() -> list[dict[str, Any]]:
    logger.info("Searching for OpenCV cameras...")
    found = []
    aliases = list(Path("/dev").glob("am_camera_*"))
    # Enumerate actual local device nodes, never IP addresses or network cameras.
    for target in sorted(Path("/dev").glob("video[0-9]*"), key=lambda p: p.name):
        info = _probe_one(str(target))
        if info is not None:
            info["aliases"] = [
                str(alias) for alias in aliases if alias.resolve() == target.resolve()
            ]
            found.append(info)
    logger.info("Found %s OpenCV cameras.", len(found))
    return found


def find_and_print_cameras(camera_type_filter: str | None = None) -> list[dict[str, Any]]:
    """
    Finds available cameras based on an optional filter and prints their information.

    Args:
        camera_type_filter: Optional string to filter cameras ("opencv").
                            If None, lists all cameras.

    Returns:
        A list of all available cameras matching the filter, with their metadata.
    """
    all_cameras_info: list[dict[str, Any]] = []

    if camera_type_filter:
        camera_type_filter = camera_type_filter.lower()

    if camera_type_filter is None or camera_type_filter == "opencv":
        all_cameras_info.extend(find_all_opencv_cameras())

    if not all_cameras_info:
        if camera_type_filter:
            logger.warning(f"No {camera_type_filter} cameras were detected.")
        else:
            logger.warning("No OpenCV cameras were detected.")
    else:
        print("\n--- Detected Cameras ---")
        for i, cam_info in enumerate(all_cameras_info):
            print(f"Camera #{i}:")
            for key, value in cam_info.items():
                if key == "default_stream_profile" and isinstance(value, dict):
                    print(f"  {key.replace('_', ' ').capitalize()}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key.capitalize()}: {sub_value}")
                else:
                    print(f"  {key.replace('_', ' ').capitalize()}: {value}")
            print("-" * 20)
    return all_cameras_info


def save_image(img_array, camera_identifier, images_dir, camera_type):
    """Save a standard RGB snapshot with the original per-device filename."""
    import cv2

    safe_identifier = str(camera_identifier).replace("/", "_").replace("\\", "_")
    path = images_dir / f"{camera_type.lower()}_{safe_identifier}.png"
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Failed to encode snapshot: {camera_identifier}")
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded)
    os.replace(temporary, path)
    return path


def create_camera_instance(cam_meta):
    # Snapshot uses the same requested profile and capture lifecycle as native Host.
    instance = OpenCVCamera(CameraConfig(device=str(cam_meta["id"])))
    try:
        instance.start()
    except BaseException:
        instance.close()
        raise
    return {"instance": instance, "meta": cam_meta, "last_stamp": None, "saved": False}


def process_camera_image(cam_dict, output_dir, current_time):
    import cv2
    import numpy as np

    history = cam_dict["instance"].read_frame_history()
    if not history:
        return None
    stamp, jpeg = history[-1]
    if stamp == cam_dict["last_stamp"]:
        return None
    # Native Host JPEG decodes directly to the established client RGB array.
    rgb = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError("Invalid camera JPEG")
    meta = cam_dict["meta"]
    path = save_image(rgb, meta["id"], output_dir, meta["type"])
    cam_dict.update(last_stamp=stamp, saved=True)
    return path


def cleanup_cameras(cameras_to_use):
    errors = []
    for cam_dict in cameras_to_use:
        try:
            cam_dict["instance"].close()
        except Exception as exc:
            errors.append(f"{cam_dict['meta']['id']}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))


def save_images_from_all_cameras(output_dir=None, record_time_s=6.0, camera_type=None):
    if camera_type not in (None, "opencv"):
        raise ValueError("Native camera discovery currently supports local OpenCV/V4L2 cameras")
    if not math.isfinite(record_time_s) or record_time_s < 0:
        raise ValueError("record-time-s must be finite and non-negative")
    metadata = find_and_print_cameras(camera_type)
    if not metadata:
        raise OSError("No local cameras found")
    if record_time_s == 0:
        return None  # Listing only: no camera capture or output directories.
    return save_images_from_cameras(metadata, output_dir, record_time_s)


def save_images_from_cameras(metadata, output_dir=None, record_time_s=6.0):
    if not math.isfinite(record_time_s) or record_time_s <= 0:
        raise ValueError("Snapshot duration must be finite and positive")
    if not metadata:
        raise ValueError("No cameras selected")
    for camera in metadata:
        CameraConfig(device=str(camera["id"]))
    output_dir = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else (WorkspacePaths().logs / "debug" / f"cameras-{time.time_ns()}")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Saving images to %s", output_dir)
    cameras, failed = [], []
    try:
        for meta in metadata:
            cameras.append(create_camera_instance(meta))
        start_time = time.perf_counter()
        try:
            while time.perf_counter() - start_time < record_time_s:
                for camera in cameras:
                    if camera in failed:
                        continue
                    try:
                        process_camera_image(camera, output_dir, time.perf_counter())
                    except (OSError, ValueError) as exc:
                        logger.error("Camera %s: %s", camera["meta"]["id"], exc)
                        failed.append(camera)
                time.sleep(0.02)
        except KeyboardInterrupt:
            logger.info("Capture interrupted by user.")
    finally:
        cleanup_cameras(cameras)
    print(f"Image capture finished. Images saved to {output_dir}")
    missing = [
        camera["meta"]["id"] for camera in cameras if not camera["saved"] or camera in failed
    ]
    if missing:
        raise OSError(
            f"Cameras without a healthy snapshot: {missing}; "
            f"other snapshots retained at {output_dir}"
        )
    return output_dir
