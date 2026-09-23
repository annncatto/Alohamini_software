"""Shared checkerboard and ChArUco geometry/detection helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class BoardDetection:
    object_points: np.ndarray
    image_points: np.ndarray
    point_count: int


def load_board(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("board document must be a mapping")
    target_type = document.get("target_type")
    if target_type == "checkerboard":
        columns = int(document["columns"])
        rows = int(document["rows"])
        square_size_m = float(document["square_size_m"])
        if columns < 2 or rows < 2 or not math.isfinite(square_size_m) or square_size_m <= 0.0:
            raise ValueError("invalid checkerboard geometry")
    elif target_type == "charuco":
        squares_x = int(document["squares_x"])
        squares_y = int(document["squares_y"])
        square_length_m = float(document["square_length_m"])
        marker_length_m = float(document["marker_length_m"])
        marker_id_min = int(document["marker_id_min"])
        marker_id_max = int(document["marker_id_max"])
        expected = int(document["expected_marker_count"])
        if squares_x < 2 or squares_y < 2:
            raise ValueError("invalid ChArUco square geometry")
        if not math.isfinite(square_length_m) or not 0.0 < marker_length_m < square_length_m:
            raise ValueError("marker length must be smaller than the square length")
        if marker_id_max - marker_id_min + 1 != expected:
            raise ValueError("configured ChArUco marker ID range/count disagree")
        # With the standard alternating layout, marker count is floor(N/2).
        if expected != squares_x * squares_y // 2:
            raise ValueError("marker count does not match a standard ChArUco layout")
    else:
        raise ValueError(f"unsupported target_type: {target_type!r}")
    return document


def create_charuco_board(config: dict):
    if config.get("target_type") != "charuco":
        raise ValueError("not a ChArUco board configuration")
    try:
        dictionary_id = getattr(cv2.aruco, str(config["dictionary"]))
    except AttributeError as error:
        raise ValueError(f"unknown ArUco dictionary: {config['dictionary']}") from error
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    marker_ids = np.arange(
        int(config["marker_id_min"]),
        int(config["marker_id_max"]) + 1,
        dtype=np.int32,
    )
    if marker_ids[0] < 0 or marker_ids[-1] >= len(dictionary.bytesList):
        raise ValueError("ChArUco IDs exceed the dictionary")
    if not hasattr(cv2.aruco, "CharucoDetector"):
        # OpenCV < 4.6 uses the legacy even-row pattern. Odd rows (the supplied
        # 9x7 board) are identical; never silently change printed board geometry.
        if int(config["squares_y"]) % 2 == 0 and not config.get("legacy_pattern", False):
            raise ValueError("This OpenCV requires legacy_pattern=true for even-row boards")
        board = cv2.aruco.CharucoBoard_create(
            int(config["squares_x"]),
            int(config["squares_y"]),
            float(config["square_length_m"]),
            float(config["marker_length_m"]),
            dictionary,
        )
        board.setIds(marker_ids)
        return board
    try:
        board = cv2.aruco.CharucoBoard(
            (int(config["squares_x"]), int(config["squares_y"])),
            float(config["square_length_m"]),
            float(config["marker_length_m"]),
            dictionary,
            marker_ids,
        )
    except AttributeError as error:
        raise RuntimeError("OpenCV with the aruco module is required") from error
    board.setLegacyPattern(bool(config.get("legacy_pattern", False)))
    actual_ids = board.getIds().reshape(-1)
    if not np.array_equal(actual_ids, marker_ids):
        raise ValueError("OpenCV ChArUco marker IDs do not match the configuration")
    return board


def draw_board(board, size, *, margin=0):
    method = board.generateImage if hasattr(board, "generateImage") else board.draw
    return method(size, marginSize=margin, borderBits=1)


def _checkerboard_detection(image: np.ndarray, config: dict) -> BoardDetection | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    columns, rows = int(config["columns"]), int(config["rows"])
    found, corners = cv2.findChessboardCornersSB(
        gray,
        (columns, rows),
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    )
    if not found:
        return None
    object_points = np.zeros((columns * rows, 1, 3), np.float32)
    object_points[:, 0, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    object_points *= float(config["square_size_m"])
    image_points = corners.reshape(-1, 1, 2).astype(np.float32)
    return BoardDetection(object_points, image_points, len(image_points))


def _charuco_detection(image: np.ndarray, config: dict) -> BoardDetection | None:
    board = create_charuco_board(config)
    modern = hasattr(cv2.aruco, "CharucoDetector")
    parameters = cv2.aruco.DetectorParameters() if modern else cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if modern:
        detector = cv2.aruco.CharucoDetector(board, detectorParams=parameters)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(image)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(image, board.dictionary, parameters=parameters)
        if ids is None:
            return None
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, image, board
        )
    if charuco_ids is None:
        return None
    minimum = int(config.get("minimum_charuco_corners", 8))
    if len(charuco_ids) < minimum:
        return None
    if modern:
        object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    else:
        object_points = board.chessboardCorners[charuco_ids.reshape(-1)].reshape(-1, 1, 3)
        image_points = charuco_corners.reshape(-1, 1, 2)
    return BoardDetection(
        object_points.astype(np.float32),
        image_points.astype(np.float32),
        len(charuco_ids),
    )


def detect_board(image: np.ndarray, config: dict) -> BoardDetection | None:
    target_type = config["target_type"]
    if target_type == "checkerboard":
        return _checkerboard_detection(image, config)
    if target_type == "charuco":
        return _charuco_detection(image, config)
    raise ValueError(f"unsupported target_type: {target_type!r}")


def board_report(config: dict) -> dict:
    if config["target_type"] == "checkerboard":
        return {
            "target_type": "checkerboard",
            "columns": int(config["columns"]),
            "rows": int(config["rows"]),
            "square_size_m": float(config["square_size_m"]),
        }
    return {
        "target_type": "charuco",
        "squares_x": int(config["squares_x"]),
        "squares_y": int(config["squares_y"]),
        "square_length_m": float(config["square_length_m"]),
        "marker_length_m": float(config["marker_length_m"]),
        "dictionary": str(config["dictionary"]),
        "marker_id_min": int(config["marker_id_min"]),
        "marker_id_max": int(config["marker_id_max"]),
        "legacy_pattern": bool(config.get("legacy_pattern", False)),
    }
