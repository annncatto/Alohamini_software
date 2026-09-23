import hashlib
import sys
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "aruco"):
    pytest.skip("OpenCV with aruco is required", allow_module_level=True)


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "scripts"))

from camera_board import create_charuco_board, detect_board, draw_board, load_board  # noqa: E402

BOARD_PATH = PACKAGE / "config/cameras/boards/charuco_9x7_26mm_18p7_ids300_330.yaml"


def test_standard_a4_charuco_geometry_and_detection():
    config = load_board(BOARD_PATH)
    board = create_charuco_board(config)

    ids = board.getIds() if hasattr(board, "getIds") else board.ids
    corners = (
        board.getChessboardCorners()
        if hasattr(board, "getChessboardCorners")
        else board.chessboardCorners
    )
    assert ids.reshape(-1).tolist() == list(range(300, 331))
    assert corners.shape == (48, 3)
    # The printable bitmap is identical on OpenCV 4.5.4 and 4.13.0.
    assert hashlib.sha256(draw_board(board, (2340, 1820)).tobytes()).hexdigest() == (
        "68e4c96cb84e8490c022e3145a9a661d562c96ba8583ebfd2c8ca461794bbc5e"
    )

    image = draw_board(board, (1800, 1400), margin=20)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    detection = detect_board(image, config)

    assert detection is not None
    assert detection.point_count == 48
    assert detection.object_points.shape == (48, 1, 3)
    assert detection.image_points.shape == (48, 1, 2)
