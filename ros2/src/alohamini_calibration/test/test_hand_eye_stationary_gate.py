import math
import runpy
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/capture_hand_eye_samples"
sys.path.insert(0, str(SCRIPT.parent))
POSES_WITHIN = runpy.run_path(str(SCRIPT))["poses_within"]


def pose(x=0.0, yaw_deg=0.0):
    half = math.radians(yaw_deg) / 2.0
    return {
        "translation_m": [x, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, math.sin(half), math.cos(half)],
    }


def test_stationary_gate_requires_translation_and_rotation_to_remain_bounded():
    reference = pose()

    assert POSES_WITHIN(reference, pose(x=0.001, yaw_deg=0.5), 0.0015, math.radians(0.75))
    assert not POSES_WITHIN(reference, pose(x=0.002, yaw_deg=0.5), 0.0015, math.radians(0.75))
    assert not POSES_WITHIN(reference, pose(x=0.001, yaw_deg=1.0), 0.0015, math.radians(0.75))
