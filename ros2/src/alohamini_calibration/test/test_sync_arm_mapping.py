import importlib.util
import math
import sys
import xml.etree.ElementTree as ET
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

from alohamini.model import get_robot_model

SCRIPT = Path(__file__).parents[1] / "scripts/sync_arm_mapping"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_loader(
    "sync_arm_mapping", SourceFileLoader("sync_arm_mapping", str(SCRIPT))
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def calibration():
    return {
        f"arm_{side}_{joint}": {
            "id": motor_id,
            "drive_mode": 0,
            "homing_offset": -100 + motor_id,
            "range_min": 500,
            "range_max": 3500,
        }
        for side in ("left", "right")
        for motor_id, joint in enumerate(MODULE.ARM_JOINTS, start=1)
    }


def template(side):
    joints = {}
    for motor_id, joint in enumerate(MODULE.ARM_JOINTS, start=1):
        joints[joint] = {
            "id": motor_id,
            "model": "sts3095" if joint in ("shoulder_lift", "elbow_flex") else "sts3250",
            "reference_tick": 2048,
            "reference_q_rad": 0.0,
            "sign": 1,
        }
    joints["gripper"].update(
        {
            "sign": -1,
            "reference_q_rad": 0.32,
            "joint_per_encoder_ratio": 1.0,
            "urdf_closed_rad": 0.32,
            "urdf_open_rad": -1.8030294104,
            "closed_tick": 2048,
            "open_tick": 3432,
        }
    )
    return {
        "schema_version": 1,
        "robot_model": "alohamini2pro",
        "side": side,
        "ticks_per_revolution": 4096,
        "joints": joints,
    }


def host_observation(document, ticks):
    motors = {}
    observation = {"_images": []}
    for name, entry in document.items():
        metadata = {
            "id": entry["id"],
            "homing_offset": entry["homing_offset"],
            "range_min": entry["range_min"],
            "range_max": entry["range_max"],
            "drive_mode": entry["drive_mode"],
            "normalization": "degrees",
        }
        motors[name] = metadata
        midpoint = (entry["range_min"] + entry["range_max"]) / 2.0
        observation[f"{name}.pos"] = (ticks[name] - midpoint) * 360.0 / 4095.0
    observation["_robot_metadata"] = {
        "schema_version": 1,
        "robot_model": "alohamini2pro",
        "motors": motors,
    }
    observation["_motor_feedback"] = {
        "motors": {name: {"position_raw": tick, "packet_error": 0} for name, tick in ticks.items()}
    }
    return observation


def test_json_and_host_capture_build_machine_specific_mapping():
    document = calibration()
    MODULE.validate_calibration(document)
    expected = {name: 2000 + entry["id"] for name, entry in document.items()}
    observation = host_observation(document, expected)

    captured = MODULE.median_stable_ticks(
        [observation, observation, observation], document, max_spread=2
    )
    mapping = MODULE.build_side_mapping(
        template("right"),
        document,
        captured,
        "right",
        {
            "captured_at": "2026-08-19T00:00:00+00:00",
            "host": "192.168.3.73",
            "observation_port": 5556,
            "command_port_used": False,
            "calibration_sha256": "abc",
        },
    )

    pan = mapping["joints"]["shoulder_pan"]
    assert pan["reference_tick"] == pytest.approx(expected["arm_right_shoulder_pan"], abs=1)
    assert pan["motor_calibration"]["homing_offset"] == -99
    assert pan["safe_q_min_rad"] == pytest.approx(
        (500 - pan["reference_tick"]) * 2.0 * math.pi / 4096
    )
    assert mapping["machine_profile"]["command_port_used"] is False
    assert mapping["reference_capture"]["q_rad"] == [0.0] * 6


def test_capture_rejects_motion_and_json_host_mismatch():
    document = calibration()
    ticks = {name: 2048 for name in document}
    first = host_observation(document, ticks)
    moved = host_observation(document, {**ticks, "arm_left_shoulder_pan": 2060})

    with pytest.raises(ValueError, match="moved by"):
        MODULE.median_stable_ticks([first, moved], document, max_spread=4)

    first["_robot_metadata"]["motors"]["arm_left_shoulder_pan"]["range_min"] = 501
    with pytest.raises(ValueError, match="disagree"):
        MODULE.ticks_from_observation(first, document)


def test_calibration_requires_all_arm_entries():
    document = calibration()
    document.pop("arm_right_gripper")
    with pytest.raises(ValueError, match="lacks arm_right_gripper"):
        MODULE.validate_calibration(document)


def test_mapping_uses_raw_feedback_and_checks_actual_offset():
    document = calibration()
    ticks = {name: 2048 for name in document}
    observation = host_observation(document, ticks)
    observation["arm_left_shoulder_pan.pos"] = 100.0
    assert MODULE.ticks_from_observation(observation, document)["arm_left_shoulder_pan"] == 2048
    observation["_robot_metadata"]["motors"]["arm_left_shoulder_pan"]["homing_offset"] += 1
    with pytest.raises(ValueError, match="homing_offset"):
        MODULE.ticks_from_observation(observation, document)


def test_default_geometry_uses_model_without_device_ticks():
    model = get_robot_model("alohamini2pro")
    semantic = ET.parse(model.description_path("semantic"))
    for side in ("left", "right"):
        template = MODULE.default_template(side)
        assert "reference_capture" not in template
        home = {
            j.attrib["name"]: float(j.attrib["value"])
            for j in semantic.find(f"group_state[@name='home'][@group='{side}_arm']").findall(
                "joint"
            )
        }
        for name, entry in template["joints"].items():
            assert "reference_tick" not in entry
            if name != "gripper":
                key = f"{side}_{'wrist_yaw_joint' if name == 'wrist_yaw' else name}"
                assert entry["reference_q_rad"] == home[key]


def test_generated_mapping_loads_in_current_bridge_without_historical_claims():
    from alohamini_bridge.mapping import JointMapper

    document = calibration()
    ticks = {name: 2000 for name in document}
    source = {"captured_at": "now", "host": "127.0.0.1"}
    mappings = {}
    for side in ("left", "right"):
        geometry = MODULE.default_template(side)
        geometry["gripper_capture"] = {"old": True}
        geometry["joints"]["shoulder_pan"]["direction_validation"] = "old robot"
        mappings[side] = MODULE.build_side_mapping(geometry, document, ticks, side, source)
        assert "gripper_capture" not in mappings[side]
        assert "direction_validation" not in mappings[side]["joints"]["shoulder_pan"]
    mapper = JointMapper(
        mappings,
        {
            "mechanism": {"physical_min_mm": 0, "physical_max_mm": 600},
            "urdf": {"q_at_physical_min_m": -0.3, "q_at_physical_max_m": 0.3},
        },
    )
    for side in ("left", "right"):
        for joint in MODULE.ARM_JOINTS:
            ros_name = f"{side}_{'wrist_yaw_joint' if joint == 'wrist_yaw' else joint}"
            assert mapper.encoders[ros_name].position_from_tick(2000) == pytest.approx(
                mappings[side]["joints"][joint]["reference_q_rad"]
            )


def test_remote_copy_is_targeted_and_bounded(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        MODULE.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    MODULE.pull_json(
        "pi5@192.168.8.161",
        "Alohamini_workspace/calibration/robots/AlohaMiniRobot.json",
        tmp_path / "robot.json",
    )
    assert len(calls) == 1 and calls[0][1]["timeout"] == 30
    assert "--protect-args" in calls[0][0][0]
    with pytest.raises(ValueError):
        MODULE.pull_json("pi5@host;other", "robot.json", tmp_path / "robot.json")
