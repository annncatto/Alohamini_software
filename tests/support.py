import json


def state_payload():
    return {
        "arm_left_shoulder_pan.pos": -12.5,
        "lift_axis.height_mm": 120.0,
        "_images": [],
        "_image_encoding": "jpeg",
        "_robot_metadata": {"schema_version": 1, "robot_model": "alohamini2pro", "motors": {}},
        "_host_timing": {"state_sample_monotonic_s": 100.125},
        "_safety": {"version": 1, "host_session_id": "host-session", "control_owner": None},
        "_motor_feedback": {
            "version": 1,
            "motors": {
                "arm_left_shoulder_pan": {
                    "current_raw": 10,
                    "current_ma": 65.0,
                    "sample_started_s": 100.126,
                    "sample_finished_s": 100.128,
                }
            },
        },
    }


def state_frames(token, payload=None):
    return [token, json.dumps(state_payload() if payload is None else payload).encode()]
