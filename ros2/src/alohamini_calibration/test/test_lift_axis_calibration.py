import importlib.util
import json
import sys
import threading
from copy import deepcopy
from importlib.machinery import SourceFileLoader
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/calibrate_lift_axis"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_loader(
    "calibrate_lift_axis", SourceFileLoader("calibrate_lift_axis", str(SCRIPT))
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fit_recovers_direction_lead_and_home_mapping():
    samples = [
        {"physical_height_mm": 0.0, "raw_tick": 100, "extended_ticks": 500.0},
        {"physical_height_mm": 131.0, "raw_tick": 100, "extended_ticks": -3596.0},
        {"physical_height_mm": 262.0, "raw_tick": 100, "extended_ticks": -7692.0},
    ]

    result = MODULE.fit_calibration(samples, 4096, -0.3)

    assert result["encoder_direction_sign"] == -1
    assert result["measured_lead_mm_per_revolution"] == pytest.approx(131.0)
    assert result["home_extended_ticks"] == 500.0
    assert result["urdf_q_at_home_m"] == -0.3
    assert result["rms_fit_error_mm"] == pytest.approx(0.0, abs=1e-10)


def test_fit_requires_motion_and_two_points():
    with pytest.raises(ValueError, match="at least two"):
        MODULE.fit_calibration([], 4096, -0.3)
    with pytest.raises(ValueError, match="span enough motion"):
        MODULE.fit_calibration(
            [
                {"physical_height_mm": 0.0, "raw_tick": 1, "extended_ticks": 10.0},
                {"physical_height_mm": 1.0, "raw_tick": 1, "extended_ticks": 10.0},
            ],
            4096,
            -0.3,
        )


def observation(sequence=1, *, session="host", reference=1, extended=500, raw=500):
    return {
        "lift_axis.homed": True,
        "lift_axis.height_mm": 0.0,
        "lift_axis.raw_tick": raw,
        "lift_axis.extended_ticks": extended,
        "lift_axis.zero_extended_ticks": 500,
        "lift_axis.reference_sequence": reference,
        "_images": [],
        "_robot_metadata": {
            "schema_version": 1,
            "robot_model": "alohamini2pro",
            "lift_axis": {
                "ticks_per_revolution": 4096,
                "lead_mm_per_revolution": 131,
                "direction_sign": -1,
                "soft_min_mm": 0,
                "soft_max_mm": 600,
            },
        },
        "_safety": {
            "version": 1,
            "phase": "ready",
            "feedback_valid": True,
            "host_session_id": session,
        },
        "_host_timing": {
            "clock_id": session,
            "state_sample_started_monotonic_s": sequence * 0.02,
            "state_sample_finished_monotonic_s": sequence * 0.02 + 0.005,
            "host_clock_reference": {"monotonic_s": sequence * 0.02 + 0.006},
        },
        "_motor_feedback": {
            "version": 1,
            "motors": {"lift_axis": {"packet_error": 0, "velocity_raw": 0, "moving": 0}},
        },
    }


def test_capture_rejects_reference_change_and_motion(monkeypatch):
    monkeypatch.setattr(MODULE.time, "sleep", lambda seconds: None)
    for second in (observation(2, reference=2), observation(2, extended=506, raw=506)):
        values = iter([observation(), second])
        client = SimpleNamespace(receive=lambda values=values: next(values))
        with pytest.raises(ValueError, match="reference changed|moved"):
            MODULE.capture_point(client, 0, 2)
    value = observation()
    value["_motor_feedback"]["motors"]["lift_axis"]["velocity_raw"] = 1
    with pytest.raises(ValueError, match="stationary"):
        MODULE.capture_point(SimpleNamespace(receive=lambda: value), 0, 1)


def test_raw_tick_median_crosses_wrap_without_inventing_midpoint(monkeypatch):
    monkeypatch.setattr(MODULE.time, "sleep", lambda seconds: None)
    values = iter([observation(1, extended=4095, raw=4095), observation(2, extended=4096, raw=0)])
    result = MODULE.capture_point(SimpleNamespace(receive=lambda: next(values)), 0, 2)
    assert result["raw_tick"] in (0, 4095)
    assert result["extended_ticks"] == 4095.5


@pytest.mark.parametrize("fault", ["session", "metadata", "stale", "protected", "backwards"])
def test_state_client_rejects_discontinuous_feedback(monkeypatch, fault):
    import hardware_capture

    first, second = observation(2), observation(3)
    if fault == "session":
        second = observation(3, session="another")
    elif fault == "metadata":
        second["_robot_metadata"]["lift_axis"]["lead_mm_per_revolution"] = 100
    elif fault == "stale":
        second["_host_timing"]["host_clock_reference"]["monotonic_s"] += 1
    elif fault == "protected":
        second["_safety"]["feedback_valid"] = False
    else:
        second = observation(1)
    values = iter([first, second])

    class ReadOnly:
        def __init__(self, *args, **kwargs):
            assert kwargs["request_window"] == 1

        def read(self, *, include_images):
            assert include_images is False
            return SimpleNamespace(payload=next(values), round_trip_s=0.01)

        def close(self):
            pass

    monkeypatch.setattr(hardware_capture, "HostClient", ReadOnly)
    client = hardware_capture.StateClient("127.0.0.1", 5556, 1)
    assert client.receive() == first
    with pytest.raises(ValueError):
        client.receive()
    client.close()


def test_state_client_skips_duplicate_sample(monkeypatch):
    import hardware_capture

    first, second = observation(), observation(2)
    values = iter([first, deepcopy(first), second])
    fake = SimpleNamespace(
        read=lambda **kwargs: SimpleNamespace(payload=next(values), round_trip_s=0.01)
    )
    monkeypatch.setattr(hardware_capture, "HostClient", lambda *args, **kwargs: fake)
    monkeypatch.setattr(hardware_capture.time, "sleep", lambda seconds: None)
    client = hardware_capture.StateClient("127.0.0.1", 5556, 1)
    assert client.receive() == first
    assert client.receive() == second


def test_lift_candidate_matches_existing_ros_mapping_schema():
    from alohamini.calibration.lift import LiftCalibration

    samples = [
        {"physical_height_mm": 0.0, "raw_tick": 100, "extended_ticks": 500.0},
        {"physical_height_mm": 262.0, "raw_tick": 100, "extended_ticks": -7692.0},
    ]
    result = MODULE.build_document(
        samples, observation(), SimpleNamespace(urdf_q_at_home_m=-0.3, host="127.0.0.1", port=5556)
    )
    mechanism, urdf = result["mechanism"], result["urdf"]
    mapping = LiftCalibration(
        mechanism["physical_min_mm"] / 1000,
        mechanism["physical_max_mm"] / 1000,
        urdf["q_at_physical_min_m"],
        urdf["q_at_physical_max_m"],
    )
    assert mapping.height_to_position(0.262) == pytest.approx(-0.038)
    assert result["status"] == "candidate_requires_review"
    assert result["source"]["command_port_used"] is False


def test_interrupted_lift_capture_keeps_raw_samples(tmp_path, monkeypatch):
    class Client:
        closed = False

        def receive(self):
            return observation()

        def close(self):
            self.closed = True

    client = Client()
    monkeypatch.setattr(MODULE, "StateClient", lambda *args: client)
    monkeypatch.setattr(MODULE.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    answers = iter(["", KeyboardInterrupt()])

    def answer(prompt):
        value = next(answers)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", answer)
    output = tmp_path / "lift.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_lift_axis",
            "--host",
            "127.0.0.1",
            "--samples-per-point",
            "1",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(KeyboardInterrupt):
        MODULE.main()
    assert not output.exists()
    import yaml

    assert len(yaml.safe_load(output.with_suffix(".samples.yaml").read_text())["samples"]) == 1
    assert client.closed


def test_native_client_uses_only_local_state_endpoint():
    import hardware_capture
    import zmq

    ready, requests, errors = Queue(), [], []

    def server():
        with zmq.Context() as context, context.socket(zmq.ROUTER) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            ready.put(socket.bind_to_random_port("tcp://127.0.0.1"))
            try:
                for sequence in (1, 2):
                    if not socket.poll(3000):
                        raise TimeoutError("missing local calibration request")
                    identity, token = socket.recv_multipart()
                    requests.append(token)
                    socket.send_multipart(
                        [identity, token, json.dumps(observation(sequence)).encode()]
                    )
            except Exception as error:
                errors.append(error)

    worker = threading.Thread(target=server)
    worker.start()
    client = hardware_capture.StateClient("127.0.0.1", ready.get(timeout=3), 2)
    try:
        assert client.receive()["lift_axis.extended_ticks"] == 500
        assert client.receive()["lift_axis.reference_sequence"] == 1
        assert client.client._command_socket is None
    finally:
        client.close()
        worker.join(timeout=4)
    assert not worker.is_alive() and not errors
    assert len(requests) == 2 and all(token.endswith(b":state") for token in requests)
