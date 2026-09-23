"""Read-only, session-bound hardware calibration samples from the native Host."""

import math
import time

from alohamini.client import HostClient


def finite_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


class StateClient:
    def __init__(self, host: str, port: int, timeout_sec: float) -> None:
        self.client = HostClient(
            host, port=port, timeout_s=timeout_sec, expected_model="alohamini2pro", request_window=1
        )
        self.timeout_sec = timeout_sec
        self.session = None
        self.last_sample = None
        self.metadata = None

    def receive(self) -> dict:
        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            snapshot = self.client.read(include_images=False)
            observation = snapshot.payload
            safety, timing = observation["_safety"], observation["_host_timing"]
            if (
                safety.get("version") != 1
                or observation.get("_motor_feedback", {}).get("version") != 1
                or safety.get("feedback_valid") is not True
                or safety.get("phase") not in ("ready", "active")
                or safety.get("fault")
            ):
                raise ValueError("Host feedback is invalid or protected")
            session = safety.get("host_session_id")
            if not isinstance(session, str) or not session or timing.get("clock_id") != session:
                raise ValueError("Host clock/session identity is missing")
            if self.session is not None and self.session != session:
                raise ValueError("Host session changed; restart calibration")
            metadata = observation["_robot_metadata"]
            if self.metadata is not None and metadata != self.metadata:
                raise ValueError("Host calibration metadata changed; restart calibration")
            started = finite_float(timing["state_sample_started_monotonic_s"], "sample start")
            finished = finite_float(timing["state_sample_finished_monotonic_s"], "sample finish")
            now = finite_float(timing["host_clock_reference"]["monotonic_s"], "Host time")
            if not 0 <= started <= finished <= now or now - started + snapshot.round_trip_s > 0.25:
                raise ValueError("Host feedback is stale or has invalid timestamps")
            if self.last_sample is not None and finished < self.last_sample:
                raise ValueError("Host sample time moved backwards")
            if finished == self.last_sample:
                time.sleep(0.01)
                continue
            self.session, self.metadata, self.last_sample = session, metadata, finished
            return observation
        raise TimeoutError("timed out waiting for a new Host feedback sample")

    def close(self) -> None:
        self.client.close()
