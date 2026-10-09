"""Named fixed targets shared by recording, learning and deployment."""

import json
import math
import time
from copy import deepcopy
from pathlib import Path

from alohamini._validation import finite_number
from alohamini.client import control_feedback_valid

VELOCITIES = ("x.vel", "y.vel", "theta.vel")


def dimensions(selection, robot_model):
    from alohamini.datasets.record import state_names

    names = state_names(robot_model)
    groups = {
        side: [n for n in names if n.startswith(side + "_")] for side in ("arm_left", "arm_right")
    }
    groups.update(base=list(VELOCITIES), lift_axis=["lift_axis.height_mm"])
    if selection is None:
        return []
    if isinstance(selection, str):
        selection = selection.split(",")
    result = set()
    for item in selection:
        if item in groups:
            result.update(groups[item])
        elif item in names:
            result.add(item)
        else:
            raise ValueError(f"Unknown fixed dimension: {item}")
    if not result:
        raise ValueError("Select at least one fixed dimension")
    return [n for n in names if n in result]


def validate(config, robot_model):
    if config is None:
        return {}
    if (
        not isinstance(config, dict)
        or type(config.get("version")) is not int
        or config["version"] != 1
    ):
        raise ValueError("Unsupported fixed-dimension configuration")
    targets = config.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("Fixed configuration requires named targets")
    if set(dimensions(list(targets), robot_model)) != set(targets):
        raise ValueError("Saved fixed targets require canonical dimension names")
    for name, value in targets.items():
        finite_number(value, name)
        if name in VELOCITIES and value != 0:
            raise ValueError("Fixed base dimensions require zero velocity")
    return dict(targets)


def dataset_info(root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") == "v3.0":
        return json.loads((root / "meta/alohamini.json").read_text())["source_info"]
    return info


def recording_config(root, selection, robot_model, *, resume):
    selected = dimensions(selection, robot_model)
    if not resume:
        return None, selected
    saved = dataset_info(root).get("fixed_dimensions")
    targets = validate(saved, robot_model)
    if selection is not None and set(selected) != set(targets):
        raise ValueError("Resume must retain the dataset's fixed dimensions; use a new dataset")
    return saved, list(targets)


def capture(snapshot, selected):
    if not control_feedback_valid(snapshot):
        raise ValueError("Fresh feedback required to capture fixed targets")
    targets = {n: 0.0 if n in VELOCITIES else float(snapshot.payload[n]) for n in selected}
    config = dict(version=1, targets=targets, source="recording_start_feedback")
    validate(config, snapshot.robot_model)
    return config


def limits(name, metadata):
    """Conservative preparation speed and arrival tolerance in declared Host units."""
    if name in VELOCITIES:
        return 0.0, {"x.vel": 0.02, "y.vel": 0.02, "theta.vel": 2.0}[name]
    if name == "lift_axis.height_mm":
        return 20.0, 3.0
    normalization = metadata["motors"][name.removesuffix(".pos")]["normalization"]
    return (10.0, 2.0) if normalization != "range_0_100" else (20.0, 2.0)


class FixedGuard:
    """Abort sustained loss of the prepared pose; never hide actual feedback."""

    def __init__(self, config, metadata):
        self.targets = validate(config, metadata["robot_model"])
        self.metadata = metadata
        self.outside_since = None

    def check(self, snapshot):
        bad = [
            n
            for n, v in self.targets.items()
            if abs(snapshot.payload[n] - v) > limits(n, self.metadata)[1]
        ]
        if not bad:
            self.outside_since = None
        elif self.outside_since is None:
            self.outside_since = time.monotonic()
        elif time.monotonic() - self.outside_since >= 0.5:
            raise RuntimeError("Fixed target no longer held: " + ", ".join(bad))


def restore(
    client,
    robot_model,
    config,
    *,
    timeout_s=60.0,
    cancelled=None,
    expected_metadata=None,
    expected_snapshot=None,
):
    """Prepare fixed targets with bounded ramps; caller retains the returned lease.

    This is joint-space preparation, not collision planning. On any failure the
    last owned command is stopped. No commands are sent using stale feedback.
    """
    import numpy as np

    from alohamini.apps.replay import check_target_ranges
    from alohamini.apps.teleoperation import ready_units, stop_owned_robot
    from alohamini.datasets.record import state_names

    targets = validate(config, robot_model)
    if not targets:
        return None
    finite_number(timeout_s, "preparation timeout")
    if timeout_s <= 0:
        raise ValueError("Preparation timeout must be positive")
    names = state_names(robot_model)
    identity = None
    started = time.monotonic()
    initial = None
    stable_since = None
    last_request = -math.inf
    print("Restoring fixed targets: " + json.dumps(targets, ensure_ascii=False), flush=True)
    try:
        while time.monotonic() - started < timeout_s:
            if cancelled is not None and cancelled():
                raise InterruptedError("Fixed-target preparation cancelled")
            snapshot = client.read()
            if ready_units(
                snapshot, robot_model, client.client_id
            ) is None or not control_feedback_valid(snapshot):
                raise RuntimeError(
                    "Fresh controllable feedback required for fixed-target preparation"
                )
            if initial is None:
                if (
                    expected_metadata is not None
                    and snapshot.payload["_robot_metadata"] != expected_metadata
                ):
                    raise RuntimeError("Host metadata changed before fixed-target preparation")
                if expected_snapshot is not None and (
                    any(
                        snapshot.payload["_safety"][key]
                        != expected_snapshot.payload["_safety"][key]
                        for key in ("host_session_id", "control_epoch")
                    )
                    or snapshot.payload.get("lift_axis.reference_sequence")
                    != expected_snapshot.payload.get("lift_axis.reference_sequence")
                ):
                    raise RuntimeError("Host context changed before fixed-target preparation")
                initial = deepcopy(snapshot)
                metadata = initial.payload["_robot_metadata"]
                baseline = {n: 0.0 if n in VELOCITIES else float(initial.payload[n]) for n in names}
                final = {**baseline, **targets}
                check_target_ranges(np.asarray([[final[n] for n in names]]), names, snapshot)
                started_motion = time.monotonic()
            if (
                snapshot.payload["_robot_metadata"] != metadata
                or snapshot.payload["_safety"]["host_session_id"]
                != initial.payload["_safety"]["host_session_id"]
                or snapshot.payload.get("lift_axis.reference_sequence")
                != initial.payload.get("lift_axis.reference_sequence")
                or snapshot.payload["_safety"]["control_epoch"]
                != initial.payload["_safety"]["control_epoch"]
            ):
                raise RuntimeError("Host context changed during fixed-target preparation")
            elapsed = time.monotonic() - started_motion
            action = dict(baseline)
            ramp_done = True
            for name, value in targets.items():
                speed, _ = limits(name, metadata)
                delta = value - baseline[name]
                step = min(abs(delta), speed * elapsed) if speed else abs(delta)
                action[name] = baseline[name] + math.copysign(step, delta)
                ramp_done &= step >= abs(delta)
            submitted = client.send_command(action, based_on=snapshot)
            if submitted is not None:
                identity = submitted
            fresh = snapshot.request_started_s > last_request
            last_request = snapshot.request_started_s
            at_target = all(
                abs(snapshot.payload[n] - v) <= limits(n, metadata)[1] for n, v in targets.items()
            )
            if fresh and submitted is not None and ramp_done and at_target:
                if stable_since is None:
                    stable_since = time.monotonic()
                if time.monotonic() - stable_since >= 0.5:
                    print("Fixed targets ready", flush=True)
                    return identity
            else:
                stable_since = None
            time.sleep(0.02)
        raise TimeoutError("Fixed targets did not settle before the preparation timeout")
    except BaseException:
        stop_owned_robot(client, robot_model, identity)
        raise
