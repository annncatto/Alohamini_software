"""Passive tracking diagnostics from existing teleoperation snapshots."""

import csv
import logging
import math
import threading
import time
from copy import deepcopy
from pathlib import Path
from queue import Empty, Full, Queue

from alohamini.calibration.encoder import HostPositionUnits
from alohamini.paths import WorkspacePaths


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def target_tick(units, value):
    """Invert reported wire values without truncating a round-tripped encoder tick."""
    if number(value) is None:
        raise ValueError("Non-finite target")
    first, last = units.from_tick(units.range_min), units.from_tick(units.range_max)
    if not min(first, last) <= value <= max(first, last):
        raise ValueError("Target outside calibration")
    return round(
        units.range_min + (value - first) * (units.range_max - units.range_min) / (last - first)
    )


class TrackingAnalysis:
    """Align feedback only to a target already issued before its sampling began.

    A preceding snapshot is usable across adjacent Host cycles only. Missing
    cycles may hide another target; such pairs remain unscored. Angles describe
    STS output encoder coordinates (4096 ticks/turn), not ROS joint coordinates.
    """

    FIELDS = (
        "session",
        "epoch",
        "owner",
        "state_sequence",
        "joint",
        "motor_id",
        "sample_started_s",
        "sample_finished_s",
        "command_sequence",
        "command_at_s",
        "requested_tick",
        "command_tick",
        "measured_tick",
        "error_deg",
        "target_reduction_deg",
        "velocity_ticks_s",
        "current_a",
        "joint_hold",
        "pairing",
    )

    def __init__(self):
        self.previous = None
        self.stats = {}

    def rows(self, payload):
        safety = payload["_safety"]
        metadata = payload["_robot_metadata"]
        sequence = payload["_host_timing"]["state_sequence"]
        context = (safety["host_session_id"], safety["control_epoch"], safety["control_owner"])
        previous = self.previous
        if previous is not None and (previous[0] != context or previous[2] != metadata):
            previous = None
        if previous is not None and sequence <= previous[1]:
            return []
        self.previous = (context, sequence, metadata, safety)
        result = []
        feedback = payload.get("_motor_feedback", {}).get("motors", {})
        for name, motor in metadata["motors"].items():
            if not name.startswith("arm_") or name.endswith("gripper"):
                continue
            sample = feedback.get(name, {})
            errors = sample.get("field_errors", {})
            started, finished = (
                number(sample.get(key)) for key in ("sample_started_s", "sample_finished_s")
            )
            raw = sample.get("position_raw")
            valid = (
                safety.get("feedback_valid") is True
                and sample.get("packet_error") == 0
                and not errors.get("position_raw")
                and type(raw) is int
                and 0 <= raw < 4096
                and started is not None
                and finished is not None
                and started <= finished
            )
            selected = safety
            at = number(selected.get("accepted_at_monotonic_s"))
            paired = valid and at is not None and at <= started
            pairing = "current" if paired else "unavailable"
            if not paired and valid and previous is not None and sequence == previous[1] + 1:
                candidate = previous[3]
                at = number(candidate.get("accepted_at_monotonic_s"))
                if at is not None and at <= started:
                    selected, paired, pairing = candidate, True, "previous_cycle"
            row = dict.fromkeys(self.FIELDS)
            row.update(
                session=context[0],
                epoch=context[1],
                owner=context[2],
                state_sequence=sequence,
                joint=name,
                motor_id=motor["id"],
                sample_started_s=started,
                sample_finished_s=finished,
                measured_tick=raw if valid else None,
                command_sequence=selected.get("command", {}).get("sequence"),
                command_at_s=selected.get("accepted_at_monotonic_s"),
                joint_hold=name in selected.get("joint_holds", {}),
                pairing=pairing if valid else "invalid_feedback",
            )
            for source, column, scale in (
                ("current_ma", "current_a", 0.001),
                ("velocity_raw", "velocity_ticks_s", 1),
            ):
                value = number(sample.get(source))
                if (
                    value is not None
                    and not errors.get(source)
                    and not (source == "current_ma" and errors.get("current_raw"))
                    and sample.get("packet_error") == 0
                ):
                    row[column] = value * scale
            try:
                units = HostPositionUnits(
                    **{
                        key: motor[key]
                        for key in ("normalization", "range_min", "range_max", "drive_mode")
                    }
                )
                key = f"{name}.pos"
                row["command_tick"] = target_tick(units, selected["accepted_targets"][key])
                row["requested_tick"] = target_tick(units, selected["requested_targets"][key])
                row["target_reduction_deg"] = (
                    (row["requested_tick"] - row["command_tick"]) * 360 / 4096
                )
                if paired:
                    delta = row["command_tick"] - raw
                    if abs(delta) >= 2048:
                        row["pairing"] = "ambiguous_wrap"
                    else:
                        row["error_deg"] = delta * 360 / 4096
            except (KeyError, TypeError, ValueError):
                row["pairing"] = "missing_target_or_calibration"
            result.append(row)
            self._accumulate(row)
        return result

    def _accumulate(self, row):
        key = (row["session"], row["epoch"], row["owner"], row["joint"], row["motor_id"])
        stats = self.stats.setdefault(
            key,
            dict(
                samples=0,
                scored=0,
                absolute=0.0,
                squared=0.0,
                maximum=0.0,
                currents=0,
                current_sum=0.0,
                current_max=0.0,
                holds=0,
            ),
        )
        stats["samples"] += 1
        stats["holds"] += int(row["joint_hold"])
        if row["error_deg"] is not None:
            error = abs(row["error_deg"])
            stats["scored"] += 1
            stats["absolute"] += error
            stats["squared"] += error * error
            stats["maximum"] = max(stats["maximum"], error)
        if row["current_a"] is not None:
            current = abs(row["current_a"])
            stats["currents"] += 1
            stats["current_sum"] += current
            stats["current_max"] = max(stats["current_max"], current)

    def summary(self):
        for key, stats in self.stats.items():
            count, currents = stats["scored"], stats["currents"]
            yield dict(
                zip(("session", "epoch", "owner", "joint", "motor_id"), key, strict=True),
                samples=stats["samples"],
                scored_samples=count,
                mae_deg=stats["absolute"] / count if count else None,
                rmse_deg=math.sqrt(stats["squared"] / count) if count else None,
                max_error_deg=stats["maximum"] if count else None,
                mean_current_a=stats["current_sum"] / currents if currents else None,
                max_current_a=stats["current_max"] if currents else None,
                hold_samples=stats["holds"],
            )


class TrackingLog:
    """Bounded background CSV writer; never owns a robot or sends commands."""

    def __init__(self, root=None):
        self.root = (
            Path(root)
            if root is not None
            else (
                WorkspacePaths().logs
                / "tracking"
                / f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
            )
        )
        self.root.mkdir(parents=True, exist_ok=False)
        self.queue = Queue(maxsize=128)
        self.stopping = threading.Event()
        self.dropped = 0
        self.error = None
        self.thread = threading.Thread(target=self._write, name="AlohaMiniTracking", daemon=True)
        self.thread.start()
        print(f"[TRACKING] {self.root}", flush=True)

    def submit(self, payload):
        if self.error is not None or self.stopping.is_set():
            return
        try:
            self.queue.put_nowait(deepcopy(payload))
        except Full:
            self.dropped += 1
            if self.dropped == 1:
                logging.warning(
                    "Tracking queue full; diagnostic samples dropped, teleoperation continues"
                )

    def _write(self):
        analysis = TrackingAnalysis()
        try:
            with (self.root / "samples.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=analysis.FIELDS)
                writer.writeheader()
                while not self.stopping.is_set() or not self.queue.empty():
                    try:
                        payload = self.queue.get(timeout=0.05)
                    except Empty:
                        continue
                    writer.writerows(analysis.rows(payload))
                stream.flush()
            summaries = list(analysis.summary())
            if summaries:
                with (self.root / "summary.csv").open("x", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=[*summaries[0], "dropped_snapshots"])
                    writer.writeheader()
                    writer.writerows(dict(row, dropped_snapshots=self.dropped) for row in summaries)
        except Exception as exc:
            self.error = exc
            logging.warning("Tracking log stopped: %s; teleoperation continues", exc)

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive() or self.error is not None:
            logging.warning("Tracking log incomplete: %s", self.root)
        else:
            print(f"[TRACKING] Saved: {self.root}; dropped_snapshots={self.dropped}", flush=True)
