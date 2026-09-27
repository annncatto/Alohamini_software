"""Training samples from AlohaMini recordings and their LeRobot v3 exports."""

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from alohamini.datasets.images import image_rgb
from alohamini.datasets.native import StateSelection
from alohamini.datasets.tools import check_dataset
from alohamini.learning.processor import DEFAULT_IMAGE_SIZE, image_tensor
from alohamini.policies.configuration import PolicyFeature


def capture_timeline(root, episode):
    """Read unfiltered Host-clock timing for inspection before training review.

    Run check_dataset first. Camera timestamps describe read completion, not
    exposure. Missing timestamps remain NaN; never substitute the PC clock.
    This view does not resample frames or change state/action pairing.
    """
    if type(episode) is not int or episode < 0:
        raise ValueError("Episode index must be a nonnegative integer")
    root = Path(root).expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    cameras = info.get("cameras", info["robot_metadata"]["cameras"])
    records = []
    with (root / "episodes" / f"episode_{episode:06d}" / "safety.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("frame_index") is not None:
                records.append(record)
    indices = np.array([r["frame_index"] for r in records], dtype=int)
    stamps = {
        camera: np.array(
            [
                (r.get("host_timing") or {})
                .get("camera_capture_monotonic_s", {})
                .get(camera, np.nan)
                for r in records
            ],
            dtype=float,
        )
        for camera in cameras
    }
    if stamps:
        physical = np.median(np.array(list(stamps.values())), axis=0)
    else:
        physical = np.array(
            [(r.get("host_timing") or {}).get("state_sample_monotonic_s", np.nan) for r in records],
            dtype=float,
        )
    nominal = indices / info["fps"]
    capture = physical - physical[0] if len(physical) else physical
    span = capture[-1] if len(capture) else np.nan
    return {
        "fps": info["fps"],
        "frame_index": indices,
        "nominal_time_s": nominal,
        "capture_time_s": capture,
        "drift_s": capture - nominal,
        "camera_intervals_s": {name: np.diff(values) for name, values in stamps.items()},
        "alignment_error_s": np.array([r.get("alignment_error_s") for r in records], dtype=float),
        "measured_fps": (len(physical) - 1) / span if span > 0 else np.nan,
    }


class AlohaMiniDataset(Dataset):
    """PyTorch dataset over AlohaMini recordings and their LeRobot v3 exports.

    By default each item is one recorded row. ``delta_indices`` selects per-field
    row offsets, e.g. {"observation.state": [-1, 0], "action": [0, 1, 2]}.
    Windowed fields gain a leading time dimension and a ``<key>_is_pad`` mask;
    boundary values are repeated, as in LeRobot's DatasetReader. Offsets follow
    recorded row order, not nearest physical timestamps. No interpolation,
    resampling or new image/state/action pairing is performed.

    Episodes and recorded control interruptions bound windows. Unusable input
    fields exclude only anchors whose requested windows need those fields;
    physical rows are never removed or renumbered. ``sample_indices`` maps
    Dataset indices to physical ``rows``/``records``/``locations`` indices.
    ``boundaries`` records reasons once per boundary, not once per sample.
    Camera intervals and normal gripper transitions do not split them. Timing
    warnings remain available in ``report`` and raw timestamps in ``records``.
    ``chunk_size`` is shorthand for action offsets range(chunk_size), retained
    for existing ACT/AM-ACT callers. Numeric stored fields can also be selected
    through delta_indices; absent fields such as rewards are never synthesized.
    """

    def __init__(
        self,
        root,
        *,
        episodes,
        chunk_size=None,
        delta_indices=None,
        state=StateSelection.DEFAULT,
        cameras=None,
        image_size=DEFAULT_IMAGE_SIZE,
        review_note="",
        include_task=False,
    ):
        self.root = Path(root).expanduser().resolve()
        self.report = check_dataset(self.root, decode_images=True)
        if not self.report["valid"]:
            raise ValueError(f"Dataset integrity check failed: {self.report['issues']}")
        if self.report["warnings"]:
            codes = sorted(
                {issue["code"] for issue in self.report["issues"] if issue["severity"] == "warning"}
            )
            logging.getLogger(__name__).warning(
                "Dataset has %d warning(s): %s. Continuing with sample filtering; "
                "timestamps are not resampled. Full report is available as samples.report.",
                self.report["warnings"],
                ", ".join(codes),
            )
        self.review_note = review_note
        self.include_task = include_task
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self._v3 = None
        if self.info.get("codebase_version") == "v3.0":
            from alohamini.learning.lerobot import LeRobotSource

            self._v3 = LeRobotSource(self.root, self.info)
            self.info = self._v3.info
        self.episodes = list(episodes)
        if not self.episodes or len(set(self.episodes)) != len(self.episodes):
            raise ValueError("Select unique, nonempty episode indices")
        if any(
            type(n) is not int
            or not 0
            <= n
            < (len(self._v3.episodes) if self._v3 else self.report["summary"]["episodes"])
            for n in self.episodes
        ):
            raise ValueError("Episode index outside dataset")
        if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        if delta_indices is not None and not isinstance(delta_indices, Mapping):
            raise ValueError("delta_indices must map field names to integer row offsets")
        self.delta_indices = {}
        for key, offsets in (delta_indices or {}).items():
            if not isinstance(key, str):
                raise ValueError("Window field names must be strings")
            try:
                offsets = list(offsets)
            except TypeError as exc:
                raise ValueError(f"{key}: offsets must be a nonempty sequence of integers") from exc
            if not offsets or any(type(i) is not int for i in offsets):
                raise ValueError(f"{key}: offsets must be a nonempty sequence of integers")
            self.delta_indices[key] = offsets
        if chunk_size is not None:
            if "action" in self.delta_indices:
                raise ValueError("Specify either chunk_size or action delta_indices, not both")
            self.delta_indices["action"] = list(range(chunk_size))
        self.state = state
        if (
            self._v3
            and state != "none"
            and (self.info["features"].get("observation.state") != self._v3.original_state)
        ):
            raise ValueError(
                "Selected v3 has no original state coordinates; "
                "use state='none' or export default state"
            )
        self.selection = None if state == "none" else StateSelection(self.info, state)
        available_cameras = self.info.get("cameras", self.info["robot_metadata"]["cameras"])
        self.cameras = list(available_cameras if cameras is None else cameras)
        if len(set(self.cameras)) != len(self.cameras) or any(
            c not in available_cameras for c in self.cameras
        ):
            raise ValueError("Select available, unique camera names")
        if len(image_size) != 2 or any(type(n) is not int or n < 32 for n in image_size):
            raise ValueError("image_size must contain height/width >= 32")
        self.image_size = tuple(image_size)
        self.rows, self.records, self.locations = [], [], []
        self.segment_starts, self.segment_ends = [], []
        self.boundaries = []
        self.excluded = 0
        self.table_sha256 = {}
        self.input_features = {
            f"observation.images.{c}": PolicyFeature("VISUAL", (3, *self.image_size))
            for c in self.cameras
        }
        if self.selection:
            self.input_features["observation.state"] = PolicyFeature(
                "STATE", tuple(self.selection.feature["shape"])
            )
        self.output_features = {}
        if "action" in self.info["features"]:
            self.output_features["action"] = PolicyFeature(
                "ACTION", tuple(self.info["features"]["action"]["shape"])
            )
        self.sample_keys = [*self.input_features, *self.output_features]
        for key in self.delta_indices:
            if key in self.sample_keys:
                continue
            feature = self.info["features"].get(key)
            if key == "observation.state" or key.startswith("observation.images."):
                raise ValueError(f"Window field is not selected by state/cameras: {key}")
            if feature is None:
                raise ValueError(f"Window field is not recorded: {key}")
            if feature["dtype"] not in {
                "float16",
                "float32",
                "float64",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "bool",
            }:
                raise ValueError(f"Window field must be numeric: {key}")
            self.sample_keys.append(key)
            if key.startswith("observation."):
                self.input_features[key] = PolicyFeature("ENV", tuple(feature["shape"]))
        if not self.sample_keys:
            raise ValueError("Select at least one recorded field")
        columns = [k for k in self.sample_keys if k != "observation.state"]
        if include_task:
            columns.append("task")
        if self.selection:
            columns.extend(k for k, _, _, _ in self.selection.columns)
            columns.extend(mask for _, _, _, mask in self.selection.columns if mask)
            if any(mask for _, _, _, mask in self.selection.columns):
                columns.append("motor_feedback.sample_finished_s")
        for key in self.sample_keys:
            mask = self._feedback_mask(key)
            if mask:
                columns.extend((mask, "motor_feedback.sample_finished_s"))
        for episode in self.episodes:
            directory = self.root / "episodes" / f"episode_{episode:06d}"
            table_path = directory / "frames.parquet"
            if self._v3:
                rows, safety_path, paths = self._v3.read_episode(episode, columns)
            else:
                safety_path = directory / "safety.jsonl"
                paths = (table_path, safety_path)
                rows = pq.read_table(table_path, columns=list(dict.fromkeys(columns))).to_pylist()
            for path in paths:
                relative = str(path.relative_to(self.root))
                if relative not in self.table_sha256:
                    with path.open("rb") as stream:
                        self.table_sha256[relative] = hashlib.file_digest(
                            stream, "sha256"
                        ).hexdigest()
            records = []
            pending_reasons = []
            with safety_path.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    if record.get("frame_index") is None:
                        pending_reasons.extend(self._interruptions(record.get("safety") or {}))
                        event = record.get("event") or {}
                        if event.get("type") == "watchdog_recovered":
                            pending_reasons.append("watchdog_stop")
                        elif event.get("type") == "sequence_boundary":
                            reason = event.get("reason")
                            pending_reasons.append(
                                reason.strip()
                                if isinstance(reason, str) and reason.strip()
                                else "explicit_boundary"
                            )
                        continue
                    records.append(
                        {
                            **{
                                k: record.get(k)
                                for k in (
                                    "host_timing",
                                    "safety",
                                    "alignment_error_s",
                                    "client_timing",
                                )
                            },
                            "boundary_reasons": pending_reasons,
                        }
                    )
                    pending_reasons = []
            start = len(self.rows)
            previous = None
            for index, (row, record) in enumerate(zip(rows, records, strict=True)):
                reasons = self._boundary_reasons(previous, record)
                if reasons:
                    self.segment_starts.extend([start] * (len(self.rows) - start))
                    self.segment_ends.extend([len(self.rows)] * (len(self.rows) - start))
                    start = len(self.rows)
                    self.boundaries.append(
                        {
                            "episode_index": episode,
                            "frame_index": index,
                            "reasons": reasons,
                        }
                    )
                self.rows.append(row)
                self.records.append(record)
                self.locations.append((episode, index))
                previous = record
            self.segment_starts.extend([start] * (len(self.rows) - start))
            self.segment_ends.extend([len(self.rows)] * (len(self.rows) - start))
        self.field_validity = {
            key: [self._field_usable(i, key) for i in range(len(self.rows))]
            for key in self.sample_keys
        }
        self.sample_indices = []
        self._used_rows = {key: set() for key in self.sample_keys}
        for i in range(len(self.rows)):
            windows, padding = self._get_query_indices(i)
            requested = {key: windows.get(key, [i]) for key in self.sample_keys}
            if not all(
                self.field_validity[key][j] for key, indices in requested.items() for j in indices
            ):
                continue
            if "action_is_pad" in padding and padding["action_is_pad"].all():
                continue
            self.sample_indices.append(i)
            for key, indices in requested.items():
                self._used_rows[key].update(indices)
        self.excluded = len(self.rows) - len(self.sample_indices)
        if not self.sample_indices:
            raise ValueError("No usable samples for the selected fields and windows")

    def __len__(self):
        return len(self.sample_indices)

    @staticmethod
    def _interruptions(safety):
        return [
            reason
            for key, reason in (
                ("fault", "fault"),
                ("watchdog_active", "watchdog_stop"),
                ("joint_holds", "joint_protection"),
            )
            if safety.get(key)
        ]

    @classmethod
    def _boundary_reasons(cls, previous, record):
        reasons = list(record["boundary_reasons"])
        if previous is None:
            return list(dict.fromkeys(["episode_start", *reasons]))
        before, after = previous["safety"] or {}, record["safety"] or {}
        for key, reason in (
            ("host_session_id", "host_restart"),
            ("control_epoch", "control_epoch_changed"),
            ("watchdog_events", "watchdog_stop"),
            ("joint_hold_events", "joint_protection"),
        ):
            if (
                before.get(key) is not None
                and after.get(key) is not None
                and before[key] != after[key]
            ):
                reasons.append(reason)
        old, new = cls._interruptions(before), cls._interruptions(after)
        if old != new:
            reasons.extend(new or ["control_recovered"])
        return list(dict.fromkeys(reasons))

    def _feedback_mask(self, key):
        if key.startswith("observation.motor_"):
            mask = "motor_feedback." + key.removeprefix("observation.motor_") + "_valid"
            if mask in self.info["features"]:
                return mask
        return None

    @staticmethod
    def _fresh_feedback(row, timing, indices):
        now = timing.get("state_sample_monotonic_s")
        finished = row["motor_feedback.sample_finished_s"]
        # A per-motor read may finish just after the whole-bus midpoint.
        return now is not None and all(-0.05 <= now - finished[j] <= 0.25 for j in indices)

    def _field_usable(self, index, key):
        row, record = self.rows[index], self.records[index]
        safety, timing = record["safety"] or {}, record["host_timing"] or {}
        try:
            if key == "action":
                # Feedback validity describes measurements, not the human's target.
                return safety.get("feedback_valid") is not None and not self._interruptions(safety)
            if key == "observation.state":
                self.selection.frame(row)
                if any(k == "observation.state" for k, _, _, _ in self.selection.columns):
                    if safety.get("feedback_valid") is not True:
                        return False
                indices = [j for _, j, _, mask in self.selection.columns if mask]
                return not indices or self._fresh_feedback(row, timing, indices)
            mask = self._feedback_mask(key)
            if mask:
                return all(v == 1 for v in row[mask]) and self._fresh_feedback(
                    row, timing, range(len(row[mask]))
                )
            # File integrity (including images) is checked before this field view.
            return True
        except (KeyError, ValueError, IndexError, TypeError):
            return False

    def _value(self, index, key):
        row = self.rows[index]
        if key.startswith("observation.images."):
            camera = key.removeprefix("observation.images.")
            episode = self.root / "episodes" / f"episode_{self.locations[index][0]:06d}"
            return image_tensor(
                self._v3.image(row[key], camera)
                if self._v3
                else image_rgb(episode, camera, row[key]),
                self.image_size,
            )
        if key == "observation.state":
            return torch.from_numpy(self.selection.frame(row))
        # Preserve integer/bool labels; policy preprocessing owns normalization.
        return torch.as_tensor(row[key], dtype=getattr(torch, self.info["features"][key]["dtype"]))

    def observation(self, index):
        """Return the current observation without temporal expansion."""
        row = self.sample_indices[index]
        return {key: self._value(row, key) for key in self.input_features}

    def _get_query_indices(self, index):
        """Query physical row offsets within an episode/control segment."""
        start, end = self.segment_starts[index], self.segment_ends[index]
        indices = {
            key: [max(start, min(end - 1, index + delta)) for delta in offsets]
            for key, offsets in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.tensor(
                [index + delta < start or index + delta >= end for delta in offsets],
                dtype=torch.bool,
            )
            for key, offsets in self.delta_indices.items()
        }
        return indices, padding

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        index = self.sample_indices[index]
        indices, sample = self._get_query_indices(index)
        if self.include_task:
            task = self.rows[index]["task"]
            if not isinstance(task, str) or not task.strip():
                raise ValueError("Language-conditioned samples require a nonempty task")
            sample["task"] = task
        for key in self.sample_keys:
            if key not in indices:
                sample[key] = self._value(index, key)
                continue
            # Decode only the requested image rows, once each even when padded.
            values = {i: self._value(i, key) for i in dict.fromkeys(indices[key])}
            sample[key] = torch.stack([values[i] for i in indices[key]])
        return sample

    def statistics(self, *, keys=None):
        """Empirical statistics of selected fields; each used physical value counted once."""
        keys = list(self._used_rows) if keys is None else list(keys)
        unknown = set(keys) - self._used_rows.keys()
        if unknown:
            raise ValueError(f"Unknown statistics fields: {sorted(unknown)}")
        means, m2s, counts = {}, {}, {}
        for key in dict.fromkeys(keys):
            indices = self._used_rows[key]
            for i in sorted(indices):
                value = self._value(i, key).double()
                if key.startswith("observation.images."):
                    value = value.flatten(1).T
                else:
                    value = value[None]
                batch_mean = value.mean(0)
                batch_m2 = (value - batch_mean).square().sum(0)
                count = counts.get(key, 0)
                total = count + len(value)
                delta = batch_mean - means.get(key, batch_mean)
                m2s[key] = (
                    m2s.get(key, 0) + batch_m2 + delta.square() * (count * len(value) / total)
                )
                means[key] = means.get(key, batch_mean) + delta * (len(value) / total)
                counts[key] = total
        stats = {}
        for key, mean in means.items():
            std = (m2s[key] / counts[key]).clamp_min(0).sqrt()
            if key.startswith("observation.images."):
                mean, std = mean[:, None, None], std[:, None, None]
            stats[key] = {"mean": mean.float().tolist(), "std": std.float().tolist()}
        return stats
