"""Shared training views over native recordings and their local v3 exports."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from alohamini.datasets.images import image_rgb
from alohamini.datasets.native import StateSelection
from alohamini.datasets.tools import check_dataset
from alohamini.policies.configuration import PolicyFeature

DEFAULT_IMAGE_SIZE = (480, 640)  # Height, width; retain the original ACT input resolution.


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


def image_tensor(rgb, size):
    tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float() / 255
    if tuple(tensor.shape[1:]) != tuple(size):
        tensor = F.interpolate(
            tensor[None], size=size, mode="bilinear", align_corners=False, antialias=True
        )[0]
    return tensor


class NativeSamples(Dataset):
    """Native/v3 observations and recorded targets, padded only at segment ends.

    No new image/state/action pairing, interpolation or action shift is performed.
    Gaps, repeated camera timestamps and safety interruptions split action chunks.
    Checker warnings are advisory; structural errors prevent loading. Optional
    ``review_note`` and the original report are saved with training checkpoints.
    """

    def __init__(
        self,
        root,
        *,
        episodes,
        chunk_size=100,
        state=StateSelection.DEFAULT,
        cameras=None,
        image_size=DEFAULT_IMAGE_SIZE,
        review_note="",
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
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self._v3 = None
        if self.info.get("codebase_version") == "v3.0":
            from alohamini.learning.lerobot import LeRobotSource

            self._v3 = LeRobotSource(self.root, self.info)
            self.info = self._v3.info
        if "action" not in self.info["features"]:
            raise ValueError("Policy training requires the action feature")
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
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
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
        if (
            not self.cameras
            or len(set(self.cameras)) != len(self.cameras)
            or any(c not in available_cameras for c in self.cameras)
        ):
            raise ValueError("Select available, unique camera names")
        if len(image_size) != 2 or any(type(n) is not int or n < 32 for n in image_size):
            raise ValueError("image_size must contain height/width >= 32")
        self.image_size = tuple(image_size)
        self.rows, self.records, self.locations, self.segment_ends = [], [], [], []
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
        self.output_features = {
            "action": PolicyFeature("ACTION", tuple(self.info["features"]["action"]["shape"]))
        }
        columns = ["action", *[f"observation.images.{c}" for c in self.cameras]]
        if self.selection:
            columns.extend(k for k, _, _, _ in self.selection.columns)
            columns.extend(mask for _, _, _, mask in self.selection.columns if mask)
            if any(mask for _, _, _, mask in self.selection.columns):
                columns.append("motor_feedback.sample_finished_s")
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
            with safety_path.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    if record.get("frame_index") is not None:
                        records.append(
                            {
                                k: record.get(k)
                                for k in (
                                    "host_timing",
                                    "safety",
                                    "alignment_error_s",
                                    "client_timing",
                                )
                            }
                        )
            start = len(self.rows)
            previous = None
            for index, (row, record) in enumerate(zip(rows, records, strict=True)):
                stamps = (record["host_timing"] or {}).get("camera_capture_monotonic_s", {})
                safety = record["safety"] or {}
                valid = (
                    safety.get("feedback_valid") is True
                    and not safety.get("fault")
                    and not safety.get("watchdog_active")
                    and not safety.get("joint_holds")
                    and all(c in stamps and np.isfinite(stamps[c]) for c in self.cameras)
                )
                # Gripper current-limited holding is part of normal grasping;
                # retain it, unlike a joint fault/hold or watchdog interruption.
                try:
                    if self.selection:
                        self.selection.frame(row)
                        now = (record["host_timing"] or {}).get("state_sample_monotonic_s")
                        for _, j, _, mask in self.selection.columns:
                            # The recorded state timestamp is the bus-read midpoint;
                            # an individual motor read can finish slightly after it.
                            if mask and (
                                now is None
                                or not -0.05
                                <= now - row["motor_feedback.sample_finished_s"][j]
                                <= 0.25
                            ):
                                valid = False
                except (KeyError, ValueError, IndexError):
                    valid = False
                continuous = previous is not None and all(
                    0.5 / self.info["fps"]
                    <= stamps.get(c, -np.inf) - previous[c]
                    <= 1.5 / self.info["fps"]
                    for c in self.cameras
                )
                if continuous:
                    last_safety = self.records[-1]["safety"] or {}
                    continuous = all(
                        safety.get(key) == last_safety.get(key)
                        for key in (
                            "host_session_id",
                            "control_epoch",
                            "control_owner",
                            "watchdog_events",
                            "joint_hold_events",
                        )
                    )
                if not valid or not continuous:
                    self.segment_ends.extend([len(self.rows)] * (len(self.rows) - start))
                    start = len(self.rows)
                if not valid:
                    self.excluded += 1
                    previous = None
                    continue
                self.rows.append(row)
                self.records.append(record)
                self.locations.append((episode, index))
                previous = stamps
            self.segment_ends.extend([len(self.rows)] * (len(self.rows) - start))
        if not self.rows:
            raise ValueError("No usable samples after feedback, timing and safety checks")

    def __len__(self):
        return len(self.rows)

    def observation(self, index):
        row = self.rows[index]
        episode = self.root / "episodes" / f"episode_{self.locations[index][0]:06d}"
        result = {
            f"observation.images.{c}": image_tensor(
                self._v3.image(row[f"observation.images.{c}"], c)
                if self._v3
                else image_rgb(episode, c, row[f"observation.images.{c}"]),
                self.image_size,
            )
            for c in self.cameras
        }
        if self.selection:
            result["observation.state"] = torch.from_numpy(self.selection.frame(row))
        return result

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample = self.observation(index)
        end = min(index + self.chunk_size, self.segment_ends[index])
        actions = [row["action"] for row in self.rows[index:end]]
        count = len(actions)
        actions += [actions[-1]] * (self.chunk_size - count)
        sample["action"] = torch.tensor(actions, dtype=torch.float32)
        sample["action_is_pad"] = torch.arange(self.chunk_size) >= count
        return sample

    def statistics(self):
        """Population mean/std of this split only; each physical row counted once."""
        sums, squares, counts = {}, {}, {}
        for i, row in enumerate(self.rows):
            observation = self.observation(i)
            observation["action"] = torch.tensor(row["action"], dtype=torch.float32)
            for key, value in observation.items():
                value = value.double()
                if key.startswith("observation.images."):
                    value = value.flatten(1).T
                else:
                    value = value[None]
                sums[key] = sums.get(key, 0) + value.sum(0)
                squares[key] = squares.get(key, 0) + value.square().sum(0)
                counts[key] = counts.get(key, 0) + len(value)
        stats = {}
        for key in sums:
            mean = sums[key] / counts[key]
            std = (squares[key] / counts[key] - mean.square()).clamp_min(0).sqrt()
            if key.startswith("observation.images."):
                mean, std = mean[:, None, None], std[:, None, None]
            stats[key] = {"mean": mean.float().tolist(), "std": std.float().tolist()}
        return stats
