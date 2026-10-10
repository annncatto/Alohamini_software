"""Training samples from AlohaMini recordings and their LeRobot v3 exports."""

import hashlib
import json
import logging
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from alohamini.datasets.images import image_path, image_rgb
from alohamini.datasets.record import StateSelection
from alohamini.datasets.tools import check_dataset
from alohamini.datasets.video_reader import VideoFrameCache
from alohamini.learning.logging import log_stage
from alohamini.learning.processor import DEFAULT_IMAGE_SIZE, image_tensor
from alohamini.policies.configuration import PolicyFeature


class DatasetInspection:
    """Reuse one integrity check for unchanged train/validation views in a run.

    Never persisted or shared globally. File identity, size, modification/change
    times and directory membership must still match before reusing a result.
    """

    def __init__(self):
        self.root = self.stamp = self.report = None

    @staticmethod
    def snapshot(root):
        result = {}
        for path in [root, *root.rglob("*")]:
            stat = path.lstat()
            result[str(path.relative_to(root))] = (
                stat.st_dev,
                stat.st_ino,
                stat.st_mode,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        return result

    def check(self, root):
        stamp = self.snapshot(root)
        if root != self.root or stamp != self.stamp:
            report = check_dataset(root, decode_images=True)
            if self.snapshot(root) != stamp:
                raise RuntimeError(
                    "Dataset changed during integrity inspection; retry on stable data"
                )
            self.root, self.stamp, self.report = root, stamp, report
        return deepcopy(self.report)


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


def inspect_feedback(root, episodes, *, state="joint_velocity,joint_current"):
    """Read raw feedback for review, without constructing or filtering training windows.

    Missing measurements remain NaN in this diagnostic view and are listed by
    episode/frame. These values are never fed into the training Dataset.
    """
    root = Path(root).expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    source = None
    if info.get("codebase_version") == "v3.0":
        from alohamini.learning.lerobot import LeRobotSource

        source = LeRobotSource(root, info)
        info = source.info
    selection = StateSelection(info, state)
    columns = list(
        dict.fromkeys(
            [key for key, _, _, _ in selection.columns]
            + [mask for _, _, _, mask in selection.columns if mask]
        )
    )
    episodes = list(episodes)
    loaded = source.read_episodes(episodes, columns) if source else None
    values, locations, issues, starts = [], [], [], []
    for episode in episodes:
        rows = (
            next(loaded)[0]
            if source
            else pq.read_table(
                root / "episodes" / f"episode_{episode:06d}" / "frames.parquet", columns=columns
            ).to_pylist()
        )
        starts.append(len(values))
        for frame, row in enumerate(rows):
            try:
                value = selection.frame(row)
            except (KeyError, ValueError, IndexError, TypeError) as exc:
                value = np.full(len(selection.columns), np.nan, dtype=np.float32)
                issues.append({"episode": episode, "frame": frame, "message": str(exc)})
            values.append(value)
            locations.append((episode, frame))
    return {
        "names": selection.feature["names"],
        "units": selection.units,
        "values": np.asarray(values, dtype=np.float32).reshape(-1, len(selection.columns)),
        "locations": locations,
        "episode_starts": starts,
        "issues": issues,
    }


class AlohaMiniDataset(Dataset):
    """PyTorch dataset over AlohaMini recordings and their LeRobot v3 exports.

    By default each item is one recorded row. ``delta_indices`` selects per-field
    row offsets, e.g. {"observation.state": [-1, 0], "action": [0, 1, 2]}.
    Windowed fields gain a leading time dimension and a ``<key>_is_pad`` mask;
    boundary values are repeated, as in LeRobot's DatasetReader. Offsets follow
    recorded row order, not nearest physical timestamps. No interpolation,
    resampling or new image/state/action pairing is performed.

    Only episodes bound windows. All selected fields must be usable throughout
    the selected episodes; invalid values raise with their episode/frame location.
    Safety events never remove samples. ``sample_indices`` maps
    Dataset indices to physical ``rows``/``records``/``locations`` indices.
    Control events and camera intervals do not split windows. Timing
    warnings remain available in ``report`` and raw timestamps in ``records``.
    ``chunk_size`` is shorthand for action offsets range(chunk_size), retained
    for existing ACT/AM-ACT callers. Numeric stored fields can also be selected
    through delta_indices; absent fields such as rewards are never synthesized.
    """

    sample_filter = "episode_windows_v2"

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
        drop_n_last_frames=0,
        review_note="",
        include_task=False,
        video_cache_size=8,
        video_backend="pyav",
        camera_workers=0,
        return_uint8=False,
        inspection=None,
    ):
        self.root = Path(root).expanduser().resolve()
        if type(camera_workers) is not int or camera_workers < 0:
            raise ValueError("camera_workers must be a nonnegative integer")
        if type(return_uint8) is not bool:
            raise ValueError("return_uint8 must be a boolean")
        self.camera_workers = camera_workers
        self.return_uint8 = return_uint8
        self.video_cache = VideoFrameCache(video_cache_size, backend=video_backend)
        with log_stage(f"Checking dataset integrity and decoding media: {self.root}"):
            self.report = (
                inspection.check(self.root)
                if inspection is not None
                else check_dataset(self.root, decode_images=True)
            )
        if not self.report["valid"]:
            raise ValueError(f"Dataset integrity check failed: {self.report['issues']}")
        logging.getLogger(__name__).info(
            "Dataset check: errors=%d warnings=%d", self.report["errors"], self.report["warnings"]
        )
        if self.report["warnings"]:
            codes = sorted(
                {issue["code"] for issue in self.report["issues"] if issue["severity"] == "warning"}
            )
            logging.getLogger(__name__).warning(
                "Dataset has %d warning(s): %s. No samples are removed by quality warnings; "
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
        self.selection = (
            None if state == "none" else StateSelection(self.info, state, exclude_fixed=True)
        )
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
        for key in self.sample_keys:
            mask = self._feedback_mask(key)
            if mask:
                columns.append(mask)
        logging.getLogger(__name__).info("Loading selected episodes and sample windows")
        loaded = self._v3.read_episodes(self.episodes, columns) if self._v3 else None
        for number, episode in enumerate(self.episodes, 1):
            directory = self.root / "episodes" / f"episode_{episode:06d}"
            table_path = directory / "frames.parquet"
            if self._v3:
                rows, safety_path, paths = next(loaded)
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
                    if record.get("frame_index") is None:
                        continue
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
            for index, (row, record) in enumerate(zip(rows, records, strict=True)):
                self.rows.append(row)
                self.records.append(record)
                self.locations.append((episode, index))
            self.segment_starts.extend([start] * (len(self.rows) - start))
            self.segment_ends.extend([len(self.rows)] * (len(self.rows) - start))
            if number % 10 == 0 or number == len(self.episodes):
                logging.getLogger(__name__).info(
                    "Loaded episodes %d/%d; rows=%d", number, len(self.episodes), len(self.rows)
                )
        logging.getLogger(__name__).info("Validating selected fields in every selected episode")
        for i in range(len(self.rows)):
            for key in self.sample_keys:
                self._validate_field(i, key)
            if self.include_task:
                self._validate_field(i, "task")
        logging.getLogger(__name__).info("Constructing episode-bounded sample windows")
        self.sample_indices = []
        self._used_rows = {key: set() for key in self.sample_keys}
        if type(drop_n_last_frames) is not int or drop_n_last_frames < 0:
            raise ValueError("drop_n_last_frames must be a nonnegative integer")
        for i in range(len(self.rows)):
            if i >= self.segment_ends[i] - drop_n_last_frames:
                continue
            windows, padding = self._get_query_indices(i)
            requested = {key: windows.get(key, [i]) for key in self.sample_keys}
            if "action_is_pad" in padding and padding["action_is_pad"].all():
                continue
            self.sample_indices.append(i)
            for key, indices in requested.items():
                self._used_rows[key].update(indices)
        self.excluded = len(self.rows) - len(self.sample_indices)
        if not self.sample_indices:
            raise ValueError("No samples for the selected episodes and window configuration")
        logging.getLogger(__name__).info(
            "Samples ready: samples=%d window_excluded=%d quality_filtered=0",
            len(self),
            self.excluded,
        )

    def __len__(self):
        return len(self.sample_indices)

    def _feedback_mask(self, key):
        if key.startswith("observation.motor_"):
            mask = "motor_feedback." + key.removeprefix("observation.motor_") + "_valid"
            if mask in self.info["features"]:
                return mask
        return None

    def _validate_field(self, index, key):
        row, record = self.rows[index], self.records[index]
        try:
            if key.startswith("observation.images."):
                # Media integrity and frame correspondence were checked above.
                return
            if key == "task":
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError("expected nonempty task text")
                return
            if key == "observation.state":
                self.selection.frame(row)
                if any(k == "observation.state" for k, _, _, _ in self.selection.columns):
                    if (record["safety"] or {}).get("feedback_valid") is not True:
                        raise ValueError("recorded observation.state feedback_valid is not true")
                return
            value = np.asarray(row[key], dtype=np.float64)
            if value.shape != tuple(self.info["features"][key]["shape"]):
                raise ValueError("shape differs from recorded feature definition")
            if not np.isfinite(value).all():
                raise ValueError("contains non-finite values")
            mask = self._feedback_mask(key)
            if mask and (
                np.shape(row[mask]) != value.shape or not np.all(np.asarray(row[mask]) == 1)
            ):
                raise ValueError(f"{mask} marks unavailable measurements")
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            episode, frame = self.locations[index]
            raise ValueError(
                f"Episode {episode}, frame {frame}: {key} unavailable: {exc}. "
                "Repair the data or explicitly select different episodes/input fields. "
                "No samples were silently filtered."
            ) from exc

    def _value(self, index, key):
        row = self.rows[index]
        if key.startswith("observation.images."):
            camera = key.removeprefix("observation.images.")
            episode = self.root / "episodes" / f"episode_{self.locations[index][0]:06d}"
            reference = row[key]
            if self._v3:
                rgb = self._v3.image(reference, camera, video_reader=self.video_cache.read_tensor)
            elif isinstance(reference, dict) and "frame_index" in reference:
                rgb = self.video_cache.read_tensor(
                    image_path(episode, camera, reference), reference["frame_index"]
                )
            else:
                rgb = image_rgb(episode, camera, reference)
            return image_tensor(rgb, self.image_size, return_uint8=self.return_uint8)
        if key == "observation.state":
            return torch.from_numpy(self.selection.frame(row))
        # Preserve integer/bool labels; policy preprocessing owns normalization.
        return torch.as_tensor(row[key], dtype=getattr(torch, self.info["features"][key]["dtype"]))

    def observation(self, index):
        """Return the current observation without temporal expansion."""
        row = self.sample_indices[index]
        return {key: self._value(row, key) for key in self.input_features}

    def _get_query_indices(self, index):
        """Query physical row offsets within an episode."""
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

        def field(key):
            if key not in indices:
                return self._value(index, key)
            # Decode only the requested image rows, once each even when padded.
            values = {i: self._value(i, key) for i in dict.fromkeys(indices[key])}
            return torch.stack([values[i] for i in indices[key]])

        videos = []
        for key in self.sample_keys:
            if self.camera_workers and key.startswith("observation.images."):
                ref = self.rows[index][key]
                # Parquet embedded-image row-group caches remain single-threaded.
                if (self._v3 and ref[1] is None) or (
                    not self._v3 and isinstance(ref, dict) and "frame_index" in ref
                ):
                    videos.append(key)
                    continue
            sample[key] = field(key)
        if len(videos) > 1 and self.camera_workers > 1:
            self.video_cache.prepare_process()
            with ThreadPoolExecutor(max_workers=min(self.camera_workers, len(videos))) as pool:
                sample.update(zip(videos, pool.map(field, videos), strict=True))
        else:
            sample.update((key, field(key)) for key in videos)
        return sample

    def action_metadata(self, index):
        """Numeric action targets/masks for loss counting; never decode images."""
        row = self.sample_indices[index]
        indices, padding = self._get_query_indices(row)
        if "action" not in indices:
            return {"action": self._value(row, "action")}
        return {
            "action": torch.stack([self._value(i, "action") for i in indices["action"]]),
            "action_is_pad": padding["action_is_pad"],
        }

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
                value = self._value(i, key)
                if key.startswith("observation.images.") and value.dtype == torch.uint8:
                    value = value.float() / 255
                value = value.double()
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
