"""OpenPI transforms with an explicit AlohaMini coordinate contract.

Raw 18-D coordinates are retained. Unlike the Trossen ALOHA example, this
adapter does not apply another robot's joint signs or gripper geometry.
"""

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from alohamini.datasets.native import state_names
from alohamini.datasets.statistics import ExactQuantileStats, diagnose_statistics

from .image_tools import resize_with_pad

ARM_DELTA_DIMS = (*range(6), *range(7, 13))
CAMERAS = {
    "base_0_rgb": "forward",
    "left_wrist_0_rgb": "wrist_left",
    "right_wrist_0_rgb": "wrist_right",
}


def relative_actions(state, actions, delta_dims=ARM_DELTA_DIMS):
    """All steps relative to the current state, not adjacent-action differences."""
    actions = np.array(actions, dtype=np.float64, copy=True)
    actions[..., list(delta_dims)] -= np.asarray(state)[..., None, list(delta_dims)]
    return actions


def fit_statistics(samples, *, delta_dims=ARM_DELTA_DIMS, report_path=None):
    """Fit training windows after delta conversion; never average episode quantiles.

    Repeated end-of-sequence targets are retained, matching the author's window
    sampling. This reads numeric rows only, without decoding images. Float64
    centered moments replace cancellation-prone squared-moment accumulation.
    """
    names = state_names(samples.info["robot_metadata"]["robot_model"])
    if samples.selection is None or samples.selection.feature["names"] != names:
        raise ValueError("PI0.5 AlohaMini recipe requires the original 18-D position state")
    if samples.info["features"]["action"]["names"] != names:
        raise ValueError("State and action coordinates must match")
    if "action" not in samples.delta_indices:
        raise ValueError("Select an action window before fitting PI0.5 statistics")
    trackers = {key: ExactQuantileStats() for key in ("state", "actions")}
    for anchor in samples.sample_indices:
        indices, _ = samples._get_query_indices(anchor)
        state = samples.selection.frame(samples.rows[anchor]).astype(np.float64)
        actions = np.asarray([samples.rows[i]["action"] for i in indices["action"]], np.float64)
        trackers["state"].update(state[None])
        trackers["actions"].update(relative_actions(state, actions, delta_dims))
    stats = {key: tracker.get_statistics() for key, tracker in trackers.items()}
    if report_path is not None:
        import json
        from pathlib import Path

        report = dict(
            method="float64_centered_exact_linear_v2",
            source_sha256=samples.table_sha256,
            state_feature=samples.selection.feature,
            state_units=samples.selection.units,
            delta_dims=list(delta_dims),
            windows=samples.delta_indices,
            fields={key: diagnose_statistics(values, names=names) for key, values in stats.items()},
        )
        with Path(report_path).expanduser().open("x") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    for key, values in stats.items():
        narrow = np.flatnonzero((values["q99"] - values["q01"]) < 1e-6).tolist()
        if narrow:
            logging.getLogger(__name__).warning(
                "%s has narrow q01/q99 intervals at dimensions %s; inspect physical ranges. "
                "Coordinates are not zeroed, clipped or silently rescaled.",
                key,
                narrow,
            )
    return {key: {name: value.tolist() for name, value in v.items()} for key, v in stats.items()}


@dataclass
class PI05Processor:
    """Shared offline/online transforms; tokenizer reads a local SentencePiece asset.

    Quantiles use OpenPI's additive 1e-6 epsilon, without clipping or scale floors.
    Subtractions are float64; model inputs become float32 only afterwards.
    State is tokenized before zero-padding, as in ModelTransformFactory.
    """

    stats: dict
    tokenizer: object
    action_dim: int = 32
    delta_dims: tuple = ARM_DELTA_DIMS
    span_floors: dict | None = None

    def __post_init__(self):
        self.stats = {
            key: {name: np.asarray(values[name], np.float64) for name in ("q01", "q99")}
            for key, values in self.stats.items()
        }
        if set(self.stats) != {"state", "actions"}:
            raise ValueError("PI0.5 requires separate state and actions statistics")
        for values in self.stats.values():
            if any(v.shape != (18,) or not np.isfinite(v).all() for v in values.values()):
                raise ValueError("AlohaMini PI0.5 statistics must contain 18 finite coordinates")
            if (values["q99"] < values["q01"]).any():
                raise ValueError("q99 cannot be smaller than q01")
        self.span_floors = self.span_floors or {}
        if set(self.span_floors) - self.stats.keys():
            raise ValueError("Unknown span_floors field")
        self.spans = {}
        for key, values in self.stats.items():
            floor = np.asarray(self.span_floors.get(key, np.zeros(18)), np.float64)
            if floor.shape != (18,) or not np.isfinite(floor).all() or (floor < 0).any():
                raise ValueError("span_floors require 18 finite nonnegative physical values")
            self.spans[key] = np.maximum(values["q99"] - values["q01"], floor) + 1e-6
        self.delta_dims = tuple(self.delta_dims)
        if (
            self.action_dim < 18
            or len(set(self.delta_dims)) != len(self.delta_dims)
            or any(type(i) is not int or not 0 <= i < 18 for i in self.delta_dims)
        ):
            raise ValueError("Invalid action padding or delta dimensions")

    def normalize(self, key, value):
        s = self.stats[key]
        value = np.asarray(value, np.float64)
        if value.shape[-1] != 18 or not np.isfinite(value).all():
            raise ValueError(f"{key}: expected finite 18-D coordinates")
        return ((value - s["q01"]) / self.spans[key] * 2 - 1).astype(np.float32)

    def restore_actions(self, actions, state):
        """Differentiable inverse, followed by restoration of physical targets."""
        if actions.shape[-1] != self.action_dim:
            raise ValueError("Model action width does not match processor")
        s = self.stats["actions"]
        low = torch.as_tensor(s["q01"], device=actions.device, dtype=torch.float64)
        span = torch.as_tensor(self.spans["actions"], device=actions.device, dtype=torch.float64)
        values = (actions[..., :18].double() + 1) / 2 * span + low
        state = torch.as_tensor(state, device=actions.device, dtype=torch.float64)
        offset = torch.zeros_like(state)
        offset[..., list(self.delta_dims)] = state[..., list(self.delta_dims)]
        return (values + offset.unsqueeze(-2)).float()

    def prepare(self, state, images, prompt, actions=None, *, device="cpu"):
        """One physical sample: HWC uint8 RGB, original state and target chunk."""
        state = np.asarray(state, np.float64)
        if state.shape != (18,) or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Expected 18-D state and a nonempty task prompt")
        if "forward" not in images or set(images) - set(CAMERAS.values()):
            raise ValueError("Select forward and optional wrist_left/wrist_right cameras")
        tokens, mask = self.tokenizer.tokenize(prompt, self.normalize("state", state))
        rgb, image_masks = {}, {}
        for target, source in CAMERAS.items():
            if source in images:
                image = np.asarray(images[source])
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(f"{source}: expected HWC uint8 RGB")
                image = resize_with_pad(image, 224, 224)
                rgb[target] = (
                    torch.as_tensor(image.copy(), device=device).permute(2, 0, 1).float()[None]
                    / 127.5
                    - 1
                )
            else:
                rgb[target] = torch.full((1, 3, 224, 224), -1.0, device=device)
            image_masks[target] = torch.tensor([source in images], device=device)
        padded_state = np.pad(self.normalize("state", state), (0, self.action_dim - 18))
        observation = SimpleNamespace(
            images=rgb,
            image_masks=image_masks,
            state=torch.as_tensor(padded_state[None], device=device),
            tokenized_prompt=torch.as_tensor(tokens[None], device=device, dtype=torch.long),
            tokenized_prompt_mask=torch.as_tensor(mask[None], device=device, dtype=torch.bool),
            token_ar_mask=None,
            token_loss_mask=None,
        )
        if actions is None:
            return observation, None
        actions = np.asarray(actions)
        if actions.ndim != 2:
            raise ValueError("Expected an action chunk shaped [time, 18]")
        targets = self.normalize("actions", relative_actions(state, actions, self.delta_dims))
        targets = np.pad(targets, ((0, 0), (0, self.action_dim - 18)))
        return observation, torch.as_tensor(targets[None], device=device)


def sample_inputs(samples, index):
    """Read an AlohaMini/LeRobot v3 training row without the ACT image-resize recipe."""
    from alohamini.datasets.images import image_rgb

    anchor = samples.sample_indices[index]
    row = samples.rows[anchor]
    episode = samples.root / "episodes" / f"episode_{samples.locations[anchor][0]:06d}"
    images = {}
    for camera in samples.cameras:
        value = row[f"observation.images.{camera}"]
        images[camera] = (
            samples._v3.image(value, camera) if samples._v3 else image_rgb(episode, camera, value)
        )
    indices, _ = samples._get_query_indices(anchor)
    return {
        "state": samples.selection.frame(row),
        "images": images,
        "prompt": row.get("task") or samples.info.get("task", ""),
        "actions": np.asarray([samples.rows[i]["action"] for i in indices["action"]]),
    }
