"""Shared tensor transforms for training and inference; no robot execution logic."""

import numpy as np
import torch
import torch.nn.functional as F

from alohamini.policies.configuration import NormalizationMode

DEFAULT_IMAGE_SIZE = (480, 640)  # Height, width; original AlohaMini ACT resolution.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def validate_statistics(stats, expected, names, *, manifest=None):
    if set(stats) != {*expected, "action"}:
        raise ValueError("Checkpoint is missing normalization statistics")
    for key, values in stats.items():
        shape = (
            (3, 1, 1)
            if key.startswith("observation.images.")
            else ((len(names),) if key == "action" else expected[key][1])
        )
        if any(np.asarray(values[k]).shape != shape for k in ("mean", "std")):
            raise ValueError(f"Checkpoint normalization shape mismatch: {key}")


def act_statistics(samples):
    """Fit numeric training fields and use fixed RGB statistics for ACT/AM-ACT.

    Checkpoint loading must use saved statistics instead of calling this recipe.
    Dataset.statistics() remains an empirical view for other policies/analysis.
    """
    image_keys = {
        key for key, feature in samples.input_features.items() if feature.type == "VISUAL"
    }
    stats = samples.statistics(keys=[key for key in samples.sample_keys if key not in image_keys])
    for key in image_keys:
        stats[key] = {
            "mean": [[[value]] for value in IMAGENET_MEAN],
            "std": [[[value]] for value in IMAGENET_STD],
        }
    return stats


def image_tensor(rgb, size=DEFAULT_IMAGE_SIZE):
    """Convert decoded HWC uint8 RGB to CHW float32 in [0, 1]."""
    tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float() / 255
    if tuple(tensor.shape[1:]) != tuple(size):
        tensor = F.interpolate(
            tensor[None], size=size, mode="bilinear", align_corners=False, antialias=True
        )[0]
    return tensor


class Processor:
    """Normalize declared fields, preserving masks, labels and autograd.

    Modes are resolved per field from the policy configuration. Unlisted fields
    pass through unchanged apart from device placement. With no explicit modes,
    only fields present in stats use MEAN_STD. Statistics remain fixed; neither
    validation nor inference fits new statistics.
    """

    def __init__(self, stats, device="cpu", *, modes=None):
        self.device = torch.device(device)
        self.modes = {
            key: NormalizationMode(mode)
            for key, mode in (
                modes if modes is not None else dict.fromkeys(stats, "MEAN_STD")
            ).items()
        }
        self.stats = {}
        for key, mode in self.modes.items():
            if mode == NormalizationMode.IDENTITY:
                continue
            if key not in stats or not {"mean", "std"} <= stats[key].keys():
                raise ValueError(f"{key}: MEAN_STD requires mean and std statistics")
            values = {
                name: torch.tensor(stats[key][name], dtype=torch.float32, device=self.device)
                for name in ("mean", "std")
            }
            if (
                values["mean"].shape != values["std"].shape
                or not torch.isfinite(values["mean"]).all()
                or not torch.isfinite(values["std"]).all()
                or (values["std"] < 0).any()
            ):
                raise ValueError(f"{key}: Invalid normalization statistics")
            self.stats[key] = values

    @classmethod
    def from_config(cls, config, stats, device="cpu"):
        features = {**config.input_features, **config.output_features}
        modes = {
            key: config.normalization_mapping.get(feature.type, "IDENTITY")
            for key, feature in features.items()
        }
        return cls(stats, device, modes=modes)

    def __call__(self, batch):
        result = {}
        for key, value in batch.items():
            value = value.to(self.device)
            if not key.endswith("_is_pad") and key in self.stats:
                stats = self.stats[key]
                value = (value - stats["mean"]) / (stats["std"] + 1e-8)
            result[key] = value
        return result

    def unnormalize(self, key, tensor):
        """Restore recorded units without detaching, clipping or execution scaling."""
        if key not in self.modes:
            raise KeyError(f"No normalization mode configured for {key}")
        if self.modes[key] == NormalizationMode.IDENTITY:
            return tensor
        stats = self.stats[key]
        return tensor * stats["std"] + stats["mean"]

    def action(self, tensor, *, context=None):
        return self.unnormalize("action", tensor)


def scale_action(action, dims=(), scale=1.0):
    """Scale selected physical outputs after unnormalization for execution."""
    if not dims or scale == 1.0:
        return action
    action = action.clone()
    action[..., list(dims)] *= scale
    return action
