"""Framework-independent feature descriptions for local policies."""

import math
from dataclasses import dataclass, field
from enum import Enum


class NormalizationMode(str, Enum):
    IDENTITY = "IDENTITY"
    MEAN_STD = "MEAN_STD"


@dataclass
class PolicyFeature:
    type: str
    shape: tuple[int, ...]

    def __post_init__(self):
        self.shape = tuple(self.shape)
        if self.type not in {"VISUAL", "STATE", "ENV", "ACTION"}:
            raise ValueError(f"Unknown feature type: {self.type}")
        if not self.shape or any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError("Feature dimensions must be positive integers")


@dataclass
class PolicyConfig:
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    def __post_init__(self):
        for features in (self.input_features, self.output_features):
            for key, value in features.items():
                if isinstance(value, dict):
                    features[key] = PolicyFeature(**value)
        self.normalization_mapping = {
            key: NormalizationMode(mode) for key, mode in self.normalization_mapping.items()
        }
        if any(type(n) is not int or n < 1 for n in (self.chunk_size, self.n_action_steps)):
            raise ValueError("Chunk and execution lengths must be positive")
        if self.dim_model % self.n_heads or self.dim_model % 4:
            raise ValueError("dim_model must be divisible by n_heads and 4")
        if self.temporal_ensemble_coeff is not None and not math.isfinite(
            self.temporal_ensemble_coeff
        ):
            raise ValueError("Temporal ensemble coefficient must be finite")
        if getattr(self, "allow_partial_pretrained_load", False) or getattr(
            self, "use_dataset_input_features", False
        ):
            raise ValueError("Native checkpoints require an exact feature/weight match")
        if any(
            not math.isfinite(getattr(self, name)) or getattr(self, name) < 0
            for name in (
                "optimizer_lr",
                "optimizer_lr_backbone",
                "optimizer_weight_decay",
                "kl_weight",
            )
        ):
            raise ValueError("Optimizer and loss weights must be finite and nonnegative")
        action = self.output_features.get("action")
        if action is None or action.type != "ACTION" or len(action.shape) != 1:
            raise ValueError("Policy requires a one-dimensional action feature")
        if self.robot_state_feature is not None and (
            self.robot_state_feature.type != "STATE" or len(self.robot_state_feature.shape) != 1
        ):
            raise ValueError("observation.state must be a one-dimensional STATE feature")
        for feature in self.image_features.values():
            if len(feature.shape) != 3 or feature.shape[0] != 3:
                raise ValueError("Images must use CHW RGB features")
        if hasattr(self, "inference_action_scale"):
            if not math.isfinite(self.inference_action_scale):
                raise ValueError("Inference scale must be finite")
            indices = [
                *self.inference_action_scale_dims,
                *(i for group in self.action_loss_groups.values() for i in group),
            ]
            if any(type(i) is not int or not 0 <= i < action.shape[0] for i in indices):
                raise ValueError("Action scaling/loss-group indices are outside the action vector")

    @property
    def image_features(self):
        return {k: v for k, v in self.input_features.items() if v.type == "VISUAL"}

    @property
    def robot_state_feature(self):
        return self.input_features.get("observation.state")

    @property
    def env_state_feature(self):
        return self.input_features.get("observation.environment_state")

    @property
    def action_feature(self):
        return self.output_features["action"]
