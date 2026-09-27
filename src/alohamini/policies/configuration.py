"""Framework-independent feature descriptions for local policies."""

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
        if hasattr(self, "normalization_mapping"):
            self.normalization_mapping = {
                key: NormalizationMode(mode) for key, mode in self.normalization_mapping.items()
            }
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
