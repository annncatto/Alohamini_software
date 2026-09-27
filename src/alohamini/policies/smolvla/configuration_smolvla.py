# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from alohamini.policies.configuration import NormalizationMode, PolicyFeature


@dataclass
class SmolVLAConfig:
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    device: str = "cpu"

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = (
        False  # Whether to use special image tokens around image features.
    )

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = (
        4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    )
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: dict | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    def __post_init__(self):
        for features in (self.input_features, self.output_features):
            for key, value in features.items():
                if isinstance(value, dict):
                    features[key] = PolicyFeature(**value)
        self.normalization_mapping = {
            key: NormalizationMode(value) for key, value in self.normalization_mapping.items()
        }
        if any(
            type(n) is not int or n < 1
            for n in (
                self.chunk_size,
                self.n_action_steps,
                self.num_steps,
                self.max_state_dim,
                self.max_action_dim,
                self.tokenizer_max_length,
            )
        ):
            raise ValueError("Chunk, decoding and feature dimensions must be positive integers")
        if self.n_obs_steps != 1 or not self.use_cache:
            raise ValueError("SmolVLA requires one observation and prefix KV caching")
        if self.rtc_config is not None:
            raise ValueError("RTC is not supported by the SmolVLA executor")
        if self.adapt_to_pi_aloha:
            raise ValueError("Trossen ALOHA transforms do not describe AlohaMini coordinates")
        if self.empty_cameras != 0:
            raise ValueError("SmolVLA uses explicitly recorded cameras only")

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        if self.robot_state_feature is None or self.robot_state_feature.type != "STATE":
            raise ValueError("SmolVLA requires observation.state")
        for feature, limit in (
            (self.robot_state_feature, self.max_state_dim),
            (self.action_feature, self.max_action_dim),
        ):
            if len(feature.shape) != 1 or feature.shape[0] > limit:
                raise ValueError("SmolVLA state/action feature exceeds its padded dimension")
        if not self.image_features:
            raise ValueError("SmolVLA requires at least one RGB camera")
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type="VISUAL",
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def build_scheduler(self, optimizer, num_training_steps):
        """LeRobot CosineDecayWithWarmupSchedulerConfig.build without its registry."""
        from alohamini.learning.optim import make_scheduler, resolve_optimization

        return make_scheduler(resolve_optimization({"steps": num_training_steps}, self), optimizer)

    @property
    def image_features(self):
        return {key: value for key, value in self.input_features.items() if value.type == "VISUAL"}

    @property
    def robot_state_feature(self):
        return self.input_features.get("observation.state")

    @property
    def action_feature(self):
        return self.output_features["action"]
