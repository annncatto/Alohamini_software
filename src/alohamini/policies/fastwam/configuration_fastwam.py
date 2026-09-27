# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alohamini.policies.configuration import NormalizationMode, PolicyConfig

ACTION = "action"
OBS_STATE = "observation.state"

WAN22_DIFFUSERS_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_T5_TOKENIZER_ID = "google/umt5-xxl"


def default_video_dit_config(action_dim: int) -> dict[str, Any]:
    return {
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1.0e-6,
        "seperated_timestep": True,
        "use_gradient_checkpointing": False,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": action_dim,
        "action_group_causal_mask_mode": "group_diagonal",
        "fp32_attention": True,
    }


def default_action_dit_config(action_dim: int) -> dict[str, Any]:
    return {
        "action_dim": action_dim,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1.0e-6,
        "use_gradient_checkpointing": False,
        "fp32_attention": True,
    }


@dataclass
class FastWAMConfig(PolicyConfig):
    """Configuration for the FastWAM video/action policy.

    Args:
        action_dim (int): Number of scalar action channels per timestep.
        proprio_dim (int | None): Number of proprioception channels used as an
            extra text-context token; inferred from the selected state when omitted.
        action_horizon (int): Number of actions predicted by one policy call.
        num_video_frames (int): Raw video sampling window (in dataset frames). The
            model actually operates on `model_video_frames` frames after subsampling
            by `action_video_freq_ratio`.
        action_video_freq_ratio (int): Actions are sampled at this multiple of the
            video frame rate. Video frames are taken every `action_video_freq_ratio`-th
            raw frame, so the model sees `(num_video_frames - 1) // ratio + 1` frames
            spanning the same time window as `action_horizon` actions (ratio actions
            per video frame).
        image_size (tuple[int, int]): Concatenated image size as `(height, width)`.
        context_len (int): Maximum text embedding token length.
        video_dit_config (dict[str, Any] | None): Wan video expert config.
        action_dit_config (dict[str, Any] | None): Action expert config.
        use_gradient_checkpointing (bool): Enable activation checkpointing in both DiT
            experts (trades compute for memory; propagated into the DiT configs).
        freeze_video_expert (bool): Freeze the ~5B Wan video expert
            (`model.video_expert`) so only the action expert + proprio encoder train.
            Cuts the AdamW optimizer footprint substantially; the video expert keeps its
            pretrained weights. (If enabled, also set `loss.lambda_video=0` to skip the
            now-gradient-free video loss compute.)
    """

    n_obs_steps: int = 1
    action_dim: int | None = None
    proprio_dim: int | None = None
    action_horizon: int = 32
    n_action_steps: int = 32
    num_video_frames: int = 33
    action_video_freq_ratio: int = 4
    image_size: tuple[int, int] = (224, 448)
    context_len: int = 128
    device: str = "cpu"
    vae_model_id: str = WAN22_DIFFUSERS_MODEL_ID
    tokenizer_model_id: str = WAN_T5_TOKENIZER_ID
    text_encoder_model_id: str = WAN22_DIFFUSERS_MODEL_ID
    tokenizer_max_len: int = 128
    load_text_encoder: bool = True
    mot_checkpoint_mixed_attn: bool = False
    torch_dtype: str = "bfloat16"
    prompt_template: str = (
        "A video recorded from a robot's point of view executing the following instruction: {task}"
    )
    num_inference_steps: int = 10
    inference_seed: int | None = 42
    rand_device: str = "cpu"
    text_cfg_scale: float = 1.0
    negative_prompt: str = ""
    sigma_shift: float | None = None
    tiled: bool = False
    fp32_attention: bool = True
    use_gradient_checkpointing: bool = False
    freeze_video_expert: bool = False
    toggle_action_dimensions: list[int] = field(default_factory=list)
    video_scheduler: dict[str, float | int] = field(
        default_factory=lambda: {
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": 1000,
        }
    )
    action_scheduler: dict[str, float | int] = field(
        default_factory=lambda: {
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": 1000,
        }
    )
    loss: dict[str, float] = field(
        default_factory=lambda: {"lambda_video": 1.0, "lambda_action": 1.0}
    )
    video_dit_config: dict[str, Any] | None = None
    action_dit_config: dict[str, Any] | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    optimizer_lr: float = 1.0e-4
    optimizer_weight_decay: float = 1.0e-2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_dim is None:
            self.action_dim = self.action_feature.shape[0]
        if self.proprio_dim is None and self.robot_state_feature is not None:
            self.proprio_dim = self.robot_state_feature.shape[0]
        self.image_size = tuple(self.image_size)
        self.toggle_action_dimensions = [int(dim) for dim in self.toggle_action_dimensions]
        self.video_dit_config = self.video_dit_config or default_video_dit_config(self.action_dim)
        self.action_dit_config = self.action_dit_config or default_action_dit_config(
            self.action_dim
        )
        self.video_dit_config["fp32_attention"] = bool(self.fp32_attention)
        self.action_dit_config["fp32_attention"] = bool(self.fp32_attention)
        self.video_dit_config["use_gradient_checkpointing"] = bool(self.use_gradient_checkpointing)
        self.action_dit_config["use_gradient_checkpointing"] = bool(self.use_gradient_checkpointing)
        if self.context_len != self.tokenizer_max_len:
            raise ValueError("context_len must match tokenizer_max_len")
        self.validate_features()

    def validate_features(self) -> None:
        if self.n_obs_steps != 1 or self.num_video_frames <= 1 or self.n_action_steps < 1:
            raise ValueError("FastWAM needs one current observation and a nonempty future video")
        if self.toggle_action_dimensions:
            raise ValueError(
                "AlohaMini uses recorded Host units; LIBERO gripper toggles are not applicable"
            )
        if self.text_cfg_scale != 1 or self.negative_prompt or self.tiled:
            raise ValueError(
                "This FastWAM action path supports CFG=1, no negative prompt, untiled VAE"
            )
        if self.action_horizon != self.num_video_frames - 1:
            raise ValueError("Video and action windows must span the same recorded interval")
        if self.action_dim <= 0:
            raise ValueError(f"`action_dim` must be positive, got {self.action_dim}.")
        if self.action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {self.action_horizon}.")
        if self.n_action_steps > self.action_horizon:
            raise ValueError("`n_action_steps` cannot exceed `action_horizon`.")
        if self.action_video_freq_ratio <= 0:
            raise ValueError(
                f"`action_video_freq_ratio` must be positive, got {self.action_video_freq_ratio}."
            )
        # Video frames are subsampled by action_video_freq_ratio; the resulting model frame
        # count must satisfy T % 4 == 1 for the VAE temporal tokenization (mirrors the
        # original FastWAM dataset asserts).
        if (self.num_video_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                f"`num_video_frames - 1` ({self.num_video_frames - 1}) must be divisible by "
                f"`action_video_freq_ratio` ({self.action_video_freq_ratio})."
            )
        if ((self.num_video_frames - 1) // self.action_video_freq_ratio) % 4 != 0:
            raise ValueError(
                f"Subsampled video transitions ({(self.num_video_frames - 1) // self.action_video_freq_ratio}) "
                "must be divisible by 4 for VAE tokenization (i.e. model_video_frames % 4 == 1)."
            )
        if self.action_horizon % ((self.num_video_frames - 1) // self.action_video_freq_ratio) != 0:
            raise ValueError(
                f"`action_horizon` ({self.action_horizon}) must be divisible by the number of "
                f"video transitions ({(self.num_video_frames - 1) // self.action_video_freq_ratio})."
            )
        if not self.image_features:
            raise ValueError("FastWAM requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("FastWAM requires `action` in output_features.")
        action_shape = tuple(self.action_feature.shape)
        if action_shape != (self.action_dim,):
            raise ValueError(
                f"FastWAM action feature shape must be ({self.action_dim},), got {action_shape}."
            )
        if self.proprio_dim is not None:
            state_feature = self.robot_state_feature
            if state_feature is None:
                raise ValueError("FastWAM requires `observation.state` when `proprio_dim` is set.")
            state_shape = tuple(state_feature.shape)
            if state_shape != (self.proprio_dim,):
                raise ValueError(
                    f"FastWAM state feature shape must be ({self.proprio_dim},), got {state_shape}."
                )
        height, width = self.image_size
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError("FastWAM composite image dimensions must be multiples of 32")
        if width % len(self.image_features):
            raise ValueError("Composite width must divide evenly between cameras")

    @property
    def model_video_frames(self) -> int:
        """Number of video frames the model actually operates on, after subsampling the
        raw `num_video_frames` window by `action_video_freq_ratio` (e.g. 33 -> 9)."""
        return (self.num_video_frames - 1) // self.action_video_freq_ratio + 1

    @property
    def observation_delta_indices(self) -> list[int]:
        # Load the video frames the model is supervised on: the future window subsampled by
        # action_video_freq_ratio (e.g. [0, 4, 8, ..., 32] -> 9 frames). Each video frame is
        # thus `action_video_freq_ratio` actions apart, while actions load at the full rate
        # (`action_delta_indices` = range(action_horizon)). Returning None would load only the
        # current frame, making the video target a static repeat (degenerate supervision).
        return list(range(0, self.num_video_frames, self.action_video_freq_ratio))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
