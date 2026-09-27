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

from collections import deque
from typing import Any

import torch
from torch import Tensor

from .configuration_fastwam import FastWAMConfig
from .wan import (
    ActionDiT,
    FastWAM,
    MoT,
    WanVideoDiT,
    build_wan_tokenizer,
    load_pretrained_wan_text_encoder,
    load_pretrained_wan_vae,
)

OBS_STATE = "observation.state"


class FastWAMPolicy(torch.nn.Module):
    """FastWAM tensor policy with video supervision and action-only inference.

    Attention backend: FastWAM's DiT uses ``torch.nn.functional.scaled_dot_product_attention``
    (SDPA) for all attention. It does not use FlashAttention, because MoT routing requires
    arbitrary boolean ``[query, key]`` masks that the FlashAttention varlen API cannot express;
    installing ``flash-attn`` has no effect on the FastWAM path. (SDPA may still dispatch to
    PyTorch's own flash/mem-efficient/math kernel internally, unrelated to the ``flash-attn`` package.)

    Args:
        config (FastWAMConfig): FastWAM policy configuration.
        dataset_stats (dict[str, dict[str, Tensor]] | None): Optional training
            dataset statistics passed by the training/evaluation stack.
    """

    config_class = FastWAMConfig
    name = "fastwam"

    def __init__(
        self,
        config: FastWAMConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        config.validate_features()
        self.config = config
        self.dataset_stats = dataset_stats
        self.model = self._build_core_model(config)
        if config.freeze_video_expert and getattr(self.model, "video_expert", None) is not None:
            # Freeze the ~5B Wan video expert; get_optim_params filters on requires_grad,
            # so its params drop out of the optimizer (and DDP skips them).
            self.model.video_expert.requires_grad_(False)
            # The transformer blocks are re-parented onto the MoTLayers (single FSDP owner), so
            # `video_expert.requires_grad_` no longer reaches them — freeze them via the layers.
            mot = getattr(self.model, "mot", None)
            if mot is not None and getattr(mot, "layers", None) is not None:
                for layer in mot.layers:
                    if "video" in layer.blocks:
                        layer.blocks["video"].requires_grad_(False)
        self.reset()

    def get_optim_params(self) -> list[Tensor]:
        # Return the trainable tensors directly (a single param group). The optimizer
        # builder wraps these in a param group; returning a bare {"params": [...]} dict
        # instead would make `list(...)` yield the key string "params".
        params = (
            list(self.model.dit.parameters())
            if hasattr(self.model, "dit")
            else list(self.model.parameters())
        )
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            params.extend(list(proprio_encoder.parameters()))
        return [p for p in params if p.requires_grad]

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    def _batch_to_training_sample(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Assemble future video, frozen text context and current proprioception."""
        sample = dict(batch)
        if "video" not in sample:
            sample["video"] = _stack_video_from_images(batch, self.config, training=True)
        pads = [batch[k + "_is_pad"] for k in self.config.image_features if k + "_is_pad" in batch]
        if pads:
            sample["image_is_pad"] = torch.stack(pads).any(dim=0)
        if "context" not in sample or "context_mask" not in sample:
            prompt = _prompt_from_batch(batch=batch, config=self.config)
            if prompt is None:
                raise KeyError(
                    "FastWAM training requires a `task`/`prompt` to encode text context, "
                    "or precomputed `context`/`context_mask` in the batch."
                )
            sample["context"], sample["context_mask"] = self.model.encode_prompt(prompt)
        if self.config.proprio_dim is not None and "proprio" not in sample:
            state = sample.get(OBS_STATE)
            if state is not None:
                # The reader gives current state [B, D]; build_inputs expects
                # per-frame [B, T, D] and uses frame 0, so add a T=1 axis.
                sample["proprio"] = state.unsqueeze(1) if state.ndim == 2 else state
        return sample

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        """Return weighted video/action loss and separate logging metrics."""
        sample = self._batch_to_training_sample(batch)
        loss, metrics = self.model.training_loss(sample)
        return loss * batch.get("_loss_weight", 1.0), dict(metrics or {})

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **_: Any) -> Tensor:
        """Predict a chunk of actions from the current FastWAM observation.

        Args:
            batch (dict[str, Tensor]): Inference batch with `input_image` or
                image observation keys, plus `context/context_mask` or `prompt`.

        Returns:
            Tensor: Action chunk with shape `[B, action_horizon, action_dim]`.
        """

        self.eval()
        infer_kwargs = _batch_to_infer_kwargs(batch=batch, config=self.config)
        batch_size = _infer_kwargs_batch_size(infer_kwargs)
        if batch_size == 1:
            action = _action_from_model_output(self.model.infer_action(**infer_kwargs))
        else:
            action = torch.cat(
                [
                    _action_from_model_output(
                        self.model.infer_action(
                            **_slice_infer_kwargs(infer_kwargs, index=i, batch_size=batch_size)
                        )
                    )
                    for i in range(batch_size)
                ],
                dim=0,
            )
        return action.to(device=batch_device(batch), dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def _build_core_model(self, config: FastWAMConfig) -> FastWAM:
        """Build trainable experts; load frozen Wan VAE and text assets locally.

        The algorithm adapter loads trainable weights from model.safetensors.
        Frozen assets are excluded from checkpoints and must remain available.
        """
        dtype = _dtype_from_name(config.torch_dtype)
        device = config.device
        video_expert = WanVideoDiT(**config.video_dit_config).to(device=device, dtype=dtype)
        action_expert = ActionDiT(**config.action_dit_config).to(device=device, dtype=dtype)
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
        )
        text_encoder = (
            load_pretrained_wan_text_encoder(
                model_id=config.text_encoder_model_id, torch_dtype=dtype, device=device
            )
            if config.load_text_encoder
            else None
        )
        return FastWAM(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=load_pretrained_wan_vae(
                model_id=config.vae_model_id, torch_dtype=dtype, device=device
            ),
            text_encoder=text_encoder,
            tokenizer=build_wan_tokenizer(
                model_id=config.tokenizer_model_id, tokenizer_max_len=config.tokenizer_max_len
            ),
            text_dim=int(config.video_dit_config["text_dim"]),
            proprio_dim=config.proprio_dim,
            device=device,
            torch_dtype=dtype,
            video_train_shift=float(config.video_scheduler["train_shift"]),
            video_infer_shift=float(config.video_scheduler["infer_shift"]),
            video_num_train_timesteps=int(config.video_scheduler["num_train_timesteps"]),
            action_train_shift=float(config.action_scheduler["train_shift"]),
            action_infer_shift=float(config.action_scheduler["infer_shift"]),
            action_num_train_timesteps=int(config.action_scheduler["num_train_timesteps"]),
            loss_lambda_video=float(config.loss["lambda_video"]),
            loss_lambda_action=float(config.loss["lambda_action"]),
        )


def _scalar(value: Any) -> Any:
    """Unwrap a 0-/1-element tensor (e.g. from DataLoader collation) to a Python scalar."""
    return value.item() if isinstance(value, Tensor) else value


def _batch_to_infer_kwargs(batch: dict[str, Tensor], config: FastWAMConfig) -> dict[str, Any]:
    return {
        "prompt": _prompt_from_batch(batch=batch, config=config),
        "input_image": _input_image_from_batch(batch, config),
        "action_horizon": config.action_horizon,
        "proprio": batch.get("proprio", batch.get(OBS_STATE)),
        "context": batch.get("context"),
        "context_mask": batch.get("context_mask"),
        "negative_prompt": batch.get("negative_prompt", config.negative_prompt),
        "text_cfg_scale": float(_scalar(batch.get("text_cfg_scale", config.text_cfg_scale))),
        "num_inference_steps": int(
            _scalar(batch.get("num_inference_steps", config.num_inference_steps))
        ),
        "sigma_shift": batch.get("sigma_shift", config.sigma_shift),
        "seed": batch.get("seed", config.inference_seed),
        "rand_device": batch.get("rand_device", config.rand_device),
        "tiled": bool(batch.get("tiled", config.tiled)),
    }


def _prompt_from_batch(batch: dict[str, Tensor], config: FastWAMConfig) -> Any:
    prompt = batch.get("prompt")
    if prompt is not None:
        return prompt

    task = batch.get("task")
    if task is None:
        return None
    if isinstance(task, str):
        return config.prompt_template.format(task=task)
    if isinstance(task, (list, tuple)):
        return [config.prompt_template.format(task=str(item)) for item in task]
    return config.prompt_template.format(task=str(task))


def _action_from_model_output(output: Any) -> Tensor:
    action = output["action"] if isinstance(output, dict) else output
    if action.ndim == 2:
        action = action.unsqueeze(0)
    return action


def _infer_kwargs_batch_size(infer_kwargs: dict[str, Any]) -> int:
    image = infer_kwargs["input_image"]
    if not isinstance(image, Tensor):
        raise TypeError(f"`input_image` must be a tensor, got {type(image).__name__}.")
    if image.ndim == 3:
        return 1
    if image.ndim == 4:
        return int(image.shape[0])
    raise ValueError(f"`input_image` must be [B,C,H,W] or [C,H,W], got {tuple(image.shape)}.")


def _slice_infer_kwargs(
    infer_kwargs: dict[str, Any], *, index: int, batch_size: int
) -> dict[str, Any]:
    return {
        key: _slice_infer_value(value, index=index, batch_size=batch_size)
        for key, value in infer_kwargs.items()
    }


def _slice_infer_value(value: Any, *, index: int, batch_size: int) -> Any:
    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        return value[index : index + 1]
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[index]
    return value


def _dtype_from_name(name: str) -> torch.dtype:
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype `{name}`.")
    return dtype_map[name]


def batch_device(batch: dict[str, Any]) -> torch.device:
    for value in batch.values():
        if isinstance(value, Tensor):
            return value.device
    return torch.device("cpu")


def _resize_frames(frames: Tensor, size: tuple[int, int]) -> Tensor:
    """Resize a frame tensor to `size` (H, W), tolerating a leading temporal/batch stack.

    `interpolate` only accepts a single leading batch dim (`[N, C, H, W]`), but FastWAM camera
    tensors arrive as `[B, C, H, W]` (live eval) or `[B, T, C, H, W]` (temporal stack), so flatten
    any leading dims into the batch, resize, then restore. A no-op when already at `size`.
    """
    if tuple(frames.shape[-2:]) == size:
        return frames
    lead = frames.shape[:-3]
    flat = frames.reshape(-1, *frames.shape[-3:])
    flat = torch.nn.functional.interpolate(
        flat, size=size, mode="bilinear", align_corners=False, antialias=True
    )
    return flat.reshape(*lead, *flat.shape[-3:])


def _stack_video_from_images(
    batch: dict[str, Tensor], config: FastWAMConfig, *, training=False
) -> Tensor:
    # Exclude the `*_is_pad` companion tensors that delta-timestamp loading adds alongside
    # each camera (shape [B, T]); they share the `observation.images.` prefix but are not frames.
    image_keys = sorted(
        k for k in batch if k.startswith("observation.images.") and not k.endswith("_is_pad")
    )
    if not image_keys:
        raise KeyError("FastWAM batch must contain `video` or `observation.images.*` keys.")
    if training and any(
        batch[key].ndim != 5 or batch[key].shape[1] != config.model_video_frames
        for key in image_keys
    ):
        raise ValueError("FastWAM training requires real future camera windows")
    per_cam = (int(config.image_size[0]), int(config.image_size[1]) // len(image_keys))
    images = [_resize_frames(batch[key], per_cam) for key in image_keys]
    # Cameras concatenate along width (last dim) in both the single-frame and temporal case.
    image = torch.cat(images, dim=-1) if len(images) > 1 else images[0]
    if image.ndim == 4:
        image = image.unsqueeze(2)
    elif image.ndim == 5:
        # [B, T, C, H, W]: temporal stack from delta-timestamp loading -> [B, C, T, H, W].
        image = image.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(
            f"Expected image batch [B,C,H,W] or temporal [B,T,C,H,W], got {tuple(image.shape)}."
        )
    return image


def _input_image_from_batch(batch: dict[str, Tensor], config: FastWAMConfig) -> Tensor:
    if "input_image" in batch:
        return _prepare_infer_image(batch["input_image"], config)
    video = batch.get("video")
    if video is None:
        video = _stack_video_from_images(batch, config)
    if video.ndim == 5:
        return _prepare_infer_image(video[:, :, 0], config)
    if video.ndim == 4:
        return _prepare_infer_image(video, config)
    raise ValueError(f"Cannot build input image from tensor with shape {tuple(video.shape)}.")


def _prepare_infer_image(image: Tensor, config: FastWAMConfig) -> Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor [B,C,H,W] or [C,H,W], got {tuple(image.shape)}.")

    # Resize to the full configured resolution (no-op when the video path already produced it, but
    # also covers a directly-supplied `input_image`). The model owns its input resolution — see
    # `_stack_video_from_images` — so we resize rather than assert on a mismatch.
    target_h, target_w = int(config.image_size[0]), int(config.image_size[1])
    return _resize_frames(image, (target_h, target_w))
