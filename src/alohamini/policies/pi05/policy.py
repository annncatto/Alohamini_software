"""Tensor-level PI0.5 training/inference API; never connects to a robot."""

from collections import deque
from pathlib import Path

import torch

from .configuration_pi05 import PI05Config
from .modeling_pi05 import PI0Pytorch


class PI05Policy:
    """Physical inputs/outputs around the author's flow-matching network.

    ``model`` exposes the original unreduced flow loss for custom trainers.
    ``loss`` follows the author's mean reduction, including padded coordinates
    and repeated tail targets. Dataset statistics and tokenizer are explicit:
    base checkpoint normalization from another robot must not be reused.
    """

    def __init__(self, model, processor, *, n_action_steps=None, num_inference_steps=10):
        self.model, self.processor = model, processor
        self.config = model.config
        self.n_action_steps = n_action_steps or self.config.action_horizon
        self.num_inference_steps = num_inference_steps
        if (
            not 1 <= self.n_action_steps <= self.config.action_horizon
            or type(num_inference_steps) is not int
            or num_inference_steps < 1
            or processor.action_dim != self.config.action_dim
        ):
            raise ValueError("Invalid execution horizon, denoising steps or processor width")
        self.reset()

    @classmethod
    def from_pretrained(cls, path, processor, *, config=None, device="cpu", **kwargs):
        """Load official converted PyTorch weights; Orbax requires conversion first."""
        from safetensors.torch import load_model

        path = Path(path).expanduser().resolve()
        weights = path / "model.safetensors" if path.is_dir() else path
        if not weights.is_file():
            if (path / "params" / "_METADATA").exists():
                raise ValueError(
                    "This is an OpenPI Orbax checkpoint; convert it before PyTorch loading"
                )
            raise FileNotFoundError(weights)
        config = config or PI05Config()
        model = PI0Pytorch(config)
        # load_model handles the author's tied PaliGemma embedding / lm_head.
        load_model(model, weights, strict=True, device="cpu")
        model.to(device).eval()
        return cls(model, processor, **kwargs)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def reset(self):
        self._actions = deque()

    def loss(self, state, images, prompt, actions):
        if len(actions) != self.config.action_horizon:
            raise ValueError("Training action window must match action_horizon")
        observation, targets = self.processor.prepare(
            state, images, prompt, actions, device=self.device
        )
        return self.model(observation, targets).mean()

    @torch.inference_mode()
    def predict_action_chunk(self, state, images, prompt):
        self.model.eval()
        observation, _ = self.processor.prepare(state, images, prompt, device=self.device)
        actions = self.model.sample_actions(
            self.device, observation, num_steps=self.num_inference_steps
        )
        physical = self.processor.restore_actions(actions, state)[0]
        if not torch.isfinite(physical).all():
            raise ValueError("PI0.5 returned non-finite actions")
        return physical

    def select_action(self, state, images, prompt):
        if not self._actions:
            chunk = self.predict_action_chunk(state, images, prompt)
            self._actions.extend(chunk[: self.n_action_steps])
        return self._actions.popleft()
