"""Diffusion Policy sampling, limits normalization and checkpoint assembly."""

from copy import deepcopy

import numpy as np
from safetensors.torch import load_file, save_file

from alohamini.learning.processor import Processor, limits_statistics, validate_statistics


class DiffusionAlgorithm:
    include_task = False
    fsdp_classes = ()  # EMA needs a separate sharded-state integration.

    @property
    def config_class(self):
        from .configuration_diffusion import DiffusionConfig

        return DiffusionConfig

    def apply_preset(self, settings):
        cfg = deepcopy(settings)
        cfg.setdefault("state", "auto")
        cfg.setdefault("seed", 42)
        cfg.setdefault("batch_size", 64)
        cfg.setdefault("image_size", [96, 96])
        model = cfg.setdefault("model", {})
        model.setdefault("crop_shape", [84, 84])
        cfg["optimizer"] = {"type": "adamw", **cfg.get("optimizer", {})}
        cfg["scheduler"] = {
            "type": "diffusers_cosine",
            "warmup_steps": 500,
            **(cfg.get("scheduler") or {}),
        }
        cfg["paper_preset"] = {
            "name": "diffusion-unet-robot",
            "source": "https://github.com/real-stanford/diffusion_policy",
            "implementation": "LeRobot/AlohaMini 524420cde405b7bfacb9b08f804186789eb5d742",
            "reference": "train_diffusion_unet_hybrid_workspace.yaml",
            "reference_revision": "5ba07ac6661db573af695b419a7947ecb704690f",
            "overrides": deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides", {k: v for k, v in settings.items() if k != "paper_preset"}
                )
            ),
            "observation": "history ending at current row; no timestamp resampling",
            "action": "recorded named Host targets, with dataset units; no TCP conversion",
            "normalization": "numeric limits [-1,1]; RGB [0,1] to [-1,1]",
            "adaptation": "AlohaMini cameras at 96x96, crop 84x84 instead of lift_image_abs 84/76; recorded action coordinates",
            "reference_recipe": {
                "n_obs_steps": 2,
                "horizon": 16,
                "n_action_steps": 8,
                "ddpm_steps": 100,
                "ema_power": 0.75,
                "group_norm": True,
            },
        }
        return cfg

    def options(self, settings, device):
        options = dict(settings.get("model", {}))
        self.mask_loss = options.get("do_mask_loss_for_padding", False)
        return options

    def sample_spec(self, options, *, cameras=()):
        n = options.get("n_obs_steps", 2)
        horizon = options.get("horizon", 16)
        observations = list(range(1 - n, 1))
        return dict(
            delta_indices={
                "observation.state": observations,
                **{f"observation.images.{c}": observations for c in cameras},
                "action": list(range(1 - n, 1 - n + horizon)),
            },
            drop_n_last_frames=options.get("drop_n_last_frames", 7),
        )

    def statistics(self, samples, options=None):
        return limits_statistics(samples)

    @staticmethod
    def validate_statistics(stats, expected, names, *, manifest=None):
        validate_statistics(stats, expected, names)
        for key, values in stats.items():
            shape = np.asarray(values["mean"]).shape
            for field in ("min", "max"):
                if field not in values or np.asarray(values[field]).shape != shape:
                    raise ValueError(f"{key}: missing or invalid {field} statistics")

    def build(self, options, stats=None):
        from .modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy(self.config_class(**deepcopy(options)))

    def processor(self, model, stats, device):
        return Processor.from_config(model.config, stats, device)

    def loss_counts(self, batch):
        # Masked loss uses valid targets; unmasked DDPM loss uses repeated tails.
        return {
            "_loss_weight": int((~batch["action_is_pad"]).sum())
            if getattr(self, "mask_loss", False)
            else len(batch["action"])
        }

    def initialize(self, model, settings):
        if settings.get("pretrained_path"):
            from pathlib import Path

            self.load(Path(settings["pretrained_path"]).expanduser(), model)

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        options = deepcopy(options)
        options["pretrained_backbone_weights"] = None
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("ACT temporal ensembling does not apply to Diffusion")
        if n_action_steps is not None:
            options["n_action_steps"] = n_action_steps
        return options

    def save(self, path, model, manifest, model_state=None):
        state = model.state_dict() if model_state is None else model_state
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in state.items()}, path / "model.safetensors"
        )

    def load(self, path, model):
        model.load_state_dict(load_file(path / "model.safetensors"), strict=True)
