"""FastWAM assembly with local assets and explicit video/action sampling."""

from copy import deepcopy
from pathlib import Path

from safetensors.torch import load_file, save_file

from alohamini.learning.processor import Processor, limits_statistics, validate_statistics


class FastWAMProcessor:
    def __init__(self, config, stats, device):
        self.numeric = Processor.from_config(config, stats, device)

    def __call__(self, batch):
        tasks = batch.get("task")
        if not tasks or any(not isinstance(t, str) or not t.strip() for t in tasks):
            raise ValueError("FastWAM requires a nonempty task for each sample")
        result = self.numeric({k: v for k, v in batch.items() if k != "task"})
        result["task"] = tasks
        return result

    def action(self, tensor, *, context=None):
        return self.numeric.action(tensor)


class FastWAMAlgorithm:
    include_task = True
    fsdp_classes = ()
    validate_statistics = staticmethod(validate_statistics)

    def statistics(self, samples, options=None):
        return limits_statistics(samples)

    @property
    def config_class(self):
        from .configuration_fastwam import FastWAMConfig

        return FastWAMConfig

    def apply_preset(self, settings):
        cfg = deepcopy(settings)
        cfg.setdefault("state", "auto")
        cfg.setdefault("image_size", [224, 224])
        cfg.setdefault("batch_size", 16)
        cfg.setdefault("mixed_precision", "bfloat16")
        cfg["scheduler"] = {
            "type": "diffusers_cosine",
            "warmup_steps": 0,
            **(cfg.get("scheduler") or {}),
        }
        cfg["paper_preset"] = {
            "name": "fastwam-libero-2cam-alohamini",
            "paper": "https://arxiv.org/abs/2603.16666",
            "source": "https://github.com/yuantianyuan01/FastWAM",
            "reference_revision": "7faa71108368fbb3b6885649f112af607427a2d4",
            "implementation": "LeRobot/AlohaMini 524420cde405b7bfacb9b08f804186789eb5d742",
            "data": {
                "images": "9 RGB frames at offsets 0,4,...,32; sorted cameras concatenated horizontally to 224x448; Wan maps [0,1] to [-1,1]",
                "action": "32 recorded Host targets, dataset field names and units; no LIBERO TCP/gripper transforms",
                "state": "current selected feedback, not future proprioception",
                "normalization": "training-only limits [-1,1]; constant dimensions unamplified",
                "language": "instruction prompt template; UMT5, 128 tokens",
                "time": "row offsets at recorded FPS; no implicit resampling",
            },
            "reference_recipe": {"batch_size": 16, "lr": 1e-4, "weight_decay": 0.01},
            "adaptations": [
                "Recorded AlohaMini joint/base/lift targets replace LIBERO end-effector actions",
                "Copied release uses action flow shift 5; current author config uses 1",
                "Local converted FastWAM safetensors base plus frozen Wan VAE/UMT5 assets",
                "No benchmark success claim; train steps and camera views are dataset-specific",
                "Step-budget training replaces the reference task's 10-epoch schedule",
            ],
            "overrides": deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides", {k: v for k, v in settings.items() if k != "paper_preset"}
                )
            ),
        }
        return cfg

    def options(self, settings, device):
        options = dict(settings.get("model", {}))
        options["device"] = device
        if not settings.get("resume") and not settings.get("pretrained_path"):
            raise ValueError("FastWAM requires --policy.path to a local converted base checkpoint")
        if not settings.get("resume"):
            weights = Path(settings["pretrained_path"]).expanduser() / "model.safetensors"
            if not weights.is_file():
                raise FileNotFoundError(weights)
        for key in ("vae_model_id", "text_encoder_model_id", "tokenizer_model_id"):
            if key not in options or not Path(options[key]).expanduser().is_dir():
                raise ValueError(f"FastWAM requires --policy.{key} pointing to local assets")
            options[key] = str(Path(options[key]).expanduser().resolve())
        if not options.get("load_text_encoder", True):
            raise ValueError("The dataset trainer requires the text encoder for task strings")
        return options

    def sample_spec(self, options, *, cameras=()):
        if not cameras:
            raise ValueError("FastWAM requires recorded cameras")
        return dict(
            delta_indices={
                "action": list(range(options.get("action_horizon", 32))),
                **{
                    f"observation.images.{c}": list(
                        range(
                            0,
                            options.get("num_video_frames", 33),
                            options.get("action_video_freq_ratio", 4),
                        )
                    )
                    for c in cameras
                },
            },
            include_task=True,
        )

    @staticmethod
    def loss_counts(batch):
        # Each sample averages its own valid action/video targets before weighting.
        return {"_loss_weight": len(batch["action"])}

    def build(self, options, stats=None):
        from .modeling_fastwam import FastWAMPolicy

        return FastWAMPolicy(self.config_class(**deepcopy(options)), stats)

    def processor(self, model, stats, device):
        return FastWAMProcessor(model.config, stats, device)

    def initialize(self, model, settings):
        """Load compatible core weights; only embodiment heads may be reinitialized."""
        path = Path(settings["pretrained_path"]).expanduser()
        state = load_file(path / "model.safetensors")
        own = model.state_dict()
        import re

        remapped = {}
        for key, value in state.items():
            key = re.sub(
                r"model\.mot\.mixtures\.([^.]+)\.blocks\.(\d+)\.",
                r"model.mot.layers.\2.blocks.\1.",
                key,
            )
            remapped[key] = value
        allowed = (
            "model.proprio_encoder.",
            "model.mot.mixtures.action.action_encoder.",
            "model.mot.mixtures.action.head.",
        )
        wrong = [k for k, v in remapped.items() if k in own and v.shape != own[k].shape]
        for key in wrong:
            if not key.startswith(allowed):
                raise ValueError(f"FastWAM base architecture mismatch: {key}")
            del remapped[key]
        missing = set(own) - remapped.keys()
        unexpected = remapped.keys() - own.keys()
        if unexpected or any(not key.startswith(allowed) for key in missing):
            raise ValueError(
                f"Incompatible FastWAM base: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        model.load_state_dict(remapped, strict=False)
        if missing:
            import logging

            logging.getLogger(__name__).info("Initialized embodiment heads: %s", sorted(missing))

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        options = deepcopy(options)
        options["device"] = device
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("ACT temporal ensembling does not apply to FastWAM")
        if n_action_steps is not None:
            options["n_action_steps"] = n_action_steps
        for key in ("vae_model_id", "text_encoder_model_id", "tokenizer_model_id"):
            if not Path(options[key]).is_dir():
                raise FileNotFoundError(
                    f"FastWAM checkpoint requires its frozen assets: {options[key]}"
                )
        return options

    def save(self, path, model, manifest, model_state=None):
        if model_state is not None:
            raise ValueError("FastWAM sharded export is not adapted yet")
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            path / "model.safetensors",
        )

    def load(self, path, model):
        model.load_state_dict(load_file(path / "model.safetensors"), strict=True)
