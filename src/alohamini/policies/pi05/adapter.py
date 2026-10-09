"""Training/checkpoint adapter around the OpenPI model and transforms."""

import shutil
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_model, save_model

from alohamini.policies.configuration import PolicyConfig

from .configuration_pi05 import PI05Config
from .processor_pi05 import ARM_DELTA_DIMS, PI05Processor, fit_statistics


@dataclass
class PI05TrainingConfig(PolicyConfig):
    network: dict = field(default_factory=dict)
    tokenizer_path: str = ""
    n_action_steps: int = 50
    num_inference_steps: int = 10
    delta_dims: tuple = ARM_DELTA_DIMS
    span_floors: dict | None = None
    optimizer_lr: float = 2.5e-5
    optimizer_weight_decay: float = 1e-10
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0
    scheduler_type: str = "warmup_cosine"
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()
        core = PI05Config(**self.network)
        self.network = asdict(core)
        if self.robot_state_feature is None or self.robot_state_feature.shape != (18,):
            raise ValueError("PI0.5 AlohaMini requires the original 18-D position state")
        if self.action_feature.shape != (18,) or core.action_dim < 18:
            raise ValueError("PI0.5 requires 18 physical action coordinates")
        cameras = {key.removeprefix("observation.images.") for key in self.image_features}
        if "forward" not in cameras or cameras - {"forward", "wrist_left", "wrist_right"}:
            raise ValueError("PI0.5 requires forward and optional wrist_left/wrist_right cameras")
        if not 1 <= self.n_action_steps <= core.action_horizon or self.num_inference_steps < 1:
            raise ValueError("Invalid PI0.5 execution horizon or sampling steps")

    @property
    def action_delta_indices(self):
        return list(range(self.network["action_horizon"]))


class PI05BatchProcessor:
    def __init__(self, config, stats, device):
        from .tokenizer import PaligemmaTokenizer

        self.config, self.device = config, device
        self.transform = PI05Processor(
            stats,
            PaligemmaTokenizer(config.tokenizer_path, config.network["max_token_len"]),
            action_dim=config.network["action_dim"],
            delta_dims=config.delta_dims,
            span_floors=config.span_floors,
        )

    def __call__(self, batch):
        states = batch["observation.state"]
        tasks = batch.get("task")
        if not isinstance(tasks, (list, tuple)) or len(tasks) != len(states):
            raise ValueError("PI0.5 requires one task prompt per observation")
        observations, targets = [], []
        for i, state in enumerate(states):
            images = {}
            for key in self.config.image_features:
                rgb = batch[key][i].detach().cpu()
                if rgb.dtype != torch.uint8:
                    if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                        raise ValueError(f"{key}: expected CHW RGB in [0, 1]")
                    rgb = rgb.mul(255).round().byte()
                # OpenPI owns resize/padding and [-1, 1] scaling; raw bytes need
                # neither a float round trip nor ImageNet normalization.
                images[key.removeprefix("observation.images.")] = rgb.permute(1, 2, 0).numpy()
            observation, target = self.transform.prepare(
                state.detach().cpu().numpy(),
                images,
                tasks[i],
                batch["action"][i].detach().cpu().numpy() if "action" in batch else None,
                device=self.device,
            )
            observations.append(observation)
            if target is not None:
                targets.append(target)
        first = observations[0]
        values = {}
        for key, value in vars(first).items():
            if isinstance(value, dict):
                values[key] = {
                    name: torch.cat([getattr(o, key)[name] for o in observations]) for name in value
                }
            else:
                values[key] = (
                    None if value is None else torch.cat([getattr(o, key) for o in observations])
                )
        result = {
            "observation": SimpleNamespace(**values),
            "_action_context": states.to(self.device),
        }
        if targets:
            result["action"] = torch.cat(targets)
        return result

    def action(self, tensor, *, context=None):
        if context is None:
            raise ValueError("PI0.5 delta actions require their original observation state")
        single = tensor.ndim == 2
        result = self.transform.restore_actions(tensor[:, None] if single else tensor, context)
        return result[:, 0] if single else result


class PI05TrainModel(torch.nn.Module):
    name = "pi05"

    def __init__(self, config):
        super().__init__()
        from .modeling_pi05 import PI0Pytorch

        self.config = config
        self.network = PI0Pytorch(PI05Config(**config.network))
        self.reset()

    def get_optim_params(self):
        return self.parameters()

    def forward(self, batch):
        # Author's mean includes padded coordinates and repeated tail targets.
        loss = self.network(batch["observation"], batch["action"]).mean()
        return loss * batch.get("_loss_weight", 1.0), {"flow_loss": loss.detach().item()}

    def reset(self):
        self._actions = deque()
        self.action_context = None

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        return self.network.sample_actions(
            next(self.parameters()).device,
            batch["observation"],
            num_steps=self.config.num_inference_steps,
        )

    @torch.no_grad()
    def select_action(self, batch):
        if not self._actions:
            self.action_context = batch["_action_context"].detach().clone()
            chunk = self.predict_action_chunk(batch)
            self._actions.extend(chunk[:, : self.config.n_action_steps].transpose(0, 1))
        return self._actions.popleft()


class PI05Algorithm:
    config_class = PI05TrainingConfig
    include_task = True
    fsdp_classes = ()

    @staticmethod
    def apply_preset(settings):
        result = deepcopy(settings)
        defaults = dict(
            steps=30000, batch_size=32, state="joint_position,base_velocity,lift_height"
        )
        for key, value in defaults.items():
            result.setdefault(key, value)
        result.setdefault(
            "paper_preset",
            {
                "name": "openpi-pi05-alohamini-v1",
                "source": "https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac",
                "protocol": (
                    "Released OpenPI ALOHA transforms and PyTorch training defaults; "
                    "not paper pretraining"
                ),
                "data": {
                    "state": "Original 18-D AlohaMini coordinates, encoded into language tokens",
                    "action": (
                        "12 arm joints relative to the current state; "
                        "grippers/base/lift remain absolute targets/velocities"
                    ),
                    "images": "RGB; resize with aspect-preserving padding to 224x224; [-1,1]",
                    "language": "PaliGemma task + quantized state, length 200 by default",
                    "normalization": (
                        "Train-window q01/q99; float64 transforms; additive 1e-6; "
                        "no automatic clipping"
                    ),
                    "time": "50 future recorded rows by default; no timestamp resampling",
                    "loss": (
                        "Mean over batch/time/padded action width, "
                        "retaining repeated episode-tail targets"
                    ),
                },
                "adaptations": [
                    "No Trossen signs/gripper geometry",
                    "Configured dataset image size is saved explicitly",
                    "No JAX/EMA implementation; explicit overrides saved in train_config.json",
                ],
            },
        )
        return result

    def options(self, settings, device):
        options = dict(settings.get("model", {}))
        options.setdefault("n_action_steps", options.get("network", {}).get("action_horizon", 50))
        if not settings.get("resume") and not settings.get("pretrained_path"):
            raise ValueError("PI0.5 requires --policy.path to local converted PyTorch base weights")
        return options

    def sample_spec(self, options, *, cameras=()):
        return dict(
            delta_indices={
                "action": list(range(options.get("network", {}).get("action_horizon", 50)))
            },
            include_task=True,
        )

    def statistics(self, samples, options=None):
        return fit_statistics(samples, delta_dims=(options or {}).get("delta_dims", ARM_DELTA_DIMS))

    def build(self, options, stats=None):
        return PI05TrainModel(self.config_class(**options))

    def processor(self, model, stats, device):
        return PI05BatchProcessor(model.config, stats, device)

    @staticmethod
    def loss_counts(batch):
        return {"_loss_weight": len(batch["action"])}

    def initialize(self, model, settings):
        path = Path(settings["pretrained_path"]).expanduser()
        weights = path / "model.safetensors" if path.is_dir() else path
        load_model(model.network, weights, strict=True, device="cpu")

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        options = deepcopy(options)
        assets = (path / options["tokenizer_path"]).resolve()
        if not assets.is_relative_to(path) or not assets.is_file():
            raise ValueError("PI0.5 checkpoint is missing its local tokenizer")
        options["tokenizer_path"] = str(assets)
        if temporal_ensemble_coeff not in (None, "checkpoint"):
            raise ValueError("ACT temporal ensembling does not apply to PI0.5")
        if n_action_steps is not None:
            options["n_action_steps"] = n_action_steps
        return options

    def save(self, path, model, manifest, model_state=None):
        if model_state is not None:
            raise ValueError("PI0.5 sharded export is not adapted yet")
        shutil.copyfile(model.config.tokenizer_path, path / "tokenizer.model")
        manifest["config"]["tokenizer_path"] = "tokenizer.model"
        save_model(model.network, path / "model.safetensors")

    def load(self, path, model):
        load_model(model.network, path / "model.safetensors", strict=True)

    def validate_statistics(self, stats, expected, names, *, manifest=None):
        if manifest is not None and (manifest.get("state_feature") or {}).get("names") != names:
            raise ValueError("PI0.5 requires original position state matching action coordinates")
        # PI05Processor validates quantile shapes, ordering and finiteness.
        PI05Processor(stats, tokenizer=None)
