"""ACT-family assembly and loss reduction, preserving the existing model semantics."""

from copy import deepcopy

from safetensors.torch import load_file, save_file

from alohamini.learning.checkpoint import (
    model_directory,
    read_checkpoint,
    resolve_pretrained,
    validate_pretrained,
)
from alohamini.learning.metrics import MetricSpec, loss_count
from alohamini.learning.processor import Processor, act_statistics, validate_statistics


def loss_counts(batch):
    return {
        "_reconstruction_weight": int((~batch["action_is_pad"]).sum()),
        "_kl_weight": len(batch["action"]),
    }


class ACTAlgorithm:
    fsdp_classes = ("ACTEncoderLayer", "ACTDecoderLayer", "BasicBlock", "Bottleneck")
    include_task = False
    loss_counts = staticmethod(loss_counts)
    kind = "act"

    def metric_specs(self, config):
        specs = [
            MetricSpec(
                "loss_l1",
                "l1_loss",
                "valid_action_elements",
                loss_count("_reconstruction_weight", config.action_feature.shape[0]),
                console="l1",
            )
        ]
        if config.use_vae:
            specs.extend(
                [
                    MetricSpec(
                        "loss_kl",
                        "kld_loss",
                        "samples",
                        loss_count("_kl_weight"),
                        console="kl",
                        phases=("train",),
                    ),
                    MetricSpec(
                        "loss_kl_weighted",
                        "kld_loss",
                        "samples",
                        loss_count("_kl_weight"),
                        scale=config.kl_weight,
                        console="kl_w",
                        phases=("train",),
                    ),
                ]
            )
        return specs

    def apply_preset(self, settings):
        if settings.get("pretrained_path") and not settings.get("resume"):
            if not (model_directory(settings["pretrained_path"]) / "policy.json").is_file():
                raise ValueError("--policy.path requires an AlohaMini checkpoint model directory")
        return resolve_pretrained(settings, expected_kind=self.kind)

    validate_pretrained = staticmethod(validate_pretrained)

    def statistics(self, samples, options=None):
        return act_statistics(samples)

    validate_statistics = staticmethod(validate_statistics)

    @property
    def config_class(self):
        from .configuration_act import ACTConfig

        return ACTConfig

    def options(self, settings, device):
        options = dict(settings.get("model", {}))
        options.setdefault("n_action_steps", options.get("chunk_size", 100))
        return options

    def sample_spec(self, options, *, cameras=()):
        return dict(
            delta_indices={"action": list(range(options.get("chunk_size", 100)))},
            include_task=self.include_task,
        )

    def build(self, options, stats=None):
        from .modeling_act import ACTPolicy

        return ACTPolicy(self.config_class(**options))

    def processor(self, model, stats, device):
        return Processor.from_config(model.config, stats, device)

    def initialize(self, model, settings):
        if settings.get("pretrained_path") and not settings.get("resume"):
            path = model_directory(settings["pretrained_path"])
            if not (path / "policy.json").is_file():
                raise ValueError("--policy.path requires an AlohaMini checkpoint model directory")
            path, _ = read_checkpoint(path)
            self.load(path, model)

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        options = deepcopy(options)
        options["pretrained_backbone_weights"] = None
        if n_action_steps is not None:
            options["n_action_steps"] = n_action_steps
        if temporal_ensemble_coeff != "checkpoint":
            options["temporal_ensemble_coeff"] = temporal_ensemble_coeff
        return options

    def save(self, path, model, manifest, model_state=None):
        state = model.state_dict() if model_state is None else model_state
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in state.items()},
            path / "model.safetensors",
        )

    def load(self, path, model):
        model.load_state_dict(load_file(path / "model.safetensors"), strict=True)


class AMACTAlgorithm(ACTAlgorithm):
    kind = "am_act"

    def metric_specs(self, config):
        specs = super().metric_specs(config)[1:]  # KL follows the ACT convention.
        excluded = set(config.fixed_action_dims) | set(config.discrete_action_dims)
        if config.action_loss_groups:
            # The combined L1 is a weighted group mean, not an element mean
            # across the concatenation of groups (which can overlap).
            specs.insert(
                0,
                MetricSpec(
                    "loss_l1",
                    "l1_loss",
                    "valid_action_steps_for_weighted_group_mean",
                    loss_count("_reconstruction_weight"),
                    console="l1",
                ),
            )
            for name, dims in config.action_loss_groups.items():
                width = len([d for d in dims if d not in excluded])
                if width:
                    specs.append(
                        MetricSpec(
                            f"loss_l1_{name}",
                            f"l1_loss_{name}",
                            "valid_group_action_elements",
                            loss_count("_reconstruction_weight", width),
                        )
                    )
        else:
            width = config.action_feature.shape[0] - len(excluded)
            specs.insert(
                0,
                MetricSpec(
                    "loss_l1",
                    "l1_loss",
                    "valid_continuous_action_elements",
                    loss_count("_reconstruction_weight", width),
                    console="l1",
                ),
            )
        if config.discrete_action_dims:
            heads = len(config.discrete_action_dims)
            count = loss_count("_reconstruction_weight", heads)
            specs.extend(
                [
                    MetricSpec(
                        "loss_classification",
                        "classification_loss",
                        "valid_head_targets",
                        count,
                        console="ce",
                    ),
                    MetricSpec(
                        "loss_classification_weighted",
                        "classification_loss",
                        "valid_head_targets",
                        count,
                        scale=config.discrete_action_loss_weight,
                        console="ce_w",
                    ),
                ]
            )
            for dim in config.discrete_action_dims:
                specs.append(
                    MetricSpec(
                        f"loss_classification_dim_{dim}",
                        f"classification_loss_dim_{dim}",
                        "valid_action_steps",
                        loss_count("_reconstruction_weight"),
                    )
                )
        return specs

    @property
    def config_class(self):
        from alohamini.policies.am_act.configuration_am_act import AMACTConfig

        return AMACTConfig

    def build(self, options, stats=None):
        from alohamini.policies.am_act.modeling_am_act import AMACTPolicy

        return AMACTPolicy(self.config_class(**options), dataset_stats=stats)
