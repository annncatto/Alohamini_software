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

    def processor(self, model, stats, device):
        from alohamini.policies.am_act.processor import AMACTProcessor

        return AMACTProcessor.from_config(model.config, stats, device)

    def prepare_options(self, options, samples):
        from alohamini.policies.am_act.classification import prepare_options

        prepare_options(options, samples)

    def apply_preset(self, settings):
        resolved = super().apply_preset(settings)
        if settings.get("pretrained_path"):
            _, manifest = read_checkpoint(settings["pretrained_path"])
            if "base_classification" not in manifest["config"]:
                resolved["model"].setdefault("base_classification", False)
                resolved["model"].setdefault("discrete_action_weighting", "none")
            if resolved["model"].get("discrete_action_class_weights") != manifest["config"].get(
                "discrete_action_class_weights"
            ):
                resolved["model"]["discrete_action_weight_source"] = "manual"
        # Record the reference/adaptation boundary with the experiment. Concrete
        # field order, calibration, statistics and resolved options are separately
        # saved in the native checkpoint; dimensions never establish semantics.
        resolved.setdefault(
            "paper_preset",
            {
                "name": "am-act-host-v1",
                "source": "https://arxiv.org/abs/2304.13705",
                "implementation_reference": "AlohaMini native AM-ACT 45dfb7f",
                "reference_protocol": "ACT Action/State posterior, standard Gaussian prior, zero-z inference; "
                "Host recordings are not the paper benchmark dataset",
                "adaptation": "Named absolute Host targets; optional image-conditioned CVAE and fixed-speed base classification",
                "fields_units": "arm/gripper .pos use dataset motor normalization; x/y.vel m/s, "
                "theta.vel deg/s, lift_axis.height_mm mm",
                "images": "RGB CHW; configured resize; shared ResNet L4; ImageNet MEAN_STD by default",
                "language": "unused",
                "normalization": "training numeric MEAN_STD; classified constant axes use std=1; "
                "resolved mapping/statistics saved in checkpoint",
                "temporal": "one current observation; future action rows within episode; padded targets excluded",
                "training_defaults": "chunk=100, execute=100, latent=32, KL=10, AdamW lr=1e-5, "
                "weight_decay=1e-4; conditional experiment overrides in config",
                "overrides": deepcopy(settings.get("model", {})),
            },
        )
        return resolved

    def checkpoint_options(self, *args, **kwargs):
        options = super().checkpoint_options(*args, **kwargs)
        options.setdefault("base_classification", False)
        options.setdefault("discrete_action_weighting", "none")
        return options

    def statistics(self, samples, options=None):
        stats = super().statistics(samples, options)
        # A never-moving classified axis still needs an invertible transform.
        # This scale is saved in checkpoint statistics and used by both processors.
        for dim in (options or {}).get("discrete_action_dims", []):
            if stats["action"]["std"][dim] < 1e-8:
                stats["action"]["std"][dim] = 1.0
        return stats

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
            for head, dim in enumerate(config.discrete_action_dims):
                specs.append(
                    MetricSpec(
                        f"loss_classification_dim_{dim}",
                        f"classification_loss_dim_{dim}",
                        "valid_action_steps",
                        loss_count("_reconstruction_weight"),
                    )
                )
                for actual in range(len(config.discrete_action_values[head])):
                    prefix = f"classification_dim_{dim}_class_{actual}"
                    specs.append(
                        MetricSpec(
                            f"{prefix}_recall",
                            f"{prefix}_recall",
                            "class_targets",
                            None,
                            count_source=f"{prefix}_support",
                            macro_group=f"classification_dim_{dim}_macro_recall",
                        )
                    )
                    for prediction in range(len(config.discrete_action_values[head])):
                        key = f"classification_confusion_dim_{dim}_{actual}_{prediction}"
                        specs.append(MetricSpec(key, key, "counts", None, reduction="sum"))
        if config.latent_kl_warmup_steps:
            specs = [s for s in specs if s.name != "loss_kl_weighted"]
            specs.append(
                MetricSpec(
                    "loss_kl_weighted",
                    "kld_loss_weighted",
                    "samples",
                    loss_count("_kl_weight"),
                    console="kl_w",
                    phases=("train",),
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
