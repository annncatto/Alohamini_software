"""SmolVLA assembly; tokenization, assets and valid-action reduction stay together."""

from copy import deepcopy

from alohamini.learning.metrics import MetricSpec, loss_count
from alohamini.learning.processor import validate_statistics

from .preset import apply_preset
from .processor_smolvla import SmolVLAProcessor, fit_statistics


class SmolVLAAlgorithm:
    fsdp_classes = ()
    include_task = True
    apply_preset = staticmethod(apply_preset)
    validate_statistics = staticmethod(validate_statistics)

    @staticmethod
    def metric_specs(config):
        width = min(config.action_feature.shape[0], config.max_action_dim)
        specs = [
            MetricSpec(
                "loss_flow",
                "loss",
                "valid_action_elements",
                loss_count("_reconstruction_weight", width),
                console="flow",
            )
        ]
        # These upstream diagnostics average over every position, including
        # masked zeroes. They are not valid-element means or additive losses.
        for source in (
            "losses_after_forward",
            "losses_after_in_ep_bound",
            "losses_after_rm_padding",
        ):
            specs.append(
                MetricSpec(
                    source,
                    source,
                    "action_elements_including_padding"
                    if source == "losses_after_forward"
                    else "action_elements_including_masked_zeroes",
                    lambda batch, counts: (
                        batch["action"].shape[0] * batch["action"].shape[1] * width
                    ),
                )
            )
        return specs

    def statistics(self, samples, options=None):
        return fit_statistics(samples)

    @staticmethod
    def loss_counts(batch):
        return {"_reconstruction_weight": int((~batch["action_is_pad"]).sum())}

    @property
    def config_class(self):
        from .configuration_smolvla import SmolVLAConfig

        return SmolVLAConfig

    def options(self, settings, device):
        options = dict(settings.get("model", {}))
        options["device"] = device
        options.setdefault("n_action_steps", options.get("chunk_size", 50))
        if not settings.get("resume") and not settings.get("pretrained_path"):
            raise ValueError(
                "SmolVLA fine-tuning requires --policy.path to a local base checkpoint"
            )
        return options

    def sample_spec(self, options, *, cameras=()):
        return dict(
            delta_indices={"action": list(range(options.get("chunk_size", 50)))},
            include_task=True,
        )

    def build(self, options, stats=None):
        from .modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy(self.config_class(**options))

    def processor(self, model, stats, device):
        return SmolVLAProcessor(
            model.config, stats, model.model.vlm_with_expert.processor.tokenizer, device
        )

    def initialize(self, model, settings):
        model.load_base_weights(settings["pretrained_path"])

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        options = deepcopy(options)
        assets = (path / options["vlm_model_name"]).resolve()
        if not assets.is_relative_to(path) or not assets.is_dir():
            raise ValueError("SmolVLA checkpoint is missing its local backbone/tokenizer assets")
        options.update(vlm_model_name=str(assets), load_vlm_weights=False, device=device)
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("ACT temporal ensembling does not apply to SmolVLA")
        if n_action_steps is not None:
            options["n_action_steps"] = n_action_steps
        return options

    def save(self, path, model, manifest, model_state=None):
        from safetensors.torch import save_model

        if model_state is not None:
            raise ValueError("SmolVLA sharded export is not adapted yet")
        assets = path / "vlm_assets"
        backbone = model.model.vlm_with_expert
        backbone.config.save_pretrained(assets)
        backbone.processor.save_pretrained(assets)
        manifest["config"].update(vlm_model_name="vlm_assets", load_vlm_weights=False)
        save_model(model, path / "model.safetensors")

    def load(self, path, model):
        from safetensors.torch import load_model

        load_model(model, path / "model.safetensors", strict=True)
