"""Local policy checkpoints and the adapter to the shared protected evaluator."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from alohamini.apps.replay import check_calibration
from alohamini.datasets.images import decode_host_image
from alohamini.datasets.native import StateSelection, motor_feedback_frame, state_names
from alohamini.learning.data import image_tensor


def make_policy(kind, options, stats=None):
    """Instantiate a native model without LeRobot, Hub access or device fallback."""
    if kind == "act":
        from alohamini.policies.act.configuration_act import ACTConfig
        from alohamini.policies.act.modeling_act import ACTPolicy

        return ACTPolicy(ACTConfig(**options))
    if kind == "am_act":
        from alohamini.policies.am_act.configuration_am_act import AMACTConfig
        from alohamini.policies.am_act.modeling_am_act import AMACTPolicy

        return AMACTPolicy(AMACTConfig(**options), dataset_stats=stats)
    raise ValueError(f"Unknown native policy: {kind}")


class Processor:
    """MEAN_STD and physical-action scaling, matching the migrated processors."""

    def __init__(self, stats, device="cpu", *, scale_dims=(), scale=1.0):
        self.device = torch.device(device)
        self.stats = {
            key: {
                k: torch.tensor(v, dtype=torch.float32, device=self.device)
                for k, v in values.items()
            }
            for key, values in stats.items()
        }
        self.scale_dims, self.scale = list(scale_dims), scale
        for values in self.stats.values():
            if (
                not torch.isfinite(values["mean"]).all()
                or not torch.isfinite(values["std"]).all()
                or (values["std"] < 0).any()
            ):
                raise ValueError("Invalid normalization statistics")

    def __call__(self, batch):
        result = {}
        for key, value in batch.items():
            value = value.to(self.device)
            if key != "action_is_pad":
                stats = self.stats[key]
                value = (value - stats["mean"]) / (stats["std"] + 1e-8)
            result[key] = value
        return result

    def action(self, tensor, *, execution=True):
        stats = self.stats["action"]
        result = tensor * stats["std"] + stats["mean"]
        if execution and self.scale_dims:
            result = result.clone()
            result[..., self.scale_dims] *= self.scale
        return result


def save_checkpoint(path, model, stats, samples, *, training):
    """Commit a new checkpoint directory; never overwrite an existing checkpoint."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".pending")
    if path.exists():
        raise FileExistsError(path)
    staging.mkdir()
    manifest = {
        "format": "alohamini-policy",
        "version": 1,
        "kind": model.name,
        "config": asdict(model.config),
        "stats": stats,
        "source_info": samples.info,
        "state": samples.state,
        "state_feature": samples.selection.feature if samples.selection else None,
        "state_units": samples.selection.units if samples.selection else [],
        "cameras": samples.cameras,
        "image_size": samples.image_size,
        "training": training,
        "review_note": samples.review_note,
        "dataset_check": samples.report,
        "table_sha256": samples.table_sha256,
    }
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        staging / "model.safetensors",
    )
    (staging / "policy.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    staging.rename(path)


class NativePolicy:
    """reset()/select_action(snapshot) interface consumed by apps.evaluation.

    The tensor-level network remains available as ``model`` for custom training.
    This adapter assembles the same ordered fields and RGB preprocessing as training.
    """

    def __init__(
        self, checkpoint, *, device="cpu", n_action_steps=None, temporal_ensemble_coeff="checkpoint"
    ):
        path = Path(checkpoint).expanduser().resolve()
        if path.name.endswith(".pending"):
            raise ValueError("Incomplete checkpoint")
        manifest = json.loads((path / "policy.json").read_text())
        if manifest.get("format") != "alohamini-policy" or manifest.get("version") != 1:
            raise ValueError("Unsupported native checkpoint")
        self.manifest = manifest
        self.source = manifest["source_info"]
        self.robot_metadata = deepcopy(self.source["robot_metadata"])
        self.fps = self.source["fps"]
        self.names = self.source["features"]["action"]["names"]
        if self.names != state_names(self.robot_metadata["robot_model"]):
            raise ValueError("Checkpoint action coordinates must retain all native Host targets")
        self.selection = (
            None if manifest["state"] == "none" else StateSelection(self.source, manifest["state"])
        )
        if self.selection and (
            self.selection.feature != manifest["state_feature"]
            or self.selection.units != manifest["state_units"]
        ):
            raise ValueError("Checkpoint state names/units mismatch")
        options = deepcopy(manifest["config"])
        # Saved model weights are authoritative; never fetch backbone weights while loading.
        options["pretrained_backbone_weights"] = None
        if n_action_steps is not None:
            options["n_action_steps"] = n_action_steps
        if temporal_ensemble_coeff != "checkpoint":
            options["temporal_ensemble_coeff"] = temporal_ensemble_coeff
        self.model = make_policy(manifest["kind"], options)
        self.config = self.model.config
        expected = {
            f"observation.images.{c}": ("VISUAL", (3, *manifest["image_size"]))
            for c in manifest["cameras"]
        }
        if self.selection:
            expected["observation.state"] = ("STATE", tuple(self.selection.feature["shape"]))
        actual = {k: (v.type, v.shape) for k, v in self.config.input_features.items()}
        if actual != expected or self.config.action_feature.shape != (len(self.names),):
            raise ValueError("Checkpoint feature dimensions/order do not match its data contract")
        if set(manifest["stats"]) != {*expected, "action"}:
            raise ValueError("Checkpoint is missing normalization statistics")
        for key, stats in manifest["stats"].items():
            shape = (
                (3, 1, 1)
                if key.startswith("observation.images.")
                else ((len(self.names),) if key == "action" else expected[key][1])
            )
            if any(np.asarray(stats[k]).shape != shape for k in ("mean", "std")):
                raise ValueError(f"Checkpoint normalization shape mismatch: {key}")
        self.model.load_state_dict(load_file(path / "model.safetensors"), strict=True)
        self.model.to(device).eval()
        self.processor = Processor(
            manifest["stats"],
            device,
            scale_dims=getattr(self.model.config, "inference_action_scale_dims", ()),
            scale=getattr(self.model.config, "inference_action_scale", 1.0),
        )
        self.reset()

    def reset(self):
        self.model.reset()

    @torch.inference_mode()
    def predict(self, sample):
        """Offline physical action chunk; sample has no batch dimension."""
        batch = self.processor(
            {k: v[None] for k, v in sample.items() if k not in ("action", "action_is_pad")}
        )
        return self.processor.action(self.model.predict_action_chunk(batch))[0].cpu()

    def select_action(self, snapshot):
        check_calibration(self.robot_metadata, snapshot)
        observation = {}
        if self.selection:
            row = {
                "observation.state": [snapshot.payload[n] for n in self.selection.source_names],
                **motor_feedback_frame(
                    self.source["features"], snapshot.payload.get("_motor_feedback", {})
                ),
            }
            observation["observation.state"] = torch.from_numpy(self.selection.frame(row))
            for _, index, _, mask in self.selection.columns:
                if mask:
                    now = snapshot.payload.get("_host_timing", {}).get(
                        "state_sample_finished_monotonic_s"
                    )
                    if (
                        now is None
                        or not 0 <= now - row["motor_feedback.sample_finished_s"][index] <= 0.25
                    ):
                        raise ValueError("Selected motor feedback is stale")
        for camera in self.manifest["cameras"]:
            if camera not in snapshot.images:
                raise ValueError(f"Missing policy camera: {camera}")
            observation[f"observation.images.{camera}"] = image_tensor(
                decode_host_image(snapshot.images[camera]), self.manifest["image_size"]
            )
        batch = self.processor({k: v[None] for k, v in observation.items()})
        with torch.inference_mode():
            values = self.processor.action(self.model.select_action(batch)).cpu().numpy()
        if values.shape != (1, len(self.names)) or not np.isfinite(values).all():
            raise ValueError("Policy returned invalid absolute targets")
        return dict(zip(self.names, map(float, values[0]), strict=True))


def evaluate_robot(
    checkpoint,
    *,
    enable_robot=False,
    host=None,
    device="cuda",
    n_action_steps=None,
    temporal_ensemble_coeff="checkpoint",
    **kwargs,
):
    """Explicit real-hardware boundary; importing/loading a model never connects."""
    if enable_robot is not True or not host:
        raise ValueError("Real robot evaluation requires enable_robot=True and an explicit host")
    from alohamini.apps.evaluation import evaluate

    policy = NativePolicy(
        checkpoint,
        device=device,
        n_action_steps=n_action_steps,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
    )
    return evaluate(
        host,
        policy.robot_metadata["robot_model"],
        policy_factory=lambda: policy,
        fps=policy.fps,
        **kwargs,
    )
