"""Local policy checkpoints and the adapter to the shared protected evaluator."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from alohamini.apps.replay import check_calibration
from alohamini.datasets.images import decode_host_image
from alohamini.datasets.record import StateSelection, motor_feedback_frame, state_names
from alohamini.learning.checkpoint import read_checkpoint, write_checkpoint_metadata
from alohamini.learning.processor import image_tensor, scale_action
from alohamini.policies.registry import algorithm


def make_policy(kind, options, stats=None):
    """Instantiate the configured policy model."""
    return algorithm(kind).build(options, stats)


def make_processor(model, stats, device):
    return algorithm(model.name).processor(model, stats, device)


def save_checkpoint(path, model, stats, samples, *, training, model_state=None):
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
        "sample_windows": samples.delta_indices,
        "sample_boundary": "episode",
        "sample_filter": samples.sample_filter,
    }
    from alohamini.learning.fixed import action_summary

    manifest["fixed_state_excluded"] = True
    summary = action_summary(samples)
    if summary is not None:
        manifest["physical_action_summary"] = summary
    algorithm(model.name).save(staging, model, manifest, model_state)
    write_checkpoint_metadata(staging, manifest)
    staging.rename(path)


class NativePolicy:
    """reset()/select_action(snapshot) interface consumed by apps.evaluation.

    The tensor-level network remains available as ``model`` for custom training.
    This adapter assembles the same ordered fields and RGB preprocessing as training.
    """

    def __init__(
        self,
        checkpoint,
        *,
        device="cpu",
        n_action_steps=None,
        temporal_ensemble_coeff="checkpoint",
        task=None,
        fixed_dimensions=None,
        fixed_dataset=None,
        calibration_mode="strict",
    ):
        if calibration_mode not in ("strict", "normalized"):
            raise ValueError("calibration_mode must be strict or normalized")
        self.calibration_mode = calibration_mode
        self._robot_bound = False
        path, manifest = read_checkpoint(checkpoint)
        self.checkpoint_path = path
        self.checkpoint_manifest_sha256 = sha256((path / "policy.json").read_bytes()).hexdigest()
        self.manifest = manifest
        self.source = manifest["source_info"]
        self.robot_metadata = deepcopy(self.source["robot_metadata"])
        self.fps = self.source["fps"]
        self.names = self.source["features"]["action"]["names"]
        if self.names != state_names(self.robot_metadata["robot_model"]):
            raise ValueError("Checkpoint action coordinates must retain all Host targets")
        self.selection = (
            None
            if manifest["state"] == "none"
            else StateSelection(
                self.source,
                manifest["state"],
                exclude_fixed=manifest.get("fixed_state_excluded", False),
            )
        )
        if self.selection and (
            self.selection.feature != manifest["state_feature"]
            or self.selection.units != manifest["state_units"]
        ):
            raise ValueError("Checkpoint state names/units mismatch")
        self.algorithm = algorithm(manifest["kind"])
        self.task = task
        options = self.algorithm.checkpoint_options(
            path, manifest["config"], device, n_action_steps, temporal_ensemble_coeff
        )
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
        self.algorithm.validate_statistics(
            manifest["stats"], expected, self.names, manifest=manifest
        )
        self.algorithm.load(path, self.model)
        self.model.to(device).eval()
        self.processor = make_processor(self.model, manifest["stats"], device)
        from alohamini.learning.fixed import configure

        configure(self, fixed_dimensions, fixed_dataset)
        self.fixed_deployment["calibration_mode"] = calibration_mode
        self.reset()

    def bind_robot(self, snapshot):
        """Bind once before execution; later changes always require strict matching."""
        if self.calibration_mode == "strict" or self._robot_bound:
            check_calibration(self.robot_metadata, snapshot)
            return None
        from alohamini.learning.calibration import normalized_transfer

        metadata, report = normalized_transfer(self.source["robot_metadata"], snapshot)
        if self.selection:
            live_source = {**self.source, "robot_metadata": metadata}
            self.selection = StateSelection(
                live_source,
                self.manifest["state"],
                exclude_fixed=self.manifest.get("fixed_state_excluded", False),
            )
        self.robot_metadata = metadata
        self._robot_bound = True
        report.update(
            checkpoint=str(self.checkpoint_path),
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            fixed_dimensions=self.fixed_dimensions,
        )
        return report

    def reset(self):
        self.model.reset()

    def refresh_latent(self):
        """Notify AM-ACT of a task phase boundary before its next observation."""
        if not hasattr(self.model, "refresh_latent"):
            raise ValueError("This policy does not expose latent phase control")
        self.model.refresh_latent()

    def execution_action(self, tensor, *, context=None):
        """Restore physical outputs and apply the checkpoint's deployment scaling."""
        action = scale_action(
            self.processor.action(tensor, context=context),
            getattr(self.config, "inference_action_scale_dims", ()),
            getattr(self.config, "inference_action_scale", 1.0),
        )
        if self.fixed_dimensions:
            action = action.clone()
            for name, value in self.fixed_dimensions["targets"].items():
                action[..., self.names.index(name)] = value
        return action

    def fixed_input(self, sample):
        if self.fixed_state_references and "observation.state" in sample:
            sample = dict(sample)
            state = sample["observation.state"].clone()
            for index, value in self.fixed_state_references.items():
                state[..., index] = value
            sample["observation.state"] = state
        return sample

    def inference_context(self):
        precision = self.manifest.get("training", {}).get("mixed_precision")
        if precision in ("bfloat16", "float16"):
            return torch.autocast(
                next(self.model.parameters()).device.type, dtype=getattr(torch, precision)
            )
        return nullcontext()

    @torch.inference_mode()
    def predict(self, sample):
        """Offline physical action chunk; sample has no batch dimension."""
        batch = self.processor(
            {
                k: ([v] if k == "task" else v[None])
                for k, v in self.fixed_input(sample).items()
                if k not in ("action", "action_is_pad")
            }
        )
        with self.inference_context():
            return (
                self.execution_action(
                    self.model.predict_action_chunk(batch), context=batch.get("_action_context")
                )[0]
                .float()
                .cpu()
            )

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
                decode_host_image(snapshot.images[camera]),
                self.manifest["image_size"],
            )
        batch = {k: v[None] for k, v in observation.items()}
        if self.algorithm.include_task:
            batch["task"] = [self.task]
        batch = self.processor(self.fixed_input(batch))
        with torch.inference_mode(), self.inference_context():
            action = self.model.select_action(batch)
            values = (
                self.execution_action(
                    action,
                    context=getattr(self.model, "action_context", batch.get("_action_context")),
                )
                .float()
                .cpu()
                .numpy()
            )
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
    fixed_dimensions=None,
    fixed_dataset=None,
    calibration_mode="strict",
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
        fixed_dimensions=fixed_dimensions,
        fixed_dataset=fixed_dataset,
        calibration_mode=calibration_mode,
        task=kwargs.get("task"),
    )
    return evaluate(
        host,
        policy.robot_metadata["robot_model"],
        policy_factory=lambda: policy,
        fps=policy.fps,
        **kwargs,
    )
