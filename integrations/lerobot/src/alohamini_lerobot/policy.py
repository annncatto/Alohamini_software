# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AlohaMini evaluate_bi.py and LeRobot SyncInferenceEngine.
"""Local ACT checkpoints using saved processors and recorded Host coordinates."""

import json
import sys
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from alohamini._validation import finite_number
from alohamini.apps.replay import check_calibration
from alohamini.datasets.images import decode_host_image
from alohamini.datasets.native import StateSelection, motor_feedback_frame, state_names

_UNSET = object()


def _execution_config(config, *, n_action_steps=None, temporal_ensemble_coeff=_UNSET):
    """Apply evaluation overrides before ACT constructs its queue or ensembler."""
    steps = config.n_action_steps if n_action_steps is None else n_action_steps
    coefficient = (
        config.temporal_ensemble_coeff
        if temporal_ensemble_coeff is _UNSET
        else temporal_ensemble_coeff
    )
    if type(steps) is not int or not 1 <= steps <= config.chunk_size:
        raise ValueError(f"--policy.n_action_steps must be in [1, {config.chunk_size}]")
    if coefficient is not None:
        finite_number(coefficient, "--policy.temporal_ensemble_coeff")
        if steps != 1:
            raise ValueError("--policy.temporal_ensemble_coeff requires --policy.n_action_steps=1")
    return replace(config, n_action_steps=steps, temporal_ensemble_coeff=coefficient)


def _json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _processors(root):
    """Only absolute ACT preprocessing; no implicit relative-action reconstruction."""
    result = []
    allowed = (
        {
            "rename_observations_processor",
            "to_batch_processor",
            "device_processor",
            "normalizer_processor",
        },
        {"unnormalizer_processor", "device_processor"},
    )
    for filename, names in zip(
        ("policy_preprocessor.json", "policy_postprocessor.json"), allowed, strict=True
    ):
        config = _json(root / filename)
        steps = config.get("steps", [])
        if not steps:
            raise ValueError(f"Missing saved processor steps: {filename}")
        for step in steps:
            if step.get("registry_name") not in names:
                raise ValueError(f"Unsupported ACT processor: {step.get('registry_name')}")
            if step.get("config", {}).get("rename_map"):
                raise ValueError("Camera renaming must be resolved in the training export")
            if "state_file" in step:
                path = (root / step["state_file"]).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError(f"Missing or external processor state: {path}")
        result.append(config)
    return result


class LeRobotPolicy:
    """Map a platform training export to the native synchronous policy interface.

    Features and state selection come from the explicitly supplied training export,
    never guessed from checkpoint dimensions. This does not establish that an
    arbitrary checkpoint was trained on that export; the caller must pair them.
    """

    def __init__(self, policy, preprocessor, postprocessor, dataset, *, task="robot task"):
        self.policy, self.preprocessor, self.postprocessor = policy, preprocessor, postprocessor
        self.task = task
        root = Path(dataset).expanduser().resolve()
        if root.name.startswith(".pending-") or ".pending-" in root.name:
            raise ValueError("Cannot use an unfinished training export")
        info = _json(root / "meta/info.json")
        origin = _json(root / "meta/alohamini.json")
        self.source = origin.get("source_info", {})
        if (
            info.get("codebase_version") != "v3.0"
            or origin.get("version") != 1
            or self.source.get("format") != "alohamini-episodes"
            or self.source.get("version") not in (1, 2)
        ):
            raise ValueError("A platform LeRobot v3 training export is required")
        self.robot_metadata = deepcopy(self.source["robot_metadata"])
        self.fps = info["fps"]
        finite_number(self.fps, "training fps")
        if not 0 < self.fps <= 30 or self.fps != self.source.get("fps"):
            raise ValueError("Invalid or changed training FPS")
        self.names = state_names(self.robot_metadata["robot_model"])
        features = info["features"]
        action = features["action"]
        self.action_names = tuple(action["names"])
        if (
            action != self.source["features"]["action"]
            or action["dtype"] != "float32"
            or len(self.action_names) != len(self.names)
            or len(set(self.action_names)) != len(self.names)
            or set(self.action_names) != set(self.names)
            or action["shape"] != [len(self.names)]
        ):
            raise ValueError("Training action must retain all named absolute Host targets")
        self.selection = StateSelection(self.source, ",".join(origin["state_groups"]))
        if len(self.selection.source_names) != len(self.names) or set(
            self.selection.source_names
        ) != set(self.names):
            raise ValueError("Source state must retain the complete named Host state")
        if (
            self.selection.feature != features["observation.state"]
            or self.selection.units != origin["state_units"]
        ):
            raise ValueError(
                "Training state names, order or units do not match the recorded selection"
            )
        self.config = policy.config
        self.device = torch.device(self.config.device)
        inputs, outputs = self.config.input_features, self.config.output_features
        if (
            not inputs
            or set(outputs) != {"action"}
            or tuple(outputs["action"].shape) != (len(self.action_names),)
            or outputs["action"].type.value != "ACTION"
        ):
            raise ValueError("Policy input/output schema does not match training data")
        self.cameras = {}
        self.use_state = "observation.state" in inputs
        for key, feature in inputs.items():
            if key == "observation.state":
                expected = tuple(self.selection.feature["shape"])
                expected_type = "STATE"
            elif key.startswith("observation.images."):
                camera = key.removeprefix("observation.images.")
                if camera not in self.robot_metadata.get("cameras", []):
                    raise ValueError(f"Camera absent from training metadata: {camera}")
                shape = features[key]["shape"]
                if (
                    features[key]["dtype"] not in ("image", "video")
                    or len(shape) != 3
                    or shape[2] != 3
                ):
                    raise ValueError(f"Expected RGB training image: {key}")
                expected = (3, shape[0], shape[1])
                expected_type = "VISUAL"
                self.cameras[key] = camera
            else:
                raise ValueError(f"Unsupported policy input: {key}")
            if tuple(feature.shape) != expected or feature.type.value != expected_type:
                raise ValueError(f"Checkpoint/training feature shape mismatch: {key}")
        self.policy.to(self.device).eval()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint,
        dataset,
        *,
        device="cuda",
        task="robot task",
        n_action_steps=None,
        temporal_ensemble_coeff=_UNSET,
    ):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class
        from lerobot.processor import (
            PolicyProcessorPipeline,
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )

        root = Path(checkpoint).expanduser().resolve()
        if not root.is_dir() or not (root / "model.safetensors").is_file():
            raise ValueError("Provide a complete local checkpoint directory")
        config_json = _json(root / "config.json")
        if config_json.get("type") != "act" or config_json.get("use_peft", False):
            raise ValueError("This loader accepts official ACT checkpoints, not am_act or PEFT")
        processors = _processors(root)
        config = PreTrainedConfig.from_pretrained(root, local_files_only=True)
        config = _execution_config(
            config, n_action_steps=n_action_steps, temporal_ensemble_coeff=temporal_ensemble_coeff
        )
        config.device = str(torch.device(device))
        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment")
        # Full checkpoint weights include the backbone; do not fetch ImageNet weights.
        config.pretrained_backbone_weights = None
        policy = get_policy_class(config.type).from_pretrained(
            root, config=config, local_files_only=True, strict=True
        )
        pipelines = []
        for filename, saved, to_transition, to_output in zip(
            ("policy_preprocessor.json", "policy_postprocessor.json"),
            processors,
            (batch_to_transition, policy_action_to_transition),
            (transition_to_batch, transition_to_policy_action),
            strict=True,
        ):
            overrides = {}
            if any(step["registry_name"] == "device_processor" for step in saved["steps"]):
                overrides["device_processor"] = {"device": config.device}
            pipelines.append(
                PolicyProcessorPipeline.from_pretrained(
                    root,
                    config_filename=filename,
                    local_files_only=True,
                    overrides=overrides,
                    to_transition=to_transition,
                    to_output=to_output,
                )
            )
        return cls(policy, *pipelines, dataset, task=task)

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def select_action(self, snapshot):
        from lerobot.policies.utils import prepare_observation_for_inference

        check_calibration(self.robot_metadata, snapshot)
        observation = {}
        if self.use_state:
            row = {
                "observation.state": [
                    snapshot.payload[name] for name in self.selection.source_names
                ],
                **motor_feedback_frame(
                    self.source["features"], snapshot.payload.get("_motor_feedback", {})
                ),
            }
            observation["observation.state"] = self.selection.frame(row)
            for _, index, _, mask in self.selection.columns:
                if mask is None:
                    continue
                now = snapshot.payload.get("_host_timing", {}).get(
                    "state_sample_finished_monotonic_s"
                )
                finite_number(now, "Host state timestamp")
                age = now - row["motor_feedback.sample_finished_s"][index]
                if not 0 <= age <= 0.25:
                    raise ValueError("Selected motor feedback is stale")
        for key, camera in self.cameras.items():
            if camera not in snapshot.images:
                raise ValueError(f"Required policy camera is unavailable: {camera}")
            rgb = decode_host_image(snapshot.images[camera])
            if (3, *rgb.shape[:2]) != tuple(self.config.input_features[key].shape):
                raise ValueError(f"Live camera shape differs from training: {camera}")
            observation[key] = rgb
        autocast = (
            torch.autocast(device_type="cuda")
            if self.device.type == "cuda" and self.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            observation = prepare_observation_for_inference(
                observation, self.device, self.task, "alohamini_client"
            )
            observation = self.preprocessor(observation)
            if not self.use_state and self.config.type == "act":
                # LeRobot 0.6.1 ACT reads this tensor's device for its zero latent
                # even without a robot-state projection. No measured state enters
                # the model: the device carrier has zero features and skips stats.
                observation["observation.state"] = torch.empty((1, 0), device=self.device)
            action = self.postprocessor(self.policy.select_action(observation))
        if not isinstance(action, torch.Tensor) or action.shape != (1, len(self.action_names)):
            raise ValueError("Policy must produce exactly one complete action vector")
        values = action.detach().float().cpu().numpy()[0]
        if not np.isfinite(values).all():
            raise ValueError("Policy returned non-finite targets")
        return dict(zip(self.action_names, map(float, values), strict=True))


def main(argv=None):
    """Retain the existing command as an alias of the native evaluation entry."""
    from alohamini.cli import main as native_main

    return native_main(["evaluate", *(sys.argv[1:] if argv is None else argv)])
