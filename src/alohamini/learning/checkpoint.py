"""Shared native checkpoint metadata, fine-tuning configuration and weight routing.

External base checkpoints keep their algorithm-specific importers. Native checkpoints
always restore the complete policy, including robot heads, through strict loaders.
"""

import json
from copy import deepcopy
from pathlib import Path

ASSET_FIELDS = {
    "smolvla": ("vlm_model_name",),
    "pi05": ("tokenizer_path",),
    "fastwam": ("vae_model_id", "text_encoder_model_id", "tokenizer_model_id"),
}


def model_directory(path):
    path = Path(path).expanduser().resolve()
    if not (path / "policy.json").is_file() and (path / "pretrained_model/policy.json").is_file():
        path = path / "pretrained_model"
    return path


def read_checkpoint(path):
    path = model_directory(path)
    if any(part.endswith(".pending") or ".pending-" in part for part in path.parts):
        raise ValueError("Incomplete checkpoint")
    manifest = json.loads((path / "policy.json").read_text())
    if (manifest.get("format"), manifest.get("version")) != ("alohamini-policy", 1):
        raise ValueError("Unsupported AlohaMini checkpoint")
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(path / "model.safetensors")
    # Old manifests remain readable without the new inspectable sidecars.
    layout = manifest.get("layout_version", 1)
    if layout not in (1, 2):
        raise ValueError("Unsupported AlohaMini checkpoint layout")
    if layout == 2:
        for filename, expected in checkpoint_sidecars(manifest).items():
            if json.loads((path / filename).read_text()) != expected:
                raise ValueError(f"Checkpoint {filename} disagrees with policy.json")
    return path, manifest


def checkpoint_sidecars(manifest):
    """Keep policy.json authoritative; sidecars expose config and preprocessing."""
    config = {"type": manifest["kind"], **manifest["config"]}
    processor = {
        "format": "alohamini-preprocessing",
        "version": 1,
        **{
            key: manifest[key]
            for key in ("stats", "state", "state_feature", "state_units", "cameras", "image_size")
        },
        "action_feature": manifest["source_info"]["features"]["action"],
        "normalization_mapping": manifest["config"].get("normalization_mapping"),
    }
    return {"config.json": config, "preprocessing.json": processor}


def write_checkpoint_metadata(path, manifest):
    manifest["layout_version"] = 2
    manifest["assets"] = {
        key: {
            "path": manifest["config"][key],
            "storage": "external" if Path(manifest["config"][key]).is_absolute() else "bundled",
        }
        for key in ASSET_FIELDS.get(manifest["kind"], ())
    }
    for filename, content in {"policy.json": manifest, **checkpoint_sidecars(manifest)}.items():
        (path / filename).write_text(
            json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )


def _merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(result.get(key), dict) and isinstance(value, dict)
            else deepcopy(value)
        )
    return result


def resolve_pretrained(settings, *, expected_kind=None):
    """Resolve native configs before policy presets, without depending on old sources on resume."""
    cfg = deepcopy(settings)
    mode = cfg.get("normalization", "dataset")
    if mode not in ("dataset", "checkpoint"):
        raise ValueError("normalization must be dataset or checkpoint")
    if cfg.get("resume"):
        return cfg
    if mode == "checkpoint" and cfg.get("stats"):
        raise ValueError("Choose --normalization=checkpoint or --stats, not both")
    path = model_directory(cfg["pretrained_path"]) if cfg.get("pretrained_path") else None
    if path is None or not (path / "policy.json").is_file():
        if mode == "checkpoint":
            raise ValueError("--normalization=checkpoint requires a native --policy.path")
        return cfg  # External base: the algorithm's existing importer applies.
    path, manifest = read_checkpoint(path)
    kind = cfg.get("policy", expected_kind)
    if kind is not None and kind != manifest["kind"]:
        raise ValueError(f"--policy.path contains {manifest['kind']}, not a {kind} checkpoint")
    cfg["policy"] = manifest["kind"]
    cfg["pretrained_path"] = str(path)
    overrides = cfg.get("model", {})
    options = _merge(manifest["config"], overrides)
    for key in ASSET_FIELDS.get(manifest["kind"], ()):
        asset = Path(options[key]).expanduser()
        if not asset.is_absolute():
            asset = (
                Path.cwd()
                if key in overrides and overrides[key] != manifest["config"][key]
                else path
            ) / asset
        if not asset.exists():
            raise FileNotFoundError(f"Checkpoint resource {key}: {asset}")
        options[key] = str(asset.resolve())
    if "pretrained_backbone_weights" in options:
        options["pretrained_backbone_weights"] = None
    if "load_vlm_weights" in options:
        options["load_vlm_weights"] = False
    cfg["model"] = options
    if cfg.get("state", "auto") == "auto":
        cfg["state"] = manifest["state"]
    for key in ("cameras", "image_size"):
        if cfg.get(key) is None:
            cfg[key] = deepcopy(manifest[key])
    return cfg


def validate_pretrained(settings, samples):
    if settings.get("resume") or not settings.get("pretrained_path"):
        return None
    path = model_directory(settings["pretrained_path"])
    if not (path / "policy.json").is_file():
        return None
    _, manifest = read_checkpoint(path)
    expected = {
        "cameras": samples.cameras,
        "state_feature": samples.selection.feature if samples.selection else None,
        "state_units": samples.selection.units if samples.selection else [],
    }
    for key, value in expected.items():
        if json.loads(json.dumps(manifest[key])) != json.loads(json.dumps(value)):
            raise ValueError(f"Pretrained checkpoint {key} does not match training data")
    saved = manifest["source_info"]
    if saved["features"]["action"] != samples.info["features"]["action"]:
        raise ValueError("Pretrained checkpoint action coordinates do not match training data")
    for name in samples.info["features"]["action"]["names"]:
        if name.endswith(".pos"):
            motor = name.removesuffix(".pos")
            old = saved["robot_metadata"]["motors"][motor]["normalization"]
            new = samples.info["robot_metadata"]["motors"][motor]["normalization"]
            if old != new:
                raise ValueError(f"Pretrained checkpoint action units differ: {name}")
    if settings.get("normalization") == "checkpoint":
        for key in ("delta_dims", "normalization_mapping"):
            old = manifest["config"].get(key)
            new = settings.get("model", {}).get(key, old)
            if json.loads(json.dumps(old)) != json.loads(json.dumps(new)):
                raise ValueError(f"Checkpoint statistics require unchanged {key}")
    return manifest


def initialize_policy(component, model, settings):
    path = model_directory(settings["pretrained_path"]) if settings.get("pretrained_path") else None
    if path is not None and (path / "policy.json").is_file():
        path, manifest = read_checkpoint(path)
        if manifest["kind"] != model.name:
            raise ValueError("Pretrained checkpoint policy type differs")
        component.load(path, model)
    else:
        component.initialize(model, settings)
