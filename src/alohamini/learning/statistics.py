"""Inspectable dataset statistics and reusable policy normalization artifacts."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from alohamini.learning.checkpoint import resolve_pretrained, validate_pretrained
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.processor import DEFAULT_IMAGE_SIZE
from alohamini.policies.registry import algorithm


def sample_arguments(cfg, components, options):
    """Shared field/window/episode selection for statistics and training."""
    root = Path(cfg["dataset"]).expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    state = cfg.get("state", "none")
    if state == "auto":
        state = (
            "joint_position,base_velocity,lift_height"
            if "observation.state" in info["features"]
            else "none"
        )
    cameras = cfg.get("cameras")
    if cameras is None:
        cameras = info.get(
            "cameras",
            info.get("robot_metadata", {}).get(
                "cameras",
                [
                    k.removeprefix("observation.images.")
                    for k in info["features"]
                    if k.startswith("observation.images.")
                ],
            ),
        )
    args = dict(
        root=str(root),
        **components.sample_spec(options, cameras=cameras),
        state=state,
        cameras=cfg.get("cameras"),
        image_size=tuple(cfg.get("image_size", DEFAULT_IMAGE_SIZE)),
        review_note=cfg.get("review_note", ""),
        video_cache_size=cfg.get("video_cache_size", 8),
        video_backend=cfg.get("video_backend", "pyav"),
        camera_workers=cfg.get("camera_workers", 0),
        return_uint8=cfg.get("return_uint8", False),
    )
    validation = cfg.get("val_episodes", [])
    episodes = cfg.get("train_episodes")
    if episodes is None:
        count = info.get("total_episodes")
        if count is None:
            count = len(list((root / "episodes").glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]")))
        episodes = [i for i in range(count) if i not in validation]
    if set(episodes) & set(validation):
        raise ValueError("Train and validation episodes must be disjoint")
    return args, episodes, validation


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _contract(components, samples, options, kind):
    model = asdict(components.config_class(**options))
    # Device and execution cadence do not change the fitted training population.
    for key in ("device", "n_action_steps", "temporal_ensemble_coeff"):
        model.pop(key, None)
    return _canonical(
        dict(
            policy=kind,
            model=model,
            source_info=samples.info,
            source_sha256=samples.table_sha256,
            episodes=samples.episodes,
            state=samples.state,
            state_units=samples.selection.units if samples.selection else [],
            cameras=samples.cameras,
            image_size=samples.image_size,
            windows=samples.delta_indices,
            sample_boundary="episode",
            sample_filter=samples.sample_filter,
            sample_indices_sha256=hashlib.sha256(
                json.dumps(samples.sample_indices, separators=(",", ":")).encode()
            ).hexdigest(),
        )
    )


def write_statistics(path, artifact):
    """Publish a new JSON file, without replacing a previous experiment's statistics."""
    text = json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return path


def training_statistics(components, samples, options, kind, source=None, *, checkpoint=None):
    contract = _contract(components, samples, options, kind)
    if source and checkpoint is not None:
        raise ValueError("Choose a statistics file or checkpoint statistics")
    if source:
        artifact = json.loads(Path(source).expanduser().read_text())
        if (artifact.get("format"), artifact.get("version")) != (
            "alohamini-training-statistics",
            1,
        ):
            raise ValueError("--stats requires policy statistics, not raw dataset stats.json")
        if artifact.get("contract") != contract:
            saved = artifact.get("contract") or {}
            differences = sorted(k for k, value in contract.items() if saved.get(k) != value)
            raise ValueError(
                "Statistics do not match the selected data, policy or sample configuration: "
                + ", ".join(differences)
            )
        stats = artifact["stats"]
    elif checkpoint is not None:
        stats = checkpoint["stats"]
    else:
        stats = components.statistics(samples, options)
    # Policy-specific layouts (including PI0.5 delta statistics) remain with the algorithm.
    expected = {k: (v.type, v.shape) for k, v in samples.input_features.items()}
    components.validate_statistics(stats, expected, samples.info["features"]["action"]["names"])
    return _canonical(
        dict(
            format="alohamini-training-statistics",
            version=1,
            contract=contract,
            stats=stats,
            samples=len(samples),
            excluded_samples=samples.excluded,
            **({"statistics_source": "checkpoint"} if checkpoint is not None else {}),
        )
    )


def prepare_statistics(root, output, *, config=None):
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(root):
        raise ValueError("Save the statistics report outside the source dataset")
    if config is None:
        from alohamini.datasets.edit import compute_statistics
        from alohamini.datasets.tools import check_dataset

        info = json.loads((root / "meta/info.json").read_text())
        if info.get("format") != "alohamini-episodes":
            raise ValueError(
                "Basic statistics require an AlohaMini recording; "
                "use --config for policy statistics on v3"
            )
        report = check_dataset(root)
        if not report["valid"]:
            raise ValueError(f"Dataset integrity check failed: {report['issues']}")
        stats, provenance = compute_statistics(
            root,
            info,
            SimpleNamespace(
                relative_action=False,
                chunk_size=1,
                relative_exclude_joints=[],
                skip_image_video=True,
            ),
        )
        artifact = dict(
            format="alohamini-dataset-statistics",
            version=1,
            features=info["features"],
            stats=stats,
            provenance=provenance,
        )
    else:
        cfg = json.loads(Path(config).expanduser().read_text())
        if not isinstance(cfg, dict):
            raise ValueError("Training config must be a JSON object")
        cfg = resolve_pretrained(cfg)
        kind = cfg.get("policy", "act")
        components = algorithm(kind)
        cfg = components.apply_preset({**cfg, "dataset": str(root)})
        # Same defaults as the public training CLI; no model is built or downloaded.
        cfg.setdefault("state", "auto")
        options = components.options(cfg, "cpu")
        args, episodes, _ = sample_arguments(cfg, components, options)
        samples = AlohaMiniDataset(**args, episodes=episodes)
        options.update(
            input_features=samples.input_features, output_features=samples.output_features
        )
        manifest = validate_pretrained(cfg, samples)
        if hasattr(components, "prepare_options"):
            components.prepare_options(options, samples)
        artifact = training_statistics(
            components,
            samples,
            options,
            kind,
            checkpoint=manifest if cfg.get("normalization") == "checkpoint" else None,
        )
    return write_statistics(output, artifact)
