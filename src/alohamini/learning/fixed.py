"""Physical training means and explicit deployment overrides for fixed dimensions."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from alohamini.fixed import VELOCITIES, dataset_info, dimensions, validate
from alohamini.paths import WorkspacePaths


def action_summary(samples):
    cached = getattr(samples, "_physical_action_summary", None)
    if cached is not None:
        return cached
    if not hasattr(samples, "rows") or not hasattr(samples, "_used_rows"):
        return None
    indices = sorted(samples._used_rows.get("action", ()))
    if not indices:
        return None
    values = np.asarray([samples.rows[i]["action"] for i in indices], dtype=np.float64)
    result = summarize(values, samples.info["features"]["action"]["names"])
    samples._physical_action_summary = result
    return result


def summarize(values, names):
    if (
        values.ndim != 2
        or values.shape[1] != len(names)
        or not len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Cannot calculate physical means from invalid action rows")
    return dict(
        version=1,
        names=list(names),
        count=len(values),
        mean=values.mean(0).tolist(),
        min=values.min(0).tolist(),
        max=values.max(0).tolist(),
        std=values.std(0).tolist(),
        source="unique_training_action_rows",
    )


def dataset_summary(manifest, root=None):
    if root is None:
        recorded = Path(manifest.get("training", {}).get("dataset", ""))
        candidates = [recorded, WorkspacePaths().datasets / recorded.name]
        root = next((p for p in candidates if (p / "meta/info.json").is_file()), None)
    if root is None:
        raise ValueError("Checkpoint has no physical action summary; supply --fixed-dataset")
    root = Path(root).expanduser().resolve()
    info = dataset_info(root)
    source = manifest["source_info"]
    if (
        info["robot_metadata"] != source["robot_metadata"]
        or info["features"]["action"] != source["features"]["action"]
    ):
        raise ValueError("Fixed-target dataset coordinates differ from checkpoint")
    hashes = manifest.get("table_sha256", {})
    if not hashes:
        raise ValueError("Checkpoint lacks training file fingerprints for mean verification")
    for name, expected in hashes.items():
        file = (root / name).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError(f"Missing checkpoint training file: {name}")
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for part in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(part)
        if digest.hexdigest() != expected:
            raise ValueError(f"Fixed-target training file changed: {name}")
    episodes = manifest.get("training", {}).get("train_episodes")
    if not episodes:
        raise ValueError("Checkpoint must identify training episodes for fixed-target means")
    storage = json.loads((root / "meta/info.json").read_text())
    if storage.get("codebase_version") == "v3.0":
        from alohamini.learning.lerobot import LeRobotSource

        reader = LeRobotSource(root, storage)
        rows = [row for rows, _, _ in reader.read_episodes(episodes, ["action"]) for row in rows]
    else:
        rows = [
            row
            for ep in episodes
            for row in pq.read_table(
                root / f"episodes/episode_{ep:06d}/frames.parquet", columns=["action"]
            ).to_pylist()
        ]
    # Legacy checkpoints did not preserve the exact fitted row population. Recompute
    # over all physical rows of their declared training episodes, never validation.
    result = summarize(
        np.asarray([r["action"] for r in rows], dtype=np.float64),
        source["features"]["action"]["names"],
    )
    result["source"] = "verified_training_episodes_all_rows"
    result["dataset"] = str(root)
    return result


def configure(policy, selection=None, dataset=None, current=None):
    manifest = policy.manifest
    model = policy.robot_metadata["robot_model"]
    inherited = manifest["source_info"].get("fixed_dimensions")
    targets = validate(inherited, model)
    chosen = dimensions(selection, model)
    policy.fixed_current = dimensions(current, model)
    policy._fixed_current_bound = False
    if set(chosen) & set(policy.fixed_current):
        raise ValueError("A dimension cannot use both --fixed-dimensions and --fixed-current")
    # Explicit current-position selection overrides inherited targets, without
    # exposing the old target while waiting for the first controllable snapshot.
    targets = {n: v for n, v in targets.items() if n not in policy.fixed_current}
    extra = [n for n in chosen if n not in targets]
    summary = None
    if extra:
        summary = manifest.get("physical_action_summary")
        if dataset is not None or summary is None:
            summary = dataset_summary(manifest, dataset)
        if (
            summary.get("version") != 1
            or summary.get("names") != policy.names
            or len(summary.get("mean", [])) != len(policy.names)
        ):
            raise ValueError("Physical action summary does not match checkpoint coordinates")
        for name in extra:
            j = policy.names.index(name)
            targets[name] = 0.0 if name in VELOCITIES else float(summary["mean"][j])
    elif dataset is not None:
        raise ValueError("--fixed-dataset requires newly selected fixed dimensions")
    config = (
        dict(version=1, targets=targets, source="checkpoint_and_explicit_selection")
        if targets
        else None
    )
    validate(config, model)
    policy.fixed_dimensions = config
    policy.fixed_state_references = {}
    fixed_names = set(targets) | set(policy.fixed_current)
    if fixed_names and policy.selection:
        stats = manifest["stats"].get("observation.state", {})
        means = stats.get("mean")
        for j, name in enumerate(policy.selection.feature["names"]):
            joint = name.rsplit(".", 1)[0] + ".pos"
            if name in fixed_names or joint in fixed_names:
                if means is None or len(means) != len(policy.selection.feature["names"]):
                    raise ValueError("Fixed state adaptation requires saved physical state means")
                policy.fixed_state_references[j] = float(means[j])
    policy.fixed_deployment = dict(
        checkpoint=str(getattr(policy, "checkpoint_path", "")),
        checkpoint_manifest_sha256=getattr(policy, "checkpoint_manifest_sha256", None),
        inherited_fixed_dimensions=inherited,
        explicit_dimensions=chosen,
        current_dimensions=policy.fixed_current,
        training_table_sha256=manifest.get("table_sha256", {}),
        training_episodes=manifest.get("training", {}).get("train_episodes"),
        fixed_dimensions=config,
        state_references={
            policy.selection.feature["names"][j]: v
            for j, v in policy.fixed_state_references.items()
        },
        target_summary=summary,
    )


def bind_current(policy, snapshot, client_id):
    """Capture once per policy run; never silently recapture between episodes."""
    if not policy.fixed_current:
        return
    if policy._fixed_current_bound:
        saved = policy.fixed_deployment["current_capture"]
        if saved["host_session_id"] != snapshot.payload["_safety"]["host_session_id"] or saved[
            "lift_reference_sequence"
        ] != snapshot.payload.get("lift_axis.reference_sequence"):
            raise RuntimeError(
                "Host session or lift reference changed after --fixed-current capture"
            )
        return
    from alohamini.apps.teleoperation import ready_units
    from alohamini.fixed import capture

    if ready_units(snapshot, policy.robot_metadata["robot_model"], client_id) is None:
        raise ValueError("Fresh controllable Host feedback required for --fixed-current")
    captured = capture(snapshot, policy.fixed_current)
    targets = validate(policy.fixed_dimensions, snapshot.robot_model)
    targets.update(captured["targets"])
    config = dict(version=1, targets=targets, source="checkpoint_means_and_current_feedback")
    validate(config, snapshot.robot_model)
    policy.fixed_dimensions = config
    policy.fixed_deployment.update(
        fixed_dimensions=config,
        current_capture=dict(
            targets=captured["targets"],
            request_started_s=snapshot.request_started_s,
            received_s=snapshot.received_s,
            host_session_id=snapshot.payload["_safety"]["host_session_id"],
            control_epoch=snapshot.payload["_safety"]["control_epoch"],
            lift_reference_sequence=snapshot.payload.get("lift_axis.reference_sequence"),
            robot_metadata=deepcopy(snapshot.payload["_robot_metadata"]),
        ),
    )
    policy._fixed_current_bound = True
