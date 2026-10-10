"""Named mobile-base defaults and training-only, unique-frame class statistics."""

import logging
import math

import torch

BASE_CLASSES = {
    "x.vel": [-0.15, 0.0, 0.15],  # m/s, KeyboardTargets default speed level
    "y.vel": [-0.15, 0.0, 0.15],  # m/s
    "theta.vel": [-45.0, 0.0, 45.0],  # deg/s (Host coordinates)
}


def nearest_class(values, centers):
    """Nearest center; ties choose the first configured class, as in legacy AM-ACT."""
    centers = torch.as_tensor(centers, device=values.device, dtype=values.dtype)
    return (values.unsqueeze(-1) - centers).abs().argmin(-1)


def class_weights(counts, strategy="sqrt_inverse_frequency", max_ratio=5.0, reference_index=None):
    if strategy not in {"none", "inverse_frequency", "sqrt_inverse_frequency"}:
        raise ValueError("Unknown class weighting strategy")
    if not math.isfinite(max_ratio) or max_ratio < 1:
        raise ValueError("Class weight ratio must be finite and >= 1")
    count = torch.as_tensor(counts, dtype=torch.float64)
    if not torch.isfinite(count).all() or (count < 0).any() or count.sum() <= 0:
        raise ValueError("Class counts must be nonnegative with at least one observed frame")
    if strategy == "none":
        return torch.ones_like(count).tolist()
    exponent = 0.5 if strategy == "sqrt_inverse_frequency" else 1.0
    # An absent class receives the cap, never an infinite weight. It still has
    # no positive examples; weighting cannot teach a missing motion direction.
    reference_index = int(count.argmax()) if reference_index is None else reference_index
    reference = count[reference_index].clamp_min(1)
    weights = (reference / count.clamp_min(1)).pow(exponent).clamp(min=1, max=max_ratio)
    weights[count == 0] = max_ratio
    weights[reference_index] = 1
    return weights.tolist()


def prepare_options(options, samples):
    """Resolve coordinates and fit counts without decoding images or counting padding."""
    names = samples.info["features"]["action"]["names"]
    if options.get("base_classification", True) and not options.get("discrete_action_dims"):
        missing = set(BASE_CLASSES) - set(names)
        if missing:
            raise ValueError(
                f"Base classification needs named Host coordinates {sorted(missing)}; "
                "set base_classification=false for other datasets"
            )
        options["discrete_action_dims"] = [names.index(name) for name in BASE_CLASSES]
        options["discrete_action_values"] = list(BASE_CLASSES.values())
    dims = options.get("discrete_action_dims", [])
    if not dims:
        return
    centers = options["discrete_action_values"]
    if len(dims) != len(centers) or any(not 0 <= d < len(names) for d in dims):
        raise ValueError("Discrete coordinates and centers do not match the action contract")
    # This is the same selected physical population as AlohaMiniDataset.statistics.
    # Boundary padding only repeats rows already in this set; no padded copies
    # or overlapping action chunks increase a frame's frequency.
    rows = sorted(samples._used_rows["action"])
    counts = [torch.zeros(len(values), dtype=torch.int64) for values in centers]
    for start in range(0, len(rows), 8192):
        actions = torch.tensor(
            [samples.rows[i]["action"] for i in rows[start : start + 8192]], dtype=torch.float32
        )
        for i, (dim, values) in enumerate(zip(dims, centers, strict=True)):
            counts[i] += torch.bincount(
                nearest_class(actions[:, dim], values), minlength=len(values)
            )
    options["discrete_action_class_counts"] = [count.tolist() for count in counts]
    manual = (
        options.get("discrete_action_class_weights")
        and options.get("discrete_action_weight_source") != "training_frames"
    )
    if not manual:
        options["discrete_action_class_weights"] = [
            class_weights(
                count,
                options.get("discrete_action_weighting", "sqrt_inverse_frequency"),
                options.get("discrete_action_max_weight_ratio", 5.0),
                values.index(0.0) if 0.0 in values else None,
            )
            for count, values in zip(counts, centers, strict=True)
        ]
    options["discrete_action_weight_source"] = "manual" if manual else "training_frames"
    for dim, count in zip(dims, counts, strict=True):
        if (count == 0).any():
            logging.warning(
                "AM-ACT %s class counts %s: absent classes have no positive supervision",
                names[dim],
                count.tolist(),
            )
    logging.info(
        "AM-ACT discrete frame counts=%s weights=%s",
        options["discrete_action_class_counts"],
        options["discrete_action_class_weights"],
    )


def classification_metrics(logits, labels, valid, dim):
    predicted = logits.argmax(-1)
    classes = logits.shape[-1]
    matrix = torch.bincount(
        (labels[valid] * classes + predicted[valid]), minlength=classes * classes
    ).reshape(classes, classes)
    outputs = {}
    for actual in range(classes):
        prefix = f"classification_dim_{dim}_class_{actual}"
        support = matrix[actual].sum()
        outputs[f"{prefix}_support"] = support
        outputs[f"{prefix}_recall"] = matrix[actual, actual] / support.clamp_min(1)
        for prediction in range(classes):
            outputs[f"classification_confusion_dim_{dim}_{actual}_{prediction}"] = matrix[
                actual, prediction
            ]
    return outputs
