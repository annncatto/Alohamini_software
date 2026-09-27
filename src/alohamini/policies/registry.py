"""Lazy algorithm registration; no model loading at import time."""

from importlib import import_module

ALGORITHMS = {
    "diffusion": ("alohamini.policies.diffusion.adapter", "DiffusionAlgorithm"),
    "fastwam": ("alohamini.policies.fastwam.adapter", "FastWAMAlgorithm"),
    "act": ("alohamini.policies.act.adapter", "ACTAlgorithm"),
    "am_act": ("alohamini.policies.act.adapter", "AMACTAlgorithm"),
    "smolvla": ("alohamini.policies.smolvla.adapter", "SmolVLAAlgorithm"),
    "pi05": ("alohamini.policies.pi05.adapter", "PI05Algorithm"),
}


def algorithm(name):
    try:
        module, cls = ALGORITHMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown policy: {name}") from exc
    return getattr(import_module(module), cls)()
