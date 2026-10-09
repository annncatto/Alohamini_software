"""Detached diagnostics: algorithms own denominators, consumers own time windows."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


def loss_count(key, width=1):
    """Reuse the algorithm's already computed loss denominator."""
    return lambda batch, counts: counts[key] * width


@dataclass(frozen=True)
class MetricSpec:
    name: str
    source: str
    denominator: str
    count: Callable
    scale: float = 1.0
    console: str | None = None
    phases: tuple[str, ...] = ("train", "eval")
    # Policy outputs are means. Other reductions require an explicit extension,
    # rather than silently treating arbitrary scalars as batch means.
    reduction: str = "mean"

    def __post_init__(self):
        if self.reduction != "mean" or not math.isfinite(self.scale):
            raise ValueError("Loss metrics require mean reduction and a finite scale")

    def schema(self):
        return {
            key: getattr(self, key)
            for key in ("name", "source", "denominator", "scale", "console", "phases", "reduction")
        }


class MetricAccumulator:
    """Bounded device storage; one packed reduction/transfer for all components.

    Each entry is a numerator and its actual denominator, including across
    microbatches/ranks. No activation or autograd graph is retained. Validation
    uses no collective: the caller evaluates the complete dataset on one rank
    (DDP) or identically on every rank (FSDP).
    """

    def __init__(self, specs=(), *, device="cpu", phase="train"):
        self.specs = tuple(spec for spec in specs if phase in spec.phases)
        if len({s.name for s in self.specs}) != len(self.specs):
            raise ValueError("Metric names must be unique")
        self.totals = torch.zeros((len(self.specs), 2), device=device, dtype=torch.float64)

    def add(self, outputs, batch, counts):
        with torch.no_grad():
            for index, spec in enumerate(self.specs):
                count = spec.count(batch, counts)
                if not math.isfinite(count) or count < 0:
                    raise ValueError(f"{spec.name}: invalid metric denominator {count}")
                value = torch.as_tensor(
                    outputs[spec.source], device=self.totals.device, dtype=torch.float64
                ).detach()
                if value.numel() != 1:
                    raise ValueError(f"{spec.name}: expected a scalar policy metric")
                # A batch with no valid targets contributes neither sum nor count.
                if count:
                    self.totals[index, 0].add_(value.double().reshape(()) * (count * spec.scale))
                self.totals[index, 1].add_(count)

    def result(self, reduce=None):
        totals = self.totals
        if reduce is not None and self.specs:
            totals = reduce(totals, reduction="sum")
        values, statistics = {}, {}
        for spec, (numerator, count) in zip(self.specs, totals.tolist(), strict=True):
            if not math.isfinite(numerator) or not math.isfinite(count):
                raise RuntimeError(f"{spec.name}: non-finite diagnostic metric")
            if count:
                values[spec.name] = numerator / count
                statistics[spec.name] = {"sum": numerator, "count": count}
        return {"metrics": values, "metric_totals": statistics}


def format_metrics(values, specs):
    """Only declared headline components belong on the console."""
    return "".join(
        f" {spec.console}:{values[spec.name]:.4f}"
        for spec in specs
        if spec.console and spec.name in values
    )
