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
    count: Callable | None
    scale: float = 1.0
    console: str | None = None
    phases: tuple[str, ...] = ("train", "eval")
    # Losses/recalls are means with explicit denominators. Confusion entries
    # declare sum reduction; they must never be averaged across batches.
    reduction: str = "mean"
    count_source: str | None = None
    macro_group: str | None = None

    def __post_init__(self):
        if self.reduction not in {"mean", "sum"} or not math.isfinite(self.scale):
            raise ValueError("Metrics require mean or sum reduction and a finite scale")

    def schema(self):
        return {
            key: getattr(self, key)
            for key in (
                "name",
                "source",
                "denominator",
                "scale",
                "console",
                "phases",
                "reduction",
                "count_source",
                "macro_group",
            )
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
                count = (
                    outputs[spec.count_source]
                    if spec.count_source
                    else (1 if spec.reduction == "sum" else spec.count(batch, counts))
                )
                if not isinstance(count, torch.Tensor) and (not math.isfinite(count) or count < 0):
                    raise ValueError(f"{spec.name}: invalid metric denominator {count}")
                value = torch.as_tensor(
                    outputs[spec.source], device=self.totals.device, dtype=torch.float64
                ).detach()
                if value.numel() != 1:
                    raise ValueError(f"{spec.name}: expected a scalar policy metric")
                # A batch with no valid targets contributes neither sum nor count.
                if isinstance(count, torch.Tensor):
                    # Class supports are detached bincounts. Keep them on device;
                    # validate the aggregate once in result(), not once per class.
                    self.totals[index, 0].add_(
                        torch.where(count > 0, value.reshape(()) * count * spec.scale, 0)
                    )
                elif count:
                    self.totals[index, 0].add_(value.double().reshape(()) * (count * spec.scale))
                self.totals[index, 1].add_(count)

    def result(self, reduce=None):
        totals = self.totals
        if reduce is not None and self.specs:
            totals = reduce(totals, reduction="sum")
        values, statistics = {}, {}
        for spec, (numerator, count) in zip(self.specs, totals.tolist(), strict=True):
            if not math.isfinite(numerator) or not math.isfinite(count) or count < 0:
                raise RuntimeError(f"{spec.name}: non-finite diagnostic metric")
            if count:
                values[spec.name] = numerator if spec.reduction == "sum" else numerator / count
                statistics[spec.name] = {"sum": numerator, "count": count}
        for group in {s.macro_group for s in self.specs if s.macro_group}:
            recalls = [
                values[s.name] for s in self.specs if s.macro_group == group and s.name in values
            ]
            if recalls:
                values[group] = sum(recalls) / len(recalls)
        return {"metrics": values, "metric_totals": statistics}


def format_metrics(values, specs):
    """Only declared headline components belong on the console."""
    return "".join(
        f" {spec.console}:{values[spec.name]:.4f}"
        for spec in specs
        if spec.console and spec.name in values
    )
