# Copyright 2024-2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming statistics migrated from LeRobot compute_stats; no framework dependency."""

import tempfile

import numpy as np

DEFAULT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]


class RunningQuantileStats:
    """
    Maintains running statistics for batches of vectors, including mean,
    standard deviation, min, max, and approximate quantiles.

    Statistics are computed per feature dimension and updated incrementally
    as new batches are observed. Quantiles are estimated using histograms,
    which adapt dynamically if the observed data range expands.
    """

    def __init__(self, quantile_list: list[float] | None = None, num_quantile_bins: int = 5000):
        self._count = 0
        self._mean = None
        self._m2 = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = num_quantile_bins

        self._quantile_list = quantile_list
        if self._quantile_list is None:
            self._quantile_list = DEFAULT_QUANTILES
        self._quantile_keys = [f"q{int(q * 100):02d}" for q in self._quantile_list]

    def update(self, batch: np.ndarray) -> None:
        """Update the running statistics with a batch of vectors.

        Args:
            batch: An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        # Accumulate centered moments in float64. E[x²] - E[x]² loses small
        # physical variations around large offsets, even when input is float32.
        batch = batch.astype(np.float64, copy=False)
        num_elements, vector_length = batch.shape
        if num_elements == 0 or not np.isfinite(batch).all():
            raise ValueError("Statistics require nonempty, finite vectors")
        batch_mean = np.mean(batch, axis=0)
        batch_m2 = np.sum((batch - batch_mean) ** 2, axis=0)

        if self._count == 0:
            self._mean = batch_mean.copy()
            self._m2 = np.zeros(vector_length, dtype=np.float64)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError(
                    "The length of new vectors does not match the initialized vector length."
                )

            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        total = self._count + num_elements
        delta = batch_mean - self._mean
        self._m2 += batch_m2 + delta**2 * (self._count * num_elements / total)
        self._mean += delta * (num_elements / total)
        self._count = total

        self._update_histograms(batch)

    def get_statistics(self) -> dict[str, np.ndarray]:
        """Compute and return the statistics of the vectors processed so far.

        Returns:
            Dictionary containing the computed statistics.
        """
        if self._count < 1:
            raise ValueError("Cannot compute statistics without vectors.")

        variance = self._m2 / self._count

        stddev = np.sqrt(np.maximum(0, variance))
        constant = self._min == self._max
        stddev[constant] = 0

        stats = {
            "min": self._min.copy(),
            "max": self._max.copy(),
            "mean": np.where(constant, self._min, self._mean),
            "std": stddev,
            "count": np.array([self._count]),
        }

        quantile_results = self._compute_quantiles()
        for i, q in enumerate(self._quantile_keys):
            stats[q] = quantile_results[i]

        return stats

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            old_hist = self._histograms[i]

            # Create new edges with small padding to ensure range coverage
            padding = (self._max[i] - self._min[i]) * 1e-10
            new_edges = np.linspace(
                self._min[i] - padding, self._max[i] + padding, self._num_quantile_bins + 1
            )

            # Redistribute existing histogram counts to new bins
            # We need to map each old bin center to the new bins
            old_centers = (old_edges[:-1] + old_edges[1:]) / 2
            new_hist = np.zeros(self._num_quantile_bins)

            for old_center, count in zip(old_centers, old_hist, strict=False):
                if count > 0:
                    # Find which new bin this old center belongs to
                    bin_idx = np.searchsorted(new_edges, old_center) - 1
                    bin_idx = max(0, min(bin_idx, self._num_quantile_bins - 1))
                    new_hist[bin_idx] += count

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self) -> list[np.ndarray]:
        """Compute quantiles based on histograms."""
        results = []
        for q in self._quantile_list:
            target_count = q * self._count
            q_values = []

            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                q_value = self._compute_single_quantile(hist, edges, target_count)
                q_values.append(q_value)

            results.append(np.array(q_values))
        return results

    def _compute_single_quantile(
        self, hist: np.ndarray, edges: np.ndarray, target_count: float
    ) -> float:
        """Compute a single quantile value from histogram and bin edges."""
        cumsum = np.cumsum(hist)
        idx = np.searchsorted(cumsum, target_count)

        if idx == 0:
            return edges[0]
        if idx >= len(cumsum):
            return edges[-1]

        # If not edge case, interpolate within the bin
        count_before = cumsum[idx - 1]
        count_in_bin = cumsum[idx] - count_before

        # If no samples in this bin, use the bin edge
        if count_in_bin == 0:
            return edges[idx]

        # Linear interpolation within the bin
        fraction = (target_count - count_before) / count_in_bin
        return edges[idx] + fraction * (edges[idx + 1] - edges[idx])


class ExactQuantileStats(RunningQuantileStats):
    """FP64 centered moments and exact linear quantiles of all observed vectors.

    Numeric vectors spool to a temporary file; quantiles allocate at most one
    coordinate's samples at a time. No episode-quantile merging or rebinning.
    Use the histogram tracker for large image populations, not policy targets.
    """

    def __init__(self):
        super().__init__(num_quantile_bins=1)
        self._values = tempfile.TemporaryFile()

    def _adjust_histograms(self):
        pass

    def _update_histograms(self, batch):
        self._values.write(np.ascontiguousarray(batch, dtype=np.float64).tobytes())

    def _compute_quantiles(self):
        self._values.flush()
        values = np.memmap(
            self._values, dtype=np.float64, mode="r", shape=(self._count, len(self._mean))
        )
        return np.stack(
            [
                np.quantile(values[:, i], self._quantile_list, method="linear")
                for i in range(len(self._mean))
            ],
            axis=1,
        )


def diagnose_statistics(stats, *, names=None, epsilon=1e-6, span_floor=None):
    """Diagnose physical ranges; never silently suppress a constant robot joint.

    A narrow quantile range with nonzero full range is distinct from a truly
    constant coordinate. Floors are explicit per-coordinate physical quantities.
    """
    low, high = (np.asarray(stats[k], np.float64) for k in ("q01", "q99"))
    minimum, maximum = (np.asarray(stats[k], np.float64) for k in ("min", "max"))
    std = np.asarray(stats["std"], np.float64)
    span = high - low
    floor = np.zeros_like(span) if span_floor is None else np.asarray(span_floor, np.float64)
    if floor.shape != span.shape or not np.isfinite(floor).all() or (floor < 0).any():
        raise ValueError(
            "span_floor must specify one finite nonnegative physical value per dimension"
        )
    labels = names if names is not None else [str(i) for i in range(len(span))]
    if len(labels) != len(span):
        raise ValueError("Statistics names/dimensions mismatch")
    result = []
    for i, name in enumerate(labels):
        constant = maximum[i] == minimum[i]
        result.append(
            dict(
                name=name,
                constant=bool(constant),
                quantile_span=float(span[i]),
                full_range=float(maximum[i] - minimum[i]),
                narrow_quantiles=bool(span[i] <= epsilon),
                inconsistent_constant_std=bool(constant and std[i] != 0),
                scale_floor=float(floor[i]),
                normalization_gain=float(2 / (max(span[i], floor[i]) + epsilon)),
                normalized_min=float(
                    2 * (minimum[i] - low[i]) / (max(span[i], floor[i]) + epsilon) - 1
                ),
                normalized_max=float(
                    2 * (maximum[i] - low[i]) / (max(span[i], floor[i]) + epsilon) - 1
                ),
            )
        )
    return result
