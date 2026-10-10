"""Sampled uint8 RGB statistics, with mergeable per-channel histograms."""

import numpy as np

from alohamini.datasets.statistics import DEFAULT_QUANTILES


def sample_indices(length):
    """LeRobot's bounded N**0.75 sample count and uniform rounded indices."""
    count = max(min(100, length), min(int(length**0.75), 10_000))
    return np.round(np.linspace(0, length - 1, count)).astype(int).tolist()


class ImageStatistics:
    def __init__(self, payload=None):
        self.histograms = {}
        self.frames = {}
        self.sources = set()
        if payload is not None:
            if payload["version"] != 1:
                raise ValueError("Unsupported image statistics version")
            self.sources.update(payload["sources"])
            for key, values in payload["cameras"].items():
                histogram = np.asarray(values["histogram"], dtype=np.int64)
                if (
                    histogram.shape != (3, 256)
                    or (histogram < 0).any()
                    or not (histogram.sum(axis=1) == histogram.sum(axis=1)[0]).all()
                    or histogram.sum() == 0
                    or values["frames"] < 1
                ):
                    raise ValueError(f"Invalid image statistics: {key}")
                self.histograms[key] = histogram
                self.frames[key] = values["frames"]

    def update(self, key, rgb, *, source):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Image statistics require uint8 RGB")
        # Match LeRobot's spatial sampling: retain images smaller than 300 px.
        size = max(rgb.shape[:2])
        stride = size // 150 if size >= 300 else 1
        pixels = rgb[::stride, ::stride].reshape(-1, 3)
        histogram = np.stack([np.bincount(pixels[:, c], minlength=256) for c in range(3)])
        self.histograms[key] = self.histograms.get(key, 0) + histogram
        self.frames[key] = self.frames.get(key, 0) + 1
        self.sources.add(source)

    def merge(self, other):
        for key, histogram in other.histograms.items():
            self.histograms[key] = self.histograms.get(key, 0) + histogram
            self.frames[key] = self.frames.get(key, 0) + other.frames[key]
        self.sources.update(other.sources)

    def payload(self):
        return {
            "version": 1,
            "sources": sorted(self.sources),
            "sampling": "uniform rounded linspace; min(N, max(100, min(floor(N**0.75), 10000)))",
            "spatial_sampling": "stride=max(H,W)//150 if max(H,W)>=300 else 1",
            "cameras": {
                key: {"histogram": histogram.tolist(), "frames": self.frames[key]}
                for key, histogram in self.histograms.items()
            },
        }

    def result(self):
        result = {}
        levels = np.arange(256, dtype=np.float64) / 255
        for key, histogram in self.histograms.items():
            count = histogram.sum(axis=1)
            mean = (histogram * levels).sum(axis=1) / count
            variance = (histogram * (levels - mean[:, None]) ** 2).sum(axis=1) / count
            stats = {
                "mean": mean,
                "std": np.sqrt(variance),
                "min": np.array([levels[np.flatnonzero(h)[0]] for h in histogram]),
                "max": np.array([levels[np.flatnonzero(h)[-1]] for h in histogram]),
            }
            cumulative = histogram.cumsum(axis=1)
            for q in DEFAULT_QUANTILES:
                ranks = (count - 1) * q
                low, high = np.floor(ranks).astype(int), np.ceil(ranks).astype(int)
                left = np.array(
                    [np.searchsorted(h, r + 1) for h, r in zip(cumulative, low, strict=True)]
                )
                right = np.array(
                    [np.searchsorted(h, r + 1) for h, r in zip(cumulative, high, strict=True)]
                )
                stats[f"q{int(q * 100):02d}"] = (left + (right - left) * (ranks - low)) / 255
            result[key] = {name: value.reshape(3, 1, 1) for name, value in stats.items()}
            result[key]["count"] = np.array([self.frames[key]])
        return result


def sample_recording_images(episode, cameras, length):
    """Read accepted-row JPEG/legacy PNG samples before replacing the image index."""
    import pyarrow.parquet as pq

    from alohamini.datasets.images import image_rgb

    stats = ImageStatistics()
    if cameras:
        indices = set(sample_indices(length))
        keys = [f"observation.images.{camera}" for camera in cameras]
        offset = 0
        for batch in pq.ParquetFile(episode / "frames.parquet").iter_batches(
            batch_size=512, columns=keys
        ):
            for row in batch.to_pylist():
                if offset in indices:
                    for camera, key in zip(cameras, keys, strict=True):
                        stats.update(
                            key,
                            image_rgb(episode, camera, row[key]),
                            source=(
                                "host_jpeg" if row[key].endswith(".jpg") else "pre_encoding_png"
                            ),
                        )
                offset += 1
        if offset != length:
            raise ValueError("Image statistics row count differs from episode length")
    return stats


def sample_video_images(container, indices, *, offset=0):
    """Legacy fallback by frame order, without additional timestamp validation."""
    indices = set(indices)
    remaining = len(indices)
    if not remaining:
        return
    for index, frame in enumerate(container.decode(video=0)):
        if index - offset in indices:
            yield frame.to_ndarray(format="rgb24")
            remaining -= 1
            if not remaining:
                return
    raise ValueError("Missing video frames for image statistics")
