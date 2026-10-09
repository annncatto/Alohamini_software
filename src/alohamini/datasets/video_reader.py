"""Bounded, process-local video readers with exact dataset frame matching."""

import os
from collections import OrderedDict
from pathlib import Path
from threading import RLock


class _Reader:
    def __init__(self, path):
        import av

        self.lock = RLock()
        self.path = path
        self.container = av.open(str(path))
        try:
            self.stream = self.container.streams.video[0]
            # DataLoader provides process parallelism; avoid a decoder thread pool per camera.
            self.stream.codec_context.thread_count = 1
            self.rate = self.stream.average_rate
            if not self.rate or not self.stream.time_base:
                raise ValueError("Video must have a fixed frame rate and timestamps")
        except Exception:
            self.container.close()
            raise
        self.frames = None
        self.index = -1
        self.rgb = None

    def close(self):
        self.frames = self.rgb = None
        self.container.close()

    def read(self, target):
        if target == self.index and self.rgb is not None:
            return self.rgb.copy()
        # Continue nearby sequential reads; seek for random/backward requests.
        if self.frames is None or not self.index < target <= self.index + 32:
            self.container.seek(
                int(target / self.rate / self.stream.time_base), stream=self.stream, backward=True
            )
            self.frames = iter(self.container.decode(self.stream))
        for frame in self.frames:
            if frame.pts is None:
                raise ValueError("Video frame has no timestamp")
            position = frame.pts * frame.time_base * self.rate
            index = round(position)
            if abs(position - index) > 0.01:
                raise ValueError("Video frame timestamp is off the dataset timeline")
            if index == target:
                self.index = index
                self.rgb = frame.to_ndarray(format="rgb24")
                return self.rgb.copy()
            if index > target:
                break
        raise ValueError(f"Missing video frame {target}: {self.path}")


class _TorchCodecReader:
    def __init__(self, path):
        from torchcodec.decoders import VideoDecoder

        self.lock = RLock()
        self.path = path
        # Match LeRobot's backend defaults, but enforce our frame-index contract.
        self.decoder = VideoDecoder(path, seek_mode="approximate", dimension_order="NHWC")
        self.rate = self.decoder.metadata.average_fps
        if not self.rate or self.rate <= 0:
            self.close()
            raise ValueError("Video must have a fixed frame rate and timestamps")

    def read(self, target):
        try:
            frame = self.decoder.get_frame_at(target)
        except IndexError as exc:
            raise ValueError(f"Missing video frame {target}: {self.path}") from exc
        position = frame.pts_seconds * self.rate
        if abs(position - target) > 0.01:
            raise ValueError("Video frame timestamp is off the dataset timeline")
        return frame.data.numpy().copy()

    def close(self):
        # Releasing the decoder also releases its owned local-file handle.
        self.decoder = None


class VideoFrameCache:
    """LRU containers, retaining at most one RGB frame per entry, never shared across PIDs.

    Size zero selects the original open/seek/close reader for diagnostics.
    Pickling drops live handles; a forked process closes inherited copies before use.
    """

    def __init__(self, max_size=8, *, backend="pyav"):
        if type(max_size) is not int or max_size < 0:
            raise ValueError("video_cache_size must be a nonnegative integer")
        self.max_size = max_size
        if backend not in ("pyav", "torchcodec"):
            raise ValueError("video_backend must be pyav or torchcodec")
        self.backend = backend
        self.pid = os.getpid()
        self.entries = OrderedDict()
        self.lock = RLock()
        self.hits = self.misses = 0

    def __getstate__(self):
        return {"max_size": self.max_size, "backend": self.backend}

    def __setstate__(self, state):
        self.__init__(**state)

    def close(self):
        with self.lock:
            while self.entries:
                _, (_, reader) = self.entries.popitem(last=False)
                with reader.lock:
                    reader.close()

    def __del__(self):
        if hasattr(self, "entries"):
            self.close()

    def prepare_process(self):
        """Call before starting per-camera threads in a newly forked worker."""
        if self.pid != os.getpid():
            self.lock = RLock()
            for _, reader in self.entries.values():
                reader.lock = RLock()
            self.close()
            self.pid = os.getpid()
            self.hits = self.misses = 0

    def read(self, path, target):
        if type(target) is not int or target < 0:
            raise ValueError("Video frame index must be a nonnegative integer")
        if not self.max_size:
            if self.backend == "torchcodec":
                reader = _TorchCodecReader(path)
                try:
                    return reader.read(target)
                finally:
                    reader.close()
            from alohamini.datasets.images import video_rgb

            return video_rgb(path, target)
        self.prepare_process()
        path = Path(path).resolve()
        stamp = path.stat()
        identity = (stamp.st_dev, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns)
        with self.lock:
            entry = self.entries.pop(path, None)
            if entry is not None and entry[0] != identity:
                with entry[1].lock:
                    entry[1].close()
                entry = None
            if entry is None:
                self.misses += 1
                if len(self.entries) >= self.max_size:
                    _, (_, reader) = self.entries.popitem(last=False)
                    with reader.lock:
                        reader.close()
                reader_class = _Reader if self.backend == "pyav" else _TorchCodecReader
                entry = (identity, reader_class(path))
            else:
                self.hits += 1
            self.entries[path] = entry
            # Serialize requests for a single container, not different cameras.
            entry[1].lock.acquire()
        try:
            try:
                return entry[1].read(target)
            finally:
                entry[1].lock.release()
        except Exception:
            with self.lock:
                if self.entries.get(path) is entry:
                    self.entries.pop(path)
                    with entry[1].lock:
                        entry[1].close()
            raise
