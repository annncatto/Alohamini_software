from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path

import zmq

from alohamini.paths import WorkspacePaths


def record(endpoint: str, path: Path) -> None:
    with zmq.Context() as context, context.socket(zmq.SUB) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.connect(endpoint)
        stop = False

        def request_stop(_signum, _frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        last_flush = time.monotonic()
        with path.open("a", encoding="utf-8") as stream:
            while not stop:
                if not socket.poll(100):
                    continue
                payload = json.loads(socket.recv_string())
                stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
                now = time.monotonic()
                if now - last_flush >= 1.0:
                    stream.flush()
                    last_flush = now


def replay(endpoint: str, path: Path, speed: float) -> None:
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("speed must be positive")
    with zmq.Context() as context, context.socket(zmq.PUB) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(endpoint)
        previous_source_ns = None
        time.sleep(0.2)  # PUB/SUB slow-joiner allowance.
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                payload = json.loads(line)
                source_ns = int(payload.get("monotonic_ns", 0))
                if previous_source_ns is not None and source_ns > previous_source_ns:
                    time.sleep(min(0.25, (source_ns - previous_source_ns) / 1.0e9 / speed))
                payload["monotonic_ns"] = time.monotonic_ns()
                payload["replay"] = True
                socket.send_string(json.dumps(payload, separators=(",", ":")))
                previous_source_ns = source_ns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record or replay timestamped raw Joy-Con ZMQ samples."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("path", type=Path)
    record_parser.add_argument("--endpoint", default="tcp://127.0.0.1:5567")
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("path", type=Path)
    replay_parser.add_argument("--endpoint", default="tcp://127.0.0.1:5568")
    replay_parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    path = args.path.expanduser()
    if not path.is_absolute():
        path = WorkspacePaths().logs / path
    if args.command == "record":
        path.parent.mkdir(parents=True, exist_ok=True)
        record(args.endpoint, path)
    else:
        replay(args.endpoint, path, args.speed)


if __name__ == "__main__":
    main()
