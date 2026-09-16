# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect a local Parquet episode and optionally select rows near a timestamp."""

import argparse
import math


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a frames.parquet file")
    parser.add_argument("--timestamp", type=float, help="Frame-index timestamp in seconds")
    parser.add_argument("--tolerance", type=float, default=0.001, help="Seconds (default: 0.001)")
    parser.add_argument("--limit", type=int, default=5, help="Maximum rows to print")
    args = parser.parse_args(argv)
    if not math.isfinite(args.tolerance) or args.tolerance < 0 or args.limit < 1:
        parser.error("tolerance must be finite and non-negative; limit must be positive")
    if args.timestamp is not None and not math.isfinite(args.timestamp):
        parser.error("timestamp must be finite")
    import pyarrow.parquet as pq

    dataset = pq.ParquetFile(args.path)
    print("columns:", dataset.schema_arrow.names)
    print("rows:", dataset.metadata.num_rows)
    matched = printed = 0
    for batch in dataset.iter_batches(batch_size=128):
        for row in batch.to_pylist():
            if args.timestamp is None or abs(row["timestamp"] - args.timestamp) <= args.tolerance:
                matched += 1
                if printed < args.limit:
                    print(row)
                    printed += 1
        if args.timestamp is None and printed >= args.limit:
            break
    if args.timestamp is not None:
        print("Matched rows:", matched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
