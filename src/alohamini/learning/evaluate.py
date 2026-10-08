"""Offline checkpoint evaluation using the shared dataset and policy interfaces."""

import argparse
import json
from pathlib import Path

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.logging import training_logging
from alohamini.learning.policy import NativePolicy
from alohamini.learning.train import offline_evaluate
from alohamini.learning.train_config import boolean


def evaluate_checkpoint(
    checkpoint,
    dataset,
    *,
    episodes,
    device="cpu",
    batch_size=8,
    num_workers=0,
    prefetch_factor=4,
    persistent_workers=True,
    log_freq=50,
    video_cache_size=8,
):
    """Compare predicted chunks with recorded actions; never connect to a robot."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    policy = NativePolicy(checkpoint, device=device)
    manifest = policy.manifest
    samples = AlohaMiniDataset(
        dataset,
        episodes=episodes,
        **policy.algorithm.sample_spec(manifest["config"], cameras=manifest["cameras"]),
        state=manifest["state"],
        cameras=manifest["cameras"],
        image_size=tuple(manifest["image_size"]),
        video_cache_size=video_cache_size,
    )
    return {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "dataset": str(samples.root),
        "episodes": samples.episodes,
        "samples": len(samples),
        "excluded_samples": samples.excluded,
        "sample_filter": samples.sample_filter,
        "dataset_check": samples.report,
        **offline_evaluate(
            policy,
            samples,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            log_freq=log_freq,
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline policy evaluation; no robot access")
    parser.add_argument("--policy.path", dest="checkpoint", required=True)
    parser.add_argument("--dataset.root", dest="dataset", required=True)
    parser.add_argument("--dataset.episodes", dest="episodes", type=json.loads, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", type=boolean, default=True)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--video_cache_size", type=int, default=8)
    parser.add_argument("--output", type=Path, help="Optional new JSON report; never overwrites")
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    if output is not None:
        output = output.expanduser()
        if output.exists():
            raise FileExistsError(output)
    with training_logging(True):
        report = evaluate_checkpoint(**args)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output is not None:
        with output.open("x") as stream:
            stream.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    main()
