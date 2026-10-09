"""Command-line configuration for local policy training."""

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

from alohamini.learning.checkpoint import resolve_pretrained
from alohamini.learning.loading import resolve_data_pipeline
from alohamini.policies.registry import ALGORITHMS, algorithm


def boolean(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("Expected true or false")
    return value.lower() == "true"


def value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_training_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Local policy training on AlohaMini datasets and their LeRobot v3 exports"
    )
    parser.add_argument("--config", "--config_path", type=Path)
    parser.add_argument("--dataset.root", dest="dataset")
    parser.add_argument(
        "--dataset.repo_id", dest="dataset_repo_id", help="Local label only; never downloads"
    )
    parser.add_argument("--dataset.episodes", dest="train_episodes", type=json.loads)
    parser.add_argument("--dataset.eval_episodes", dest="val_episodes", type=json.loads)
    parser.add_argument("--policy.type", dest="policy", choices=tuple(ALGORITHMS))
    parser.add_argument(
        "--policy.path",
        dest="pretrained_path",
        help="Native checkpoint model directory or algorithm-specific local base weights",
    )
    parser.add_argument("--policy.device", dest="device")
    parser.add_argument("--policy.push_to_hub", type=boolean)
    parser.add_argument("--wandb.enable", type=boolean)
    parser.add_argument("--output_dir")
    parser.add_argument("--run_name")
    parser.add_argument("--state", help="auto (default), none, or comma-separated state groups")
    parser.add_argument(
        "--stats", help="Prepared policy statistics JSON; omitted: fit once before training"
    )
    parser.add_argument("--normalization", choices=("dataset", "checkpoint"))
    parser.add_argument("--mixed_precision", choices=("none", "bfloat16", "float16"))
    parser.add_argument("--distributed_backend", choices=("ddp", "fsdp2"))
    parser.add_argument("--drop_last", type=boolean)
    parser.add_argument("--cameras", type=json.loads)
    parser.add_argument("--image_size", type=json.loads)
    parser.add_argument("--video_backend", choices=("pyav", "torchcodec"))
    parser.add_argument("--return_uint8", type=boolean)
    for name in (
        "steps",
        "batch_size",
        "save_freq",
        "log_freq",
        "eval_steps",
        "eval_batch_size",
        "eval_num_workers",
        "eval_prefetch_factor",
        "eval_log_freq",
        "video_cache_size",
        "camera_workers",
        "num_workers",
        "prefetch_factor",
        "seed",
        "cpu_threads",
        "gradient_accumulation_steps",
        "num_processes",
    ):
        parser.add_argument(f"--{name}", type=int)
    for name in (
        "type",
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "grad_clip_norm",
        "momentum",
        "dampening",
        "nesterov",
    ):
        parser.add_argument(f"--optimizer.{name}", type=value, default=argparse.SUPPRESS)
    for name in ("type", "warmup_steps", "decay_steps", "decay_lr"):
        parser.add_argument(f"--scheduler.{name}", type=value, default=argparse.SUPPRESS)
    for name in ("resume", "cudnn_deterministic", "persistent_workers", "deterministic_algorithms"):
        parser.add_argument(f"--{name}", type=boolean)
    parser.add_argument("--eval_persistent_workers", type=boolean)
    launch = parser.add_mutually_exclusive_group()
    launch.add_argument(
        "--background",
        dest="background",
        action="store_true",
        default=None,
        help="Detach with dedicated log and PID files (default)",
    )
    launch.add_argument(
        "--foreground",
        dest="background",
        action="store_false",
        help="Run in the current process and display training output directly",
    )
    model_fields = {f.name for name in ALGORITHMS for f in fields(algorithm(name).config_class)} - {
        "input_features",
        "output_features",
        "normalization_mapping",
        "device",
    }
    for name in sorted(model_fields):
        parser.add_argument(f"--policy.{name}", type=value, default=argparse.SUPPRESS)
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config")
    background = args.pop("background")
    if background is None:
        # External torchrun already owns the worker lifecycle; never detach each rank.
        background = "LOCAL_RANK" not in os.environ
    elif background and "LOCAL_RANK" in os.environ:
        parser.error("torchrun workers must run in the foreground; omit --background")
    for name in ("policy.push_to_hub", "wandb.enable"):
        if args.pop(name):
            parser.error(f"--{name}=true is not supported by the local trainer")
    cfg = json.loads(config_path.expanduser().read_text()) if config_path else {}
    if not isinstance(cfg, dict):
        parser.error("Training config must be a JSON object")
    model = dict(cfg.get("model", {}))
    for name, setting in args.items():
        if name.startswith("policy."):
            model[name.removeprefix("policy.")] = setting
        elif name.startswith(("optimizer.", "scheduler.")):
            section, key = name.split(".", 1)
            cfg[section] = {**(cfg.get(section) or {}), key: setting}
        elif setting is not None:
            cfg[name] = setting
    cfg["model"] = model
    cfg = resolve_pretrained(cfg)
    cfg = algorithm(cfg.get("policy", "act")).apply_preset(cfg)
    defaults = dict(
        policy="act",
        device="cuda",
        state="auto",
        steps=100_000,
        batch_size=8,
        save_freq=20_000,
        log_freq=200,
        seed=1000,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
    )
    for name, setting in defaults.items():
        cfg.setdefault(name, setting)
    if not cfg.get("dataset"):
        parser.error("--dataset.root (or dataset in --config) is required")
    if not cfg.get("output_dir") and not cfg.get("run_name"):
        parser.error("--output_dir or --run_name is required")
    if cfg.get("resume"):
        if config_path is not None and config_path.name == "train_config.json":
            cfg["_checkpoint"] = str(config_path.expanduser().resolve().parent.parent)
        elif not cfg.get("_checkpoint"):
            parser.error(
                "Resume requires --config_path=<checkpoint>/pretrained_model/train_config.json"
            )
    return resolve_data_pipeline(cfg), background
