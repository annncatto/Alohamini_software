"""Command-line configuration for local policy training."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

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
        "--policy.path", dest="pretrained_path", help="Local pretrained base weights"
    )
    parser.add_argument("--policy.device", dest="device")
    parser.add_argument("--policy.push_to_hub", type=boolean)
    parser.add_argument("--wandb.enable", type=boolean)
    parser.add_argument("--output_dir")
    parser.add_argument("--run_name")
    parser.add_argument("--state", help="auto (default), none, or comma-separated state groups")
    parser.add_argument("--mixed_precision", choices=("none", "bfloat16", "float16"))
    parser.add_argument("--distributed_backend", choices=("ddp", "fsdp2"))
    parser.add_argument("--drop_last", type=boolean)
    parser.add_argument("--cameras", type=json.loads)
    parser.add_argument("--image_size", type=json.loads)
    for name in (
        "steps",
        "batch_size",
        "save_freq",
        "log_freq",
        "eval_steps",
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
    parser.add_argument(
        "--background", action="store_true", help="Detach with dedicated log and PID files"
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
    return cfg, background
