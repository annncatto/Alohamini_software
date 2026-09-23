"""Local training CLI with the familiar LeRobot dataset/policy option names."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

from alohamini.policies.act.configuration_act import ACTConfig
from alohamini.policies.am_act.configuration_am_act import AMACTConfig


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
        description="Local ACT/AM-ACT training on native or AlohaMini v3 data"
    )
    parser.add_argument("--config", "--config_path", type=Path)
    parser.add_argument("--dataset.root", dest="dataset")
    parser.add_argument(
        "--dataset.repo_id", dest="dataset_repo_id", help="Local label only; never downloads"
    )
    parser.add_argument("--dataset.episodes", dest="train_episodes", type=json.loads)
    parser.add_argument("--dataset.eval_episodes", dest="val_episodes", type=json.loads)
    parser.add_argument("--policy.type", dest="policy", choices=("act", "am_act"))
    parser.add_argument("--policy.device", dest="device")
    parser.add_argument("--policy.push_to_hub", type=boolean)
    parser.add_argument("--wandb.enable", type=boolean)
    parser.add_argument("--output_dir")
    parser.add_argument("--run_name")
    parser.add_argument("--state", help="auto (default), none, or native state groups")
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
    ):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--optimizer.grad_clip_norm", dest="grad_clip_norm", type=float)
    for name in ("resume", "cudnn_deterministic", "persistent_workers"):
        parser.add_argument(f"--{name}", type=boolean)
    parser.add_argument(
        "--background", action="store_true", help="Detach with dedicated log and PID files"
    )
    model_fields = {f.name for cls in (ACTConfig, AMACTConfig) for f in fields(cls)} - {
        "input_features",
        "output_features",
        "normalization_mapping",
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
        elif setting is not None:
            cfg[name] = setting
    cfg["model"] = model
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
