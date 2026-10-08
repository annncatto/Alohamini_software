"""Versioned real-world SmolVLA recipe, with explicit AlohaMini adaptations."""

from copy import deepcopy

PRESET = {
    "name": "smolvla-realworld-alohamini-v1",
    "paper": "https://arxiv.org/html/2506.01844v1#S4.SS3",
    "implementation": "lerobot==0.6.1",
    "data": {
        "state": "joint_position,base_velocity,lift_height",
        "action": "absolute_joint_position+base_velocity+absolute_lift_height",
        "units": "Recorded AlohaMini coordinates; checkpoint stores field order and units",
        "language": "Per-frame task + trailing newline; right-padded tokenizer, length 48",
        "images": "RGB [0,1]; aspect-preserving left/top padding to 512x512; then [-1,1]",
        "normalization": "Training-only MEAN_STD for state/action, epsilon 1e-8; images IDENTITY",
        "time": "Recorded FPS; action row offsets 0..49; episode boundaries retained",
    },
    "paper_settings": {
        "steps": 200_000,
        "chunk_size": 50,
        "n_action_steps": 50,
        "num_steps": 10,
        "num_vlm_layers": 16,
        "mixed_precision": "bfloat16",
        "compile_model": True,
        "train_expert_only": True,
    },
    "implementation_defaults": {
        "batch_size": 64,
        "optimizer_betas": [0.9, 0.95],
        "optimizer_lr": 1e-4,
        "scheduler_warmup_steps": 1000,
        "scheduler_decay_steps": 30000,
        "scheduler_decay_lr": 2.5e-6,
    },
    "adaptations": [
        "AlohaMini 18-D targets are not the paper's SO100/SO101 coordinates; no Trossen transform",
        "Camera names/order are recorded explicitly, not inferred as paper viewpoint equivalence",
        "Retain recorded FPS and sequence-boundary masks; no automatic temporal resampling",
        "drop_last operates on global batches, not paper episode-tail filtering",
        "Batch 64 and fine-tuning warmup/decay use released-code defaults; paper explicitly gives batch 64 for simulation and warmup 100 for pretraining",
    ],
}


def apply_preset(settings):
    """Explicit user values win; persist resolved settings and departures together."""
    result = deepcopy(settings)
    if result.get("policy") != "smolvla":
        return result
    defaults = {
        "steps": 200_000,
        "batch_size": 64,
        "state": PRESET["data"]["state"],
        "mixed_precision": "bfloat16",
        "drop_last": True,
    }
    model = {
        "chunk_size": 50,
        "n_action_steps": 50,
        "num_steps": 10,
        "num_vlm_layers": 16,
        "freeze_vision_encoder": True,
        "train_expert_only": True,
        "train_state_proj": True,
        "resize_imgs_with_padding": [512, 512],
        "compile_model": True,
        "pad_language_to": "max_length",
        "tokenizer_max_length": 48,
        **{k: v for k, v in PRESET["implementation_defaults"].items() if k != "batch_size"},
    }
    for key, value in defaults.items():
        result.setdefault(key, value)
    supplied = result.setdefault("model", {})
    for key, value in model.items():
        supplied.setdefault(key, value)
    overrides = {k: result[k] for k, v in defaults.items() if result[k] != v}
    overrides.update({f"model.{k}": supplied[k] for k, v in model.items() if supplied[k] != v})
    result["paper_preset"] = {**deepcopy(PRESET), "overrides": overrides}
    return result
