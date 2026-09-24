"""OpenPI PI0.5 dimensions and defaults, without a JAX runtime dependency."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def get_config(variant):
    # Dimensions from OpenPI models/gemma.py:get_config.
    configs = {
        "dummy": GemmaConfig(64, 4, 128, 8, 1, 16),
        "gemma_300m": GemmaConfig(1024, 18, 4096, 8, 1, 256),
        "gemma_2b": GemmaConfig(2048, 18, 16384, 8, 1, 256),
    }
    if variant not in configs:
        raise ValueError(f"Unsupported PyTorch Gemma variant: {variant}")
    return configs[variant]


@dataclass(frozen=True)
class PI05Config:
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200
    pi05: bool = True
    discrete_state_input: bool = True
    # Explicit compilation is optional; eager mode avoids compilation during loading.
    pytorch_compile_mode: str | None = None

    def __post_init__(self):
        if not self.pi05 or not self.discrete_state_input:
            raise ValueError("PI0.5 requires discrete state input and timestep adaRMSNorm")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("dtype must be float32 or bfloat16")
        if min(self.action_dim, self.action_horizon, self.max_token_len) < 1:
            raise ValueError("Model dimensions must be positive")
        get_config(self.paligemma_variant)
        get_config(self.action_expert_variant)
        if self.pytorch_compile_mode not in (
            None,
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ):
            raise ValueError("Unsupported PyTorch compile mode")
