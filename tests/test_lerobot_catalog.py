"""Optional LeRobot policy migration checks; no weights or hardware downloads."""

import ast
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "integrations/lerobot/src/alohamini_lerobot"
MANIFEST = json.loads((PACKAGE / "policies/sources.json").read_text())


class AlgorithmTree(ast.NodeTransformer):
    """Ignore relocation and registration names, not executable model operations."""

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace("alohamini_lerobot.policies", "lerobot.policies")
            if node.value.startswith("alohamini_"):
                node.value = node.value.removeprefix("alohamini_")
        return node


def algorithm_hash(tree):
    def canonical(node):
        if isinstance(node, ast.AST):
            return {
                "node": type(node).__name__,
                **{
                    key: canonical(value)
                    for key, value in ast.iter_fields(node)
                    if value is not None and value != []
                },
            }
        if isinstance(node, list):
            return [canonical(value) for value in node]
        return node

    value = json.dumps(
        canonical(tree),
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: {"literal": type(value).__name__, "value": repr(value)},
    )
    return hashlib.sha256(value.encode()).hexdigest()


def test_copied_algorithm_bodies_and_source_inventory():
    """Compare every non-dispatcher module with the recorded source AST digest."""
    assert len(MANIFEST["policies"]) == len(set(MANIFEST["policies"])) == 18
    for name, evidence in MANIFEST["files"].items():
        path = PACKAGE / name
        assert path.is_file(), name
        tree = ast.parse(path.read_text())
        if path.name in {"__init__.py", "factory.py"}:
            continue
        assert algorithm_hash(AlgorithmTree().visit(tree)) == evidence["algorithm_sha256"], name


def test_package_import_does_not_import_torch():
    env = dict(os.environ, PYTHONPATH=str(PACKAGE.parent))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import alohamini_lerobot.policies; import sys; assert 'torch' not in sys.modules",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def factory():
    pytest.importorskip("lerobot")
    pytest.importorskip("alohamini_lerobot")
    from alohamini_lerobot.policies import factory

    return factory


def test_config_registry_is_separate(factory):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig as OfficialConfig

    for name in MANIFEST["policies"]:
        config_cls = PreTrainedConfig.get_choice_class("alohamini_" + name)
        assert config_cls.__module__.startswith("alohamini_lerobot.policies.")
    assert PreTrainedConfig.get_choice_class("diffusion") is OfficialConfig
    assert factory.get_policy_class("act").__module__.startswith("lerobot.policies.act.")


def test_unprefixed_policy_does_not_silently_use_official_code(factory):
    with pytest.raises(ValueError, match="alohamini_diffusion"):
        factory.get_policy_class("diffusion")
    with pytest.raises(ValueError, match="alohamini_diffusion"):
        factory.make_policy_config("diffusion")


def tiny_diffusion(factory):
    pytest.importorskip("diffusers")
    from lerobot.configs import FeatureType, PolicyFeature

    return factory.make_policy_config(
        "alohamini_diffusion",
        device="cpu",
        push_to_hub=False,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (18,)),
            "observation.environment_state": PolicyFeature(FeatureType.ENV, (3,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (18,))},
        pretrained_backbone_weights=None,
        horizon=8,
        n_obs_steps=2,
        n_action_steps=3,
        down_dims=(16, 32),
        n_groups=8,
        diffusion_step_embed_dim=16,
        num_train_timesteps=4,
        num_inference_steps=2,
        drop_n_last_frames=0,
    )


def test_diffusion_forward_queue_reset_and_checkpoint(factory, tmp_path):
    import torch

    torch.set_num_threads(2)
    cfg = tiny_diffusion(factory)
    model_cls = factory.get_policy_class(cfg.type)
    model = model_cls(cfg)
    batch = {
        "observation.state": torch.randn(2, 2, 18),
        "observation.environment_state": torch.randn(2, 2, 3),
        "action": torch.randn(2, 8, 18),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
    }
    loss, _ = model(deepcopy(batch))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    model.eval()
    observation = {
        key: value[:, -1] for key, value in batch.items() if key.startswith("observation.")
    }
    assert model.select_action(observation).shape == (2, 18)
    assert len(model._queues["action"]) == 2
    model.reset()
    assert not model._queues["action"]
    destination = tmp_path / "checkpoint"
    model.save_pretrained(destination)
    saved = json.loads((destination / "config.json").read_text())
    assert saved["type"] == "alohamini_diffusion"
    loaded = model_cls.from_pretrained(destination, local_files_only=True, strict=True)
    assert loaded.config.type == cfg.type
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_diffusion_processor_uses_min_max_and_roundtrips(factory, tmp_path):
    import torch

    cfg = tiny_diffusion(factory)
    stats = {
        "observation.state": {"min": torch.full((18,), -10.0), "max": torch.full((18,), 10.0)},
        "action": {"min": torch.full((18,), -10.0), "max": torch.full((18,), 10.0)},
    }
    pre, post = factory.make_pre_post_processors(cfg, dataset_stats=stats)
    batch = {
        "observation.state": torch.full((1, 2, 18), 5.0),
        "action": torch.full((1, 8, 18), 5.0),
    }
    processed = pre(batch)
    torch.testing.assert_close(processed["action"], torch.full((1, 8, 18), 0.5))
    torch.testing.assert_close(post(processed["action"]), batch["action"])
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded_pre, loaded_post = factory.make_pre_post_processors(cfg, pretrained_path=tmp_path)
    torch.testing.assert_close(loaded_post(loaded_pre(batch)["action"]), batch["action"])


def test_no_optional_models_imported_for_config(factory):
    code = (
        "from alohamini_lerobot.policies import factory; import sys; "
        "assert not any(n.startswith('alohamini_lerobot.policies.') "
        "and '.modeling_' in n and '.rtc.' not in n for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_platform_v3_export_provides_history_and_action_padding(factory, tmp_path):
    pytest.importorskip("datasets")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from test_lerobot_policy import training

    _, root = training.__wrapped__(tmp_path)
    cfg = tiny_diffusion(factory)
    offsets = {
        key: [i / 30 for i in cfg.observation_delta_indices]
        for key in ("observation.state", "observation.images.forward")
    }
    offsets["action"] = [i / 30 for i in cfg.action_delta_indices]
    dataset = LeRobotDataset("local/platform_export", root=root, delta_timestamps=offsets)
    sample = dataset[0]
    assert sample["observation.state"].shape == (2, 28)
    assert sample["observation.images.forward"].shape == (2, 3, 32, 32)
    assert sample["action"].shape == (8, 18)
    assert sample["action_is_pad"].shape == (8,)
    assert sample["action_is_pad"][0]
