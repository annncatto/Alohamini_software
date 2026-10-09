"""Same pixels through each model's byte and legacy floating-image input paths."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alohamini.learning.loading import resolve_data_pipeline
from alohamini.learning.processor import Processor, image_tensor
from alohamini.policies.registry import ALGORITHMS, algorithm


class Tokenizer:
    def __call__(self, tasks, **kwargs):
        assert tasks == ["pick\n", "pick\n"]
        return {
            "input_ids": torch.tensor([[1, 2], [1, 2]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        }

    def tokenize(self, prompt, state):
        assert prompt == "pick" and state.shape == (18,)
        return np.array([1, 2]), np.array([True, True])


def assert_identical(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, SimpleNamespace):
        assert_identical(vars(a), vars(b))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_identical(a[key], b[key])
    else:
        assert a == b


def make_processor(kind, shape):
    features = dict(
        input_features={
            "observation.images.forward": {"type": "VISUAL", "shape": shape},
            "observation.state": {"type": "STATE", "shape": (18,)},
        },
        output_features={"action": {"type": "ACTION", "shape": (18,)}},
    )
    options = {**features, **({"crop_shape": None} if kind == "diffusion" else {})}
    config = algorithm(kind).config_class(**options)
    stats = {}
    for key, feature in {**config.input_features, **config.output_features}.items():
        shape = (3, 1, 1) if feature.type == "VISUAL" else (18,)
        stats[key] = {
            name: torch.full(shape, value).tolist()
            for name, value in [("mean", 0.4), ("std", 0.2), ("min", -2.0), ("max", 3.0)]
        }
    if kind == "smolvla":
        from alohamini.policies.smolvla.processor_smolvla import SmolVLAProcessor

        processor = SmolVLAProcessor(config, stats, Tokenizer())
    elif kind == "fastwam":
        from alohamini.policies.fastwam.adapter import FastWAMProcessor

        processor = FastWAMProcessor(config, stats, "cpu")
    elif kind == "pi05":
        from alohamini.policies.pi05.adapter import PI05BatchProcessor
        from alohamini.policies.pi05.processor_pi05 import PI05Processor

        processor = PI05BatchProcessor.__new__(PI05BatchProcessor)
        processor.config, processor.device = config, "cpu"
        processor.transform = PI05Processor(
            {k: {"q01": [-2.0] * 18, "q99": [3.0] * 18} for k in ("state", "actions")},
            Tokenizer(),
        )
    else:
        processor = Processor.from_config(config, stats)
    return processor, config


@pytest.mark.parametrize("kind", ALGORITHMS)
@pytest.mark.parametrize("size", [(24, 32), (16, 16)])
def test_model_specific_preprocessing_is_identical_for_compact_delivery(kind, size):
    assert resolve_data_pipeline({"policy": kind})["return_uint8"] is True
    rgb = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    key = "observation.images.forward"
    legacy = image_tensor(rgb, size)
    compact = image_tensor(rgb, size, return_uint8=True)
    assert compact.dtype == (torch.uint8 if size == (24, 32) else torch.float32)
    processor, config = make_processor(kind, (3, *size))
    if kind in ("diffusion", "fastwam"):
        legacy = legacy[None, None].repeat(2, 3, 1, 1, 1)
        compact = compact[None, None].repeat(2, 3, 1, 1, 1)
    else:
        legacy = legacy[None].repeat(2, 1, 1, 1)
        compact = compact[None].repeat(2, 1, 1, 1)
    extras = {
        "observation.state": torch.ones(2, 18),
        "action": torch.ones(2, 3, 18),
        "action_is_pad": torch.tensor([[False, False, True]] * 2),
    }
    if kind in ("smolvla", "fastwam", "pi05"):
        extras["task"] = ["pick", "pick"]
    if kind in ("diffusion", "fastwam"):
        extras[key + "_is_pad"] = torch.tensor([[True, False, False]] * 2)
    original_bytes = compact.clone()
    expected = processor({key: legacy, **extras})
    actual = processor({key: compact, **extras})
    assert_identical(actual, expected)
    assert torch.equal(compact, original_bytes)
    if kind == "smolvla":
        from alohamini.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        images, masks = SmolVLAPolicy.prepare_images(SimpleNamespace(config=config), actual)
        assert all(image.min() >= -1 and image.max() <= 1 for image in images)
        assert all(mask.all() for mask in masks)
    if kind == "pi05":
        assert actual["observation"].images["base_0_rgb"].shape == (2, 3, 224, 224)
        for image in actual["observation"].images.values():
            assert image.min() >= -1 and image.max() <= 1


@pytest.mark.parametrize("kind", ["smolvla", "pi05"])
@pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.1])
def test_float_input_validation_is_not_relaxed(kind, bad):
    processor, _ = make_processor(kind, (3, 16, 16))
    batch = {
        "observation.state": torch.zeros(2, 18),
        "task": ["pick", "pick"],
        "observation.images.forward": torch.full((2, 3, 16, 16), bad),
    }
    with pytest.raises(ValueError, match="expected.*RGB"):
        processor(batch)
