import logging

import pytest
import torch
from test_native_learning import model_options
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.loading import loader_options, make_loader
from alohamini.learning.policy import make_policy, make_processor
from alohamini.learning.validation import evaluate_loss, loss_denominators
from alohamini.policies.act.adapter import loss_counts


class Targets:
    """Deliberately unequal valid lengths and a final partial batch."""

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {
            "action": torch.full((3, 2), float(index + 1)),
            "action_is_pad": torch.arange(3) >= (3 - index),
        }

    action_metadata = __getitem__


def test_weighted_loss_invariant_to_batch_partition_and_worker_reuse(caplog):
    samples = Targets()
    totals = loss_denominators(samples, loss_counts)
    assert totals == {"_reconstruction_weight": 6, "_kl_weight": 3}

    def forward(batch):
        values = batch["action"]
        valid = ~batch["action_is_pad"]
        reconstruction = (values * valid[..., None]).sum() / (valid.sum() * 2)
        kl = values[:, 0, 0].square().mean()
        return reconstruction * batch["_reconstruction_weight"] + kl * batch["_kl_weight"], {}

    caplog.set_level(logging.INFO, logger="alohamini.learning.validation")
    for batch_size, workers in ((1, 0), (2, 0), (3, 2)):
        loader = make_loader(samples, batch_size=batch_size, num_workers=workers, prefetch_factor=2)
        first = evaluate_loss(forward, lambda b: b, loader, loss_counts, totals, device="cpu")
        iterator = loader._iterator
        second = evaluate_loss(forward, lambda b: b, loader, loss_counts, totals, device="cpu")
        assert first == pytest.approx(10 / 6 + 14 / 3)
        assert second == first
        if workers:
            assert loader._iterator is iterator
    assert "valid_action_steps:6" in caplog.text
    assert "data_s:" in caplog.text and "compute_s:" in caplog.text


def test_loader_inherits_and_overrides_settings():
    cfg = dict(
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        eval_num_workers=2,
        eval_prefetch_factor=2,
    )
    assert loader_options(cfg)["num_workers"] == 4
    options = loader_options(cfg, evaluation=True)
    assert options == dict(num_workers=2, prefetch_factor=2, persistent_workers=True)
    loader = make_loader(Targets(), device="cuda", **options)
    assert loader.pin_memory and loader.num_workers == 2 and loader.prefetch_factor == 2
    zero = make_loader(Targets(), num_workers=0)
    assert zero.prefetch_factor is None and not zero.persistent_workers
    with pytest.raises(ValueError, match="num_workers"):
        make_loader(Targets(), num_workers=-1)


def test_denominators_use_real_windows_without_decoding(recording, monkeypatch):
    samples = AlohaMiniDataset(recording, episodes=[0], state="none", chunk_size=3)

    def fail(*args, **kwargs):
        raise AssertionError("Denominator prepass decoded an image")

    monkeypatch.setattr("alohamini.learning.data.image_rgb", fail)
    counts = loss_denominators(samples, loss_counts)
    assert counts == {"_reconstruction_weight": 9, "_kl_weight": 4}


@pytest.mark.parametrize("kind", ["act", "am_act"])
def test_actual_act_validation_loss_has_no_batch_partition_bias(recording, kind):
    torch.set_num_threads(2)
    samples = AlohaMiniDataset(
        recording, episodes=[0], state="none", chunk_size=3, image_size=(32, 32)
    )
    model = make_policy(kind, model_options(state=False)).eval()
    processor = make_processor(model, samples.statistics(), "cpu")
    totals = loss_denominators(samples, loss_counts)
    values = []
    with torch.no_grad():
        for size in (1, 3, 4):
            values.append(
                evaluate_loss(
                    model,
                    processor,
                    make_loader(samples, batch_size=size),
                    loss_counts,
                    totals,
                    device="cpu",
                )
            )
    assert values == pytest.approx([values[0]] * 3, rel=1e-5)
