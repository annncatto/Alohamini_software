"""Native component contracts and real, reduced-size OpenPI integration; no robot."""

import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_native_learning import recording as recording

from alohamini.learning.policy import NativePolicy, make_policy
from alohamini.policies.registry import ALGORITHMS, algorithm


def tiny_pi05(monkeypatch):
    from transformers import PaliGemmaConfig

    from alohamini.policies.pi05 import gemma_pytorch
    from alohamini.policies.pi05.modeling_pi05 import PI0Pytorch

    def vision_config():
        config = PaliGemmaConfig()
        config.vision_config.hidden_size = 32
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 4
        config.vision_config.patch_size = 112
        return config

    monkeypatch.setattr(gemma_pytorch, "CONFIG_MAPPING", {"paligemma": vision_config})
    original = PI0Pytorch.__init__

    def initialize(self, config):
        original(self, config)
        self.paligemma_with_expert.paligemma.model.multi_modal_projector.linear = torch.nn.Linear(
            32, 64
        )

    monkeypatch.setattr(PI0Pytorch, "__init__", initialize)


def test_components_declare_distinct_reductions_and_windows():
    batch = {
        "action": torch.zeros(2, 3, 18),
        "action_is_pad": torch.tensor([[False] * 3, [False, True, True]]),
    }
    assert algorithm("act").loss_counts(batch) == {"_reconstruction_weight": 4, "_kl_weight": 2}
    assert algorithm("smolvla").loss_counts(batch) == {"_reconstruction_weight": 4}
    assert algorithm("pi05").loss_counts(batch) == {"_loss_weight": 2}
    assert algorithm("pi05").sample_spec({"network": {"action_horizon": 2}}) == {
        "delta_indices": {"action": [0, 1]},
        "include_task": True,
    }
    assert set(ALGORITHMS) >= {"act", "am_act", "smolvla", "pi05"}


def test_feature_config_has_no_act_architecture_assumptions():
    from alohamini.policies.configuration import PolicyConfig, PolicyFeature

    cfg = PolicyConfig(output_features={"action": PolicyFeature("ACTION", (18,))})
    assert cfg.action_feature.shape == (18,)


@pytest.mark.parametrize("warmup", [0, 1000])
def test_pi05_schedule_matches_openpi_pytorch(warmup):
    from alohamini.learning.optim import make_scheduler

    peak, end, decay = 2.5e-5, 2.5e-6, 30000
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=peak)
    scheduler = make_scheduler(
        {
            "steps": 10,
            "optimizer": {"lr": peak},
            "scheduler": {
                "type": "warmup_cosine",
                "warmup_steps": warmup,
                "decay_steps": decay,
                "decay_lr": end,
            },
        },
        optimizer,
    )
    for step in (0, 1, 500, 1000, 15000, 30000, 40000):
        if step < warmup:
            initial = peak / (warmup + 1)
            expected = initial + (peak - initial) * step / warmup
        else:
            progress = min(1.0, (step - warmup) / max(1, decay - warmup))
            expected = end + (peak - end) * (1 + math.cos(math.pi * progress)) / 2
        assert peak * scheduler.lr_lambdas[0](step) == pytest.approx(expected)


def test_pi05_training_resume_checkpoint_and_action_context(recording, tmp_path, monkeypatch):
    import sentencepiece as spm
    from safetensors.torch import load_file, save_model

    from alohamini.learning.data import AlohaMiniDataset
    from alohamini.learning.train_config import parse_training_args

    torch.set_num_threads(2)
    tiny_pi05(monkeypatch)
    tokenizer = tmp_path / "tokenizer.model"
    buffer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["Task: pick object, State: 0 1 2 3 4 5 6 7 8 9; Action: "] * 20),
        model_writer=buffer,
        vocab_size=40,
        hard_vocab_limit=False,
    )
    tokenizer.write_bytes(buffer.getvalue())
    recipe = algorithm("pi05")
    network = dict(
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=2,
        max_token_len=48,
    )
    model_options = dict(
        network=network, tokenizer_path=str(tokenizer), n_action_steps=2, num_inference_steps=2
    )
    data = AlohaMiniDataset(
        recording, episodes=[0], image_size=(32, 32), **recipe.sample_spec(model_options)
    )
    options = {
        **model_options,
        "input_features": data.input_features,
        "output_features": data.output_features,
    }
    base = tmp_path / "base.safetensors"
    save_model(make_policy("pi05", options).network, base)
    cfg = dict(
        dataset=str(recording),
        policy="pi05",
        pretrained_path=str(base),
        model=model_options,
        device="cpu",
        steps=2,
        batch_size=3,
        gradient_accumulation_steps=2,
        train_episodes=[0],
        val_episodes=[1],
        image_size=[32, 32],
        num_workers=0,
        save_freq=1,
        eval_steps=1,
        deterministic_algorithms=True,
        seed=32,
        cpu_threads=2,
    )

    def run(settings, label, stop=False):
        path, log, pid = (tmp_path / f"{label}{suffix}" for suffix in (".json", ".log", ".pid"))
        path.write_text(json.dumps(settings))
        with log.open("wb") as stream:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(path)],
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "PI05_TEST_STOP": str(int(stop))},
            )
        pid.write_text(str(process.pid))
        try:
            assert process.wait(timeout=120) == 0, log.read_text()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

    run({**cfg, "output_dir": str(tmp_path / "full")}, "full")
    run({**cfg, "output_dir": str(tmp_path / "resumed")}, "interrupted", stop=True)
    resumed_cfg, _ = parse_training_args(
        [
            "--config_path",
            str(tmp_path / "resumed/checkpoint/train_config.json"),
            "--resume=true",
        ]
    )
    run(resumed_cfg, "resume")
    full, resumed = (tmp_path / name / "checkpoint" for name in ("full", "resumed"))
    a, b = load_file(full / "model.safetensors"), load_file(resumed / "model.safetensors")
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    states = [
        torch.load(path.resolve().parent / "training_state/state.pt", weights_only=True)
        for path in (full, resumed)
    ]

    def identical(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                identical(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for left, right in zip(a, b, strict=True):
                identical(left, right)
        else:
            assert a == b

    identical(*states)
    policy = NativePolicy(resumed, task="pick")
    assert policy.predict(data[0]).shape == (2, 18)
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=1)))
    prepared = policy.processor(batch)
    first = policy.model.select_action(prepared)
    anchor = policy.model.action_context.clone()
    shifted = {**prepared, "_action_context": prepared["_action_context"] + 100}
    second = policy.model.select_action(shifted)
    torch.testing.assert_close(policy.model.action_context, anchor)
    assert policy.execution_action(first, context=anchor).shape == (1, 18)
    assert policy.execution_action(second, context=anchor).shape == (1, 18)
    with pytest.raises(ValueError, match="original observation state"):
        policy.execution_action(second)
    assert (
        json.loads((tmp_path / "resumed/offline-evaluation.json").read_text())["valid_action_steps"]
        > 0
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    with pytest.MonkeyPatch.context() as patch:
        tiny_pi05(patch)
        import alohamini.learning.train as training

        if os.environ.get("PI05_TEST_STOP") == "1":
            save = training.save_training_checkpoint

            def save_and_stop(*args, **kwargs):
                save(*args, **kwargs)
                raise SystemExit(0)

            patch.setattr(training, "save_training_checkpoint", save_and_stop)
        training.train(json.loads(Path(sys.argv[-1]).read_text()))
