"""Native SmolVLA contracts and small real-transformer parity, without downloads."""

import pytest
import torch
from test_native_learning import recording as _recording

from alohamini.policies.configuration import PolicyFeature
from alohamini.policies.smolvla.configuration_smolvla import SmolVLAConfig


@pytest.fixture
def recording(tmp_path):
    return _recording.__wrapped__(tmp_path)


@pytest.fixture
def pixel_recording(tmp_path, monkeypatch):
    from test_dataset import jpeg

    # Avoid the ordinary 16x24 fixture's mandatory resize to >=32 pixels;
    # this test must exercise actual uint8 delivery all the way to the model.
    monkeypatch.setattr("test_native_learning.jpeg", lambda: jpeg(shape=(32, 32, 3)))
    return _recording.__wrapped__(tmp_path)


@pytest.fixture
def tiny_assets(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        SmolVLMConfig,
        SmolVLMImageProcessor,
        SmolVLMProcessor,
        SmolVLMVideoProcessor,
    )

    config = SmolVLMConfig(
        text_config={
            "model_type": "llama",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 32,
            "pad_token_id": 0,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 16,
            "patch_size": 4,
        },
        image_token_id=31,
        scale_factor=2,
    )
    config.save_pretrained(tmp_path)
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "[PAD]": 0,
                "<fake_token_around_image>": 1,
                "<global-img>": 2,
                "[UNK]": 3,
                "test": 4,
                **{f"token_{i}": i for i in range(5, 31)},
                "<image>": 31,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        extra_special_tokens={
            "fake_image_token": "<fake_token_around_image>",
            "global_image_token": "<global-img>",
            "image_token": "<image>",
        },
    )
    SmolVLMProcessor(
        SmolVLMImageProcessor(), tokenizer, SmolVLMVideoProcessor(), image_seq_len=4
    ).save_pretrained(tmp_path)
    return str(tmp_path)


def options(assets):
    return dict(
        vlm_model_name=assets,
        chunk_size=3,
        n_action_steps=2,
        num_steps=2,
        num_vlm_layers=2,
        expert_width_multiplier=1.0,
        resize_imgs_with_padding=(16, 16),
        max_state_dim=8,
        max_action_dim=8,
        input_features={
            "observation.state": PolicyFeature("STATE", (6,)),
            "observation.images.forward": PolicyFeature("VISUAL", (3, 16, 16)),
        },
        output_features={"action": PolicyFeature("ACTION", (6,))},
    )


def batch():
    return {
        "observation.state": torch.rand(1, 6),
        "observation.images.forward": torch.rand(1, 3, 16, 16),
        "observation.language.tokens": torch.tensor([[3, 4, 0]]),
        "observation.language.attention_mask": torch.tensor([[True, True, False]]),
        "action": torch.rand(1, 3, 6),
        "action_is_pad": torch.tensor([[False, False, True]]),
    }


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_transformer_forward_backward_queue_and_official_parity(tiny_assets, device):
    from alohamini.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.set_num_threads(1)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(7)
    model = SmolVLAPolicy(SmolVLAConfig(**options(tiny_assets))).to(device)
    inputs = {key: value.to(device) for key, value in batch().items()}
    noise, time = torch.randn(1, 3, 8, device=device), torch.tensor([0.4], device=device)
    loss, _ = model(inputs, noise=noise, time=time)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.model.action_out_proj.weight.grad.isfinite().all()
    assert not any(p.requires_grad for p in model.model.vlm_with_expert.vlm.parameters())
    chunk = model.predict_action_chunk(inputs, noise=noise)
    assert chunk.shape == (1, 3, 6) and chunk.isfinite().all()
    torch.testing.assert_close(model.select_action(inputs, noise=noise), chunk[:, 0])
    torch.testing.assert_close(model.select_action(inputs, noise=noise), chunk[:, 1])
    model.reset()
    torch.testing.assert_close(model.select_action(inputs, noise=noise), chunk[:, 0])

    # Reference is optional in native-only environments, never a runtime dependency.
    pytest.importorskip("lerobot")
    from lerobot.configs import FeatureType
    from lerobot.configs import PolicyFeature as OfficialFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as OfficialConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as OfficialPolicy

    opts = options(tiny_assets)
    for field in ("input_features", "output_features"):
        opts[field] = {
            k: OfficialFeature(FeatureType(v.type), v.shape) for k, v in opts[field].items()
        }
    reference = OfficialPolicy(OfficialConfig(**opts)).to(device)
    reference.load_state_dict(model.state_dict(), strict=True)
    ref_loss, _ = reference(inputs, noise=noise, time=time)
    torch.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
    torch.testing.assert_close(
        chunk, reference.predict_action_chunk(inputs, noise=noise), rtol=0, atol=0
    )
    if device == "cuda":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        from alohamini.learning.train import update_policy

        model.zero_grad()
        result = update_policy(model, inputs, optimizer, 10.0, mixed_precision="bfloat16")
        assert result["loss"] >= 0 and result["grad_norm"] > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rtc_config": {}},
        {"adapt_to_pi_aloha": True},
        {"use_cache": False},
        {"n_action_steps": 0},
        {"num_steps": 0},
        {"use_delta_joint_actions_aloha": True},
    ],
)
def test_unsupported_semantics_fail_explicitly(kwargs):
    with pytest.raises((ValueError, NotImplementedError)):
        SmolVLAConfig(**kwargs)


def test_native_and_v3_task_mapping(recording, tmp_path):
    from alohamini.datasets.lerobotv3 import export_lerobot
    from alohamini.learning.data import AlohaMiniDataset

    target = tmp_path / "v3"
    export_lerobot(recording, target)
    for root in (recording, target):
        samples = AlohaMiniDataset(root, episodes=[0], chunk_size=3, include_task=True)
        assert samples[0]["task"] == "test"
        assert samples[0]["action"].shape == (3, 18)
        assert "task" not in samples.input_features


def test_scheduler_matches_official():
    pytest.importorskip("lerobot")
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

    cfg = SmolVLAConfig(scheduler_warmup_steps=3, scheduler_decay_steps=10)
    for steps in (5, 10, 20):
        optimizers = [
            torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=cfg.optimizer_lr)
            for _ in range(2)
        ]
        schedulers = [
            cfg.build_scheduler(optimizers[0], steps),
            CosineDecayWithWarmupSchedulerConfig(
                3, 10, cfg.optimizer_lr, cfg.scheduler_decay_lr
            ).build(optimizers[1], steps),
        ]
        for _ in range(steps):
            assert optimizers[0].param_groups[0]["lr"] == optimizers[1].param_groups[0]["lr"]
            for optimizer, scheduler in zip(optimizers, schedulers, strict=True):
                optimizer.step()
                scheduler.step()


@pytest.mark.parametrize("return_uint8", [False, True])
def test_native_checkpoint_processor_roundtrip_and_offline_eval(
    pixel_recording, tiny_assets, tmp_path, return_uint8
):
    from alohamini.learning.data import AlohaMiniDataset
    from alohamini.learning.policy import NativePolicy, make_policy, make_processor, save_checkpoint
    from alohamini.learning.train import offline_evaluate
    from alohamini.policies.smolvla.processor_smolvla import fit_statistics

    recording = pixel_recording
    samples = AlohaMiniDataset(
        recording,
        episodes=[0],
        chunk_size=3,
        image_size=(32, 32),
        include_task=True,
        return_uint8=return_uint8,
    )
    opts = options(tiny_assets)
    opts.update(
        input_features=samples.input_features,
        output_features=samples.output_features,
        max_state_dim=32,
        max_action_dim=32,
    )
    model = make_policy("smolvla", opts)
    stats = fit_statistics(samples)
    processor = make_processor(model, stats, "cpu")
    batch = next(iter(torch.utils.data.DataLoader(samples, batch_size=2)))
    processed = processor(batch)
    expected_images = batch["observation.images.forward"]
    if return_uint8:
        assert expected_images.dtype == torch.uint8
        expected_images = expected_images.float() / 255
    torch.testing.assert_close(
        processed["observation.images.forward"], expected_images, rtol=0, atol=0
    )
    loss, _ = model(processed)
    loss.backward()
    assert loss.isfinite()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, samples, training={})
    loaded = NativePolicy(checkpoint)
    assert loaded.predict(samples[0]).shape == (3, 18)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[key], rtol=0, atol=0)
    result = offline_evaluate(loaded, samples, batch_size=2)
    assert result["valid_action_steps"] > 0
    assert len(result["mae_by_action"]) == 18
    from alohamini.learning.evaluate import evaluate_checkpoint

    assert evaluate_checkpoint(checkpoint, recording, episodes=[0])["samples"] == len(samples)
    with pytest.raises(ValueError, match="task description"):
        processor({k: v for k, v in batch.items() if k != "task"})


def test_paper_preset_records_user_departures():
    from alohamini.learning.train_config import parse_training_args

    cfg, _ = parse_training_args(
        [
            "--policy.type=smolvla",
            "--dataset.root=/unused",
            "--run_name=test",
            "--batch_size=2",
            "--policy.compile_model=false",
        ]
    )
    assert cfg["steps"] == 200_000
    assert cfg["model"]["chunk_size"] == 50
    assert cfg["model"]["pad_language_to"] == "max_length"
    assert cfg["mixed_precision"] == "bfloat16"
    assert cfg["paper_preset"]["overrides"] == {"batch_size": 2, "model.compile_model": False}


def test_detached_native_smolvla_training_saves_resume_state(
    recording, tiny_assets, tmp_path, monkeypatch
):
    import json
    import shutil
    import time
    from dataclasses import asdict
    from pathlib import Path

    from safetensors.torch import save_model

    from alohamini.learning.policy import NativePolicy, make_policy
    from alohamini.learning.train import launch_training

    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    opts = options(tiny_assets)
    opts["input_features"]["observation.state"] = PolicyFeature("STATE", (18,))
    opts["input_features"]["observation.images.forward"] = PolicyFeature("VISUAL", (3, 32, 32))
    opts["output_features"]["action"] = PolicyFeature("ACTION", (18,))
    opts.update(max_state_dim=32, max_action_dim=32, compile_model=False)
    model = make_policy("smolvla", opts)
    base = tmp_path / "base"
    base.mkdir()
    save_model(model, base / "model.safetensors")
    (base / "config.json").write_text(json.dumps({"type": "smolvla", **asdict(model.config)}))
    del model
    for key in ("input_features", "output_features"):
        opts.pop(key)
    cfg = dict(
        dataset=str(recording),
        policy="smolvla",
        pretrained_path=str(base),
        run_name="smol_smoke",
        steps=2,
        save_freq=1,
        device="cpu",
        batch_size=1,
        train_episodes=[0],
        val_episodes=[1],
        image_size=[32, 32],
        model=opts,
        mixed_precision="none",
        num_workers=0,
    )
    job = launch_training(cfg)
    log = Path(job["log"])
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        content = log.read_text()
        if "Checkpoint:" in content or "Traceback" in content:
            break
        time.sleep(0.1)
    assert "Traceback" not in content and "Checkpoint:" in content, content
    policy = NativePolicy(job["checkpoint"])
    assert policy.manifest["kind"] == "smolvla"
    assert policy.manifest["training"]["paper_preset"]["overrides"]["steps"] == 2
    # Resume from latest complete checkpoint with the original total-step schedule.
    state = torch.load(
        Path(job["checkpoint"]).resolve().parent / "training_state/state.pt",
        weights_only=True,
    )
    assert state["step"] == 2 and state["scheduler"]["last_epoch"] == 2
    # Simulate interruption after step 1 in this temporary fixture, preserving the
    # uninterrupted step 2 as a reference outside the live checkpoint directory.
    final = Path(job["checkpoint"]).resolve().parent
    reference = tmp_path / "uninterrupted"
    shutil.move(str(final), reference)
    previous = final.parent / "000001"
    resume = json.loads((previous / "pretrained_model/train_config.json").read_text())
    resume.update(resume=True, _checkpoint=str(previous))
    restarted = launch_training(resume)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        content = Path(restarted["log"]).read_text()
        if "Checkpoint:" in content or "Traceback" in content:
            break
        time.sleep(0.1)
    assert "Traceback" not in content and "Checkpoint:" in content, content
    from safetensors.torch import load_file

    before = load_file(reference / "pretrained_model/model.safetensors")
    after = load_file(final / "pretrained_model/model.safetensors")
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
