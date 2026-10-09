"""Reduced real networks; frozen FastWAM assets are fixtures, never downloaded."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.learning.processor import Processor
from alohamini.policies.registry import algorithm


def check_named_action(policy, data):
    from test_dataset import jpeg

    from alohamini.model import get_robot_model

    policy.task = "pick"
    for actuator in get_robot_model(policy.robot_metadata["robot_model"]).actuators:
        motor = policy.robot_metadata["motors"].setdefault(actuator.name, {})
        motor.update(
            id=actuator.motor_id,
            model=actuator.motor_model,
            homing_offset=0,
        )
        for key, value in dict(
            normalization="raw", range_min=0, range_max=4095, drive_mode=0
        ).items():
            motor.setdefault(key, value)
    snapshot = SimpleNamespace(
        robot_model=policy.robot_metadata["robot_model"],
        payload={
            "_robot_metadata": policy.robot_metadata,
            **dict(
                zip(policy.selection.source_names, data.rows[0]["observation.state"], strict=True)
            ),
        },
        images={"forward": jpeg()},
    )
    action = policy.select_action(snapshot)
    assert list(action) == policy.names
    assert all(torch.isfinite(torch.tensor(value)) for value in action.values())
    policy.reset()


def features():
    return {
        "input_features": {
            "observation.images.forward": {"type": "VISUAL", "shape": [3, 32, 32]},
            "observation.state": {"type": "STATE", "shape": [18]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [18]}},
    }


def diffusion_options():
    return dict(
        **features(),
        horizon=8,
        n_obs_steps=2,
        n_action_steps=2,
        down_dims=[16, 32],
        diffusion_step_embed_dim=16,
        num_train_timesteps=4,
        num_inference_steps=2,
        drop_n_last_frames=0,
        spatial_softmax_num_keypoints=4,
        crop_shape=None,
    )


def fastwam_options():
    video = dict(
        patch_size=[1, 2, 2],
        in_dim=4,
        out_dim=4,
        hidden_dim=24,
        ffn_dim=48,
        freq_dim=8,
        text_dim=16,
        num_heads=2,
        attn_head_dim=12,
        num_layers=1,
        eps=1e-6,
        seperated_timestep=True,
        video_attention_mask_mode="first_frame_causal",
    )
    action = dict(
        action_dim=18,
        hidden_dim=16,
        ffn_dim=32,
        text_dim=16,
        freq_dim=8,
        num_heads=2,
        attn_head_dim=12,
        num_layers=1,
        eps=1e-6,
    )
    return dict(
        **features(),
        action_horizon=4,
        n_action_steps=2,
        num_video_frames=5,
        action_video_freq_ratio=1,
        image_size=[32, 64],
        video_dit_config=video,
        action_dit_config=action,
        torch_dtype="float32",
        num_inference_steps=2,
    )


def fake_frozen_assets(monkeypatch):
    from alohamini.policies.fastwam import modeling_fastwam as module

    class VAE(torch.nn.Module):
        temporal_downsample_factor = 4
        upsampling_factor = 16
        z_dim = 4

        def encode(self, video, **kwargs):
            if isinstance(video, list):
                video = torch.stack(video)
            video = video[:, :, ::4, ::16, ::16].float()
            return torch.cat([video, video[:, :1]], dim=1)

    class Text(torch.nn.Module):
        dim = 16

        def forward(self, ids, mask):
            return torch.ones((*ids.shape, 16), device=ids.device)

    class Tokenizer:
        def __call__(self, prompt, **kwargs):
            batch = 1 if isinstance(prompt, str) else len(prompt)
            return torch.ones(batch, 4, dtype=torch.long), torch.ones(batch, 4, dtype=torch.bool)

    monkeypatch.setattr(module, "load_pretrained_wan_vae", lambda **kwargs: VAE())
    monkeypatch.setattr(module, "load_pretrained_wan_text_encoder", lambda **kwargs: Text())
    monkeypatch.setattr(module, "build_wan_tokenizer", lambda **kwargs: Tokenizer())


def test_limits_normalization_constant_and_small_span():
    processor = Processor(
        {"action": {"min": [4.0, 2.0, -3.0], "max": [4.0, 2.00001, 7.0]}},
        modes={"action": "MIN_MAX"},
    )
    values = torch.tensor([[4.0, 2.0, 2.0], [4.0, 2.00001, 7.0]], requires_grad=True)
    normalized = processor({"action": values})["action"]
    torch.testing.assert_close(normalized[0], torch.zeros(3))
    assert normalized[1, 1] < 1e-4
    restored = processor.action(normalized)
    torch.testing.assert_close(restored, values)
    restored.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_distinct_sample_windows_and_tail_filter(recording):
    dp = algorithm("diffusion")
    opts = diffusion_options()
    data = AlohaMiniDataset(
        recording,
        episodes=[0, 1],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **dp.sample_spec(opts, cameras=["forward"]),
    )
    assert data[0]["observation.state"].shape == (2, 18)
    assert data[0]["action_is_pad"].tolist() == [True, False, False, False, False, True, True, True]
    assert data[0]["observation.images.forward"].shape == (2, 3, 32, 32)
    spec = algorithm("fastwam").sample_spec(fastwam_options(), cameras=["forward"])
    fast = AlohaMiniDataset(
        recording,
        episodes=[0, 1],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **spec,
    )
    assert fast[0]["observation.state"].shape == (18,)
    assert fast[0]["observation.images.forward"].shape == (5, 3, 32, 32)
    assert fast[0]["observation.images.forward_is_pad"].tolist() == [False] * 4 + [True]
    opts["drop_n_last_frames"] = 1
    shorter = AlohaMiniDataset(
        recording,
        episodes=[0, 1],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **dp.sample_spec(opts, cameras=["forward"]),
    )
    assert len(shorter) == len(data) - 2


def test_diffusion_real_loss_queue_ema_and_checkpoint(recording, tmp_path, monkeypatch):
    recipe = algorithm("diffusion")
    options = diffusion_options()
    data = AlohaMiniDataset(
        recording,
        episodes=[0, 1],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **recipe.sample_spec(options, cameras=["forward"]),
    )
    stats = recipe.statistics(data)
    model = recipe.build(options)
    processor = recipe.processor(model, stats, "cpu")
    batch = processor(next(iter(torch.utils.data.DataLoader(data, batch_size=2))))
    loss, _ = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.diffusion.parameters())
    assert all(p.grad is None for p in model.ema.parameters())
    model.update()
    assert int(model.ema_step) == 1
    predicted = model.predict_action_chunk(batch)
    assert predicted.shape == (2, 8, 18)
    target = tmp_path / "checkpoint"
    save_checkpoint(target, model, stats, data, training={})
    loaded = NativePolicy(target)
    check_named_action(loaded, data)
    torch.manual_seed(7)
    expected = model.predict_action_chunk(batch)
    torch.manual_seed(7)
    torch.testing.assert_close(loaded.model.predict_action_chunk(batch), expected)
    model.reset()
    monkeypatch.setattr(
        model,
        "predict_action_chunk",
        lambda batch, **kwargs: torch.arange(8.0)[None, :, None].expand(1, 8, 18),
    )
    live = processor({k: v[None] for k, v in data.observation(0).items()})
    assert model.select_action(live)[0, 0] == 1
    assert model.select_action(live)[0, 0] == 2
    model.reset()
    assert all(not q for q in model._queues.values())


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_fastwam_real_core_loss_sampling_and_checkpoint(
    recording, tmp_path, monkeypatch, precision
):
    fake_frozen_assets(monkeypatch)
    recipe = algorithm("fastwam")
    options = fastwam_options()
    options["torch_dtype"] = precision
    data = AlohaMiniDataset(
        recording,
        episodes=[0, 1],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **recipe.sample_spec(options, cameras=["forward"]),
    )
    model = recipe.build(options)
    stats = recipe.statistics(data)
    processor = recipe.processor(model, stats, "cpu")
    batch = processor(next(iter(torch.utils.data.DataLoader(data, batch_size=2))))
    sample = model._batch_to_training_sample(batch)
    assert sample["image_is_pad"][0, -1]
    assert sample["video"].shape == (2, 3, 5, 32, 64)
    loss, metrics = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and set(metrics) == {"loss_video", "loss_action"}
    assert model.model.action_expert.head.weight.grad is not None
    predicted = model.predict_action_chunk(batch)
    assert predicted.shape == (2, 4, 18)
    changed = deepcopy(batch)
    changed["observation.images.forward"][:, 1:] = 0.99
    torch.testing.assert_close(model.predict_action_chunk(changed), predicted)
    for key in ("vae_model_id", "text_encoder_model_id", "tokenizer_model_id"):
        setattr(model.config, key, str(tmp_path))
    target = tmp_path / "checkpoint"
    save_checkpoint(target, model, stats, data, training={})
    loaded = NativePolicy(target)
    check_named_action(loaded, data)
    torch.testing.assert_close(loaded.model.predict_action_chunk(batch), predicted)
    current = dict(batch)
    current["observation.images.forward"] = current["observation.images.forward"][:, 0]
    with pytest.raises(ValueError, match="future camera windows"):
        model(current)


def test_recipe_registration_and_unsupported_ensembling():
    for name in ("diffusion", "fastwam"):
        recipe = algorithm(name)
        cfg = recipe.apply_preset({"policy": name, "model": {}})
        assert cfg["paper_preset"]["implementation"]
        assert recipe.apply_preset(cfg) == cfg
        with pytest.raises(ValueError, match="temporal ensembling"):
            recipe.checkpoint_options(None, {}, "cpu", None, 0.01)


def test_diffusers_cosine_and_loss_denominators():
    from diffusers.optimization import get_cosine_schedule_with_warmup

    from alohamini.learning.optim import make_scheduler

    opt = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
    cfg = {
        "steps": 30,
        "optimizer": {"lr": 1e-4},
        "scheduler": {
            "type": "diffusers_cosine",
            "warmup_steps": 5,
            "decay_steps": 30,
            "decay_lr": 0.0,
        },
    }
    ours = make_scheduler(cfg, opt).lr_lambdas[0]
    source = get_cosine_schedule_with_warmup(opt, 5, 30).lr_lambdas[0]
    for step in range(31):
        assert ours(step) == pytest.approx(source(step), abs=1e-12)
    recipe = algorithm("diffusion")
    batch = {
        "action": torch.zeros(2, 4, 18),
        "action_is_pad": torch.tensor([[False] * 4, [False, True, True, True]]),
    }
    recipe.options({"model": {}}, "cpu")
    assert recipe.loss_counts(batch) == {"_loss_weight": 2}
    recipe.options({"model": {"do_mask_loss_for_padding": True}}, "cpu")
    assert recipe.loss_counts(batch) == {"_loss_weight": 5}
    assert algorithm("fastwam").loss_counts(batch) == {"_loss_weight": 2}


@pytest.mark.parametrize("kind", ["diffusion", "fastwam"])
def test_detached_training_resume(recording, tmp_path, monkeypatch, kind):
    from safetensors.torch import load_file, save_file

    from alohamini.learning.train_config import parse_training_args

    options = diffusion_options() if kind == "diffusion" else fastwam_options()
    if kind == "fastwam":
        fake_frozen_assets(monkeypatch)
        for key in ("vae_model_id", "text_encoder_model_id", "tokenizer_model_id"):
            options[key] = str(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    save_file(algorithm(kind).build(options).state_dict(), base / "model.safetensors")
    cfg = dict(
        policy=kind,
        dataset=str(recording),
        pretrained_path=str(base),
        model=options,
        device="cpu",
        mixed_precision="none",
        steps=2,
        train_episodes=[0],
        val_episodes=[1],
        image_size=[32, 32],
        batch_size=3,
        gradient_accumulation_steps=2,
        num_workers=0,
        save_freq=1,
        eval_steps=1,
        seed=31,
        deterministic_algorithms=True,
        scheduler={"type": "none"},
    )

    def run(settings, label, stop=False):
        config = tmp_path / f"{label}.json"
        log = tmp_path / f"{label}.log"
        config.write_text(json.dumps(settings))
        with log.open("wb") as stream:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), str(config)],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, "POLICY_TEST_STOP": str(int(stop))},
            )
        (tmp_path / f"{label}.pid").write_text(str(process.pid))
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
    full, resumed = (tmp_path / n / "checkpoint" for n in ("full", "resumed"))
    left, right = (load_file(p / "model.safetensors") for p in (full, resumed))
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    report = json.loads((tmp_path / "resumed/offline-evaluation.json").read_text())
    assert report["valid_action_steps"] > 0
    expected_keys = (
        {"loss_diffusion"}
        if kind == "diffusion"
        else {"loss_video_weighted", "loss_action_weighted"}
    )
    for filename in ("metrics.jsonl", "validation-metrics.jsonl"):
        for line in (tmp_path / "full" / filename).read_text().splitlines():
            record = json.loads(line)
            assert set(record["metrics"]) == expected_keys
            assert sum(record["metrics"].values()) == pytest.approx(record["loss"], rel=1e-5)
    assert len(list((tmp_path / "resumed").glob("metrics-from-*-schema.json"))) == 1


@pytest.mark.parametrize("kind", ["diffusion", "fastwam"])
def test_cuda_small_network(kind, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if kind == "fastwam":
        fake_frozen_assets(monkeypatch)
        options = fastwam_options()
        options.update(device="cuda", torch_dtype="bfloat16")
        frames, steps = 5, 4
        state_shape = (2, 18)
    else:
        options = diffusion_options()
        frames, steps = 2, 8
        state_shape = (2, 2, 18)
    model = algorithm(kind).build(options).to("cuda")
    batch = {
        "observation.images.forward": torch.rand(2, frames, 3, 32, 32, device="cuda"),
        "observation.state": torch.zeros(state_shape, device="cuda"),
        "action": torch.rand(2, steps, 18, device="cuda"),
        "action_is_pad": torch.zeros(2, steps, dtype=torch.bool, device="cuda"),
        "task": ["pick", "pick"],
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model.predict_action_chunk(batch)
    assert prediction.shape == (2, steps, 18) and torch.isfinite(prediction).all()


if __name__ == "__main__":
    torch.set_num_threads(2)
    with pytest.MonkeyPatch.context() as patch:
        fake_frozen_assets(patch)
        import alohamini.learning.train as training

        if os.environ.get("POLICY_TEST_STOP") == "1":
            save = training.save_training_checkpoint

            def save_and_stop(*args, **kwargs):
                save(*args, **kwargs)
                raise SystemExit(0)

            patch.setattr(training, "save_training_checkpoint", save_and_stop)
        training.train(json.loads(Path(sys.argv[-1]).read_text()))
