# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Migrated from examples/debug/test_mic.py.
import argparse
import math
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

from alohamini.paths import WorkspacePaths


def list_devices():
    import sounddevice as sd

    devs = sd.query_devices()
    print("\n=== 可用音频设备（输入设备） ===")
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            print(
                f"[{i}] {d['name']}  (in={d['max_input_channels']}, "
                f"out={d.get('max_output_channels', 0)}, default_sr={d['default_samplerate']})"
            )
    print("（若没有任何行，说明系统没有可用麦克风或驱动未就绪）\n")
    return devs


def pick_input_device(devs):
    env = os.getenv("VOICE_DEVICE_INDEX")
    if env is not None:
        if (
            env.isdigit()
            and 0 <= int(env) < len(devs)
            and devs[int(env)].get("max_input_channels", 0) > 0
        ):
            return int(env)
        raise ValueError("VOICE_DEVICE_INDEX does not select an available input device")
    # Prefer the system default input; otherwise pick the first with input channels.
    try:
        import sounddevice as sd

        default_in = sd.default.device[0]
        if (
            default_in is not None
            and 0 <= default_in < len(devs)
            and devs[default_in].get("max_input_channels", 0) > 0
        ):
            return default_in
    except Exception:
        pass
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            return i
    return None


def record_3s_wav(device_idx, samplerate=16000, channels=1, output=None):
    import sounddevice as sd

    sd.default.device = (device_idx, None)
    try:
        sr_dev = int(sd.query_devices(device_idx)["default_samplerate"])
        if abs(sr_dev - samplerate) > 1:
            print(
                f"设备默认采样率 {sr_dev}Hz，与期望 {samplerate}Hz 不同，先用设备默认 {sr_dev}Hz。"
            )
            samplerate = sr_dev
    except Exception:
        pass
    print(f"[diag] 开始录音… 设备={device_idx}, 采样率={samplerate}Hz, 通道={channels}")
    audio = sd.rec(int(3 * samplerate), samplerate=samplerate, channels=channels, dtype="int16")
    try:
        sd.wait()
    finally:
        sd.stop()
    # Compute RMS
    rms = float(np.sqrt(np.mean((audio.astype(np.float32) / 32768.0) ** 2)) + 1e-12)
    dbfs = 20 * math.log10(rms)
    print(
        f"[diag] 3秒录音完成。RMS={rms:.6f}, 约 {dbfs:.1f} dBFS"
        "（<-60 dBFS 往往表示太小/静音/麦克风静音）"
    )
    path = (
        Path(output).expanduser()
        if output
        else (WorkspacePaths().logs / "debug" / f"microphone-{time.time_ns()}.wav")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream, wave.open(stream, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    print(f"[diag] 已保存测试音频：{path}")
    return str(path)


def try_asr(wav_path, model_path):
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        print(f"⚠️ faster-whisper 未安装或不可用：{e}\n请先执行：pip install faster-whisper")
        return False
    print(f"[diag] 加载本地 ASR 模型：{model_path}")
    try:
        model = WhisperModel(str(model_path), device="cpu", local_files_only=True)
    except Exception as e:
        print("❌ ASR 模型加载失败：", e)
        return False
    print("[diag] 开始识别…")
    try:
        segments, _ = model.transcribe(wav_path, language="zh")
        text = "".join(seg.text for seg in segments).strip()
        print(
            f"[diag] 识别结果：{text!r}"
            if text
            else "[diag] 没识别到文本（可能太小/静音/非中文）。"
        )
        return True
    except Exception as e:
        print("❌ 识别失败：", e)
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="List microphones, record 3 seconds, and report RMS"
    )
    parser.add_argument("--output", help="New WAV path; default: workspace logs/debug/")
    parser.add_argument(
        "--asr-model", type=Path, help="Optional local faster-whisper model directory; no downloads"
    )
    args = parser.parse_args(argv)
    if args.asr_model is not None and not args.asr_model.expanduser().is_dir():
        parser.error("--asr-model must be an existing local model directory")
    if args.output and Path(args.output).expanduser().exists():
        parser.error("--output already exists; choose a new WAV path")
    print(">>> 诊断步骤：1) 枚举设备 2) 录3秒并测量音量")
    try:
        devs = list_devices()
        idx = pick_input_device(devs)
        if idx is None:
            print("没有可用麦克风，请检查设备和驱动。", file=sys.stderr)
            return 1
        wav = record_3s_wav(idx, output=args.output)
        if args.asr_model is not None:
            return 0 if try_asr(wav, args.asr_model.expanduser()) else 1
    except ImportError as exc:
        print(
            f"音频依赖不可用：{exc}。请安装项目的 audio-debug 扩展及系统 PortAudio。",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"音频诊断失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
