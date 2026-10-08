"""AlohaMini command-line entry points."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from alohamini.client import HostClient
from alohamini.errors import AlohaMiniError
from alohamini.paths import WorkspacePaths


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def _ensemble_coefficient(value: str) -> float | None:
    if value.lower() == "none":
        return None
    try:
        from alohamini._validation import finite_number

        coefficient = float(value)
        finite_number(coefficient, "temporal ensemble coefficient")
        return coefficient
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a finite number or none") from exc


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:2] == ["dataset", "edit"]:
        try:
            from alohamini.datasets.edit import main as edit_dataset

            edit_dataset(argv[2:])
            return 0
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            print(f"alohamini: {exc}", file=sys.stderr)
            return 1
    parser = argparse.ArgumentParser(prog="alohamini", description="AlohaMini 机器人接口")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("paths", help="显示工作文件保存路径，不创建目录或移动文件")
    commands.add_parser(
        "find-port", help="拔下单个控制板 USB 前后对比串口；先停止遥操、数采或 Host"
    )
    cameras = commands.add_parser("find-cameras", help="查找本机 USB 相机并保存预览；先停止 Host")
    cameras.add_argument("camera_type", nargs="?", choices=["opencv"])
    cameras.add_argument("--output-dir", help="保存快照的新目录；默认位于工作目录 logs/debug/")
    cameras.add_argument("--record-time-s", type=float, default=6.0, help="采样时长；0 表示仅列出")
    datasets = commands.add_parser("dataset", help="本地数据检查、修复、视频预览和导出")
    dataset_commands = datasets.add_subparsers(dest="operation", required=True)
    dataset_commands.add_parser(
        "edit", help="本地数据集编辑：删除、拆分、合并、任务、字段、统计和视频"
    )
    for operation, help_text in (
        ("check", "只读检查数据结构、反馈、图像和保护记录"),
        ("recover", "将中断前的完整帧恢复到新目录，保留原始数据"),
        ("repair", "修复 AlohaMini 或 LeRobot v3 数据，输出到新目录"),
        ("preview", "生成 MP4 预览，不改动原始帧"),
        ("export", "导出数据集到新目录，不修改原始数据"),
        ("stats", "计算基础统计或按训练配置生成可复用统计"),
    ):
        command = dataset_commands.add_parser(operation, help=help_text)
        command.add_argument("root", help="本地数据集目录")
        if operation == "check":
            command.add_argument("--decode-images", action="store_true", help="完整解码所有图像")
            command.add_argument(
                "--decode-videos", action="store_true", help="逐帧检查视频及时间戳"
            )
            command.add_argument("--output-json", help="将检查报告保存到新 JSON 文件")
            command.add_argument("--fail-on-warnings", action="store_true")
        elif operation == "preview":
            command.add_argument("--output", help="新目录；默认保存在数据集的 previews/ 下")
        elif operation == "stats":
            command.add_argument("--output", required=True, help="数据集目录外的新 JSON 文件")
        else:
            command.add_argument("--output", required=True, help="尚不存在的新目录")
        if operation == "stats":
            command.add_argument("--config", help="训练 JSON 配置；省略时仅统计原始数值字段")
        if operation == "export":
            command.add_argument(
                "--format",
                type=lambda value: "alohamini" if value == "native" else value,
                choices=("alohamini", "lerobot-v3"),
                default="alohamini",
            )
            command.add_argument(
                "--state", help="可选重组 state；默认保留原记录的关节位置、底盘速度、升降高度"
            )
            command.add_argument(
                "--vision-only",
                action="store_true",
                help="从已有图像型 v3 导出纯视觉副本，保留 action 和索引",
            )
    replayer = commands.add_parser("replay", help="通过 Host 回放本地数据集的动作目标")
    replayer.add_argument("--dataset", "--dataset.repo_id", dest="dataset_name", required=True)
    replayer.add_argument("--root", "--dataset.root", dest="root", help="本地数据集自定义目录")
    replayer.add_argument("--host", "--robot.remote_ip", dest="host", required=True)
    replayer.add_argument(
        "--robot_model",
        "--robot.robot_model",
        dest="robot_model",
        required=True,
        choices=("alohamini1", "alohamini2", "alohamini2pro"),
    )
    replayer.add_argument("--episode", "--dataset.episode", dest="episode", type=int, default=0)
    replayer.add_argument("--fps", "--replay.fps", dest="fps", type=float)
    replayer.add_argument("--speed", "--replay.speed", dest="speed", type=float, default=1.0)
    replayer.add_argument("--verbose-actions", action="store_true")
    evaluator = commands.add_parser("evaluate", help="运行本地策略并可选录制评估数据")
    evaluator.add_argument("--host", "--robot.remote_ip", dest="host", required=True)
    evaluator.add_argument(
        "--robot_model",
        "--robot.robot_model",
        dest="robot_model",
        required=True,
        choices=("alohamini1", "alohamini2", "alohamini2pro"),
    )
    policy_source = evaluator.add_mutually_exclusive_group(required=True)
    policy_source.add_argument(
        "--policy",
        dest="policy_factory",
        help="可导入的 module:factory，工厂函数返回策略对象",
    )
    policy_source.add_argument(
        "--policy.path", dest="checkpoint", help="本地 AlohaMini 策略 checkpoint"
    )
    evaluator.add_argument("--device", help="checkpoint 推理设备，默认 cuda")
    evaluator.add_argument(
        "--policy.n_action_steps",
        dest="n_action_steps",
        type=int,
        default=argparse.SUPPRESS,
        help="执行多少个动作后重新预测，省略时沿用 checkpoint",
    )
    evaluator.add_argument(
        "--policy.temporal_ensemble_coeff",
        dest="temporal_ensemble_coeff",
        type=_ensemble_coefficient,
        default=argparse.SUPPRESS,
        help="ACT 融合系数；none 关闭，0 等权融合，省略时沿用 checkpoint",
    )
    evaluator.add_argument("--fps", type=int, default=30)
    evaluator.add_argument("--episode_time", dest="episode_time_s", type=float, default=60)
    evaluator.add_argument("--num_episodes", type=int, default=1)
    evaluator.add_argument("--reset_time", dest="reset_time_s", type=float, default=10)
    evaluator.add_argument("--dataset", dest="dataset_name", help="可选，本地评估数据集名称")
    evaluator.add_argument("--task", default="robot task")
    recorder = commands.add_parser("record", help="双主臂多频率数采，本地保存，不上传")
    recorder.add_argument(
        "--dataset",
        "--dataset.repo_id",
        dest="dataset_name",
        required=True,
        help="本地数据集名称；用 --root 指定自定义目录",
    )
    recorder.add_argument("--root", "--dataset.root", dest="root")
    recorder.add_argument("--host", "--robot.remote_ip", dest="host", required=True)
    recorder.add_argument(
        "--robot_model",
        "--robot.robot_model",
        dest="robot_model",
        required=True,
        choices=("alohamini1", "alohamini2", "alohamini2pro"),
    )
    recorder.add_argument("--task", "--dataset.single_task", dest="task", required=True)
    recorder.add_argument("--fps", "--dataset.fps", dest="fps", type=int, default=30)
    recorder.add_argument(
        "--num_episodes", "--dataset.num_episodes", dest="num_episodes", type=int, default=1
    )
    recorder.add_argument(
        "--episode_time", "--dataset.episode_time_s", dest="episode_time_s", type=float, default=60
    )
    recorder.add_argument(
        "--reset_time", "--dataset.reset_time_s", dest="reset_time_s", type=float, default=10
    )
    recorder.add_argument("--teleop.id", "--leader_id", dest="leader_id")
    recorder.add_argument(
        "--teleop.arm_profile",
        "--arm_profile",
        dest="arm_profile",
        choices=("so-arm-5dof", "am-leader-6dof"),
    )
    recorder.add_argument("--calibration_dir")
    recorder.add_argument("--left_port", default="/dev/am_arm_leader_left")
    recorder.add_argument("--right_port", default="/dev/am_arm_leader_right")
    recorder.add_argument("--resume", action="store_true")
    recorder.add_argument(
        "--profile_timing",
        "--profile-timing",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
    )
    recorder.add_argument(
        "--display_data", "--display-data", type=parse_bool, nargs="?", const=True, default=False
    )
    calibration = commands.add_parser(
        "calibrate", help="本机手动标定主臂、整机或仅双臂，不使能或自动回零"
    )
    calibration.add_argument("target", choices=("leader", "robot", "arms"))
    calibration.add_argument(
        "--rehome", action="store_true", help="仅 arms：重新设置双臂零偏；默认保留"
    )
    calibration.add_argument(
        "--robot_model", required=True, choices=("alohamini1", "alohamini2", "alohamini2pro")
    )
    calibration.add_argument(
        "--id", "--teleop.id", dest="device_id", help="与遥操或 Host 一致的标定 ID"
    )
    calibration.add_argument("--calibration_dir", help="标定 JSON 所在目录；默认使用工作目录")
    calibration.add_argument("--left_port", help="左侧串口；默认使用对应主臂/从臂设备名")
    calibration.add_argument("--right_port", help="右侧串口；默认使用对应主臂/从臂设备名")
    calibration.add_argument(
        "--teleop.arm_profile",
        "--arm_profile",
        dest="arm_profile",
        choices=("so-arm-5dof", "am-leader-6dof"),
        help="主臂型号，默认按整机选择",
    )
    host = commands.add_parser(
        "host", help="启动硬件 Host：使能并回零；启动前支撑双臂、清空升降下降路径"
    )
    host.add_argument(
        "--robot_model", required=True, choices=("alohamini1", "alohamini2", "alohamini2pro")
    )
    host.add_argument(
        "--calibration", help="本机标定 JSON；默认取工作目录 calibration/robots/AlohaMiniRobot.json"
    )
    host.add_argument("--left_port", default="/dev/am_arm_follower_left")
    host.add_argument("--right_port", default="/dev/am_arm_follower_right")
    host.add_argument("--bind_host", default="0.0.0.0", help="仅在可信机器人网络监听")
    host.add_argument(
        "--profile_timing",
        "--profile-timing",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="每秒打印 Host、总线、动作和相机耗时，默认关闭",
    )
    host.add_argument(
        "--use_degrees", action="store_true", help="关节使用旧角度模式，夹爪仍为 0–100"
    )
    host.add_argument(
        "--cameras",
        nargs="*",
        default=["forward", "wrist_right"],
        choices=("forward", "backward", "chest", "wrist_left", "wrist_right"),
        help="启用的相机名，设备为 /dev/am_camera_<名称>；不填名称则不启用相机",
    )
    teleop = commands.add_parser("teleoperate", help="双主臂与键盘遥操；使用已有标定")
    teleop.add_argument(
        "--host", "--robot.remote_ip", "--remote_ip", dest="host", default="127.0.0.1"
    )
    teleop.add_argument(
        "--robot_model",
        "--robot.robot_model",
        dest="robot_model",
        required=True,
        choices=("alohamini1", "alohamini2", "alohamini2pro"),
    )
    teleop.add_argument(
        "--leader_id",
        "--teleop.id",
        dest="leader_id",
        help="主臂标定 ID；默认按型号选择 so101_leader_bi 或 am_leader_bi",
    )
    teleop.add_argument("--calibration_dir", help="左右主臂标定 JSON 所在目录")
    teleop.add_argument("--left_port", default="/dev/am_arm_leader_left")
    teleop.add_argument("--right_port", default="/dev/am_arm_leader_right")
    teleop.add_argument("--no_leader", action="store_true", help="只用键盘控制底盘与升降")
    teleop.add_argument("--no_keyboard", action="store_true", help="只用主臂；适用于无 X11 桌面")
    teleop.add_argument("--no_robot", action="store_true", help="不连接 Host，只显示主臂与键盘输入")
    teleop.add_argument("--no_preview", action="store_true", help="关闭 Rerun 和相机请求")
    teleop.add_argument(
        "--tracking", action="store_true", help="记录双臂关节跟随误差与电流到 logs/tracking/"
    )
    teleop.add_argument("--fps", type=int, default=50, help="控制频率，1–50 Hz")
    teleop.add_argument("--camera-fps", type=int, default=30, help="相机请求频率，不超过控制频率")
    teleop.add_argument(
        "--teleop.arm_profile",
        "--arm_profile",
        dest="arm_profile",
        choices=("so-arm-5dof", "am-leader-6dof"),
        help="可省略，按整机型号选择；显式指定时核对是否匹配",
    )
    inspect = commands.add_parser("inspect", help="只读查询现有 Host 状态，不发送运动命令")
    inspect.add_argument("--host", required=True, help="Host 的 IPv4 地址或主机名")
    inspect.add_argument("--port", type=int, default=5556, help="状态端口，默认 5556")
    inspect.add_argument("--model", help="预期型号，不匹配时报告错误")
    inspect.add_argument("--timeout", type=float, default=1.0, help="请求超时秒数，默认 1")
    args = parser.parse_args(argv)
    try:
        if args.command == "find-port":
            from alohamini.hardware.find_port import find_port

            find_port()
            return 0
        if args.command == "find-cameras":
            from alohamini.hardware.find_cameras import save_images_from_all_cameras

            logging.basicConfig(level=logging.INFO, format="%(message)s")
            save_images_from_all_cameras(args.output_dir, args.record_time_s, args.camera_type)
            return 0
        if args.command == "dataset":
            from alohamini.datasets.tools import (
                check_dataset,
                export_dataset,
                print_report,
                repair_dataset,
            )

            if args.operation == "check":
                report = check_dataset(
                    args.root, decode_images=args.decode_images, decode_videos=args.decode_videos
                )
                if args.output_json:
                    from pathlib import Path

                    target = Path(args.output_json).expanduser().absolute()
                    if target.resolve().is_relative_to(Path(args.root).expanduser().resolve()):
                        raise ValueError("Check report must be outside the source dataset")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("x", encoding="utf-8") as stream:
                        json.dump(report, stream, ensure_ascii=False, indent=2)
                        stream.write("\n")
            elif args.operation == "stats":
                from alohamini.learning.statistics import prepare_statistics

                path = prepare_statistics(args.root, args.output, config=args.config)
                print(f"Statistics: {path}")
                return 0
            elif args.operation == "preview":
                from alohamini.datasets.video import generate_previews

                result = generate_previews(args.root, args.output)
                print(
                    f"Previews: {result['output']} "
                    f"(generated={result['generated']}, reused={result['reused']})"
                )
                return 0
            elif args.operation == "repair":
                report = repair_dataset(args.root, args.output)
            elif args.operation == "export" and args.format == "lerobot-v3":
                from alohamini.datasets.lerobotv3 import export_lerobot
                from alohamini.datasets.record import StateSelection

                selection = args.state if args.state is not None else StateSelection.DEFAULT
                if args.vision_only and args.state is not None:
                    raise ValueError("--vision-only cannot be combined with --state")
                report = export_lerobot(
                    args.root, args.output, state=selection, vision_only=args.vision_only
                )
            else:
                if getattr(args, "state", None) is not None or getattr(args, "vision_only", False):
                    raise ValueError("--state/--vision-only require --format lerobot-v3")
                report = export_dataset(args.root, args.output, recover=args.operation == "recover")
            print_report(report, summarize_warnings=args.operation == "export")
            if args.operation == "export" and report["valid"]:
                print("Export completed; recorded timestamps were not resampled.")
            return int(
                not report["valid"]
                or (getattr(args, "fail_on_warnings", False) and report["warnings"] > 0)
            )
        if args.command == "evaluate":
            from alohamini.apps.evaluation import evaluate

            logging.basicConfig(level=logging.WARNING, format="%(message)s")
            options = vars(args).copy()
            options.pop("command")
            checkpoint = options.pop("checkpoint")
            device = options.pop("device")
            overrides = {
                key: options.pop(key)
                for key in ("n_action_steps", "temporal_ensemble_coeff")
                if key in options
            }
            if checkpoint is not None:
                from pathlib import Path

                if not (Path(checkpoint).expanduser() / "policy.json").is_file():
                    raise ValueError(
                        "Expected an AlohaMini checkpoint; "
                        "use the LeRobot fork for LeRobot checkpoints"
                    )

                def load_checkpoint():
                    from alohamini.learning.policy import NativePolicy

                    policy = NativePolicy(
                        checkpoint, device=device or "cuda", task=options["task"], **overrides
                    )
                    if policy.fps != options["fps"]:
                        raise ValueError("Evaluation FPS must match the checkpoint")
                    return policy

                options["policy_factory"] = load_checkpoint
            elif device is not None or overrides:
                raise ValueError("Checkpoint options require --policy.path, not --policy")
            evaluate(**options)
            return 0
        if args.command == "replay":
            from alohamini.apps.replay import replay

            logging.basicConfig(level=logging.WARNING, format="%(message)s")
            options = vars(args).copy()
            options.pop("command")
            replay(**options)
            return 0
        if args.command == "record":
            from alohamini.apps.recording import record

            logging.basicConfig(level=logging.WARNING, format="%(message)s")
            options = vars(args).copy()
            options.pop("command")
            record(**options)
            return 0
        if args.command == "calibrate":
            from alohamini.calibration.procedure import calibrate

            logging.basicConfig(level=logging.INFO, format="%(message)s")
            calibrate(
                args.target,
                args.robot_model,
                device_id=args.device_id,
                calibration_dir=args.calibration_dir,
                left_port=args.left_port,
                right_port=args.right_port,
                arm_profile=args.arm_profile,
                rehome=args.rehome,
            )
            return 0
        if args.command == "paths":
            paths = WorkspacePaths()
            print(
                json.dumps(
                    {
                        name: str(getattr(paths, name))
                        for name in ("root", "calibration", "datasets", "runs", "incoming", "logs")
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "host":
            from alohamini.hardware.camera import CameraConfig
            from alohamini.runtime.startup import open_host

            logging.basicConfig(level=logging.INFO, format="%(message)s")
            if len(set(args.cameras)) != len(args.cameras):
                raise ValueError("Camera names must be unique")
            robot = open_host(
                args.robot_model,
                calibration_file=args.calibration,
                left_port=args.left_port,
                right_port=args.right_port,
                bind_host=args.bind_host,
                use_degrees=args.use_degrees,
                cameras={name: CameraConfig(f"/dev/am_camera_{name}") for name in args.cameras},
            )
            robot.run(profile_timing=args.profile_timing)
            return 0
        if args.command == "teleoperate":
            from alohamini.apps.teleoperation import teleoperate

            logging.basicConfig(level=logging.INFO, format="%(message)s")
            teleoperate(
                args.host,
                args.robot_model,
                leader_id=args.leader_id,
                calibration_dir=args.calibration_dir,
                left_port=args.left_port,
                right_port=args.right_port,
                no_leader=args.no_leader,
                no_keyboard=args.no_keyboard,
                no_robot=args.no_robot,
                no_preview=args.no_preview,
                fps=args.fps,
                camera_fps=args.camera_fps,
                arm_profile=args.arm_profile,
                tracking=args.tracking,
            )
            return 0
        with HostClient(
            args.host, port=args.port, expected_model=args.model, timeout_s=args.timeout
        ) as client:
            snapshot = client.read()
        print(json.dumps(snapshot.payload, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (AlohaMiniError, ValueError, ImportError, OSError, RuntimeError, EOFError) as exc:
        print(f"alohamini: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
