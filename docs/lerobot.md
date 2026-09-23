# LeRobot 训练与策略开发

平台负责机器人和数据接口；LeRobot 可作为独立的算法与训练依赖。只在 PC 安装，树莓派 Host 不需要 LeRobot 或 PyTorch。

| 路线 | 入口 | 模型实现 |
| --- | --- | --- |
| 平台原生 ACT／AM-ACT | `python -m alohamini.learning.train` | `src/alohamini/policies/`，见 [数据整理、训练与部署](training.md) |
| 官方 LeRobot | `python -m lerobot.scripts.lerobot_train` | 已安装的 LeRobot 策略 |
| 集成层策略开发 | `alohamini-lerobot-train` | `integrations/lerobot/src/alohamini_lerobot/policies/` 中复制迁移的策略；仍依赖 LeRobot |

集成层训练入口沿用旧 LeRobot 训练文件及其数据、优化和 checkpoint 工具，不使用平台原生训练器。
原生与 LeRobot checkpoint 不可直接混用。使用 v3 存储格式本身不决定使用哪条训练路线。

## 1. 安装

先完成 [PC 安装](install.md)，再在仓库根目录执行：

```bash
conda activate alohamini
python -m pip install --require-hashes --no-build-isolation -r env/lerobot-training-linux-64.lock
python -m pip install --no-deps --no-build-isolation -e 'integrations/lerobot[training,diffusion]'
python -m pip check
```

此锁文件固定 LeRobot `0.6.1`、训练及 Diffusion 依赖，沿用 PC 的 PyTorch/CUDA；
`packaging`、`fsspec` 按 LeRobot 的兼容范围调整。安装顺序是 PC 环境在先、此扩展在后。
只加载官方 ACT、无需训练时，可改用 `env/lerobot-linux-64.lock`，并去掉 editable 安装中的 extras。

其他模型的可选依赖按下表单独安装，不安装 `lerobot[all]`。例如 π 系列：

```bash
python -m pip install 'lerobot[pi]==0.6.1'
python -m pip check
```

大型模型可能还需专用权重、Tokenizer 或特定注意力后端；不随平台打包，不自动下载全部模型。
训练锁文件尚未覆盖这些额外模型的完整环境。

## 2. 数据准备

先按 [数据整理](training.md#2-检查与整理原生数据) 完成检查和必要处理，再导出：

```bash
alohamini dataset check ~/Alohamini_workspace/datasets/task_demo_ready --decode-images --decode-videos
alohamini dataset export ~/Alohamini_workspace/datasets/task_demo_ready \
  --output ~/Alohamini_workspace/datasets/task_demo_v3 --format lerobot-v3
```

LeRobot 训练器读取 LeRobot 格式，不直接读取平台原生 episode 格式。
默认导出 state 为双臂位置 14 维、底盘速度 3 维、升降高度 1 维；action 保持原含义。
纯视觉 v3 可以用于支持无 state 的策略，但不能直接替代 Diffusion 等要求 state 的输入。
重组后的字段还须满足所选模型的维度、单位、图像、语言与动作表示要求。

LeRobot 不自动执行平台原生读取器的保护/时间过滤。保护段、无效反馈和时基问题应先处理，
不能把结构检查通过视为训练数据已经合格。保留 `meta/alohamini.json`，供后续平台部署核验。

## 3. 使用 LeRobot 训练

以下使用整理好的本地 v3。`dataset.repo_id` 是数据集标识，不是磁盘路径；实际目录由 `dataset.root` 指定。
输出目录和运行名称应尚未使用。

```bash
conda activate alohamini
mkdir -p ~/Alohamini_workspace/logs/training

nohup alohamini-lerobot-train \
  --dataset.repo_id=local/task_demo_v3 \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_v3" \
  --policy.type=alohamini_diffusion --policy.device=cuda \
  --policy.horizon=32 --policy.n_action_steps=8 \
  --policy.push_to_hub=false --save_checkpoint_to_hub=false \
  --output_dir="$HOME/Alohamini_workspace/runs/diffusion_01" \
  --steps=100000 --batch_size=2 --save_freq=10000 \
  --env_eval_freq=0 --wandb.enable=false \
  > "$HOME/Alohamini_workspace/logs/training/diffusion_01.log" 2>&1 < /dev/null &

echo $! > ~/Alohamini_workspace/logs/training/diffusion_01.pid
tail -f ~/Alohamini_workspace/logs/training/diffusion_01.log
```

上述 horizon/执行步数只是起始实验配置，不保证显存占用或收敛。
Diffusion 保留原来的观测历史、动作时间偏移、MIN_MAX 数值归一化、去噪及动作队列，
不是把 ACT 的 chunk、MEAN_STD 或时间融合直接套过去。

这里的 `--policy.push_to_hub=false` 等选项属于 **LeRobot 训练器**；平台原生训练不需要这些开关。
不上传不等于完全离线：首次加载视觉骨干、Tokenizer 或预训练模型仍可能下载文件，离线使用须先准备缓存。

若要使用官方实现，将命令开头换成 `python -m lerobot.scripts.lerobot_train`，并使用官方名字，
如 `--policy.type=act` 或 `--policy.type=diffusion`。不要把集成层的 `alohamini_*` 注册名传给未经注册的官方入口。

续训使用同一训练入口和完整保存点：

```bash
alohamini-lerobot-train \
  --config_path /path/to/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --steps=150000 \
  --policy.push_to_hub=false --save_checkpoint_to_hub=false --wandb.enable=false
```

正式续训也应按上面的方式放到后台，使用新的日志和 PID 文件。
同一训练器、配置、数据及保存的处理器配套使用；不要只复制权重文件。

## 4. 集成层策略清单

复制来源和校验信息保存在 `policies/sources.json`；模型文件保留原版权声明。
注册名使用 `alohamini_` 前缀，避免覆盖已安装 LeRobot 的同名策略。原生 ACT／AM-ACT 不在此次复制范围内。

| 策略名（均加 `alohamini_` 前缀） | 额外 LeRobot extra | 主要接入条件 |
| --- | --- | --- |
| `diffusion` | `diffusion` | state、图像或环境状态，历史观测与动作序列 |
| `vqbet` | 无专用 extra | state、图像、动作块及 VQ 训练阶段 |
| `tdmpc` | 无专用 extra | 除动作外还需算法所需的 reward/后续状态等数据 |
| `gaussian_actor` | 视编码器而定 | Actor 组件；不等于完整 SAC 训练，需 RL 算法及交互数据 |
| `pi0`、`pi05`、`pi0_fast` | `pi` | 模型对应的语言、state/action、Tokenizer 和预训练资产 |
| `smolvla` | `smolvla` | 图像、语言及模型配置所需 state |
| `multi_task_dit` | `multi-task-dit` | 任务条件与动作序列 |
| `groot` | `groot` | embodiment、模型资产及专用处理器 |
| `xvla` | `xvla` | domain、动作表示及模型资产 |
| `eo1`、`evo1` | `eo1`、`evo1` | 各自的视觉语言骨干、输入处理与权重 |
| `fastwam` | `fastwam` | 视频/动作模型及配套资产 |
| `lingbot_va`、`vla_jepa` | `lingbot-va`、`vla-jepa` | 专用视觉语言处理与动作模型 |
| `molmoact2`、`wall_x` | `molmoact2`、`wallx` | 模型专用输入、坐标/动作处理与权重 |

共 18 种策略，另保留 RTC、PiGemma 和策略内部共享代码；RTC 不是独立策略。
旧 `pi05_openpi` 目录没有 Python 源实现，未作为可用策略迁入。

源码迁移不代表所有训练任务已验证。当前验证覆盖注册隔离、源码算法对照、Diffusion 小模型及原 ACT 适配；
其他模型的真实数据训练、预训练资产加载、分布式、RL 和真机部署仍需分别验证。
标准名称的旧/官方 checkpoint 不自动改写成集成层 checkpoint；需要使用对应的官方入口，或另行明确转换配置与处理器。

开发时从对应 `configuration_*.py`、`modeling_*.py`、`processor_*.py` 入手。
Tensor 模型的 `reset()`、`select_action(batch)` 接收的是该策略的处理后输入，**不是 Host snapshot**；
动作还须经过对应 postprocessor 还原，再接机器人执行适配。

## 5. 官方 ACT 真机评估

当前真机适配器只接受官方 ACT 完整 checkpoint，使用保存的预处理、归一化统计和后处理。
新增策略未自动开放真机执行；复制模型文件不会取消动作坐标和处理器检查。

先启动 Host，清空运动路径，再在 PC 执行：

```bash
alohamini evaluate \
  --policy.path /path/to/pretrained_model \
  --training-dataset /path/to/training_export \
  --host 192.168.8.161 --robot_model alohamini2pro \
  --task "pickup" --episode_time 30
```

- `--training-dataset` 指向该模型实际使用的平台 LeRobot v3 导出目录，保留 `meta/info.json` 和 `meta/alohamini.json`。同维度不代表同字段顺序或同标定；不要替换为另一份数据的元数据。
- `--fps` 默认 30，须与训练数据一致。`--device cpu` 可使用 CPU。
- `--dataset eval_pickup` 可选记录评估回合到工作区，默认不保存。保护和退出行为与原生 `alohamini evaluate` 相同。
- `--policy.n_action_steps 10`：每次预测后执行前 10 步再重新预测，不超过 checkpoint 的 chunk_size。
- `--policy.n_action_steps 1 --policy.temporal_ensemble_coeff 0.01`：每步重新预测，融合不同动作块对同一时刻的预测。
- `--policy.temporal_ensemble_coeff none` 关闭融合；`0` 表示等权融合。省略这两个选项时沿用 checkpoint，不修改模型文件。

`alohamini-lerobot-evaluate` 保留为同一评估入口的别名。自定义 Python 策略仍使用 `alohamini evaluate --policy module:factory`。

state 按导出时的字段组合、顺序和单位生成，可使用位置、速度或电流；所选反馈缺失或过期时停止，不能用零补齐。纯视觉模型不输入真实关节 state。Host 必须开启策略所需相机，名称与分辨率须与训练一致。

动作必须仍是具名的 Host 绝对关节目标、底盘速度和升降高度。该入口不接受 `am_act` 自定义架构、PEFT、相对动作处理器或末端增量 checkpoint。

### Python 调用

```python
from alohamini_lerobot.policy import LeRobotPolicy

policy = LeRobotPolicy.from_pretrained(
    "/path/to/pretrained_model", "/path/to/training_export",
    device="cuda", task="pickup",
)
```

加载本身不连接机器人。`policy.reset()` 清空动作队列和处理器历史，`policy.select_action(snapshot)` 返回具名动作；通过原生 `run_evaluation(..., fps=int(policy.fps))` 执行时保留控制权及保护检查。不要绕过检查直接将网络输出发送到机械臂。
