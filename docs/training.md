# 数据整理、训练与部署

## 1. 准备

完成 [PC 安装](install.md)，在仓库根目录执行：

```bash
cd ~/Alohamini
conda activate alohamini
```

默认工作目录为 `~/Alohamini_workspace/`：`datasets/` 保存数据，`runs/` 保存模型，`logs/training/` 保存日志与 PID。

## 2. 检查与编辑数据

```bash
alohamini dataset edit --dataset task_demo \
  --operation.type info --operation.show_features true

alohamini dataset check ~/Alohamini_workspace/datasets/task_demo \
  --decode-images --decode-videos

alohamini dataset edit --dataset task_demo \
  --output ~/Alohamini_workspace/datasets/task_demo_ready \
  --operation.type delete_episodes --operation.episode_indices '[3, 8]'
```

episode 从 0 编号。`--dataset` 指工作区中的名称，也可换成 `--root /完整路径`。
输出目录须不存在；保留回合按原顺序重新编号，图像、state/action 和时间记录保持配对。

其他编辑操作共用 `--root`、`--output`：

| `--operation.type` | 参数示例 |
| --- | --- |
| `split` | `--operation.splits '{"train": 0.8, "val": 0.2}'` 或 `'{"train": [0, 2], "val": [1]}'` |
| `merge` | `--operation.roots '["/数据集A", "/数据集B"]'`，无需 `--root` |
| `remove_feature` | `--operation.feature_names '["observation.images.wrist_right"]'` |
| `modify_tasks` | `--operation.new_task "拿起积木"`；可加 `--operation.episode_tasks '{"2": "放下积木"}'` |
| `recompute_stats` | 默认统计数值；`--operation.skip_image_video false` 同时统计图像 |
| `convert_image_to_video` | JPEG/PNG 转 MP4；`--operation.episode_indices '[0, 2]'` 可选回合 |
| `reencode_videos` | MP4 重编码，如 `--operation.rgb_encoder.crf 23` |
| `info` | `--operation.show_features true`；无需 `--output` |

拆分结果位于 `<output>/train` 等子目录。合并要求 FPS、字段、相机、标定和动作坐标一致。
训练与验证按完整回合隔离，同一示教的不同处理副本不得分置两边。

编辑器输出 AlohaMini 数据格式版本 3，可继续训练、编辑和回放，不支持追加录制。
速度/电流训练须保留对应反馈、有效性掩码和采样时间。

视频默认 H.264、CRF 18、GOP 2、`yuv420p`，每回合每相机一个 MP4。
可用 `--operation.rgb_encoder.vcodec`、`.crf`、`.g`、`.preset`、`.pix_fmt` 调整。
转换保持帧数和时间；`yuv420p` 要求偶数图像尺寸。

`recompute_stats` 排除无效反馈，使用 FP64 数值统计和精确分位数；诊断保存在 `meta/stats_info.json`。
`--operation.relative_action true --operation.chunk_size 50` 统计动作块的关节增量分布，不改写 action。
训练归一化统计由训练集重新计算。

### 修复

```bash
alohamini dataset repair /path/to/dataset --output /path/to/dataset_repaired
alohamini dataset check /path/to/dataset_repaired --decode-images --decode-videos
```

支持 AlohaMini 和 LeRobot 数据的可确定索引、媒体引用问题；缺失图像或配对不明时拒绝修复。
`*.pending-*` 是未完成输出。修复不会消除实际漏采或时间偏差。

## 3. 选择数据与输入

| 数据 | 训练支持 |
| --- | --- |
| AlohaMini 录制或编辑结果，含 MP4 存储 | 支持位置、可用反馈组或纯视觉输入 |
| 平台导出的默认图像型 LeRobot v3 | 支持，须保留 `meta/alohamini.json` 和 `meta/safety/` |
| 平台导出的纯视觉 v3 | 支持 ACT／AM-ACT |
| 导出时重组了 state 的 v3 | 仅支持纯视觉；速度/电流训练使用 AlohaMini 数据 |
| 外部 v3 或 MP4 型 LeRobot v3 | 当前训练读取器不直接支持 |

### 导出 v3 与纯视觉副本

```bash
alohamini dataset export ~/Alohamini_workspace/datasets/task_demo_ready \
  --output ~/Alohamini_workspace/datasets/task_demo_v3 --format lerobot-v3

alohamini dataset export ~/Alohamini_workspace/datasets/task_demo_v3 \
  --output ~/Alohamini_workspace/datasets/task_demo_vision \
  --format lerobot-v3 --vision-only
```

默认 state 为双臂位置 14 维、底盘速度 3 维、升降高度 1 维。
纯视觉副本保留图像和 action，去掉数值观测输入。导出不改变配对、帧数或时间戳；图像内嵌于 Parquet。

### 输入参数

| 参数 | 含义 |
| --- | --- |
| 省略 `--state` | 有位置 state 时使用 18 维，否则使用纯视觉 |
| `--state=none` | 纯视觉，无需另存数据副本 |
| `--state=joint_position,base_velocity,lift_height` | 18 维位置、底盘速度与高度 |
| `--state=joint_velocity,joint_current` | 双臂 28 维速度/电流，含夹爪 |
| `--cameras='["forward","wrist_right"]'` | 使用指定相机；省略时使用全部记录相机 |

字段按参数顺序组装。关节位置使用标定坐标，关节速度为对应坐标单位/秒，电流为 A；
底盘速度为 m/s、deg/s，升降高度为 mm。action 为双臂绝对位置目标、底盘速度及升降高度目标。


## 4. 训练

以下命令留出 episode 1 做验证，其余回合训练。数据只有一个回合时，省略 `--dataset.eval_episodes` 和 `--eval_steps`。

### ACT

```bash
python -m alohamini.learning.train \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --policy.type=act --policy.device=cuda --policy.chunk_size=30 \
  --output_dir="$HOME/Alohamini_workspace/runs/act_01" \
  --steps=100000 --batch_size=2 --save_freq=10000 --log_freq=100 \
  --eval_steps=1000 --background
```

`--state=none` 使用纯视觉；`--image_size='[480,640]'` 设置高、宽。
ResNet18 默认加载 ImageNet 权重，首次下载缓存到工作区 `pretrained/`。
离线使用须提前准备；`--policy.pretrained_backbone_weights=null` 改为随机初始化。

### AM-ACT

```bash
python -m alohamini.learning.train \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_vision" \
  --dataset.eval_episodes='[1]' \
  --policy.type=am_act --policy.device=cuda --policy.chunk_size=30 \
  --output_dir="$HOME/Alohamini_workspace/runs/am_act_vision_01" \
  --steps=100000 --batch_size=2 --save_freq=10000 --log_freq=100 \
  --eval_steps=1000 --background
```

也支持含 state 的数据。速度/电流实验改用保留反馈的数据目录，并加 `--state=joint_velocity,joint_current`。
底盘分类头可在配置的 `model` 中设置：

```json
{
  "discrete_action_dims": [14],
  "discrete_action_values": [[-0.1, 0.0, 0.1]],
  "discrete_action_class_weights": [[1.0, 1.0, 1.0]]
}
```

第 14 维为底盘 x 速度；类别值按实际示教速度填写。分类输出会恢复为物理速度。

### SmolVLA

准备基座及视觉语言模型的配置、tokenizer 文件：

```bash
hf download lerobot/smolvla_base --local-dir "$HOME/Alohamini_workspace/pretrained/smolvla_base"
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --include '*.json' '*.jinja' '*.model' '*.txt' \
  --local-dir "$HOME/Alohamini_workspace/pretrained/smolvlm_assets"

python -m alohamini.learning.train \
  --policy.type=smolvla --policy.device=cuda \
  --policy.path="$HOME/Alohamini_workspace/pretrained/smolvla_base" \
  --policy.vlm_model_name="$HOME/Alohamini_workspace/pretrained/smolvlm_assets" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --output_dir="$HOME/Alohamini_workspace/runs/smolvla_01" \
  --batch_size=2 --policy.compile_model=false \
  --save_freq=10000 --log_freq=100 --background
```

自动预设 `smolvla-realworld-alohamini-v1`：位置 state、任务文本、512×512 等比例补边、
50 步动作块、10 步流匹配采样、冻结 VLM、BF16、AdamW 与 warmup/cosine。
训练步数默认 200000；来源及覆盖参数保存在 `paper_preset` 和训练配置中。
checkpoint 自带 tokenizer 和骨干配置。此入口不支持纯视觉、RTC、PEFT 或 ACT 时间融合。

### π0.5

准备转换后的 OpenPI PyTorch `model.safetensors` 和 PaliGemma SentencePiece tokenizer。
不支持直接读取 Orbax/JAX checkpoint。

```bash
python -m alohamini.learning.train \
  --policy.type=pi05 --policy.device=cuda \
  --policy.path=/path/to/pi05_pytorch \
  --policy.tokenizer_path=/path/to/paligemma_tokenizer.model \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --cameras='["forward","wrist_right"]' \
  --output_dir="$HOME/Alohamini_workspace/runs/pi05_01" \
  --batch_size=1 --save_freq=1000 --background
```

自动预设 `openpi-pi05-alohamini-v1`：位置 state、任务文本、224×224 等比例补边、
50 步动作块、q01/q99 归一化、AdamW 与 warmup/cosine。
双臂各 6 个关节预测相对当前 state 的增量，执行前还原为绝对目标；夹爪、底盘速度、升降高度保持原义。
网络参数用 `--policy.network='{"action_horizon":30}'` 覆盖；checkpoint 自带 tokenizer。

SmolVLA／π0.5 已完成小模型训练与保存加载验证，完整基座和真机任务尚未验收；
完整基座的显存需求须单独评估，示例 batch 不保证适用于 8 GB GPU。

### 日志与配置

```bash
tail -f ~/Alohamini_workspace/logs/training/act_01.log
cat ~/Alohamini_workspace/logs/training/act_01.pid
```

`--background` 后台运行并输出日志、PID 和配置路径。输出目录须未使用。
`--config examples/learning/act.json` 可代替长命令；先修改数据路径、输出目录和回合编号，命令行参数优先。

ACT／AM-ACT 按记录行构造动作块，在回合边界或明确控制中断处截断，尾部 padding 不参与动作损失。
普通相机抖动不分段；读取器不自动重采样。缺失反馈只影响需要该字段的样本。

### 优化器、精度与多卡

在训练命令后追加所需参数：

```text
--optimizer.type=adamw --optimizer.lr=0.0001
--optimizer.weight_decay=0.0001 --optimizer.betas='[0.9,0.999]'
--optimizer.eps=1e-8 --optimizer.grad_clip_norm=10
--gradient_accumulation_steps=4 --mixed_precision=bfloat16
```

优化器支持 `adamw`、`adam`、`sgd`，默认沿用策略配置。
ACT／AM-ACT 默认无调度，SmolVLA 使用 `cosine`，π0.5 使用 `warmup_cosine`；
`--scheduler.type=none` 禁用调度。可覆盖 `warmup_steps`、`decay_steps`、`decay_lr`。

| 配置 | 追加参数 |
| --- | --- |
| 单卡 | 默认 |
| DDP，每卡完整模型 | `--num_processes=2 --background` |
| ACT／AM-ACT FSDP2 分片 | `--distributed_backend=fsdp2 --num_processes=2 --background` |

有效 batch 通常为 `batch_size × 进程数 × 累积次数`；`steps` 按成功的优化器更新计数。
多进程尾批默认重复样本补齐，`--drop_last=true` 改为丢弃。
多节点使用 `torchrun ... -m alohamini.learning.train --config train.json`，各节点须共享输出目录并使用相同数据和资产路径。

FSDP2 需要每个进程有足够 CPU 内存构建模型，保存时同时导出分片训练状态和完整推理权重。
当前仅 ACT／AM-ACT 支持 FSDP2，已验证单卡；实际多 GPU／多节点仍需验收。ZeRO、张量并行未接入。

## 5. 保存与续训

每 `save_freq` 步及训练结束时保存 checkpoint：

| 路径 | 内容 |
| --- | --- |
| `runs/<名称>/train.json` | 训练配置 |
| `runs/<名称>/metrics*.jsonl` | 训练指标 |
| `runs/<名称>/checkpoints/<step>/pretrained_model/` | 权重、`policy.json` 和配置 |
| `runs/<名称>/checkpoints/<step>/training_state/` | 优化器、随机数、采样和恢复状态 |
| `runs/<名称>/checkpoints/last` | 最近完整保存点 |
| `runs/<名称>/checkpoint` | 最近保存点的模型目录 |

```bash
python -m alohamini.learning.train \
  --config_path="$HOME/Alohamini_workspace/runs/act_01/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true --background
```

续训保持数据、模型、输入、优化器、调度器、进程数、累积次数及精度一致。
未启用调度时可用 `--steps=150000` 增加总步数；启用调度时保持原定总步数。
仅有权重、没有 `training_state/` 的目录不能恢复完整训练状态。

## 6. 离线评估

```bash
python -m alohamini.learning.evaluate \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint \
  --dataset.root ~/Alohamini_workspace/datasets/task_demo_ready \
  --dataset.episodes '[1]' --device cuda \
  --output ~/Alohamini_workspace/runs/act_01/offline-review.json
```

使用 checkpoint 的相机、字段、图像尺寸及归一化统计，输出各动作维度的原单位 MAE 和数据检查报告。
数据须保持相同标定、动作含义和 FPS；验证回合不得与训练示教重复。离线误差不等于真机任务成功率。

## 7. 真机评估

PC 加载模型，树莓派运行 Host。停止其他运动控制程序，确认标定、相机和机器人姿态，并清空运动区域。

树莓派：

```bash
conda activate alohamini_host
alohamini host --robot_model alohamini2pro --cameras forward wrist_right
```

PC（下列命令会驱动机器人）：

```bash
conda activate alohamini
alohamini evaluate --host <PI_IP> --robot_model alohamini2pro \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint \
  --device cuda --fps 30 --episode_time 10 --num_episodes 1 \
  --policy.n_action_steps 10 --policy.temporal_ensemble_coeff none
```

`--fps` 与训练数据一致。每个动作块执行前 `n_action_steps` 步，再重新预测。
ACT／AM-ACT 每步预测并融合时，使用 `--policy.n_action_steps 1 --policy.temporal_ensemble_coeff 0.01`；
`none` 关闭融合，`0` 为等权融合。更换执行参数无需重新训练。

SmolVLA／π0.5 评估须增加 `--task "拿起物体"`，不使用 ACT 时间融合。
`--dataset eval_01 --task "拿起物体"` 保存评估回合。Ctrl+C 结束；保护事件后不会自动恢复动作。
推理耗时限制实际执行频率。

跨机器使用时复制完整 checkpoint 目录，勿只复制符号链接。
自定义策略接口见 [客户端接口](host-protocol.md#python-策略评估)，示例为 `examples/learning/custom_policy.py`。
Notebook 入口见 [Notebook](notebook.md)，LeRobot 权重使用见 [LeRobot](lerobot.md)。
