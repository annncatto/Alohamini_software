# 数据整理、策略训练与评估部署

## 1. 准备环境与数据

完成 [PC 环境安装](install.md) 后，在仓库根目录执行：

```bash
cd ~/Alohamini
conda activate alohamini
python -m pip install -e '.[learning]'

```

检查结构错误、逐相机画面、示教动作与采样时间。`VALID` 表示结构检查通过，不等于适合训练。
采样警告不强制填写确认文字。相机间隔、缺测字段和夹爪开合不切断序列；回合及明确的控制中断才作为边界。
改变格式、生成 MP4 或修改标称 FPS 都不会自动修复时间配对。

工作文件默认放在 `~/Alohamini_workspace/`：`datasets/` 存数据，`runs/` 存模型，
`logs/training/` 存训练日志、PID 和启动配置；可通过 `ALOHAMINI_WORKSPACE` 更换工作区。

## 2. 整理示教回合

先停止向源数据集录制。除只读 `info` 外，编辑必须使用尚不存在的输出目录，不修改源数据。
episode 从 **0** 编号；先查看回合数量，再选择要删除的编号：

```bash
alohamini dataset edit --dataset task_demo \
  --operation.type info --operation.show_features true

alohamini dataset edit --dataset task_demo \
  --output ~/Alohamini_workspace/datasets/task_demo_ready \
  --operation.type delete_episodes --operation.episode_indices '[3, 8]'

alohamini dataset check ~/Alohamini_workspace/datasets/task_demo_ready \
  --decode-images --decode-videos
```

`--dataset` 接受工作区中的名称，也可换成 `--root /完整路径`。保留回合按原顺序重新编号，
图像、state、action 和保护记录一同保留，不重新配对；不能删除全部回合。

其他原生数据编辑操作共用 `--root`、`--output`：

| `--operation.type` | 参数示例 |
| --- | --- |
| `split` | `--operation.splits '{"train": 0.8, "val": 0.2}'` 或 `'{"train": [0, 2], "val": [1]}'` |
| `merge` | `--operation.roots '["/数据集A", "/数据集B"]'`，无需 `--root` |
| `remove_feature` | `--operation.feature_names '["observation.images.wrist_right", "observation.motor_temperature_raw"]'` |
| `modify_tasks` | `--operation.new_task "拿起积木"`；可加 `--operation.episode_tasks '{"2": "放下积木"}'` |
| `recompute_stats` | 默认统计数值；`--operation.skip_image_video false` 同时统计图像 |
| `convert_image_to_video` | JPEG/PNG 转原生 MP4；`--operation.episode_indices '[0, 2]'` 可选回合 |
| `reencode_videos` | 对原生 MP4 重编码，如 `--operation.rgb_encoder.crf 23` |
| `info` | `--operation.show_features true`；不需要 `--output` |

拆分保持原顺序，输出在 `<output>/train` 等子目录；比例和小于 1 时余下回合不选入。
合并要求 FPS、字段、相机、标定和动作坐标一致。训练/验证应按原始完整回合隔离，
同一示教的不同处理副本不能分别放入训练集和验证集。

编辑输出是原生格式版本 3，**不是 LeRobot v3**。可继续检查、编辑、训练和回放，
但不能追加录制。删掉 action 后不能做行为克隆或回放；删掉反馈字段会同步删除其有效性掩码。
速度/电流训练须保留对应反馈、掩码和采样时间；不能单独删除仍在使用的反馈时间。

视频默认 H.264、CRF 18、GOP 2、`yuv420p`，每回合每相机一个 MP4。
可用 `--operation.rgb_encoder.vcodec`、`.crf`、`.g`、`.preset`、`.pix_fmt` 调整。
转换不增删帧、不改时间，有损编码会改变像素；`yuv420p` 要求偶数尺寸，不会自动裁剪。
只想检查录像时用 `dataset preview` 即可。编辑后预览随回合重新编号；相机字段变更后可重新生成。

`recompute_stats` 不统计无效反馈，完全不可用的维度为 `null`、`count=0`。
`--operation.relative_action true --operation.chunk_size 50` 仅统计完整动作块的相对位置分布，
不修改 action，不是末端增量转换；默认排除夹爪，底盘仍为绝对速度。
训练器会重新计算训练集归一化，不直接使用全数据集统计。

结构可确定的索引或视频问题可尝试修复到新目录：

```bash
alohamini dataset repair /path/to/dataset --output /path/to/dataset_repaired
alohamini dataset check /path/to/dataset_repaired --decode-images --decode-videos
```

修复支持原生数据和受支持的 LeRobot 数据，不猜测缺失图像或不明确的对应关系。
失败留下的 `*.pending-*` 不作为完成的数据集使用。
编辑还支持 `--config_path /path/edit.json` 和 `python -m alohamini.datasets.edit`；
命令行覆盖同名配置。编辑器直接处理原生格式，LeRobot 格式使用导出、检查和修复入口。

## 3. 选择训练数据与输入

ACT 和 AM-ACT 共用数据读取器，不需要为每种存储格式写一份训练脚本。
换 `--dataset.root` 选择数据，换 `--state` 选择输入，换 `--policy.type` 选择网络：

| 数据 | 当前训练支持 | state 选择 |
| --- | --- | --- |
| 原生录制、删除/合并等编辑结果 | 支持，包括原生 MP4 图像存储 | 默认 18 维；可选反馈组或纯视觉 |
| 平台导出的默认图像型 LeRobot v3 | 支持 | 默认 18 维；可选保留的反馈组或纯视觉 |
| 平台导出的纯视觉 v3 | 支持 | 自动无 state |
| 导出时重组了 state 的 v3 | 当前不能作为数值 state 直接训练；可用纯视觉 | 速度/电流实验优先读原生数据 |
| 外部 v3、MP4 型 v3、任意自定义格式 | 当前读取器不直接支持 | 需要存储及字段适配，不能只改目录名或版本号 |

原生 MP4 和 MP4 型 LeRobot v3 是两种存储布局，不要混为一谈。
v3 读取需要平台保留的 `meta/alohamini.json` 和 `meta/safety/`，用于标定、动作含义和时间检查。
处理后的数据还须保留真实时间、回合边界、所需图像和 action；需要的反馈不能用零伪造。

### 导出 v3 与纯视觉副本

原生数据可以直接训练，只有需要 v3 时才导出：

```bash
alohamini dataset export ~/Alohamini_workspace/datasets/task_demo_ready \
  --output ~/Alohamini_workspace/datasets/task_demo_v3 --format lerobot-v3

alohamini dataset export ~/Alohamini_workspace/datasets/task_demo_v3 \
  --output ~/Alohamini_workspace/datasets/task_demo_vision \
  --format lerobot-v3 --vision-only

alohamini dataset check ~/Alohamini_workspace/datasets/task_demo_vision --decode-images
```

默认 state 是双臂关节位置 14 维、底盘速度 3 维、升降高度 1 维。
纯视觉副本去掉数值观测输入，保留图像、action、索引及必要元数据；不是无标签数据。
导出不改变帧数、配对或时间戳。

### 输入与输出含义

| 训练参数 | 输入 |
| --- | --- |
| 省略 `--state`（`auto`） | 有原始 state 时用 18 维，无 state 时用纯视觉 |
| `--state=none` | 仅图像，普通数据集也可用，不必另复制数据 |
| `--state=joint_position,base_velocity,lift_height` | 原 18 维 state |
| `--state=joint_velocity,joint_current` | 双臂 28 维速度/电流，包含夹爪 |
| `--cameras='["forward","wrist_right"]'` | 只使用这些已记录相机；省略时使用全部 |

字段按给定顺序组装。位置沿用 Host 标定坐标，速度为对应坐标单位/秒，电流为 A，
底盘为 m/s、deg/s，升降高度为 mm；电流不等于关节力矩。
action 保持双臂绝对位置目标、底盘速度、升降绝对高度目标。
改成末端增量或方向分类标签，需要配套训练契约和部署解码器，不能直接送给现有 Host 接口。

新增存储格式应扩展数据读取层，统一输出图像、具名 state/action 和回合/时间信息；
采样、归一化和 ACT／AM-ACT 继续共用。格式兼容与动作语义兼容是两件事。

## 4. 启动训练

正式入口为 `python -m alohamini.learning.train`，无需启动 Jupyter。
以下假设整理后的数据至少有两个回合，留出编号 1 验证，其余回合训练。
只用一个回合也能训练，但不能据训练 loss 判断泛化；此时省略验证选项和 `--eval_steps`。

### ACT：原生数据或默认 v3

```bash
python -m alohamini.learning.train \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --policy.type=act --policy.device=cuda --policy.chunk_size=30 \
  --output_dir="$HOME/Alohamini_workspace/runs/act_01" \
  --steps=100000 --batch_size=2 --save_freq=10000 --log_freq=100 \
  --eval_steps=1000 --background
```

换成 `task_demo_v3` 即读默认 v3；增加 `--state=none` 即用纯视觉输入。

### AM-ACT：纯视觉 v3

```bash
python -m alohamini.learning.train \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_vision" \
  --dataset.eval_episodes='[1]' \
  --policy.type=am_act --policy.device=cuda --policy.chunk_size=30 \
  --output_dir="$HOME/Alohamini_workspace/runs/am_act_vision_01" \
  --steps=100000 --batch_size=2 --save_freq=10000 --log_freq=100 \
  --eval_steps=1000 --background
```

AM-ACT 同样可用原生或默认 v3；选择速度/电流时，改用保留反馈的原生数据并加
`--state=joint_velocity,joint_current`。ACT 与 AM-ACT 都支持有 state 和纯视觉训练。

`--background` 创建独立日志、PID、启动配置并打印路径，不需要再套 `nohup`。
例如 `act_01` 的日志与 PID 为：

```bash
tail -f ~/Alohamini_workspace/logs/training/act_01.log
cat ~/Alohamini_workspace/logs/training/act_01.pid
```

运行名称和输出目录须未使用。日志中的 loss、梯度范数、加载与更新耗时用于观察训练进展，
不是任务成功率。可用 `--dataset.episodes='[0,2,3]'` 指定训练回合；与验证回合不能重叠。
训练入口目前接收一个数据根目录，验证用其中的回合编号；若已拆成独立目录，
训练只读 train 目录，之后用离线评估脚本读取 val 目录。

当前支持单设备 FP32、AdamW，不支持 AMP 或分布式训练。
图像默认 `[480,640]`（高、宽），可用 `--image_size='[480,640]'` 明确设置。
默认 ResNet18 ImageNet 初始化，首次使用可能下载并缓存到工作区 `pretrained/`，不走 Hub。
离线机器须预先准备骨干权重；`--policy.pretrained_backbone_weights=null` 表示随机初始化。
checkpoint 加载不会再次下载骨干权重。

### 模型配置

可用 `--config examples/learning/act.json` 或 `am_act_v3.json` 代替长命令；
先修改其中的数据路径、运行名称和回合编号，命令行参数优先。
JSON 的显式 `state` 优先于自动选择。示例参数用于跑通实验，不代表收敛或泛化保证。

AM-ACT 的 `fixed_action_dims`、`action_loss_groups/weights`、`observation_state_dims`、
底盘分类头和类别权重均可配置，不自动沿用某次实验权重。例如在 JSON 的 `model` 中：

```json
{
  "discrete_action_dims": [14],
  "discrete_action_values": [[-0.1, 0.0, 0.1]],
  "discrete_action_class_weights": [[1.0, 1.0, 1.0]]
}
```

这里第 14 维是底盘 x 速度，类别值必须改成真实示教的物理速度，零方差维度不能分类。
类别中心按训练统计归一化，输出恢复物理值。
`inference_action_scale_dims/scale` 在反归一化后缩放，仅用于明确需要缩放的速度命令。

ACT／AM-ACT 从当前行 action 开始构造 chunk；回合边界和明确的控制中断会截断窗口，
尾部重复末动作并标记 padding，不计入动作损失。相机抖动和夹爪开合不分段。
读取器按记录行顺序取样，不插值、不重采样；时间警告仍需检查，不能将严重漏采视为等间隔数据。
缺测字段只排除需要它的训练起点/窗口，原始行不删除或重新编号；例如电流缺失不会排除纯视觉样本，
也不会从其他样本的未来动作中删去对应 action。均值/标准差按实际使用的训练字段逐原始行统计，不重复计入窗口重叠或 padding。
边界位置与简短原因可查 `samples.boundaries`，随 checkpoint 保存，不作为模型输入。
报告中的警告和可选 `review_note` 同样保留；文件损坏和索引错误仍须先修复。

自定义策略可用 `AlohaMiniDataset(..., delta_indices={"observation.state": [-1, 0], "action": [0, 1, 2]})`
分别指定各字段的历史/未来行偏移，返回对应的 `<字段名>_is_pad`。未指定窗口的字段返回单帧；
已有 `chunk_size=K` 写法等价于 `delta_indices={"action": list(range(K))}`。
奖励等额外标签可从已添加这些字段的 AlohaMini v3 副本读取，不会自动生成，也不改变原生采集 schema。
窗口支持不代表相应策略的训练器或部署接口已经接通。
`sample_indices` 将训练样本索引映射到原始 `rows`、`records`、`locations`；筛选样本不压缩原始时间轴。

## 5. Checkpoint 与续训

每 `save_freq` 步及最后一步保存一次：

```text
runs/act_01/
├── train.json
├── metrics.jsonl
├── checkpoints/<step>/
│   ├── pretrained_model/    # policy.json、权重、train_config.json
│   └── training_state/      # 优化器、随机数和恢复状态
├── checkpoints/last         # 最近的完整保存点
├── checkpoint               # 该保存点的 pretrained_model
└── offline-evaluation.json # 指定验证回合时生成
```

恢复到更大的总训练步数，不是额外增加这么多步：

```bash
python -m alohamini.learning.train \
  --config_path="$HOME/Alohamini_workspace/runs/act_01/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true --steps=150000 --background
```

恢复要求相同数据、模型、输入与归一化及优化配置，并使用最近完整保存点。
早期没有 `training_state/` 的权重不能精确续训，中断后未保存的更新不能恢复。
平台原生 checkpoint 不能直接交给 LeRobot `from_pretrained`。

## 6. 离线评估

训练时有验证回合会自动保存离线结果。也可独立评估任意完整 checkpoint，无需 Notebook 或机器人：

```bash
python -m alohamini.learning.evaluate \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint \
  --dataset.root ~/Alohamini_workspace/datasets/task_demo_ready \
  --dataset.episodes '[1]' --device cuda \
  --output ~/Alohamini_workspace/runs/act_01/offline-review.json
```

已拆分的验证集可换成其目录，并使用该目录内重新编号后的回合。
评估按 checkpoint 自动恢复 state 顺序、相机、图像尺寸、chunk 和训练归一化，
不从验证集重算统计。数据须保持相同标定、动作含义和 FPS；普通 v3 与原生数据可以共用对应契约。
纯视觉模型可评估有 state 的数据，但有 state 的模型不能评估已删除所需字段的数据。

输出逐 action 维度的原单位 MAE、有效动作步数、样本数量和数据检查报告。
这比较的是预测动作块与记录标签，不是闭环任务成功率，也不衡量时间融合后的实际跟踪效果。
同一次示教的处理副本仍是同一份示教，不能充当独立验证数据。

## 7. 真机部署与评估

默认仍由 **PC 加载模型并推理，树莓派运行 Host**，不把 PyTorch 或训练环境装到树莓派。
换 PC 时复制完整 `checkpoint` 所指目录（含 `policy.json` 与权重），不要只复制符号链接。
原生模型的输入/输出、标定和归一化信息已随模型保存，不需要额外指定训练数据集。

先完成离线检查，确认动作含义、相机和标定一致，支撑机器人并清空运动路径。
仅用于几步流程测试的权重不要上机。停止其他遥操、录制或回放控制程序。
在树莓派启动 Host，开启模型所需的相机；例如：

```bash
conda activate alohamini_host
alohamini host --robot_model alohamini2pro --cameras forward wrist_right
```

下面 PC 命令会连接真机并执行动作。`--fps` 必须与训练数据一致；例如 25 Hz 数据就填 25：

```bash
conda activate alohamini
alohamini evaluate --host <PI_IP> --robot_model alohamini2pro \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint \
  --device cuda --fps 30 --episode_time 10 --num_episodes 1 \
  --policy.n_action_steps 10 --policy.temporal_ensemble_coeff none
```

`n_action_steps=10` 表示预测一个 chunk 后执行前 10 步，再重新预测，不能超过 chunk_size。
若每步重预测并融合多个动作块，改成：

```text
--policy.n_action_steps 1 --policy.temporal_ensemble_coeff 0.01
```

`none` 关闭融合，`0` 为等权融合；这些执行参数不需要重新训练，也不修改 checkpoint。
每回合 reset 策略历史；保护事件后不自动恢复动作，Ctrl+C 结束评估并请求停止。
同步推理耗时会限制实际执行频率，Host 的 50 Hz 不等于策略每秒推理 50 次。

加 `--dataset eval_act_01 --task "拿起物体"` 可保存本地评估回合，之后按相同流程检查与整理。
是否成功还须结合录像、跟踪误差、推理耗时和真实任务完成情况判断。

外部 LeRobot ACT checkpoint 使用 [LeRobot 策略适配](lerobot.md)，不要与平台原生权重混用。
自定义策略参考 `examples/learning/custom_policy.py`：提供 `robot_metadata`、`reset()`、
`select_action(snapshot)`，复用现有评估执行接口；网络开发不要求继承 LeRobot Policy。
