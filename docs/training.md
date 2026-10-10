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

先审核数据，再用 `--dataset.episodes` 和 `--dataset.eval_episodes` 明确选择训练与验证回合。
训练器不根据 `fault`、`watchdog_active`、`joint_holds` 或反馈年龄阈值自动剔除窗口；
这些记录保留用于审核。所选输入存在缺测或无效 mask 时，会报告 episode、frame 和字段并停止，
由使用者修复数据、排除该 episode 或调整输入字段；不会填零或悄悄跳过。
缺失字段、形状、有限数值、单位和媒体对应关系仍执行完整性检查。

每个 episode 内按算法构造窗口，末尾重复边界值并提供 padding mask；不跨 episode 拼接。
算法自身的尾帧配置仍生效（例如 Diffusion 的 `drop_n_last_frames`），
日志中的 `window_excluded` 仅计窗口规则排除的起点，不包含安全标志过滤。
中间坏片段应在数据处理阶段显式处理；不要删除中间时间后把两侧当作连续序列。
Notebook 的原始反馈诊断图保留全部行，缺测行显示为 NaN 并列出位置。

编辑器输出 AlohaMini 数据格式版本 3，可继续训练、编辑和回放，不支持追加录制。
速度/电流训练须保留对应反馈、有效性掩码和采样时间。

视频默认 H.264、CRF 18、GOP 2、`yuv420p`，每回合每相机一个 MP4。
可用 `--operation.rgb_encoder.vcodec`、`.crf`、`.g`、`.preset`、`.pix_fmt` 调整。
转换保持帧数和时间；`yuv420p` 要求偶数图像尺寸。

`recompute_stats` 排除无效反馈，使用 FP64 数值统计和精确分位数；诊断保存在 `meta/stats_info.json`。
`--operation.relative_action true --operation.chunk_size 50` 统计动作块的关节增量分布，不改写 action。
训练归一化统计只使用训练回合；可自动计算或复用下方提前生成的统计文件。

### 修复

```bash
alohamini dataset repair /path/to/dataset --output /path/to/dataset_repaired
alohamini dataset check /path/to/dataset_repaired --decode-images --decode-videos
```

支持 AlohaMini 和 LeRobot 数据的可确定索引、媒体引用问题；缺失图像或配对不明时拒绝修复。
`*.pending-*` 是未完成输出。修复不会消除实际漏采或时间偏差。

## 3. 选择数据与输入

平台统一录制 LeRobot v3 数据集，图像按相机保存为 MP4，额外舵机反馈作为数值列保留：

| 目录 | 内容 |
| --- | --- |
| `data/` | Parquet：state、action、索引、舵机反馈及有效掩码 |
| `videos/` | 各相机的 MP4；读取器按 episode 元数据和时间索引取帧 |
| `meta/` | 字段定义、任务、统计、episode 索引，以及标定和实际采样时间等平台扩展信息 |



### 导出 v3 与纯视觉副本

新录制的数据无需迁移。旧版平台数据若已有完整的 `previews/` 视频，可在仓库根目录运行：

```bash
python scripts/migrate_dataset_v3.py /path/to/old_dataset \
  --output /path/to/dataset_v3
```


```bash
alohamini dataset export ~/Alohamini_workspace/datasets/task_demo_ready \
  --output ~/Alohamini_workspace/datasets/task_demo_vision \
  --format lerobot-v3 --vision-only
```

默认 state 为双臂位置 14 维、底盘速度 3 维、升降高度 1 维。
纯视觉副本保留视频和 action，去掉数值观测输入；视频直接复制，不改变配对、帧数或时间戳。早期导出的图片内嵌 Parquet 数据仍可读取，但不是当前采集格式。

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

### 统计准备（可选）

查看 AlohaMini 数据集全部数值字段的统计，不改写数据：

```bash
alohamini dataset stats ~/Alohamini_workspace/datasets/task_demo_ready \
  --output ~/Alohamini_workspace/logs/task_demo_stats.json
```

按训练 JSON 中的策略、输入字段、时间窗口和训练回合生成归一化统计（支持 AlohaMini 和 LeRobot v3）：

```bash
alohamini dataset stats ~/Alohamini_workspace/datasets/task_demo_ready \
  --config /path/to/train.json \
  --output ~/Alohamini_workspace/logs/task_demo_training_stats.json

python -m alohamini.learning.train --config /path/to/train.json \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --stats "$HOME/Alohamini_workspace/logs/task_demo_training_stats.json" --background
```

`--stats` 只接受带 `--config` 生成的统计；数据或样本配置不匹配时须重新生成。
省略 `--stats` 时训练自动计算。新训练均保存 `runs/<名称>/statistics.json`，checkpoint 保留实际使用的统计；恢复训练沿用 checkpoint。
统计文件只保存汇总值和来源信息，不复制图像、逐帧样本或模型权重。

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
新 AM-ACT 训练默认把 `action.names` 中的 `x.vel`、`y.vel`、`theta.vel` 分别设为三分类：

| 字段 | 类别顺序：负向、停止、正向 | 单位 |
| --- | --- | --- |
| `x.vel` | `[-0.15, 0, 0.15]` | m/s |
| `y.vel` | `[-0.15, 0, 0.15]` | m/s |
| `theta.vel` | `[-45, 0, 45]` | deg/s |

这些值对应平台键盘遥操默认速度档。更高速度档的数据仍按最近类别监督，输出为上述固定速度；使用其他遥操速度时应显式设置类别值。两个机械臂、夹爪和升降仍按连续动作回归，除非显式设置了固定维度。

类别频率从**训练集实际使用的唯一动作帧**统计，不包含验证集，不重复累计重叠 Chunk 或 padding。统计仅读取已加载的数值列，不解码视频。默认权重为：停止 `1`；正/负速度分别 `clip(sqrt(N_stop / N_motion), 1, 5)`。运动类无样本时权重为 `5` 并发出缺类提示；停止类无样本时比值分子使用 `1`。不会对这些权重再次归一化。可用 `discrete_action_weighting="none"` 或 `"inverse_frequency"` 对照，`discrete_action_max_weight_ratio` 修改上限；显式 `discrete_action_class_weights` 优先。

解析后的动作维度、物理/归一化类别值、计数、权重和来源保存在 checkpoint 的 `config.json`。训练标签在归一化前按物理速度匹配，最近中心并列时选择列表中的第一个。某个分类轴全程恒定时，其训练归一化标准差设为 `1`，以保持速度反变换可逆；外部统计或旧 checkpoint 的不可逆零标准差会被拒绝。每类 Recall、Macro Recall 和混淆矩阵计数写入训练及验证的 metrics JSONL；Macro Recall 只纳入该统计窗口有真实样本的类别，缺类不会伪记为满分。验证指标按有效 Chunk 目标汇总，与拟合权重的唯一帧口径不同。

只关闭默认底盘分类：`--policy.base_classification=false`。若配置已显式包含离散维度，还需清空 `discrete_action_dims`、`discrete_action_values`、`discrete_action_normalized_values` 和 `discrete_action_class_weights`。自定义分类示例（维度必须根据当前数据的 `action.names` 核对）：

```json
{
  "base_classification": false,
  "discrete_action_dims": [14],
  "discrete_action_values": [[-0.15, 0.0, 0.15]],
  "discrete_action_class_weights": [[1.0, 1.0, 1.0]]
}
```

旧 checkpoint 按保存的结构严格加载，不自动增加分类头。低层 `AMACTPolicy` 构造接口没有命名数据契约，需显式提供分类维度；默认自动绑定与频率拟合由训练/统计入口完成。启用 `action_loss_groups` 时必须覆盖全部未固定、未分类的连续维度。

条件 CVAE 实验可直接使用：

```bash
python -m alohamini.learning.train \
  --config=examples/learning/am_act_cvae.json \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_vision" \
  --dataset.eval_episodes='[1]' --eval_steps=1000 --background
```

该配置为移动双臂抓取的机器人实验起点，不代表 ACT 论文复现实验或已验证的最优参数：ResNet-18、30 步 Chunk、执行 10 步、纯视觉、4 Query Attention Pooling、Image+Action 后验、图像条件高斯先验、β=1、5000 个成功优化步 warmup。标准 AM-ACT 默认仍是 Action（及可选 State）后验、标准高斯、推理零 z、β=10，无 warmup。语言不输入网络；相机保持 RGB/ImageNet 归一化、数值使用训练统计，动作仍是记录的命名 Host 目标和原单位，无 TCP/相对动作转换。

每路图像在一次前向中只运行一次 ResNet。ACT 主干保留完整 L4 空间 Token；后验和先验共用压缩视觉 Token。`latent_visual_pooling="gap"` 切换为单 Token 基线；`posterior_condition="action"` 关闭后验的图像条件；`prior_type="standard_normal"` 关闭条件先验。条件先验与后验使用解析 KL；warmup 计数在 AMP 成功更新后递增，并随 checkpoint 恢复。验证动作损失使用先验，不读取真实未来动作来生成 z；与原 ACT 一样，验证损失不含 KL。

条件实验默认 `latent_inference_mode="sample"`、`latent_refresh_mode="phase"`：首次读取图像采样 z；阶段内新 Chunk 仍使用最新图像，复用同一个 z，也不重复计算潜变量池化/先验。任务执行器在阶段切换时调用 `policy.refresh_latent()`；它清空旧 z、动作队列和时序融合，下次观测重新采样。`reset()` 在新 episode 做同样处理。没有自动阶段识别，未通知新阶段时 z 保持到 reset。可用 `latent_inference_mode="mean"` 做确定性先验对照、`"zero"` 做零潜变量对照，或 `latent_refresh_mode="chunk"` 每个 Chunk 重采样。

时序融合只平均连续动作，离散底盘使用当前 Chunk 的类别；每 Chunk 重采样时重置融合，避免跨 z 平均。独立离线 Chunk 评估每批清空缓存，不用它衡量阶段一致性。先用验证集分类 Recall、停止误触发运动比例和固定图像改变 z 的动作差异检查模型，再评估真实双臂选择、阶段连续性与抓取成功率；动作多样性不等于成功率。

### Diffusion Policy

```bash
python -m alohamini.learning.train \
  --policy.type=diffusion --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --cameras='["forward","wrist_right"]' \
  --output_dir="$HOME/Alohamini_workspace/runs/diffusion_01" \
  --batch_size=2 --save_freq=10000 --log_freq=100 --background
```

默认使用 2 帧历史观测、16 步预测窗口、每次执行 8 步、DDPM 100 步去噪、
GroupNorm 与 EMA；state 必须存在。数值按训练集 min/max 归一化，
图像缩放至 96×96、训练随机裁剪至 84×84，评估使用中心裁剪。
不使用预训练视觉权重。序列末尾默认不取最后 7 个采样起点，剩余尾部重复目标参与损失。
可用 `--policy.n_action_steps`、`--policy.num_inference_steps` 调整执行块和去噪步数；
checkpoint 同时保存训练权重、EMA 权重及其更新步数。

### FastWAM

准备本地基座、Wan VAE／UMT5 和 tokenizer；下载只需执行一次：

```bash
hf download lerobot/fastwam_base \
  --revision 0a868ec1dcf6ff00bcdfa9b7196d6e211ed7e616 \
  --local-dir "$HOME/Alohamini_workspace/pretrained/fastwam_base"
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --include 'vae/*' 'text_encoder/*' \
  --local-dir "$HOME/Alohamini_workspace/pretrained/wan22_assets"
hf download google/umt5-xxl --include '*.json' '*.model' \
  --local-dir "$HOME/Alohamini_workspace/pretrained/umt5_tokenizer"

python -m alohamini.learning.train \
  --policy.type=fastwam --policy.device=cuda \
  --policy.path="$HOME/Alohamini_workspace/pretrained/fastwam_base" \
  --policy.vae_model_id="$HOME/Alohamini_workspace/pretrained/wan22_assets" \
  --policy.text_encoder_model_id="$HOME/Alohamini_workspace/pretrained/wan22_assets" \
  --policy.tokenizer_model_id="$HOME/Alohamini_workspace/pretrained/umt5_tokenizer" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --cameras='["forward","wrist_right"]' \
  --output_dir="$HOME/Alohamini_workspace/runs/fastwam_01" \
  --batch_size=1 --policy.use_gradient_checkpointing=true \
  --save_freq=1000 --log_freq=100 --background
```

训练使用任务文本、当前 state、32 步动作和间隔 4 行的 9 帧视频；两路图像按名称排序，
各缩放至 224×224 后横向拼接。推理只用当前图像，不需要未来视频。
state/action 使用训练集 min/max，动作仍是记录的关节目标、底盘速度和升降高度，
不套用 LIBERO 的末端／夹爪变换。缺少未来帧时使用边界 padding 及对应损失掩码。

此入口读取转换后的 FastWAM `model.safetensors`，不直接读取作者的 `.pt`；
采用该发布基座的 flow shift=5。基座中的机器人输入／输出层缺失或维度不匹配时重新初始化。
checkpoint 不重复打包冻结的 Wan／UMT5 文件，迁移机器时须保留配置中的资产路径；
微调时也可用上述三个资源参数指定新路径。从平台 checkpoint 微调会恢复全部机器人输入／输出层。
完整模型不适合本机 8 GB GPU；上例须在显存充足的机器上运行。
目前已验证缩小网络的训练与加载，完整基座和真机效果尚未验收。

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

命令行训练默认在后台运行，输出独立的日志、PID、配置路径和可复制的 `tail -f` 命令；输出目录须未使用。
不再需要添加 `--background`，旧命令带这个选项也兼容。需要直接在当前终端运行时使用 `--foreground`。
关闭终端或按 Ctrl+C 退出 `tail -f` 只停止日志查看，不会停止后台训练。
直接由 `torchrun` 启动时，每个 rank 在当前进程运行，不再启动新的后台任务。
`--config examples/learning/act.json` 可代替长命令；先修改数据路径、输出目录和回合编号，命令行参数优先。

训练控制台沿用 LeRobot 的打印格式：`INFO 日期 时间 源文件:行号 消息`，
初始化显示配置、数据集、策略、优化器、输出目录、样本/回合数、有效 batch size 和参数量。
数据校验告警和异常仍会输出；源码位置、配置和指标使用本平台的真实值。
同次训练的训练集与验证集复用未变化数据的完整性检查结果；文件变化会重新校验，大数据集首次校验可能较久。

`--log_freq=100` 仅在更新步数为 100 的倍数时打印区间平均指标，首步和末步不再额外打印；
`0` 关闭训练指标行，进度条、验证结果和 checkpoint 提示仍显示。字段顺序为：

```text
step:1K smpl:2K ep:4 epch:0.20 loss:3.000 grdn:4.000 lr:1.0e-05 updt_s:0.125 data_s:0.500 smp/s:3 mem_gb:1.89 l1:1.0000 kl:0.2000 kl_w:2.0000
```

`step`、`smpl`、`ep` 使用 K/M 等十进制缩写；`loss`、`grdn`、`lr` 为区间均值，
`data_s`、`updt_s` 为平均加载/更新耗时，`smp/s` 为样本吞吐量。
CUDA 下 `mem_gb` 为每步峰值已分配显存的区间均值（GiB）；多卡耗时和显存取各卡区间均值的最大值。
有效 batch size 包含卡数和梯度累积。`smpl` 保留实际消费样本计数（含重复采样和 AMP 跳过更新时消费的样本），
`epch` 是相对可用训练样本数的遍历次数，`ep` 按平均回合长度换算，不是完成的回合数。

前台与默认后台模式均使用原生 `tqdm` 进度条，显示百分比、步数、已运行时间、预计剩余时间和速度；
速度自动显示为 `step/s` 或 `s/step`，按 tqdm 默认平滑方式估算。后台日志也包含回车刷新字符。
续训进度条从 0 开始，总数为本次剩余训练步数；指标行中的 `step` 仍是累计更新步数。
设置了 `SLURM_JOB_ID` 时不显示进度条。计时不含初始化和训练循环结束后的最终离线评估。
验证完成打印总损失和实际计算的分项，例如 `step N: eval_loss=0.1234 l1:0.1234`；保存前打印 `Checkpoint policy after step N`，
完成时打印 `End of training`。训练期间不额外打印逐批验证和最终 MAE 汇总；最终 MAE 保存在 `offline-evaluation.json`。

每一步的原始指标仍写入运行目录的 `metrics.jsonl`，续训写入 `metrics-from-*.jsonl`：

```bash
tail -f ~/Alohamini_workspace/runs/act_01/metrics.jsonl
```

损失分项默认开启，无需新参数。总损失保留为 `loss`，各策略显示自己的组成：

| 策略 | 控制台分项 |
|---|---|
| ACT | `l1`、`kl`、`kl_w`（KL 乘配置权重后的贡献） |
| AM-ACT | `l1`、`kl`、`kl_w`；配置离散动作时增加 `ce`、`ce_w` |
| Diffusion | `diffusion` |
| SmolVLA、π0.5 | `flow` |
| FastWAM | `video_w`、`action_w`（模型返回的已加权项） |

训练控制台中的总损失和分项均为最近日志窗口内**成功 optimizer step 的均值**。
每一步先按策略声明的分母合并梯度累积的各 microbatch 和各 rank；ACT L1 按有效动作元素计数，KL 按样本计数。
AM-ACT 分组 L1 保留模型的组权重；分类损失保留类别权重，按有效分类目标计数。
未计算的分项不会显示为零：例如 ACT/AM-ACT 验证不计算 KL，关闭 VAE 后训练也不显示 KL。
策略损失公式、反向传播和归一化统计不受分项日志影响。

`metrics.jsonl` 的 `metrics` 保存分项值，`metric_totals` 保存各项 `sum` 与 `count`；
AM-ACT 各动作组、各离散头以及 SmolVLA 中间诊断均值也保存在其中，不全部展开到控制台。
`metrics-schema.json` 记录名称映射、权重、分母含义和汇总口径。
周期验证另存为 `validation-metrics.jsonl`，按整个验证集的对应分母汇总，不平均 batch 均值。
续训的这两类附属文件沿用 `metrics-from-*.jsonl` 的步数和运行标识，避免覆盖已有记录。
SmolVLA 中间诊断保留其包含 padding 零值的均值口径，不应当作为可相加的损失项；
π0.5 保留 padded action width 和重复尾部目标参与均值，FastWAM 保留先算各样本损失再平均的规则。

ACT／AM-ACT 按记录行构造动作块，仅在 episode 边界补齐，尾部 padding 不参与动作损失。
相机抖动和控制事件不自动分段；读取器不自动重采样。缺失反馈只影响需要该字段的样本。

### 验证与视频加载

周期验证与训练结束后的离线 MAE 共用一个按顺序读取的 DataLoader；worker、预取和锁页内存的配置规则与训练相同。
验证 batch 默认沿用训练 batch，可单独覆盖；例如在原训练命令后追加：

```text
--eval_batch_size=8 --eval_num_workers=2 --eval_prefetch_factor=2 \
--eval_persistent_workers=true --eval_log_freq=50
```

这些参数分别设置验证 batch、worker 数、每个 worker 的预取批数、跨验证保留 worker 和进度打印间隔。
`eval_*` 加载参数未指定时继承对应训练设置；`eval_log_freq` 默认 50。训练控制台遵循上述简洁格式，逐批进度不打印；
独立离线评估保留逐批进度，每轮首次和末批也打印，中间超过 10 秒会在完成当前批后报告。
worker 为 0 时关闭多进程预取与持久 worker。增加 worker/预取会增加主机内存需求，具体吞吐以测量为准。

独立离线评估日志显示 `batches`、已处理样本与有效动作步数、`data_s`（平均等待下一批的时间）、
`compute_s`（预处理、传输、模型及指标计算的平均时间）、吞吐量和剩余时间。
CUDA 在计时边界同步，避免把异步计算算成数据等待；`compute_s` 不等于纯 GPU kernel 时间。

周期 `eval_loss` 按策略声明的各项有效计数汇总。ACT/AM-ACT 的动作损失按非 padding 的动作步数加权，验证模式不计算 KL；
与旧版的 batch loss 简单平均有意不同，比较旧日志时应注意统计口径。随机策略还会受模型采样影响，改变 batch 不保证逐值相同。
离线 `mae_by_action` 继续按有效动作步数累计，字段和单位不变。

训练器自动选择已安装的 TorchCodec，没有安装时使用 PyAV，无需额外选择数据管线。
默认在每个 worker 内并行读取最多三路视频相机，保留最多 32 个解码器；缓存按进程隔离，淘汰时释放资源。
训练中的周期验证与最终离线 MAE 使用相同配置。单相机与 Parquet 内嵌图像沿用串行读取。
所有已注册策略默认以 uint8 传递原尺寸的 8 位 RGB 图像，减少 worker 交付的数据量；各模型 processor 分别适配输入。
需要 resize 的图像保留原来的 float32 插值路径，避免 uint8 插值/重新量化改变训练输入。

帧索引、时间戳容差和 episode 边界规则保持不变，不插值、不换邻近帧、不跳过解码错误。
每个解码器使用一个 FFmpeg 解码线程。实际选定的配置随训练配置和 checkpoint 保存；显式配置优先于默认值。
旧命令中的 `--video_cache_size=8` 会继续覆盖默认值，可删除该项以使用自动默认配置。

仅在排查性能时需要覆盖 `--video_backend=pyav|torchcodec`、`--camera_workers`、`--video_cache_size`
或 `--return_uint8=true|false`；缓存设为 0 表示逐次打开并关闭指定后端的解码器。
这些读取选项允许在续训时修改，训练 batch、样本顺序及模型参数的断点一致性要求不变。
图像交付类型不决定模型预处理：

| 模型 | processor 和模型内的图像处理 |
|---|---|
| ACT／AM-ACT | 转 `[0,1]`，再使用 checkpoint 的视觉归一化配置 |
| Diffusion | 转 `[0,1]`，保留配置中的归一化、resize 和 crop |
| SmolVLA | 转 `[0,1]`，保留模型内的补边及 SigLIP `[-1,1]` 映射 |
| FastWAM | 转 `[0,1]`，保留时序窗口、相机拼接和 Wan 内部处理 |
| π0.5 | 原始 uint8 直接进入 OpenPI 的补边缩放及 `[-1,1]` 映射 |

已插值的 float32 图像仍走原来的浮点路径，不为统一交付类型而重新量化。
显式 `--return_uint8=false` 可保留旧交付方式；两种方式不会更换已保存的归一化统计。

独立离线评估同样支持加载配置，CLI 默认 4 个 worker（Python API 为 0）：

```bash
python -m alohamini.learning.evaluate \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint \
  --dataset.root ~/Alohamini_workspace/datasets/task_demo_ready \
  --dataset.episodes '[1]' --device cuda --batch_size 8 \
  --num_workers 2 --prefetch_factor 2 --persistent_workers true \
  --log_freq 50 --video_cache_size 8
```

### 优化器、精度与多卡

在训练命令后追加所需参数：

```text
--optimizer.type=adamw --optimizer.lr=0.0001
--optimizer.weight_decay=0.0001 --optimizer.betas='[0.9,0.999]'
--optimizer.eps=1e-8 --optimizer.grad_clip_norm=10
--gradient_accumulation_steps=4 --mixed_precision=bfloat16
```

优化器支持 `adamw`、`adam`、`sgd`，默认沿用策略配置。
ACT／AM-ACT 默认无调度，SmolVLA 使用 `cosine`，π0.5 使用 `warmup_cosine`，
Diffusion／FastWAM 使用 `diffusers_cosine`；
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
| `runs/<名称>/checkpoints/<step>/pretrained_model/` | 权重、模型配置、预处理信息、资源与训练配置 |
| `runs/<名称>/checkpoints/<step>/training_state/` | 优化器、随机数、采样和恢复状态 |
| `runs/<名称>/checkpoints/last` | 最近完整保存点 |
| `runs/<名称>/checkpoint` | 最近保存点的模型目录 |

模型目录包含 `model.safetensors`、`policy.json`、`config.json`、`preprocessing.json`
和 `train_config.json`。`policy.json` 保存完整描述与资源清单；后两个 JSON 分别展示模型配置
和归一化统计、字段、单位、相机及图像尺寸，与完整描述保持一致。旧版仅含 `policy.json`
及权重／所需资源的模型目录仍可读取，无需转换。SmolVLA 的 tokenizer／骨干配置、π0.5 的
tokenizer 使用目录内相对路径；FastWAM 的冻结资源保留外部引用。

```bash
python -m alohamini.learning.train \
  --config_path="$HOME/Alohamini_workspace/runs/act_01/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true --background
```

ACT、AM-ACT、Diffusion、SmolVLA、π0.5、FastWAM 均可从平台 checkpoint 微调。
`--policy.path` 接受模型目录或其上级步数目录，策略类型自动读取，也可显式指定并校验：

```bash
python -m alohamini.learning.train \
  --policy.path="$HOME/Alohamini_workspace/runs/act_01/checkpoint" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/new_task_ready" \
  --output_dir="$HOME/Alohamini_workspace/runs/act_finetune_01" \
  --steps=20000 --batch_size=2 --background
```

模型结构、相机选择、图像尺寸和 state 选择默认沿用 checkpoint，显式参数可覆盖；
权重严格匹配，动作字段顺序、单位和输入语义不匹配时会报错。加载完整模型时不再下载 ImageNet 骨干。

| 微调参数 | 归一化统计 |
| --- | --- |
| 省略或 `--normalization=dataset` | 按本次训练数据重算 |
| `--stats=/path/to/statistics.json` | 使用与本次训练配置匹配的统计文件 |
| `--normalization=checkpoint` | 保留来源 checkpoint 的统计；不能同时指定 `--stats`，动作增量及归一化方式须保持一致 |

这是新训练：步数从 0 开始，优化器和调度器重新初始化，`--steps` 是新训练的总步数。
恢复原运行使用上面的 `--resume=true`，同时恢复模型、统计和训练状态，不依赖最初微调来源的路径。

平台格式中的 `config.json` 不代表能直接由 LeRobot 或各策略官方加载器读取。
外部基座仍按各策略章节的格式导入；ACT／AM-ACT 目前仅接受平台 checkpoint。

续训保持数据、模型、输入、优化器、调度器、进程数、累积次数及精度一致。
未启用调度时可用 `--steps=150000` 增加总步数；启用调度时保持原定总步数。
仅有权重、没有 `training_state/` 的目录不能恢复完整训练状态。

当前样本规则版本为 `episode_windows_v2`。新训练的统计按所选 episode 中实际用于算法窗口的
数据重算；绝对值字段每个物理行只计一次，π0.5 的派生增量继续按其算法窗口统计。
旧规则生成的 `--stats` 文件须重新生成。
旧 checkpoint 推理继续使用其保存的统计；从旧过滤规则或未记录规则版本的保存点 `--resume`
会明确报错，避免改变原实验的样本范围。需要精确续训时使用原版本训练器；
改用新规则训练时通过 `--policy.path` 开始新运行。

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

SmolVLA／π0.5／FastWAM 评估须增加 `--task "拿起物体"`。Diffusion／FastWAM 不使用 ACT 时间融合。
`--dataset eval_01 --task "拿起物体"` 保存评估回合。Ctrl+C 结束。持续反馈中断或关节保护时暂停，恢复后清空旧策略缓存并继续；Host 重启、标定变化或其他客户端接管时结束评估。
推理耗时限制实际执行频率。

跨机器使用时复制完整 checkpoint 目录，勿只复制符号链接。
自定义策略接口见 [客户端接口](host-protocol.md#python-策略评估)，示例为 `examples/learning/custom_policy.py`。
Notebook 入口见 [Notebook](notebook.md)，LeRobot 权重使用见 [LeRobot](lerobot.md)。

### 固定维度的训练与推理

数据没有声明固定维度时，可在评估命令中显式补选，例如：

```bash
alohamini evaluate --host <PI_IP> --robot_model alohamini2pro \
  --policy.path ~/Alohamini_workspace/runs/pretrained_model \
  --device cuda --fps 30 --episode_time 10 \
  --fixed-dimensions arm_right \
  --fixed-dataset ~/Alohamini_workspace/datasets/module_to_shape_4_cleaned
```

支持左右臂、升降、底盘组和具名关节，字段与采集选项相同。已有固定配置继续继承，补选只增加固定维度。
新增位置目标使用训练 **action 的物理坐标均值**；旧模型相应 state 输入采用 checkpoint 的
**state 均值**，避免微小测量差异被归一化放大。真实反馈仍用于安全检查和记录。
固定底盘仍为零速度。把训练中活动的维度改为固定，会改变部署条件，不等同于恢复原任务效果。
保持遥操摆好的当前位置，`--fixed-current` 不需要 `--fixed-dataset`，也不读取训练数据来决定位置。支持 `arm_left`、`arm_right`、
`lift_axis`、`base` 和单个具名动作字段。模型加载后，Host 握手的首次新鲜、可控制反馈决定目标，
随后确认稳定再开始推理；本次运行的后续回合沿用同一目标，不重新采样。底盘仍固定为零速度。
选中字段会覆盖 checkpoint 中继承的固定目标，其余继承目标保持不变。

两种来源可以分开指定，例如 `--fixed-dimensions arm_right --fixed-current lift_axis`，
表示右臂用原有固定目标或训练动作均值、升降保持启动时高度。同一字段不能同时出现在两组选项中。
仅省略 `--fixed-dataset` 不会改变 `--fixed-dimensions` 原有的均值语义。


### 换机器后的标定兼容

默认 `--calibration-mode strict` 要求 checkpoint 与 Host 的标定一致。
换到另一台同型号机器、希望按关节范围比例迁移策略时，可在评估命令追加：

```bash
--calibration-mode normalized
```


