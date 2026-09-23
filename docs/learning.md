# 本地 ACT／AM-ACT

模型位于 `src/alohamini/policies/`，可直接修改和训练，不依赖 LeRobot 或 Hugging Face。
PC 使用 `alohamini` 环境和 PyTorch；树莓派 `alohamini_host` 不安装训练依赖。

## 环境

在已安装 PC 环境的仓库根目录执行：

```bash
conda activate alohamini
python -m pip install -e '.[learning,notebook]'
python -m ipykernel install --user --name alohamini --display-name 'AlohaMini'
```

在 VS Code/Jupyter 中打开 `examples/learning/local_act.ipynb`，选择 AlohaMini 内核。
完整 PC 依赖见 `env/pc-linux-64.lock`；PyTorch 的 CUDA 构建须支持本机 GPU。
ACT／AM-ACT 默认使用 `pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1"`，
首次训练建模时可能从 PyTorch 下载并缓存视觉骨干权重，不需要 Hugging Face 账号。
权重目录为 `~/Alohamini_workspace/pretrained/`（随 `ALOHAMINI_WORKSPACE` 变化），
训练之间共用，不提交进 Git；离线机器可提前将官方权重文件复制到此目录。
设为 `null` 可显式从随机初始化训练；评估加载本地 checkpoint 不会下载骨干权重。
这与 [ACT 作者实现](https://github.com/tonyzhaozh/act/blob/main/detr/models/backbone.py) 的预训练 ResNet 初始化一致。
训练默认图像尺寸为 `[480, 640]`（高、宽），与原论文一致；数据读取和训练入口共用此默认值。
显式配置其他尺寸仍有效，已有 checkpoint 的推理尺寸保持不变。
当前示例仍是 AlohaMini 适配实验：动作维度、采集 FPS 和可选纯视觉输入与论文设置不同，不能视为论文结果复现。

## 检查与训练

```bash
alohamini dataset check ~/Alohamini_workspace/datasets/task_demo --decode-images
```

编辑 `examples/learning/act.json` 中的数据路径、训练/验证 episode 和运行名称。
`chunk_size` 和视觉初始化属于训练配置；`n_action_steps` 与时间融合在 Notebook
评估单元或命令行评估时选择，不需要重新训练。
训练和验证按完整 episode 分开；仅有一个 episode 时，可设置 `val_episodes=[]`、
`overfit_smoke=true` 检查流程，但不能据此评估泛化能力。
数据警告会显示并随 checkpoint 保存，不要求填写确认文字；`review_note` 仅为可选实验备注。结构错误仍阻止训练，异常间隔和保护记录仍参与样本过滤与分段。

```bash
python - <<'PY'
import json
from pathlib import Path
from alohamini.learning.train import launch_training
job = launch_training(json.loads(Path('examples/learning/act.json').read_text()))
print(job)
print('tail -f', job['log'])
PY
```

训练在后台运行。日志、PID、启动配置保存在 `~/Alohamini_workspace/logs/training/`；
权重、配置、归一化统计、数据检查报告和离线评估保存在 `~/Alohamini_workspace/runs/<run_name>/`。
运行名称不能重复；恢复已有训练须显式使用 `--resume=true`。当前支持单设备 FP32
ACT／AM-ACT、AdamW、定期 checkpoint 和断点续训，不支持 AMP 或分布式训练。

`state="none"` 为纯视觉；也可选 `joint_position,base_velocity,lift_height`（原 18 维），
或 `joint_velocity,joint_current`（双臂 28 维，包含夹爪）。顺序按配置保留。
速度单位随原关节坐标，为该坐标单位/秒；电流单位 A，不等同于关节力矩。
action 始终保持原有双臂绝对目标、底盘速度和升降目标高度。

读取器不改变原始配对，不补造遗漏动作；每个 chunk 从当前行的 action 开始。
episode 边界、相机重复帧或偏离标称周期较大的间隔会截断 chunk，后面重复末动作并标为 padding，
不计入损失。它不进行时间重采样；小幅时间抖动和累计时基偏差仍需人工审查。
图像按配置缩放，所有均值/标准差仅由训练 episode 计算。

## 命令行训练与恢复

正式脚本为 `python -m alohamini.learning.train`，沿用常用的 LeRobot 长选项。
可读原生数据和平台导出的图像型 v3；不需要安装 LeRobot。
纯视觉数据集会自动选择无 state 输入，也可显式写 `--state=none`。
正常 18 维数据集默认使用原 state。已有 JSON 配置仍可用 `--config`，命令行参数优先。

以下训练 AM-ACT；改成 `--policy.type=act` 即训练 ACT。输出和日志名称须尚未使用：

```bash
cd ~/Alohamini
conda activate alohamini
mkdir -p ~/Alohamini_workspace/logs/training

nohup python -u -m alohamini.learning.train \
  --dataset.repo_id=local/pickup_01_vision \
  --dataset.root="$HOME/Alohamini_workspace/datasets/pickup_01_vision" \
  --policy.type=am_act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$HOME/Alohamini_workspace/runs/am_act_vision_01" \
  --steps=100000 \
  --batch_size=2 \
  --save_freq=10000 \
  --log_freq=100 \
  --wandb.enable=false \
  > ~/Alohamini_workspace/logs/training/am_act_vision_01.log 2>&1 < /dev/null &

echo $! > ~/Alohamini_workspace/logs/training/am_act_vision_01.pid
tail -f ~/Alohamini_workspace/logs/training/am_act_vision_01.log
```

`repo_id` 仅作本地标识，可省略；不下载、不上传。Hub/W&B 开关只接受 `false`。
省略 episode 选择时使用全部回合；可用 `--dataset.episodes='[0,1,2]'` 和
`--dataset.eval_episodes='[3]'` 分离训练/验证，`--eval_steps=1000` 定期计算验证损失。
不指定验证回合仍可训练，但不能据训练 loss 判断泛化。
本例的 pickup_01 仅用于跑通流程；正式训练应换成有足够连续示教的数据。

每 `save_freq` 步和最后一步保存一次，包括权重、归一化、训练配置、优化器及随机数状态。
`checkpoints/last` 指向最近的完整 checkpoint，`checkpoint` 指向其中的模型目录，
可直接供原生 `alohamini evaluate --policy.path .../checkpoint` 使用。
这是平台原生 checkpoint，不是可以直接交给 LeRobot `from_pretrained` 的模型格式。

恢复时指定保存的配置，`steps` 表示希望达到的总步数，不是额外步数：

```bash
python -m alohamini.learning.train \
  --config_path="$HOME/Alohamini_workspace/runs/am_act_vision_01/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true --steps=150000 --background
```

`--background` 自动创建独立 `.log`、`.pid`、启动配置并打印 `tail -f` 命令，不再额外套 `nohup`。
恢复会检查数据指纹、batch size、模型和归一化契约；不允许静默换数据或改变优化设置。
早期只保存权重、没有 `training_state/` 的 checkpoint 不能精确续训。
断电或中断后只能恢复到最近的完整保存点，之后尚未保存的更新不在 checkpoint 内。

## AM-ACT

将 `policy` 改为 `am_act` 即使用迁移后的模型。保留了 `fixed_action_dims`、
`action_loss_groups/weights`、`observation_state_dims`、底盘分类头及其类别权重。
未指定这些选项时，不会自动猜测底盘动作类别或沿用某次实验的权重。

例如，为第 14 维底盘 x 速度启用分类，需要在 `model` 中加入：

```json
{
  "discrete_action_dims": [14],
  "discrete_action_values": [[-0.1, 0.0, 0.1]],
  "discrete_action_class_weights": [[1.0, 1.0, 1.0]]
}
```

上述数值仅说明格式；必须改成实际示教的物理速度类别。零方差维度不能用于分类。
分类中心按训练统计归一化，输出再恢复物理值。`inference_action_scale_dims/scale`
在反归一化后缩放；仅为实际需要缩放的速度命令配置，不要用于绝对关节位置。

## v3 纯视觉数据与 AM-ACT

在仓库根目录、`alohamini` 环境执行。先按原来的 state 格式导出，不加 `--state`。
以下使用新目录，已有导出不会被覆盖：

```bash
alohamini dataset export ~/Alohamini_workspace/datasets/pickup_01 \
  --output ~/Alohamini_workspace/datasets/pickup_01_lerobot18 \
  --format lerobot-v3

alohamini dataset export ~/Alohamini_workspace/datasets/pickup_01_lerobot18 \
  --output ~/Alohamini_workspace/datasets/pickup_01_vision \
  --format lerobot-v3 --vision-only

alohamini dataset check ~/Alohamini_workspace/datasets/pickup_01_vision --decode-images
```

第一个数据集的 state 是双臂位置 14 维、底盘速度 3 维、升降高度 1 维。
纯视觉副本只保留图像输入、action 和索引列；18 维 action 不变，升降仍是绝对高度目标，
不是上升/停止/下降分类。原始时间、保护和标定记录作为元数据保留，不输入模型。
两个步骤均不改变配对、帧数或时间戳，也不会修复采样间隔警告。

训练器可直接读取平台导出的图像型 v3，不需要安装 LeRobot、登录或上传 Hub。
当前不支持用此读取器训练缺少 AlohaMini 标定/时间记录的外部 v3，或 MP4 型 v3。
图像按需读取；样本过滤、间隔分段、padding 和训练集归一化与原生格式共用实现。

`examples/learning/am_act_v3.json` 已配置纯视觉 AM-ACT。
`pickup_01` 只有一个 episode，因此示例是单回合过拟合流程测试，不是泛化评估。
这份数据还有采样间隔警告；按当前连续性规则，30 步 chunk 中最多只有 6 步有效，
其余为不计入损失的 padding。格式转换不会增加可用的连续动作长度。
正式实验应采集多个 episode，设置不重叠的 `train_episodes`/`val_episodes`，
删除 `overfit_smoke`，并按收敛情况调整训练步数。示例不自动启用底盘或升降分类头。

```bash
python - <<'PY'
import json
from pathlib import Path
from alohamini.learning.train import launch_training
settings = json.loads(Path('examples/learning/am_act_v3.json').read_text())
job = launch_training(settings)
print(job)
print('tail -f', job['log'])
PY
```

日志为 `~/Alohamini_workspace/logs/training/am_act_pickup01_vision_smoke.log`，
PID 同目录、扩展名 `.pid`。重新训练须换 `run_name`。
有验证回合时，训练结束会保存 `offline-evaluation.json`；不会自动连接机器人。

## 离线检查与真机评估

Notebook 展示图像、配对时间、速度/电流、单批次损失和预测轨迹。
离线指标是各 action 维度原单位的 MAE，不把混合单位合并成一个精度数值，也不代表任务成功率。

模型输出通过已有原生评估循环执行，不另建机器人控制通道。
**下面命令会连接机器人并执行动作**；应先完成离线检查、核对标定、支撑机器人并准备停止手段。
冒烟测试的几步训练权重不应用于真机。

```bash
alohamini evaluate --host <PI_IP> --robot_model alohamini2pro \
  --policy.path ~/Alohamini_workspace/runs/act_local_01/checkpoint \
  --policy.n_action_steps 1 --policy.temporal_ensemble_coeff 0.01 \
  --fps 30 --episode_time 10
```

原生 checkpoint 自带输入/输出契约，无需 `--training-dataset`。
`n_action_steps` 表示预测一次后执行多少步；时间融合启用时须为 1，`none` 关闭融合。
每个 episode 都 reset；保护事件后不自动恢复动作。实际步频仍受同步推理耗时约束。
Notebook 真机单元默认 `ENABLE_ROBOT=False`，必须显式改为 True 并提供 Host 地址。

自定义执行接口见 `examples/learning/custom_policy.py`：`robot_metadata`、`reset()`、
`select_action(snapshot)`。网络开发直接使用普通 PyTorch；不要求继承 LeRobot 的 Policy。
旧 LeRobot checkpoint 仍通过可选适配器加载，不能直接当作原生 checkpoint 使用。
