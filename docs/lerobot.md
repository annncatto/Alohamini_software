# LeRobot 策略

PC 可选安装，Host 环境无需变动。当前本地加载器接受官方 ACT 完整 checkpoint，使用其保存的预处理、归一化统计和后处理；不下载权重、不上传数据。

## 安装

在仓库根目录、已有 PC 环境中执行：

```bash
conda activate alohamini
python -m pip install --require-hashes --no-build-isolation -r env/lerobot-linux-64.lock
python -m pip install --no-deps --no-build-isolation -e integrations/lerobot
```

## 运行

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

## Python / Notebook

```python
from alohamini_lerobot.policy import LeRobotPolicy

policy = LeRobotPolicy.from_pretrained(
    "/path/to/pretrained_model", "/path/to/training_export",
    device="cuda", task="pickup",
)
```

加载本身不连接机器人。`policy.reset()` 清空动作队列和处理器历史，`policy.select_action(snapshot)` 返回具名动作；通过原生 `run_evaluation(..., fps=int(policy.fps))` 执行时保留控制权及保护检查。不要绕过检查直接将网络输出发送到机械臂。
