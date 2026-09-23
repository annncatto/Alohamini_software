# 编辑本地数据集

在 PC 的 `alohamini` 环境运行。直接处理平台原生数据，不需要 LeRobot、PyTorch 或 Hugging Face 账号。先停止对该数据集的录制；除只读 `info` 外，均须指定尚不存在的输出目录，原数据不会被修改。

## 删除指定回合

episode 编号从 **0** 开始。例如排除第 0、2 个 episode：

```bash
alohamini dataset edit \
  --dataset pickup_01 \
  --output ~/Alohamini_workspace/datasets/pickup_01_filtered \
  --operation.type delete_episodes \
  --operation.episode_indices '[0, 2]'

alohamini dataset check ~/Alohamini_workspace/datasets/pickup_01_filtered \
  --decode-images --decode-videos
```

`--dataset` 使用工作区中的名称；也可改用 `--root /完整/数据集路径`。保留的回合按原顺序重新编号，全局索引与保护记录同步更新，不重新配对图像、state 和 action。不能删除全部回合。

## 其他操作

同样使用 `--root` 和 `--output`，按下表替换操作及参数。

| `--operation.type` | 参数示例 |
| --- | --- |
| `split` | `--operation.splits '{"train": 0.8, "val": 0.2}'` 或 `'{"train": [0, 2], "val": [1]}'` |
| `merge` | `--operation.roots '["/数据集A", "/数据集B"]'`，无需 `--root` |
| `remove_feature` | `--operation.feature_names '["observation.images.wrist_right", "observation.motor_temperature_raw"]'` |
| `modify_tasks` | `--operation.new_task "拿起积木"`；可加 `--operation.episode_tasks '{"2": "放下积木"}'` |
| `recompute_stats` | 默认统计数值字段；`--operation.skip_image_video false` 也统计图像 |
| `convert_image_to_video` | JPEG/PNG 转原生 MP4 数据集；可用 `--operation.episode_indices '[0, 2]'` 选回合 |
| `reencode_videos` | 对原生 MP4 数据集重新编码，例如 `--operation.rgb_encoder.crf 23` |
| `info` | `--operation.show_features true` 查看字段，不需要 `--output` |

拆分保持原顺序、不随机打乱，输出位于 `<output>/train` 等子目录；比例和小于 1 时余下回合不选入。合并要求帧率、字段、相机、标定及动作坐标一致，保留每回合边界。删除电机反馈时，其有效性掩码也会一起删除；不能单独删除仍在使用的反馈时间。

视频默认 H.264、CRF 18、GOP 2、`yuv420p`，每回合每相机一个 MP4。可用 `--operation.rgb_encoder.vcodec`、`.crf`、`.g`、`.preset`、`.pix_fmt` 调整编码。转换不增删帧、不补帧、不改变采集时间；有损压缩可能改变像素。`yuv420p` 要求偶数尺寸，工具不会偷偷裁剪或缩放。仅需检查录像时仍可用 `alohamini dataset preview`，无需转换原始图像。

## 使用编辑结果

- 输出仍是平台原生数据，编辑版本为 3，不是 LeRobot v3。可继续检查、编辑、预览和训练；保留完整 action 时可回放。录制脚本仍输出原有版本，编辑副本不可用于追加录制。
- 删除字段不改写保护日志中的原始事实。纯视觉训练可去掉 state；速度/电流训练应保留对应反馈、有效性掩码和采样时间。删掉 action 后不能做行为克隆或回放。
- 旧统计不会被冒充为编辑后的统计；需要时对编辑结果运行 `recompute_stats`。无效反馈不计入统计，某维完全不可用时为 `null`、`count=0`。训练归一化仍只从训练集计算。
- `--operation.relative_action true --operation.chunk_size 50` 统计回合内完整动作块的相对位置分布，不修改 action；默认排除 gripper，底盘速度始终保持绝对速度。它不是末端增量转换，也不会绕过训练前的保护与时间检查。
- 有效的 MP4 预览随回合重新编号；删除字段或转换视频后，可用 `dataset preview` 重新生成预览。失败时输出保留在 `*.pending-*`，不要作为完成的数据集使用。

`--config_path /path/edit.json` 支持 JSON 配置，命令行覆盖同名配置项。也可使用 `python -m alohamini.datasets.edit` 调用同一入口。此编辑器接受原生数据；LeRobot 格式仍使用平台已有的导出、检查和修复入口。
