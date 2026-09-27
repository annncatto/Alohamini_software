# LeRobot 兼容

导出数据供 LeRobot 使用：

```bash
alohamini dataset export ~/Alohamini_workspace/datasets/pickup_01 \
  --output ~/Alohamini_workspace/datasets/pickup_01_lerobot \
  --format lerobot-v3
```

默认保留 18 维 state、action 和反馈，图像内嵌于 Parquet。
`--vision-only` 导出纯视觉副本；保留原数据，输出目录须不存在。

LeRobot 训练和评估在独立仓库及其环境中运行，按该仓库文档操作。
LeRobot checkpoint 不能直接用于 `alohamini evaluate --policy.path`。

在本项目训练这些数据，见[训练与部署](training.md)。
