# Notebook 入口

Notebook 用于数据与图像检查、时间曲线、速度/电流分析、state/action 组装、单批次损失和预测轨迹对比。
正式的数据整理、训练、离线评估与真机部署见 [训练与部署](training.md)，无需先运行 Notebook。

完成 [PC 安装](install.md) 后，注册 Notebook 内核：

```bash
conda activate alohamini
python -m ipykernel install --user --name alohamini --display-name 'AlohaMini'
```

使用 VS Code 的 Jupyter 扩展或已有 JupyterLab 打开
`examples/learning/local_act.ipynb`，选择 **AlohaMini** 内核。
PC 环境包含内核依赖，不包含 JupyterLab 服务。

打开后设置数据路径与回合划分。`RUN_TRAINING` 启动训练，`ENABLE_ROBOT` 启用真机评估；仅分析数据时保持关闭。
