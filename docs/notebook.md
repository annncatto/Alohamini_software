# Notebook 入口

Notebook 用于数据与图像检查、时间曲线、速度/电流分析、state/action 组装、单批次损失和预测轨迹对比。
正式的数据整理、训练、离线评估与真机部署见 [训练与部署](training.md)，无需先运行 Notebook。

在 PC 已有 `alohamini` 环境安装可选组件：

```bash
conda activate alohamini
python -m pip install -e '.[notebook]'
python -m ipykernel install --user --name alohamini --display-name 'AlohaMini'
```

使用 VS Code 的 Jupyter 扩展或已有 JupyterLab 打开
`examples/learning/local_act.ipynb`，选择 **AlohaMini** 内核。
`.[notebook]` 提供内核和 Notebook 执行依赖，不包含 JupyterLab 服务。

先检查数据路径、回合划分及 `RUN_TRAINING`、`ENABLE_ROBOT`：
数据分析不需要启动训练或连接机器人；只有明确要训练或真机评估时才启用相应开关。
Notebook 调用正式库代码，不另外实现采集、驱动、训练或机器人执行循环。
