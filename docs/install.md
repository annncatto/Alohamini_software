# 安装

PC 使用 `alohamini` 环境进行遥操、数采和开发；树莓派使用轻量的 `alohamini_host` 环境连接硬件。

## 准备

两台机器均需安装 Conda，并下载本仓库。以下命令在仓库根目录执行；已有对应环境时无需重复创建。

## 1. PC

适用于 Linux x86_64（glibc 2.28+）。GPU 训练与推理需要兼容 CUDA 13 的 NVIDIA 驱动。

Ubuntu / Debian 首次安装时，先准备键盘依赖的编译工具：

```bash
sudo apt install build-essential linux-libc-dev
```

创建并安装环境：

```bash
conda create -n alohamini --file env/explicit-pc-linux-64.txt
conda activate alohamini
python -m pip install --require-hashes --no-build-isolation -r env/pc-linux-64.lock
python -m pip install --no-deps --no-build-isolation -e '.[pc]'
```

PC 环境包含遥操、数采、Notebook 及四种策略的训练依赖。
基座权重和 tokenizer 按需准备，见[训练与部署](training.md)。
键盘遥操需要 X11 桌面；仅使用主臂时可加 `--no_keyboard`。

## 2. 树莓派

适用于 64 位 ARM Linux（aarch64）。

```bash
conda create -n alohamini_host --file env/explicit-linux-aarch64.txt
conda activate alohamini_host
python -m pip install --require-hashes --no-build-isolation -r env/host-linux-aarch64.lock
python -m pip install --no-deps --no-build-isolation -e '.[host]'
```

Host 环境只包含舵机、相机和通信依赖，不安装 PyTorch 或 ROS。

## 3. 检查安装

在各自环境中执行：

```bash
python -m pip check
alohamini --help
```

随后按 [AlohaMini 使用手册](alohamini.md) 配置串口权限、设备名称和标定，再启动 Host 与遥操。

日常使用只需激活对应环境，不必重复安装。若移动了源码目录，需在新位置重新执行对应的 `pip install ... -e` 命令。

ROS2 相机客户端的构建与启动见 [ROS2 相机](ros2.md)；树莓派环境无需变动。
