# 开发

核心协议解析只依赖 Python 标准库。网络客户端额外使用 pyzmq；模块导入不会启动网络连接或加载机器人硬件。

开发使用[安装指南](docs/install.md)中的 `alohamini` 环境。Python 源码保持 3.10 兼容，主开发环境使用 3.12.13；调整公共接口时应覆盖这两个版本。

## 目录

```text
Alohamini/
├── environment.yml          # Conda 直接依赖版本
├── .python-version          # 主开发解释器版本
├── env/                     # 按架构锁定的 Conda / pip 依赖
├── pyproject.toml           # 核心包、入口与检查工具配置
├── src/alohamini/
│   ├── cli.py               # 命令行入口
│   ├── errors.py            # 公共异常
│   ├── client.py            # 共用机器人客户端
│   ├── protocol.py          # 已部署 Host 协议编解码
│   ├── model/               # 型号定义与资产加载
│   ├── hardware/            # 硬件读写
│   ├── runtime/             # Host 控制与保护
│   ├── calibration/         # 标定与单位转换
│   ├── datasets/            # 本地数据存储、检查与格式导出
│   └── apps/                # 遥操、数采与可视化
├── tests/                   # 核心测试与跨组件测试
└── docs/                    # 安装运维与接口文档
```

核心运行代码放在 `src/alohamini/`；命令行入口只负责参数与输出，不复制协议逻辑，框架依赖不得进入通用接口。测试辅助代码不随运行包分发。

`apps/` 调用客户端和数据接口组织操作流程；`datasets/` 不依赖应用入口。Host、硬件和协议不反向导入应用或数据处理代码。仅进行文件格式转换的导出器属于 `datasets/`；需要学习框架运行时的适配代码属于 `integrations/`。

旧实现留在原仓库和 Git 历史中。迁入功能成为正式实现，不另建历史副本。兼容层必须有实际接入需求，只保留必要的接口适配；调用方迁移完成后删除，不能持续容纳旧业务逻辑。

ROS2、学习框架与仿真扩展分别归属同级 `ros2/`、`integrations/` 和 `simulation/`，随实际功能建立。各扩展管理自身依赖与测试，不放入核心包。

个人开发计划与进度存放于 Obsidian，不放在源码仓库。通用型号资产以根目录 `models/` 为权威来源；单机标定与型号资产分开保存。

## 检查

在仓库根目录运行离线测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

未安装 pyzmq 时跳过本机网络测试。网络测试只绑定 `127.0.0.1` 的临时端口，不连接机器人。具体测试可使用 `unittest discover -p` 选择。

迁移差分检查可通过 `ALOHAMINI_SOURCE_REPO`、`ALOHAMINI_ROS_SOURCE` 指向可信的旧仓库，再运行 `unittest discover -s tests -p test_migration.py`。检查从指定工作树读取原函数进行比较，不加载旧硬件入口；未设置路径时跳过这组检查。

可选代码检查：

```bash
ruff check src tests
ruff format --check src tests
```

新增行为应覆盖对应输入、输出和失败边界。公共接口说明随行为变化更新；机器人实测应明确设备、软件版本及测试范围，不能由离线测试结果替代。
