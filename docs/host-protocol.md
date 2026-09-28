# 客户端接口

`alohamini.client.HostClient` 读取状态和图像、发送具名运动目标。命令行只读查询使用 `alohamini inspect`。

## Python 读取

将地址和型号换成实际配置；树莓派启动方式见[使用手册](alohamini.md)。

```python
from alohamini.client import HostClient

with HostClient("192.168.8.161", expected_model="alohamini2pro") as robot:
    snapshot = robot.read()
    print(snapshot.robot_model)
    print(snapshot.payload["_safety"])

    snapshot = robot.read(include_images=True)
    forward_jpeg = snapshot.images.get("forward")
```

## 请求与响应

客户端为 DEALER，Host 为 ROUTER，默认 TCP 端口为 5556。

- 状态请求：一个以 `:state` 结尾的唯一 token。
- 含图像请求：一个以 `:full` 结尾的唯一 token。
- 录制图像请求：`<请求 ID>:<episode UUID>:record`，使用独立的有界相机游标。
- DEALER 收到的响应：`[token, state_json, camera_name, jpeg, ...]`。
- ROUTER 的客户端 identity 属于路由信封，不是 DEALER 响应内容。
- 状态请求必须只返回 token 和 JSON，且 `_images=[]`。
- 图像名字与次序必须和 JSON 的 `_images` 完全一致。

默认至多 3 个在途请求，`request_window` 范围为 1–16。较新的匹配响应取代更早的未答复请求。超时清除在途请求，保留连接和已验证的控制上下文；协议错误关闭连接。两者均不返回缓存状态。

录制开始调用 `set_recording_cameras(True)`，随后用 `read(include_images=True)` 读取有界队列中的图像组。每次启用生成新的 episode UUID，录制图像限制为一个在途请求。`_camera_buffer.pending=true` 表示图像组未齐备；结束后调用 `set_recording_cameras(False)`。

## 命令

`send_command(targets, based_on=snapshot)` 使用 PUSH 连接 Host 的 5555 端口。必须显式配置 `expected_model`，控制循环开始前调用 `connect_control()` 建立状态和动作连接；不向缺少会话/epoch 信息的 Host 发送匿名命令。

- `targets` 是非空映射，允许该型号的机械臂 `*.pos`、`x.vel`、`y.vel`、`theta.vel`、`lift_axis.height_mm` 和 `lift_axis.stop`。数值必须有限；不补齐未指定轴，不提供升降原始速度旁路。
- `lift_axis.stop=1` 仅停止升降，与高度目标互斥。Host 先发送零速度，等待至少 100 ms，再以此后采集的本机高度反馈保持位置；没有有效高度参考时只停止，不创建参考。须结合后续反馈确认静止。
- 关节位置沿用 Host 标定与归一化；底盘 x/y 为 m/s、theta 为 deg/s，升降目标为 mm。ROS 接口的单位转换见 [ROS2](ros2.md)。
- JSON 包含目标及 `_command`，后者携带 `client_id`、单调递增 `sequence`、`host_session_id`、`control_epoch`。
- 命令绑定 `based_on` 快照的会话与控制权；epoch 改变后须使用新快照。自身看门狗释放可恢复，Host 重启或其他客户端接管不可直接恢复旧动作。
- 消息先完整校验再发送。命令 socket 按需创建，启用 `IMMEDIATE`、`CONFLATE` 和零 linger；连接不可用时不会缓存离线目标，没有自动重发。
- 返回的 `CommandIdentity` 表示消息已排入通信队列，不是 Host 接受或电机执行回执。后续目标可能覆盖尚未发送的目标。
- 通道暂时不可写时立即返回 `None`，保留连接；本次目标未发送，不应记录为已发送动作。下一周期根据新状态生成目标，不重发旧目标。

动作发送非阻塞，状态读取由 `timeout_s` 限制等待。超过 Host 看门狗时长仍无有效反馈时，`send_command()` 返回 `None`；短暂超时不会立即禁发。应用负责策略队列管理，Host 负责硬件保护和命令准入。

`close()` 丢弃未发送消息并关闭连接，不隐式发送位置目标、失能或回零；断开控制者后的停止由 Host watchdog 负责。

## 状态与单位

`snapshot.payload` 保留响应字段，包括未知扩展字段：

| 字段 | 含义 |
| --- | --- |
| `_robot_metadata` | `schema_version=1`、型号、标定映射元数据和启用的相机 |
| `_host_timing` | Host 提供的采样时间及其时钟域 |
| `_safety` | Host 保护、命令来源、控制权和会话信息 |
| `_motor_feedback` | 可选电机反馈，可能只有部分字段 |
| `*.pos` | 现有 Host 的关节位置值，单位由相应电机的 normalization 决定 |
| `lift_axis.height_mm` | 现有 Host 的升降高度，毫米 |
| `lift_axis.homed` | 连续编码器高度参考是否有效；不表示存在高度传感器或下限位开关 |
| `lift_axis.raw_tick` | 当前单圈编码器读数；参考无效时省略 |
| `lift_axis.extended_ticks`、`lift_axis.zero_extended_ticks` | 同一 Host 跟踪器的连续计数与零高度计数；参考无效时省略 |
| `lift_axis.reference_sequence` | 本次 Host 会话内建立高度参考的次数；重新建立参考后递增 |

使用 `_motor_feedback` 前检查版本、读取时间和有效值，缺失字段不补零。

升降机构参数位于 `_robot_metadata.lift_axis`：`ticks_per_revolution`、`lead_mm_per_revolution`（已含传动比）和 `direction_sign`。高度满足 `height_mm = direction_sign × (extended_ticks - zero_extended_ticks) × lead_mm_per_revolution / ticks_per_revolution`。连续计数只在同一 Host 会话、同一有效参考内使用；采样工具应检查 `reference_sequence`，不能跨重连拼接。

`round_trip_s` 为客户端请求往返耗时。PC 与 Host 单调时间不能直接相减；Host 重启后须另建时间序列。

## 错误与资源边界

`ProtocolError` 表示协议错误；`ModelMismatchError` 表示型号不符；`ResponseTimeoutError` 表示超时；`alohamini.errors.ConnectionError` 表示传输错误；`CommandRejectedError` 表示客户端当前条件不允许发送。读取错误不返回上一帧。

重复 JSON 键、非有限数值、不支持的机器人元数据版本和不一致的图像信封会被拒绝。JSON 最大 1 MiB，单帧最大 8 MiB，一次响应最大 32 MiB、最多 16 个相机。JPEG 以字节返回，客户端不解码或验证图像像素。

一个客户端对象只在一个线程中使用，结束时调用 `close()` 或使用 `with`。超时限制针对网络请求，不是实时调度保证。当前连接支持 IPv4 地址或主机名，应只用于可信网络；该协议没有身份认证或加密。

## Python 策略评估

```bash
alohamini evaluate --host 192.168.8.161 --robot_model alohamini2pro \
  --policy my_policy:load --episode_time 30
```

`my_policy:load` 是开发者可导入模块中的无参数工厂函数，不是模型目录。
返回对象提供：

- `robot_metadata`：训练数据记录的型号、标定与升降参数，不应以启动时读取的配置代替。
- `reset()`：清空策略历史、动作队列和预处理器状态；每回合调用一次。
- `select_action(snapshot)`：接收 `HostSnapshot`，返回全部机械臂 `*.pos`、底盘三轴速度和升降高度的 `{字段名: Python float}`。沿用上述 Host 单位，不接受无字段名的向量或末端增量。

策略负责模型加载、图像解码、预后处理和输出坐标转换。示例见 `examples/learning/custom_policy.py`。

默认一回合、60 秒、30 Hz；`--num_episodes`、`--reset_time` 设置回合数和复位时间。`--dataset 名称 --task "任务"` 保存输入、已接受目标和实际采样时间；实际频率受推理耗时限制。

评估使用 Host 已开启的相机。关节保护或持续失联时暂停，恢复后清空旧策略缓存并继续；Host 重启、其他客户端接管、标定或升降参考变化时结束。结束时在仍持有控制权的条件下请求位置保持和底盘归零。Host 看门狗在推理期间持续生效。

## 相机订阅（5557）

Host 开启相机时，在同一监听地址的 TCP 5557 提供独立发布通道。使用 ZMQ SUB 订阅 `camera/<相机名>`；消息为 `[topic, metadata_json, jpeg]`。`HostClient` 不订阅此通道。

- `schema_version=1`；`camera_name` 与 topic 一致，`encoding=jpeg`，`width/height` 为旋转后的图像尺寸。
- `host_session_id` 与 5556 状态中的 Host 会话一致；每路 `sequence` 在本次会话内递增。客户端应在会话改变时重置序号检查。
- `capture_monotonic_s` 标记相机读取完成，不是曝光时间；`capture_unix_ns` 由同机时钟对 `host_clock_reference` 换算。跨机器使用墙上时间前须同步系统时钟。
- JPEG 可直接用于 ROS `CompressedImage`。

订阅仅使用已开启的相机，不取得运动控制权，也不消耗数采游标。无订阅时不额外编码；慢订阅者可能丢帧，不阻塞 Host 控制。`--profile_timing` 下的 `[HOST CAMERA STREAM avg ms/frame]` 单独报告订阅流的编码耗时。
