# 客户端接口

`alohamini.client.HostClient` 支持已部署 AlohaMini Host 的 multipart 状态协议和 JSON 命令协议，不依赖 LeRobot 或 ROS。`alohamini inspect` 仅调用读取接口，不连接命令端口。

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

客户端默认保留至多 3 个在途请求，`request_window` 可设为 1–16。较新的匹配响应取代它之前尚未答复的请求，过期或不匹配 token 不返回给调用方。状态/图像模式切换时清除旧请求窗口，但保留连接与录制游标；超时或协议错误后丢弃连接和命令上下文。下次读取使用新的连接与 token，不返回缓存状态。

每个 episode 开始调用 `set_recording_cameras(True)`，再以 `read(include_images=True)` 获取录制图像。每次启用都生成新的 episode UUID；状态请求仍为 `:state`。图像组尚未齐备时 `_camera_buffer.pending=true`，响应保留状态、不伪造图像；不回填 episode 开始前的帧。结束后调用 `set_recording_cameras(False)` 恢复实时图像请求。

## 命令

`send_command(targets, based_on=snapshot)` 使用 PUSH 连接 Host 的 5555 端口。必须显式配置 `expected_model`，并先读取包含有效版本 1 控制权元数据的状态；不向缺少会话/epoch 信息的 Host 发送匿名命令。

- `targets` 是非空映射，仅允许该型号的机械臂 `*.pos`、`x.vel`、`y.vel`、`theta.vel` 和 `lift_axis.height_mm`。数值必须有限；不补齐未指定轴，不提供升降原始速度旁路。
- 关节位置沿用 Host 标定与归一化；底盘 x/y 为 m/s、theta 为 deg/s，升降目标为 mm。它们不是 Native Host 内部的统一 SI 类型。
- JSON 包含目标及 `_command`，后者携带 `client_id`、单调递增 `sequence`、`host_session_id`、`control_epoch`。
- 命令使用产生它的快照会话；检测到 Host 重启、epoch 改变、其他客户端占用或读取失败后，不自动给旧动作换发新标识。同一会话/epoch 下，更新状态不会仅因为推理耗时超过 250 ms 就禁止发送。
- 消息先完整校验再发送。命令 socket 按需创建，启用 `IMMEDIATE`、`CONFLATE` 和零 linger；连接不可用时不会缓存离线目标，没有自动重发。
- 返回的 `CommandIdentity` 表示消息已排入通信队列，不是 Host 接受或电机执行回执。后续目标可能覆盖尚未发送的目标。

网络发送最多等待 `timeout_s`；该同步客户端不应直接阻塞要求非阻塞运行的事件循环。`based_on` 不代表对传感器年龄或任务语义的保证，Host 负责真实反馈监督和最终命令准入。关节接触保持不一律禁止反向退让命令；策略暂停与动作队列清理属于推理应用。

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

`_motor_feedback` 的缺失字段不补值，未知版本不推断其含义。JSON 解析成功不代表每个电机反馈有效；使用者应依据该字段的版本、读取时间及具体数值进行判断。

读取开始和收到响应的时间使用客户端单调时钟，`round_trip_s` 是二者差值。它不能与 Host 单调时间直接相减，不能证明所有传感器同步或反馈足够新。Host 重启后的单调时间也不能直接拼接成同一个采样序列。

## 错误与资源边界

`ProtocolError` 表示协议错误；`ModelMismatchError` 表示型号不符；`ResponseTimeoutError` 表示超时；`alohamini.errors.ConnectionError` 表示传输错误；`CommandRejectedError` 表示客户端当前条件不允许发送。读取错误不返回上一帧。

重复 JSON 键、非有限数值、不支持的机器人元数据版本和不一致的图像信封会被拒绝。JSON 最大 1 MiB，单帧最大 8 MiB，一次响应最大 32 MiB、最多 16 个相机。JPEG 以字节返回，客户端不解码或验证图像像素。

一个客户端对象只在一个线程中使用，结束时调用 `close()` 或使用 `with`。超时限制针对网络请求，不是实时调度保证。当前连接支持 IPv4 地址或主机名，应只用于可信网络；该协议没有身份认证或加密。

历史单帧 base64 响应不属于此客户端接口。

## 相机订阅（5557）

Host 开启相机时，在同一监听地址的 TCP 5557 提供独立发布通道。使用 ZMQ SUB 订阅 `camera/<相机名>`；消息为 `[topic, metadata_json, jpeg]`。`HostClient` 不订阅此通道。

- `schema_version=1`；`camera_name` 与 topic 一致，`encoding=jpeg`，`width/height` 为旋转后的图像尺寸。
- `host_session_id` 与 5556 状态中的 Host 会话一致；每路 `sequence` 在本次会话内递增。客户端应在会话改变时重置序号检查。
- `capture_monotonic_s` 标记相机读取完成，不是曝光时间；`capture_unix_ns` 由同机时钟对 `host_clock_reference` 换算。跨机器使用墙上时间前须同步系统时钟。
- JPEG 使用标准颜色约定，可直接作为 ROS `CompressedImage`；不要套用 5556 历史图像通道的颜色处理。

订阅仅使用已开启的相机，不取得运动控制权，也不消耗数采游标。无订阅时不额外编码；慢订阅者可能丢帧，不阻塞 Host 控制。`--profile_timing` 下的 `[HOST CAMERA STREAM avg ms/frame]` 单独报告订阅流的编码耗时。
