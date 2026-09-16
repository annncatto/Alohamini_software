# ROS2 相机

树莓派按使用手册启动 Host；PC 使用 ROS2 Humble 的系统 Python，不在 Conda 环境中运行 ROS 节点。

## 构建

在已安装 ROS2 Humble、colcon 和 rosdep 的 PC 终端执行：

```bash
source /opt/ros/humble/setup.bash
cd ~/Alohamini/ros2
rosdep install --from-paths src --ignore-src -r -y
/usr/bin/python3 -m colcon build --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/local_setup.bash
```

`alohamini_core` 从平台根目录安装同一份 Python 源码和模型资产，无需另装 LeRobot、PyTorch 或新建 Conda 环境。

## 启动

```bash
export ROS_LOG_DIR="${ALOHAMINI_WORKSPACE:-$HOME/Alohamini_workspace}/logs/ros2"
ros2 launch alohamini_camera camera.launch.py host:=<PI_IP>
```

默认跟随 Host 实际开启的相机；只订阅部分相机时追加 `cameras:="[forward, wrist_right]"`。该节点只接收图像，不发送运动命令，不打开本地相机。Host 重启后自动识别新会话。

每路相机的话题：

- `/alohamini/cameras/<name>/image_raw/compressed`：标准 JPEG。
- `/alohamini/cameras/<name>/image_raw`：RGB，仅在有订阅者时解码。
- `/alohamini/cameras/<name>/camera_info`：有可用内参时发布。

默认使用 PC 接收时间。需要 Host 采集时间时追加 `timestamp_mode:=host_wall`，并先同步 PC 与树莓派系统时钟；移动相机标定不能把接收时间当作曝光时间。

## 标定文件

默认读取 `~/Alohamini_workspace/calibration/cameras/`，支持 `ALOHAMINI_WORKSPACE`；自定义目录用 `calibration_dir:=/绝对路径`。

- `intrinsics/<相机名>.yaml`：ROS CameraInfo 格式，名称、尺寸和光学坐标系须匹配。带候选或未验收状态的文件不用于发布 CameraInfo；没有内参仍可接收图像。
- `extrinsics/<文件名>.yaml`：`mount_link` → `optical_frame` 的外参，默认不发布 TF。

显式发布已验收外参：

```bash
ros2 launch alohamini_camera camera.launch.py host:=<PI_IP> \
  enable_extrinsics:=true extrinsics:=forward.yaml,wrist_right.yaml
```

外参须有 `accepted_` 开头的状态、`T_mount_link_from_camera_optical.xyz_m`，以及明确的 `quaternion_xyzw` 或 `quaternion_wxyz`。默认拒绝候选结果，不覆盖标定文件中的坐标系。
