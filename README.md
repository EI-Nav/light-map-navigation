# Light-Map-Navigation

## 一. 项目介绍

本项目实现了基于OpenStreetMap地图（无需提前建图）的具身智能导航系统。

### 1.1 rm_simulation 话题接口

| **Topic name**      | **Type**                        | **Note**                         |
|:-------------------:|:-------------------------------:|:--------------------------------:|
| /camera_sensor/image_raw |      sensor_msgs/msg/Image      |     相机消息类型      |
|       /livox/lidar       | livox_ros_driver2/msg/CustomMsg | Mid360 自定义消息类型 |
| /livox/lidar/pointcloud  |   sensor_msgs/msg/PointCloud2   |     点云消息类型      |
|        /livox/imu        |       sensor_msgs/msg/Imu       |  Gazebo 插件仿真 IMU  |
| /cmd_vel | geometry_msgs/msg/Twist | 差速小车运动控制接口 |

### 1.2 架构图

![image-20240718102111552](doc/image-20240718102111552.png)

## 二. 环境配置

当前开发环境为 Ubuntu22.04, ROS2 humble, Gazebo Classic 11.10.0

### 方式一：使用 Docker

本项目已配置基础 Dockerfile，并可使用 [Dev Container](https://containers.dev/) 进行仿真测试和开发。

> DevContainer 的一个特征是将代码的运行环境与代码本身完全隔离开来，在效果上类似将 workspace 挂载到了容器中。  
> 在 Docker-image 中仅仅包含有关系统的配置（例如修改 .baserc 或安装依赖包等），其本身不存储任何项目代码和工作空间，做到了代码与环境的完全隔离。  
> 可以通过 devcontainer.json 配置文件，快速修改和分发容器配置。  
> 与 VSCode 深度融合，一键启动，无需任何命令。

1. 安装 Docker、NVIDIA Container Toolkit并进行配置

    ```bash
    # docker 一键安装
    wget http://fishros.com/install -O fishros && . fishros
    
    # nvidia-container-toolkit 安装
    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
    
    # 配置docker container可以使用显卡
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

    

2. 下载镜像并安装

    ```sh
    # 百度网盘下载
    链接: https://pan.baidu.com/s/1HO5n8JxQ_h3IeL7lKmnSog?pwd=2cwm 提取码: 2cwm 
    
    # 加载镜像
    docker load -i light-map-navigation.tar
    ```
    
3. 在宿主机 VSCode 中 安装 `ms-vscode-remote.remote-containers` 插件

4. 使用VSCode打开本项目，输入快捷键 `Ctrl+Shift+P`， 输入并点击 `Dev Containers:Rebuild and Reopen in Container`

### 方式二：源码安装

1. 克隆仓库

    ```sh
    git clone https://github.com/EI-Nav/light-map-navigation.git --depth=1
    git submodule init
    git submodule update
    ```

2. 安装 [Livox SDK2](https://github.com/Livox-SDK/Livox-SDK2)

    ```sh
    sudo apt install cmake
    ```

    ```sh
    git clone https://github.com/Livox-SDK/Livox-SDK2.git
    cd ./Livox-SDK2/
    mkdir build
    cd build
    cmake .. && make -j
    sudo make install
    ```

3. 安装依赖

    ```sh
    cd light-map-navigation

    rosdep install -r --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y
    ```

4. 编译

    ```sh
    colcon build --symlink-install
    ```

## 三. 运行

### 3.1 可选参数

1. `world`:

    - 仿真模式
        - `RMUL` - [2024 Robomaster 3V3 场地](https://bbs.robomaster.com/forum.php?mod=viewthread&tid=22942&extra=page%3D1)
        - `RMUC` - [2024 Robomaster 7V7 场地](https://bbs.robomaster.com/forum.php?mod=viewthread&tid=22942&extra=page%3D1)

    - 真实环境
        - 自定，world 等价于 `.pcd(ICP使用的点云图)` 文件和 `.yaml(Nav使用的栅格地图)` 的名称

2. `mode`:
   - `mapping` - 边建图边导航
   - `nav` - 已知全局地图导航

3. `lio`:
   - `fastlio` - 使用 [Fast_LIO](https://github.com/LihanChen2004/FAST_LIO/tree/ROS2)，里程计约 10Hz
   - `pointlio` - 使用 [Point_LIO](https://github.com/LihanChen2004/Point-LIO/tree/RM2024_SMBU_auto_sentry)，可以输出100+Hz的Odometry，对导航更友好，但相对的，CPU占用会更高

4. `localization` (仅 `mode:=nav` 时本参数有效)
   - `slam_toolbox` - 使用 [slam_toolbox](https://github.com/SteveMacenski/slam_toolbox) localization 模式定位，动态场景中效果更好
   - `amcl` - 使用 [AMCL](https://navigation.ros.org/configuration/packages/configuring-amcl.html) 经典算法定位
   - `icp` - 使用 [icp_registration](https://github.com/baiyeweiguang/CSU-RM-Sentry/tree/main/src/rm_localization/icp_registration)，仅在第一次启动或者手动设置 /initialpose 时进行点云配准。获得初始位姿后只依赖 LIO 进行定位，没有回环检测，在长时间运行后可能会出现累积误差。

    Tips:
    1. 若使用 AMCL 算法定位时，启动后需要在 rviz2 中手动给定初始位姿。
    2. 若使用 slam_toolbox 定位，需要提供 .posegraph 地图，详见 [如何保存 .pgm 和 .posegraph 地图？](https://gitee.com/SMBU-POLARBEAR/pb_rmsimulation/issues/I9427I)
    3. 若使用 ICP_Localization 定位，需要提供 .pcd 点云图

5. `lio_rviz`:
   - `True` - 可视化 FAST_LIO 或 Point_LIO 的点云图

6. `nav_rviz`:
   - `True` - 可视化 navigation2

### 3.2 仿真模式示例

- 边建图边导航

    ```sh
    ros2 launch rm_nav_bringup bringup_sim.launch.py \
    world:=TOSM \
    mode:=mapping \
    lio:=fastlio \
    lio_rviz:=False \
    nav_rviz:=True \
    use_sim_time:=True
    ```

- 已知全局地图导航（**当前重点关注**）

    ```sh
    # 运行机器人导航系统及仿真环境
    ros2 launch rm_nav_bringup bringup_sim.launch.py \
    world:=OSM \
    mode:=nav \
    lio:=fastlio \
    localization:=icp \
    lio_rviz:=False \
    nav_rviz:=True \
    use_sim_time:=True

    # bash中需要单独启动分割节点
    ros2 launch linefit_ground_segmentation_ros segmentation.launch.py

    # 发布机器人当前位置节点
    ros2 run llm_delivery robot_pose_pub_node
    
    # 运行探索节点
    ros2 run llm_exploration_py get_unit_num_service

    # 运行单元号识别节点
    ros2 run llm_exploration_py find_unit_server
    
    # 运行配送服务节点
    ros2 run llm_delivery llm_delivery_node
    ```

### 3.3 真实模式示例

- 边建图边导航

    ```sh
    ros2 launch rm_nav_bringup bringup_real.launch.py \
    world:=YOUR_WORLD_NAME \
    mode:=mapping  \
    lio:=fastlio \
    lio_rviz:=False \
    nav_rviz:=True
    ```

    Tips:

    1. 保存点云 pcd 文件：需先在 [fastlio_mid360.yaml](src/rm_nav_bringup/config/reality/fastlio_mid360_real.yaml) 中 将 `pcd_save_en` 改为 `true`，并设置 .pcd 文件的路径，运行时新开终端输入命令 `ros2 service call /map_save std_srvs/srv/Trigger`，即可保存点云文件。
    2. 保存地图：请参考 [如何保存 .pgm 和 .posegraph 地图？](https://gitee.com/SMBU-POLARBEAR/pb_rmsimulation/issues/I9427I)。地图名需要与 `YOUR_WORLD_NAME` 保持一致。

- 已知全局地图导航

    ```sh
    ros2 launch rm_nav_bringup bringup_real.launch.py \
    world:=kjds_2 \
    mode:=nav \
    lio:=fastlio \
    localization:=icp \
    lio_rviz:=False \
    nav_rviz:=True
    ```

    Tips: 栅格地图文件和 pcd 文件需具为相同名称，分别存放在 `src/rm_nav_bringup/map` 和 `src/rm_nav_bringup/PCD` 中，启动导航时 world 指定为文件名前缀即可。

### 3.4 小工具 - 键盘控制

```sh
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=cmd_vel_chassis
```

## 四. 实车适配关键参数

1. 雷达 ip

    本导航包已内置 [livox_ros_driver2](https://gitee.com/SMBU-POLARBEAR/livox_ros_driver2_humble)，可直接修改 [MID360_config.json](./src/rm_nav_bringup/config/reality/MID360_config.json) - `lidar_configs` - `ip`

2. 测量机器人底盘正中心到雷达的相对坐标

    x, y 距离比较重要，将影响云台旋转时解算到 base_link 的坐标准确性

    填入 [measurement_params_real.yaml](./src/rm_nav_bringup/config/reality/measurement_params_real.yaml)

    若雷达倾斜放置，无需在此处填入 rpy，而是将点云旋转角度填入 [MID360_config.json](./src/rm_nav_bringup/config/reality/MID360_config.json) - `extrinsic_parameter`

3. 测量雷达与地面的垂直距离

    此参数影响点云分割效果

    填入 [segmentation_real.yaml](./src/rm_nav_bringup/config/reality/segmentation_real.yaml) - `sensor_height`

4. nav2_params

    参数很多，比较重要的是 robot_radius 和 速度相关参数。详见 [nav2官方文档](https://docs.nav2.org/)

## 五. 机器人模型

四轮小车，采用差速运动模型。

## 六. 存在的问题

- ~~局部规划可能存在问题，在转弯的时候有可能会撞墙。~~
- ~~无法穿行桥洞类区域（树下）。~~
- 在某些case下，规控很难到达waypoint。(反复横跳现象)
- icp定位模式下坐标系有问题，小车处于地图的下方。

## 七. TODO

- [x] 在仿真环境中，将SLAM导航方式调好。
- [x] 基于LLM，实现基本的自然语言交互导航（导航到几号楼）。
- [ ] 更加智能的基于自然语言交互的决策模块。
- [ ] 基于因子图的多模态定位模块。
- [x] 在楼附近进行探索，找到指定的单元门。
- [ ] 配送请求生成器
- [ ] 导航结果评估器
- [ ] easy and hard world models

## 致谢

参考的仓库:

https://gitee.com/SMBU-POLARBEAR/pb_rm_simulation/tree/master

https://github.com/LihanChen2004/pb_rm_simulation?tab=readme-ov-file
