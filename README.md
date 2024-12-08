# replan部分的分支
## 新增功能包
- `IPM`
- `osm_planner` 

## 安装依赖

### ipm功能包和osm_planner功能包依赖安装
```bash
# vision-msgs用于调用IPM算法
sudo apt-get install ros-humble-vision-msgs
# 安装tf库
sudo apt-get install ros-humble-tf-transformations 
# scipy库 用于调用KDTree存储地图
pip install scipy==1.10.0 -i https://pypi.mirrors.ustc.edu.cn/simple/
# 安装基础的接口用于调用segment_image
sudo apt-get install ros-humble-example-interfaces
```

## 使用步骤
```bash
# step1：启动仿真环境
# step2：在conda中启动分割的action
python action_counter.py
# step3：启动IPM服务
ros2 launch ipm_image_node start_ipm.launch.py
# step4：启动osm_planner node
ros2 run osm_planner osm_planner_opti_node
```

## 参数配置
### 相机相关配置
* 相机话题设置：位于Segment Anything的`action_counter.py`中
* 修改相机内参：位于`ipm_obstacle_server.py`中的`109行`起的`self.camera_info`，同时修改RGB图像所在的坐标系
* 修改机器人的激光雷达话题：在`ipm_obs_use.py`中修改`self.livox_sub`订阅中的激光雷达话题。

## Grounded_SAM改动点
0. 代码参考：`src/grounded_sam`
1. 在`grounded_sam_func_contour.py`中，对于原有代码中的分割结果做处理，从原来的逐项素分割变成输出分割区域的轮廓,然后把代码封装一个函数。
   * 该python文件最后封装的函数需要返回Json格式的分割结果 
2. 在`action_counter.py`中，通过调用上述代码，封装成了一个`action`，通过调用封装好的函数，返回Json格式的分割结果供后续处理。

## IPM功能包
* 主要功能在`src/ipm_package/ipm_image_node`功能包中

## Navigation2有关costmap的改动点
主要在`nav2_params_sim.yaml`中进行修改，需要添加模拟生成的点云和`Scan`来利用`nav2`自带的工具生成costmap。
1. 全局代价地图生成
   ```bash
      stvl_layer:
        plugin: "spatio_temporal_voxel_layer/SpatioTemporalVoxelLayer"
        # https://github.com/SteveMacenski/spatio_temporal_voxel_layer
        enabled:                  true
        voxel_decay:              0.5                               # 如果是线性衰减，单位为秒；如果是指数衰减，则为 e 的 n 次方
        decay_model:              0                                 # 衰减模型，0=线性，1=指数，-1=持久
        voxel_size:               0.1                              # 每个体素的尺寸，单位为米
        track_unknown_space:      true                              # default space is unknown
        mark_threshold:           0                                 # voxel height
        update_footprint_enabled: true
        combination_method:       1                                 # 1=max, 0=override
        origin_z:                 0.0                               # 单位为米
        publish_voxel_map:        true                              # default false, 是否发布体素地图
        transform_tolerance:      0.2                               # 单位为秒
        mapping_mode:             false                             # default off, saves map not for navigation
        map_save_duration:        60.0                              # deault 60s, how often to autosave
        observation_sources:      sem_mark sem_clear
      sem_mark:
          data_type: PointCloud2
          topic: /sem_obstacles_points_use
          marking: true
          clearing: false
          obstacle_range: 10.0                                       # meters
          min_obstacle_height: -1.0                                  # default 0, meters
          max_obstacle_height: 0.5                                  # default 3, meters
          expected_update_rate: 0.0                                 # default 0, if not updating at this rate at least, remove from buffer
          observation_persistence: 0.0                              # default 0, use all measurements taken during now-value, 0=latest
          inf_is_valid: false                                       # default false, for laser scans
          filter: "voxel"                                           # default passthrough, apply "voxel", "passthrough", or no filter to sensor data, recommend on
          voxel_min_points: 0                                       # default 0, minimum points per voxel for voxel filter
          clear_after_reading: true                                 # default false, clear the buffer after the layer gets readings from it
        sem_clear:
          enabled: true                                             # default true, can be toggled on/off with associated service call
          data_type: PointCloud2
          topic: /sem_obstacles_points_use
          marking: false
          clearing: true
          max_z: 8.0                                                # default 10, meters
          min_z: -1.0                                                # default 0, meters
          vertical_fov_angle: 1.029                                 # 垂直视场角，单位为弧度，For 3D lidars it's the symmetric FOV about the planar axis.
          vertical_fov_padding: 0.05                                # 3D Lidar only. Default 0, in meters
          horizontal_fov_angle: 6.29                                # 3D 激光雷达水平视场角
          decay_acceleration: 5.0                                   # default 0, 1/s^2.
          model_type: 1                                             # 0=深度相机，1=3D激光雷达
   ```
2. 局部代价地图生成
   ```bash
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observati1on_sources: scan sem_scan
        # observati1on_sources: scan
        sem_scan:
          topic: /sem_obstacles_scan
          raytrace_max_range: 6.0
          obstacle_max_range: 6.0
          obstacle_min_range: 0.1
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          inf_is_valid: true
          data_type: "LaserScan"
        scan:
          topic: /scan
          raytrace_max_range: 6.0
          obstacle_max_range: 6.0
          obstacle_min_range: 0.1
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          inf_is_valid: true
          data_type: "LaserScan"
   ```