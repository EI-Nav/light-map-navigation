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
1. 在`grounded_sam_func_contour.py`中，对于原有代码中的分割结果做处理，从原来的逐项素分割变成输出分割区域的轮廓,然后把代码封装一个函数。
   * 该python文件最后封装的函数需要返回Json格式的分割结果 
2. 在`action_counter.py`中，通过调用上述代码，封装成了一个`action`，通过调用封装好的函数，返回Json格式的分割结果供后续处理。