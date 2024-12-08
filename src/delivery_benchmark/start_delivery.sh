#!/bin/bash

# 检查是否提供了参数文件路径
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <path_to_params_file>"
  exit 1
fi

# 从命令行参数获取参数文件路径
params_file=$1

# 检查文件是否存在
if [ ! -f "$params_file" ]; then
  echo "Error: File not found!"
  exit 1
fi

# 获取文件名
filename=$(basename -- "$params_file")
filename_without_extension="${filename%.*}"

# 构建 result 目录下的 CSV 文件路径
csv_file="result/${filename_without_extension}.csv"

echo "Reading from: $params_file"
echo "Writing to: $csv_file"

# 如果存在旧的CSV文件，将其备份
if [ -f "$csv_file" ]; then
  timestamp=$(date +"%Y%m%d_%H%M%S")
  backup_file="result/${filename_without_extension}_backup_${timestamp}.csv"
  mv "$csv_file" "$backup_file"
  echo "Backup created: $backup_file"
fi

# 遍历文本文件中的每一行
while IFS= read -r line || [[ -n "$line" ]]
do
  echo "Starting navigation and simulation environment..."
  
  # 启动机器人导航系统及仿真环境
  ros2 launch rm_nav_bringup bringup_sim.launch.py \
    world:=OSM \
    mode:=nav \
    lio:=fastlio \
    localization:=icp \
    lio_rviz:=False \
    nav_rviz:=True \
    use_sim_time:=True &
  sim_pid=$!  # 获取仿真环境进程ID
  
  # 等待仿真环境启动完成
  sleep 20  # 根据实际需要调整时间
  
  echo "Starting exploration nodes..."
  
  # 启动探索节点
  ros2 run llm_exploration_py get_unit_num_service &
  exploration_pid=$!  # 获取探索节点进程ID
  
  # 启动单元号识别节点
  ros2 run llm_exploration_py find_unit_server &
  find_unit_pid=$!  # 获取单元号识别节点进程ID
  
  # 等待探索和单元号识别节点启动
  sleep 5  # 根据实际需要调整时间

  echo "Starting task record node..."

  python3 task_record.py --filename "$csv_file" &

  sleep 1
  
  echo "Starting delivery service node with parameter: $line"

  ros2 run llm_delivery robot_pose_pub_node &

  sleep 1
  
  # 启动配送服务节点并传递当前的指令
  ros2 run llm_delivery llm_delivery_node "$line" &
  
  # 等待配送任务完成
  sleep 600  # 可根据实际任务的耗时调整
  
  echo "Stopping exploration nodes and simulation environment..."

  kill $(pgrep rviz2)

  sleep 5
  
  # 等待所有进程关闭
  wait $sim_pid 2>/dev/null

  killall /usr/bin/python3

  ros2 daemon stop

  ros2 daemon start
  
done < "$params_file"