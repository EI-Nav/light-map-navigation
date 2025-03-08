#!/bin/bash

apt install psmisc -y

# Check if parameter file path is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <path_to_params_file>"
  exit 1
fi

# Get parameter file path from command line argument
params_file=$1

# Check if file exists
if [ ! -f "$params_file" ]; then
  echo "Error: File not found!"
  exit 1
fi

# Get filename
filename=$(basename -- "$params_file")
filename_without_extension="${filename%.*}"

# Build CSV file path in the result directory
csv_file="result/${filename_without_extension}.csv"

echo "Reading from: $params_file"
echo "Writing to: $csv_file"

# Backup old CSV file if it exists
if [ -f "$csv_file" ]; then
  timestamp=$(date +"%Y%m%d_%H%M%S")
  backup_file="result/${filename_without_extension}_backup_${timestamp}.csv"
  mv "$csv_file" "$backup_file"
  echo "Backup created: $backup_file"
fi

# Iterate through each line in the text file
while IFS= read -r line || [[ -n "$line" ]]
do
  echo "Starting navigation and simulation environment..."
  
  # Start robot navigation system and simulation environment
  ros2 launch classic_nav_bringup bringup_sim.launch.py \
    world:=MEDIUM_OSM \
    mode:=nav \
    lio:=fastlio \
    localization:=icp \
    lio_rviz:=False \
    nav_rviz:=True \
    use_sim_time:=True &
  sim_pid=$!  # Get simulation environment process ID
  
  # Wait for simulation environment to start
  sleep 20  # Adjust time according to actual needs
  
  echo "Starting exploration nodes..."
  
  # Start delivery service related nodes
  ros2 launch delivery_bringup delivery_system_sim.launch.py &
  
  sleep 5  # Adjust time according to actual needs
  
  # Start delivery service node and pass the current command
  ros2 run delivery_executor delivery_executor_action_client --instruction "$line" --log-file "$csv_file" &
  
  # Wait for delivery task to complete
  sleep 900  # Can be adjusted according to actual task duration
  
  echo "Stopping exploration nodes and simulation environment..."

  kill $(pgrep rviz2)

  sleep 5
  
  # Wait for all processes to close
  wait $sim_pid 2>/dev/null

  killall /usr/bin/python3

  ros2 daemon stop

  ros2 daemon start
  
done < "$params_file"