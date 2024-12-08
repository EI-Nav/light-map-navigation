from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='osrm_server',
            executable='osrm_server_node',
            name='osrm_server',
            output='screen',
            parameters=[
                # Add any necessary parameters here
            ],
        ),
    ])