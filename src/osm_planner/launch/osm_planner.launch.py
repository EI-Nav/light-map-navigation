from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    robot_pose_node = Node(
        package='llm_delivery',
        executable='robot_pose_pub_node',
        name='robot_pose_pub_node',
        output='screen',
        parameters=[]  
    )
    
    ipm_obs_use = Node(
        package='ipm_image_node',
        executable='ipm_obs_use',
        name='ipm_obs_use_node',
        output='screen',
        parameters=[]  
    )

    ipm_obs_server = Node(
        package='ipm_image_node',
        executable='ipm_obstacle_server',
        name='ipm_obstacle_server',
        output='screen',
        parameters=[]  
    )


    ld = LaunchDescription()
    ld.add_action(robot_pose_node)
    ld.add_action(ipm_obs_server)
    ld.add_action(ipm_obs_use)
        
    return ld
