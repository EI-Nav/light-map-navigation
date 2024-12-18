import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
import numpy as np


import os
import sys
from gazebo_simulator_node import GazeboSimulator
sys.path.append(os.path.join(os.getcwd(), "src/llm_obj_nav/llm_obj_nav"))
from instructnav.objnav_agent import HM3D_Objnav_Agent
from instructnav.mapper import Instruct_Mapper
from instructnav.mapping_utils.transform import gazebo_camera_intrinsic



class objnav_benchmark(Node):
    def __init__(self):
        super().__init__('objnav_benchmark_node')

        # 声明 ROS2 参数
        self.declare_parameter("eval_episodes", 30)
        self.declare_parameter("mapper_resolution", 0.05)
        self.declare_parameter("path_resolution", 0.2)
        self.declare_parameter("path_scale", 5)

        # 获取参数
        self.eval_episodes = self.get_parameter("eval_episodes").get_parameter_value().integer_value
        self.mapper_resolution = self.get_parameter("mapper_resolution").get_parameter_value().double_value
        self.path_resolution = self.get_parameter("path_resolution").get_parameter_value().double_value
        self.path_scale = self.get_parameter("path_scale").get_parameter_value().integer_value

        self.gazebo_env = GazeboSimulator()

        gazebo_mapper = Instruct_Mapper(gazebo_camera_intrinsic(),
                                        pcd_resolution = self.mapper_resolution,
                                        grid_resolution=self.path_resolution,
                                        grid_size=self.path_scale
                                        )
        
        self.obj_agent = HM3D_Objnav_Agent(self.gazebo_env, gazebo_mapper)

        self.eval_episodes = 500 # 定义循环次数

        # self.timer = self.create_timer(1.0, self.run)  # 每秒调用一次

        # rclpy.spin(self.gazebo_env)
        # rclpy.spin_once(self.gazebo_env)

    def run(self):
        self.get_logger().info(f"Make Run...")
        rclpy.spin_once(self.gazebo_env)
        self.obj_agent.reset()
        self.obj_agent.make_plan()
        for _ in range(self.eval_episodes):
            self.obj_agent.step()
            self.get_logger().info(f"Episode completed. Running next episode...")
            rclpy.spin_once(self.gazebo_env)



def main(args=None):
    rclpy.init(args=args)

    obj_benchmark = objnav_benchmark()
    obj_benchmark.run()

    rclpy.shutdown()
    

if __name__ == '__main__':
    main()

