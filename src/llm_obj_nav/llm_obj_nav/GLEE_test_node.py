#! /home/lab417/.conda/envs/InstructNav/bin/python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge
import numpy as np

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src/llm_obj_nav/llm_obj_nav"))
from instructnav.cv_utils.image_percevior import GLEE_Percevior  # 确保 glee_detector 模块在你的路径中



class GLEETester(Node):
    def __init__(self):
        super().__init__('glee_tester')

        # 初始化 GLEE Perceiver
        self.glee_perceiver = GLEE_Percevior()

        # ROS 订阅和发布
        self.image_sub = self.create_subscription(Image, '/d435_0/color/image_raw', self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, '/glee/perceived_image', 10)
        self.bridge = CvBridge()

        self.get_logger().info("GLEE tester node initialized.")

    def image_callback(self, msg):
        self.get_logger().info(f"Receive Image")
        """
        接收摄像头图像并运行 GLEE 模型进行目标检测与分割。
        """
        # 将 ROS 图像消息转换为 OpenCV 图像
        # cv_image = self.ros_image_to_cv2(msg)
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 检查并确保格式正确（如 RGB 格式）
        if cv_image.ndim == 2:  # 灰度图
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        elif cv_image.shape[2] == 3:  # 确保 RGB 格式
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Unexpected cv_image format!")

        # 确保类型为 np.ndarray
        if not isinstance(cv_image, np.ndarray):
            raise TypeError("cv_image is not a valid numpy array!")
        print(3)

        # 使用 GLEE_Percevior 进行感知处理
        pred_classes, pred_masks, pred_confidences, visualizations = self.glee_perceiver.perceive(
            cv_image
        )

        # 打印检测到的类别信息
        self.get_logger().info(f"Detected classes: {pred_classes}")
        self.get_logger().info(f"Confidences: {pred_confidences}")

        # 将可视化结果发布到 ROS
        vis_image = visualizations[0]
        vis_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
        self.image_pub.publish(vis_msg)


def main(args=None):
    rclpy.init(args=args)
    glee_tester = GLEETester()

    try:
        rclpy.spin(glee_tester)
    except KeyboardInterrupt:
        glee_tester.get_logger().info('Shutting down GLEE tester node.')
    finally:
        glee_tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
