import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from example_interfaces.action import Fibonacci  # 使用标准的Fibonacci action
from rclpy.action import ActionServer, CancelResponse, GoalResponse
import cv2
import numpy as np
import json
import os
import sys
import tf2_ros

# 确保process_image函数路径正确
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Grounded_Segment_Anything.grounded_sam_func_contour import process_image

text_prompt_used = "road"
camera_class = 1 # 0 for RGB camera, 1 for depth camera

class ImageSegmentationActionServer(Node):
    def __init__(self):
        super().__init__('image_segmentation_action_server')

        # 使用Fibonacci Action来实现图像分割
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'segment_image',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        if camera_class == 0:
            self.subscription = self.create_subscription(
                Image,
                '/camera_sensor/image_raw',
                self.image_callback,
                10
            )

        elif camera_class == 1:
            # 订阅图像话题接收摄像头数据
            self.subscription = self.create_subscription(
                Image,
                '/color/image_raw',
                self.image_callback,
                10
            )
        
        # 用于存储最新的帧
        self.latest_frame = None

        # 初始化TF缓冲区和监听器获取变换
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def image_callback(self, msg):
        """图像话题的回调函数,存储最新的图像。"""
        try:
            self.latest_frame = self.ros_image_to_cv2(msg)
            self.get_logger().info('Received new image frame.')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")

    def ros_image_to_cv2(self, img_msg):
        """将ROS图像消息转换为OpenCV格式,不使用CvBridge。"""
        if img_msg.encoding == "bgr8":
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
        elif img_msg.encoding == "rgb8":
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img_msg.encoding == "mono8":
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width)
        else:
            raise ValueError(f"Unsupported image encoding: {img_msg.encoding}")
        return img

    def goal_callback(self, goal_request):
        """接受或拒绝目标请求。"""
        self.get_logger().info('Received a segmentation goal request.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """处理活动目标的取消请求。"""
        self.get_logger().info('Received request to cancel segmentation.')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """在收到目标请求时执行分割任务。"""
        self.get_logger().info('Executing image segmentation goal...')

        if self.latest_frame is None:
            goal_handle.abort()
            result = Fibonacci.Result()
            result.sequence = [0]  # 表示失败
            return result

        try:
            # 保存最新帧到文件
            image_path = "/home/huo/Grounded_Segment_Anything/latest_frame.jpg"
            cv2.imwrite(image_path, self.latest_frame)
            self.get_logger().info(f"Saved latest image frame to {image_path}")

            # 调用process_image函数进行图像分割
            json_data = process_image(input_image=image_path, text_prompt=text_prompt_used)

            # 分割成功，设置结果为成功
            goal_handle.succeed()
            result = Fibonacci.Result()
            result.sequence = [1]  # 表示成功
            result.sequence.extend([ord(c) for c in json.dumps(json_data, indent=4)])
            return result
        except Exception as e:
            # 分割失败，设置结果为失败
            goal_handle.abort()
            result = Fibonacci.Result()
            result.sequence = [0]  # 表示失败
            result.sequence.extend([ord(c) for c in str(e)])
            self.get_logger().error(f"Segmentation failed. Error: {str(e)}")
            return result

def main(args=None):
    rclpy.init(args=args)

    # 创建并运行节点
    image_segmentation_action_server = ImageSegmentationActionServer()
    rclpy.spin(image_segmentation_action_server)

    # 关闭节点
    image_segmentation_action_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
