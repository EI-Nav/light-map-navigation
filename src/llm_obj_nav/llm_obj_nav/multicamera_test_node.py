import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import time

class MultiCameraSampler(Node):
    def __init__(self):
        super().__init__('multi_camera_sampler')
        self.bridge = CvBridge()
        
        # 六个相机的订阅者
        self.rgb_topics = [
            '/d435_0/color/image_raw',
            '/d435_1/color/image_raw',
            '/d435_2/color/image_raw',
            '/d435_3/color/image_raw',
            '/d435_4/color/image_raw',
            '/d435_5/color/image_raw'
        ]
        self.depth_topics = [
            '/d435_0/depth/image_raw',
            '/d435_1/depth/image_raw',
            '/d435_2/depth/image_raw',
            '/d435_3/depth/image_raw',
            '/d435_4/depth/image_raw',
            '/d435_5/depth/image_raw'
        ]

        self.rgb_images = [None] * 6
        self.depth_images = [None] * 6
        self.received_count = 0

        # 订阅所有相机的话题
        for i in range(6):
            self.create_subscription(Image, self.rgb_topics[i], self.create_rgb_callback(i), 10)
            self.create_subscription(Image, self.depth_topics[i], self.create_depth_callback(i), 10)
        
        self.get_logger().info("Multi-camera sampler initialized, waiting for synchronized images...")

    def create_rgb_callback(self, index):
        """动态生成 RGB 图像回调函数"""
        def callback(msg):
            self.rgb_images[index] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.check_all_received()
        return callback

    def create_depth_callback(self, index):
        """动态生成深度图像回调函数"""
        def callback(msg):
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_images[index] = (depth_raw / 5.0 * 255.0).astype(np.uint8)
            self.check_all_received()
        return callback

    def check_all_received(self):
        """检查是否接收到了所有相机的 RGB 和深度数据"""
        self.received_count += 1
        if all(img is not None for img in self.rgb_images) and all(img is not None for img in self.depth_images):
            self.get_logger().info("All six RGB and depth images received.")
            self.process_images()
            # 重置接收器
            self.rgb_images = [None] * 6
            self.depth_images = [None] * 6
            self.received_count = 0

    def process_images(self):
        """处理六个相机的图像数据"""
        # 存储 RGB 图像
        for i, rgb_image in enumerate(self.rgb_images):
            filename = f"rgb_camera_{i}.jpg"
            cv2.imwrite(filename, rgb_image)
            self.get_logger().info(f"Saved RGB image from camera {i} to {filename}.")

        # 存储深度图像
        for i, depth_image in enumerate(self.depth_images):
            filename = f"depth_camera_{i}.png"
            cv2.imwrite(filename, depth_image)
            self.get_logger().info(f"Saved depth image from camera {i} to {filename}.")
        
        self.get_logger().info("All images processed and saved successfully.")

def main(args=None):
    rclpy.init(args=args)
    sampler = MultiCameraSampler()
    try:
        rclpy.spin(sampler)
    except KeyboardInterrupt:
        sampler.get_logger().info("Shutting down multi-camera sampler.")
    finally:
        sampler.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
