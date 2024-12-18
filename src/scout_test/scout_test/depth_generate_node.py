import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2  # 用于显示图像

class DepthImageSubscriber(Node):
    def __init__(self):
        super().__init__('depth_image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/d435_0/depth/image_raw',  # 替换为你的深度图像话题
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        # 检查图像编码是否为 16UC1
        if msg.encoding != '16UC1':
            self.get_logger().error(f"Unsupported encoding: {msg.encoding}")
            return
        
        # 将图像数据转换为 NumPy 数组
        try:
            depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
            depth_image_in_meters = depth_image.astype(np.float32) / 1000.0
        except ValueError as e:
            self.get_logger().error(f"Error reshaping depth image: {e}")
            return
        

        # 打印部分像素值（例如前 10x10 像素）
        self.get_logger().info(f"Top-left 10x10 pixel values:\n{depth_image_in_meters[:10, :10]}")

        # 显示图像（需归一化到 8 位以便可视化）
        normalized_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("Depth Image (Normalized)", normalized_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    depth_image_subscriber = DepthImageSubscriber()
    rclpy.spin(depth_image_subscriber)
    depth_image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()