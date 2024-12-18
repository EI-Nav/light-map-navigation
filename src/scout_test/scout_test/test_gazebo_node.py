import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped, PoseStamped
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

import math
from tf_transformations import euler_from_quaternion, quaternion_from_euler


class GazeboSimulator(Node):
    def __init__(self):
        super().__init__('gazebo_simulator_node')

        # 初始化变量
        self.position = [0.0, 0.0, 0.0]  # [x, y, z] 坐标
        self.rotation = [0.0, 0.0, 0.0, 0.0]  # 四元数 [x, y, z, w]
        self.rgb = None  # 存储 RGB 图像
        self.depth = None  # 存储深度图像

        self.camera_position = [0.0, 0.0, 0.0]  # 相机的 [x, y, z] 坐标
        self.camera_rotation = [0.0, 0.0, 0.0, 0.0]  # 相机的四元数 [x, y, z, w]

        # 用于临时存储接收到的最新数据
        self._latest_rgb = [None] * 6  # 存储6个相机的RGB图像
        self._latest_depth = [None] * 6  # 存储6个相机的深度图像
        self._latest_position = [0.0, 0.0, 0.0]
        self._latest_rotation = [0.0, 0.0, 0.0, 0.0]

        # 创建 CvBridge 对象用于图像转换
        self.bridge = CvBridge()

        # TF2 相关
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.navigator = BasicNavigator()

        # 订阅6个相机的RGB图像
        self.rgb_subscribers = []
        for i in range(6):
            topic = f'/d435_{i}/color/image_raw'
            self.rgb_subscribers.append(
                self.create_subscription(
                    Image,
                    topic,
                    lambda msg, idx=i: self.rgb_callback(msg, idx),
                    10
                )
            )

        # 订阅6个相机的深度图像
        self.depth_subscribers = []
        for i in range(6):
            topic = f'/d435_{i}/depth/image_raw'
            self.depth_subscribers.append(
                self.create_subscription(
                    Image,
                    topic,
                    lambda msg, idx=i: self.depth_callback(msg, idx),
                    10
                )
            )

        # 订阅里程计信息
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/Odometry',  # 替换为实际的里程计话题
            self.odom_callback,
            10
        )

        self.get_logger().info("GazeboSimulatorNode initialized and subscriptions set.")

    def rgb_callback(self, msg, camera_id):
        """处理 RGB 图像消息"""
        try:
            # 将 ROS 图像消息转换为 OpenCV 格式并存储
            self._latest_rgb[camera_id] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().info(f"Received RGB image from camera {camera_id}.")
        except Exception as e:
            self.get_logger().error(f"Failed to process RGB image from camera {camera_id}: {e}")

    def depth_callback(self, msg, camera_id):
        """处理深度图像消息"""
        try:
            # 将 ROS 图像消息转换为 OpenCV 格式并存储
            self._latest_depth[camera_id] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.get_logger().info(f"Received Depth image from camera {camera_id}.")
        except Exception as e:
            self.get_logger().error(f"Failed to process Depth image from camera {camera_id}: {e}")

    def get_panorama(self):
        """返回每个相机的RGB图像和深度图像"""
        panorama_rgb = {i: self._latest_rgb[i] for i in range(6)}
        panorama_depth = {i: self._latest_depth[i] for i in range(6)}

        return {"rgb": panorama_rgb, "depth": panorama_depth}

    def rotate_panoramic(self):
        goal  = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = self._latest_position[0]
        goal.pose.position.y = self._latest_position[1]
        goal.pose.position.z = 0.0

        _, _, current_yaw = euler_from_quaternion(self._latest_rotation)
        target_yaw = current_yaw + math.radians(30)  # 30度转换为弧度

        if target_yaw > math.pi:
            target_yaw -= 2 * math.pi  # 保证 yaw 角度在 -pi 到 pi 之间

        goal.pose.orientation = quaternion_from_euler(0, 0, target_yaw)

        self.navigator.goToPose(goal)
        
        # Wait for the navigator to complete the task
        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            # if feedback:
            #     node.get_logger().info(f"Feedback: {feedback}")
        
        # Get the result of the navigation
        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info(f"success rotation")
            return True
        
        else:
            return False

    def get_next_action(self, wp):
        goal  = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = wp[0]
        goal.pose.position.y = wp[2]
        goal.pose.position.z = wp[1]
        goal.pose.orientation.w = 1
        goal.pose.orientation.x = 0
        goal.pose.orientation.y = 0 
        goal.pose.orientation.z = 0

        self.navigator.goToPose(goal)

        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            # if feedback:
            #     node.get_logger().info(f"Feedback: {feedback}")
        
        # Get the result of the navigation
        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info(f"success go to next action")
            return True
        
        else:
            return False

    def step(self):
        # 更新位置信息和旋转信息
        self.position = self._latest_position[:]
        self.rotation = self._latest_rotation[:]

        # 更新图像数据
        self.rgb = self._latest_rgb[0]
        self.depth = self._latest_depth[0]

        # 更新相机的位姿
        self.update_camera_pose()

        self.get_logger().info(
            f"Step executed. Position: {self.position}, Rotation: {self.rotation}"
        )

        if self.rgb is not None:
            self.get_logger().info("RGB image updated.")
        else:
            self.get_logger().warn("RGB image not received yet.")

        if self.depth is not None:
            self.get_logger().info("Depth image updated.")
        else:
            self.get_logger().warn("Depth image not received yet.")

        obs = {
            "rgb": self.rgb,
            "depth": self.depth
        }
        return obs

    def reset(self):
        # 重置位置和旋转信息
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0, 0.0]

        # 清空图像数据
        self.rgb = None
        self.depth = None

        # 重置相机信息
        self.camera_position = [0.0, 0.0, 0.0]
        self.camera_rotation = [0.0, 0.0, 0.0, 0.0]

        self.get_logger().info("Node state has been reset.")
        obs = {
            "rgb": self.rgb,
            "depth": self.depth
        }
        return obs

    def update_camera_pose(self):
        try:
            # 查询 lidar_odom 到 d435_0_link 的坐标变换
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'lidar_odom',  # 父坐标系
                'd435_0_link',  # 子坐标系
                rclpy.time.Time()  # 查询最新时间的变换
            )

            # 提取平移和旋转信息
            self.camera_position = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ]

            self.camera_rotation = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]

            self.get_logger().info(
                f"Camera pose updated: Position = {self.camera_position}, Rotation = {self.camera_rotation}"
            )

        except Exception as e:
            self.get_logger().warn(f"Failed to lookup transform: {e}")



def main(args=None):
    rclpy.init(args=args)

    gazebo_simulator = GazeboSimulator()

    gazebo_simulator.get_logger().info("Starting panoramic rotation...")

    # 调用 rotate_panoramic 方法进行旋转
    for i in range(6):
        success = gazebo_simulator.rotate_panoramic()
        if success:
            gazebo_simulator.get_logger().info("Panoramic rotation succeeded!")
        else:
            gazebo_simulator.get_logger().warn("Panoramic rotation failed!")

    # 关闭 ROS 2 节点
    gazebo_simulator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()