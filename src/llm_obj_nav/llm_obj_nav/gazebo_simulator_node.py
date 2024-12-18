import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

import math
from math import radians, cos, sin, atan2, sqrt
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import quaternion

class CameraPosition:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"CameraPosition(x={self.x}, y={self.y}, z={self.z})"


class GazeboSimulator(Node):
    def __init__(self):
        super().__init__('gazebo_simulator_node')

        # 初始化 ROS2 参数
        self.declare_parameter('position', [0.0, 0.0, 0.0])  # [x, y, z] 坐标
        self.declare_parameter('rotation', [0.0, 0.0, 0.0, 0.0])  # 四元数 [x, y, z, w]
        self.declare_parameter('rgb', None)  # 存储 RGB 图像
        self.declare_parameter('depth', None)  # 存储深度图像

        self.camera_position = CameraPosition()  # 相机的 [x, y, z] 坐标
        self.camera_rotation = quaternion.quaternion(1,0,0,0)  # 相机的四元数 [x, y, z, w]

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

        # 标志位
        self.is_rgb_receive = False
        self.is_depth_receive = False
        self.is_odom_receive = False

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


    def odom_callback(self, msg):
        """处理里程计信息"""
        # 提取位置信息并存储到最新数据
        self._latest_position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ]

        # 提取旋转的四元数
        self._latest_rotation = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        self.is_odom_receive = True
        self.get_logger().info("Received Odometry Information.")

        # 更新参数服务器
        position_param = rclpy.parameter.Parameter('position', rclpy.Parameter.Type.DOUBLE_ARRAY, self._latest_position)
        rotation_param = rclpy.parameter.Parameter('rotation', rclpy.Parameter.Type.DOUBLE_ARRAY, self._latest_rotation)
        
        self.set_parameters([position_param, rotation_param])


    def rgb_callback(self, msg, camera_id):
        """处理 RGB 图像消息"""
        try:
            # 将 ROS 图像消息转换为 OpenCV 格式并存储
            self._latest_rgb[camera_id] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.is_rgb_receive = True
            # 更新参数服务器
            rgb_param = rclpy.parameter.Parameter('rgb', rclpy.Parameter.Type.STRING, str(self._latest_rgb[camera_id]))
            self.set_parameters([rgb_param])
        except Exception as e:
            self.get_logger().error(f"Failed to process RGB image from camera {camera_id}: {e}")

    def depth_callback(self, msg, camera_id):
        """处理深度图像消息"""
        try:
            # 将 ROS 图像消息转换为 OpenCV 格式并存储
            self._latest_depth[camera_id] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.is_depth_receive = True
            # 更新参数服务器
            depth_param = rclpy.parameter.Parameter('depth', rclpy.Parameter.Type.STRING, str(self._latest_depth[camera_id]))
            self.set_parameters([depth_param])
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
        
        # Get the result of the navigation
        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info(f"success rotation")
            return True
        
        else:
            return False

    def step(self):
        # 更新位置信息和旋转信息
        self.position = self.get_parameter('position').value
        self.rotation = self.get_parameter('rotation').value

        # 更新图像数据
        self.rgb = self.get_parameter('rgb').value
        self.depth = self.get_parameter('depth').value

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
        # 重置图像和位姿信息
        self._latest_rgb = [None] * 6
        self._latest_depth = [None] * 6
        self._latest_position = [0.0, 0.0, 0.0]
        self._latest_rotation = [0.0, 0.0, 0.0, 0.0]

        # 清空图像数据
        self.rgb = None
        self.depth = None

        # 重置相机信息
        self.camera_position = CameraPosition()
        self.camera_rotation = quaternion.quaternion(1,0,0,0)

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
            self.camera_position.x = transform.transform.translation.x
            self.camera_position.y = transform.transform.translation.y
            self.camera_position.z = transform.transform.translation.z
            
            self.rotation = quaternion.from_float_array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])

            self.get_logger().info(
                f"Camera pose updated: Position = {self.camera_position}, Rotation = {self.camera_rotation}"
            )

        except Exception as e:
            self.get_logger().warn(f"Failed to lookup transform: {e}")

def main(args=None):
    rclpy.init(args=args)
    sim_env = GazeboSimulator()
    rclpy.spin(sim_env)

if __name__ == '__main__':
    main()
