import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import PoseStamped,Pose
from nav_msgs.msg import Odometry
from scout_interfaces.msg import RobotPose
import tf2_ros
import math

def quaternion_to_yaw(quaternion):
    # 提取四元数分量
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    # 计算yaw角度
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return yaw


class TF2Listener(Node):
    def __init__(self):
        super().__init__('tf2_listener')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 订阅 /odom 话题
        self.odom_sub = self.create_subscription(
            Odometry,
            '/Odometry',
            self.odom_callback,
            10
        )

        # 创建一个发布器，用于发布转换后的Odometry
        self.odom_pub = self.create_publisher(RobotPose, '/robot_pose', 10)

    def odom_callback(self, msg):
        # 提取从Odometry中的pose
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose

        # 尝试进行坐标系转换
        try:  
            transform = self.tf_buffer.lookup_transform('map', 'lidar_odom', rclpy.time.Time())
            # print(transform)
            # 获取(x,y)
            transformed_pose = do_transform_pose(pose_stamped.pose, transform)
            yaw = quaternion_to_yaw(transformed_pose.orientation)
            # print(yaw)
            robotpose = RobotPose()
            robotpose.x = transformed_pose.position.x
            robotpose.y = transformed_pose.position.y
            robotpose.yaw = yaw

            self.odom_pub.publish(robotpose)
            self.get_logger().info(f"Transformed Pose: {robotpose}")
        except tf2_ros.LookupException as e:
            self.get_logger().warn(f"Transform not available: {e}")
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(f"Extrapolation exception: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TF2Listener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()