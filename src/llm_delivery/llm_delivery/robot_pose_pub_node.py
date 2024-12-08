# File path: tf2_listener.py

import math
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry
# from scout_interfaces.msg import RobotPose  # Commented out as it's not used

def quaternion_to_yaw(quaternion) -> float:
    """Convert quaternion to yaw angle."""
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    # Calculate yaw angle
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return yaw


class TF2Listener(Node):
    """Node for listening to transformations and processing Odometry data."""

    def __init__(self) -> None:
        super().__init__('tf2_listener')

        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to the /Odometry topic
        self.odom_sub = self.create_subscription(
            Odometry,
            '/Odometry',
            self.odom_callback,
            10
        )

        # Create a publisher for the transformed pose
        self.robot_pose_pub = self.create_publisher(Pose, '/robot_position', 10)

    def odom_callback(self, msg: Odometry) -> None:
        """Callback function to handle incoming Odometry messages."""
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose

        # Attempt to transform the pose to the 'map' frame
        try:  
            transform = self.tf_buffer.lookup_transform('map', 'lidar_odom', rclpy.time.Time())
            transformed_pose = do_transform_pose(pose_stamped.pose, transform)

            # Publish the transformed pose
            self.robot_pose_pub.publish(transformed_pose)
            # self.get_logger().info(f"Transformed Pose: {transformed_pose}")
        except LookupException as e:
            self.get_logger().warn(f"Transform not available: {e}")
        except ExtrapolationException as e:
            self.get_logger().warn(f"Extrapolation exception: {e}")


def main(args=None) -> None:
    """Main entry point for the TF2Listener node."""
    rclpy.init(args=args)
    node = TF2Listener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()