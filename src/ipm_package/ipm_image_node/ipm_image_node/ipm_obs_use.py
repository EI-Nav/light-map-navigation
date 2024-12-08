import rclpy
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener
import numpy as np
from sensor_msgs_py.point_cloud2 import read_points, create_cloud_xyz32
import tf_transformations


class SimulatedLidarPublisher(Node):
    def __init__(self):
        super().__init__('simulated_lidar_publisher')

        # Initialize subscribers for static obstacle data
        self.scan_sub = self.create_subscription(LaserScan, '/sem_obstacles_scan', self.scan_callback, 10)
        self.pointcloud_sub = self.create_subscription(PointCloud2, '/sem_obstacles_points', self.pointcloud_callback, 10)
        
        # Subscribe to /livox/lidar/pointclouds to get the timestamp
        self.livox_sub = self.create_subscription(PointCloud2, '/livox/lidar/pointcloud', self.livox_callback, 10)
        
        # Publishers for simulated single-line lidar and pointcloud from robot's perspective
        self.scan_use_pub = self.create_publisher(LaserScan, '/sem_obstacles_scan_use', 10)
        self.pointcloud_use_pub = self.create_publisher(PointCloud2, '/sem_obstacles_points_use', 10)

        # Initialize TF2 buffer and listener for robot's transformation
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Store static obstacles data
        self.static_scan_data = None
        self.static_pointcloud_data = None
        self.latest_timestamp = None  # Store the latest timestamp from /livox/lidar/pointclouds

        # Timer to publish simulated data at regular intervals
        self.timer = self.create_timer(0.1, self.publish_simulated_scan_and_pointcloud)

    def livox_callback(self, msg):
        # Store the latest timestamp from /livox/lidar/pointclouds
        self.latest_timestamp = msg.header.stamp

    def scan_callback(self, msg):
        # Store the static scan data received
        self.static_scan_data = msg

    def pointcloud_callback(self, msg):
        # Store the static pointcloud data received
        self.static_pointcloud_data = msg

    def transform_pointcloud_to_robot_frame(self, pointcloud):
        try:
            # Retrieve transformation from 'map' to 'base_link'
            transform = self.tf_buffer.lookup_transform('base_link', 'map', rclpy.time.Time())
            trans_matrix = self.get_transform_matrix(transform)

            # Transform points to robot frame
            transformed_points = []
            for p in read_points(pointcloud, field_names=('x', 'y', 'z'), skip_nans=True):
                point = np.array([p[0], p[1], p[2], 1.0])
                transformed_point = np.dot(trans_matrix, point)[:3]
                transformed_points.append(transformed_point)
            return transformed_points
        except Exception as e:
            self.get_logger().warning(f"Transform lookup failed: {str(e)}")
            return []

    def get_transform_matrix(self, transform):
        t = transform.transform.translation
        q = transform.transform.rotation
        translation = np.array([t.x, t.y, t.z])
        rotation = np.array([q.x, q.y, q.z, q.w])
        transform_matrix = tf_transformations.quaternion_matrix(rotation)
        transform_matrix[:3, 3] = translation
        return transform_matrix

    def filter_points_in_fov(self, points, max_distance=10.0):
        """Filter points based only on distance, removing FOV restrictions for 360 degrees."""
        filtered_points = []
        for p in points:
            distance = np.sqrt(p[0] ** 2 + p[1] ** 2)
            if distance <= max_distance:
                filtered_points.append(p)
        return filtered_points


    def publish_simulated_scan_and_pointcloud(self):
        if self.static_pointcloud_data is None or self.static_scan_data is None or self.latest_timestamp is None:
            return  # Wait until all required data is available

        # Transform static obstacles to robot's frame
        transformed_points = self.transform_pointcloud_to_robot_frame(self.static_pointcloud_data)
        if not transformed_points:
            return

        # Filter points based on range (no FOV restriction)
        filtered_points = self.filter_points_in_fov(transformed_points)

        # Use the latest timestamp from /livox/lidar/pointclouds
        header = self.static_pointcloud_data.header
        header.stamp = self.latest_timestamp  # Set consistent timestamp
        header.frame_id = 'base_link'

        # Publish the transformed pointcloud data
        pointcloud_use = create_cloud_xyz32(header, filtered_points)
        self.pointcloud_use_pub.publish(pointcloud_use)

        # Convert LaserScan data to points and transform to robot frame
        scan_points = []
        angle = self.static_scan_data.angle_min
        for r in self.static_scan_data.ranges:
            if self.static_scan_data.range_min <= r <= self.static_scan_data.range_max:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = 0.0
                scan_points.append([x, y, z, 1.0])  # Add 1.0 for matrix multiplication
            angle += self.static_scan_data.angle_increment

        # Transform LaserScan points to robot frame
        transformed_scan_points = []
        for point in scan_points:
            transformed_point = np.dot(self.get_transform_matrix(
                self.tf_buffer.lookup_transform('base_link', 'map', rclpy.time.Time())
            ), point)[:3]
            transformed_scan_points.append(transformed_point)

        # Filter the transformed points based only on range
        filtered_scan_points = self.filter_points_in_fov(transformed_scan_points)

        # Rebuild LaserScan message for 360 degrees
        scan_use = LaserScan()
        scan_use.header = self.static_scan_data.header
        scan_use.header.stamp = self.latest_timestamp  # Use the consistent timestamp
        scan_use.header.frame_id = 'base_link'
        scan_use.angle_min = -np.pi
        scan_use.angle_max = np.pi
        scan_use.angle_increment = np.pi / 180  # 1 degree per increment
        scan_use.range_min = 0.2
        scan_use.range_max = 10.0
        num_readings = int((scan_use.angle_max - scan_use.angle_min) / scan_use.angle_increment)
        scan_use.ranges = [scan_use.range_max] * num_readings  # Initialize ranges to max

        for point in filtered_scan_points:
            angle = np.arctan2(point[1], point[0])
            distance = np.sqrt(point[0] ** 2 + point[1] ** 2)
            if scan_use.angle_min <= angle <= scan_use.angle_max:
                index = int((angle - scan_use.angle_min) / scan_use.angle_increment)
                if distance < scan_use.ranges[index]:  # Update if closer distance
                    scan_use.ranges[index] = distance

        self.scan_use_pub.publish(scan_use)
        self.get_logger().info("Published 360-degree LaserScan and PointCloud2 with synchronized timestamp.")



def main(args=None):
    rclpy.init(args=args)
    node = SimulatedLidarPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

