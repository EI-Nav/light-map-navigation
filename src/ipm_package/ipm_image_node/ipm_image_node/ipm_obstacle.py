import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import PolygonStamped, Point32
from std_msgs.msg import Header
import json
import numpy as np
import tf2_ros as tf2
import time
from builtin_interfaces.msg import Time
from ipm_library.ipm import IPM
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from sensor_msgs.msg import CameraInfo, PointCloud2, LaserScan
from shape_msgs.msg import Plane

class ImageSegmentationClient(Node):
    def __init__(self):
        super().__init__('image_segmentation_client')
        self.client = self.create_client(Trigger, 'segment_image')
        self.polygon_pub = self.create_publisher(PolygonStamped, 'navigable_region', 10)
        self.world_coordinates = []
        self.segmentation_received = False

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for segmentation service...')

        self.ipm_projection = IPMContourProjection()

        # 定时器设置，每隔0.1秒调用一次
        self.timer = self.create_timer(0.1, self.send_request)
        
    def send_request(self):
        request = Trigger.Request()
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f'Service response: {future.result().message}')
            try:
                response_message = future.result().message
                if 'World coordinates: ' in response_message:
                    json_data_str = response_message.split('World coordinates: ')[-1]
                    self.world_coordinates = json.loads(json_data_str)
                    self.segmentation_received = True
                    self.get_logger().info(f"Parsed world coordinates: {self.world_coordinates}")
                    self.ipm_projection.load_contours_from_data(self.world_coordinates)
                else:
                    self.get_logger().error("No 'World coordinates: ' found in the response message.")
            except json.JSONDecodeError as e:
                self.get_logger().error(f"Failed to decode JSON from response: {str(e)}")
        else:
            self.get_logger().error(f"Service call failed: {future.exception()}")


class IPMContourProjection(Node):
    def __init__(self):
        super().__init__('ipm_contour_projection')
        self.scan_pub = self.create_publisher(LaserScan, '/sem_obstacles_scan', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/sem_obstacles_points', 10)
        self.tf_buffer = tf2.Buffer()
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self, spin_thread=True)

        self.camera_info = CameraInfo(
            header=Header(frame_id='d435_color_optical_frame'),
            width=640,
            height=480,
            k=[462.1379699707031, 0.0, 320.0, 0.0, 462.1379699707031, 240.0, 0.0, 0.0, 1.0],
            d=[0.0, 0.0, 0.0, 0.0, 0.0]  # Assuming no distortion
        )

        self.ipm = IPM(self.tf_buffer, self.camera_info, distortion=True)
        self.plane = Plane()
        self.plane.coef[2] = 1.0

        self.scan_angle_min = -np.pi / 2
        self.scan_angle_max = np.pi / 2
        self.scan_angle_increment = np.pi / 180
        self.scan_range_min = 0.2
        self.scan_range_max = 10.0
        self.noise_std_dev = 0.01
        self.initial_transformed_points = None

    def load_contours_from_data(self, contours_data):
        all_contour_points = []
        for obj in contours_data:
            if 'contour' in obj:
                contours = obj['contour']
                for contour in contours:
                    contour_points = np.array(contour).reshape(-1, 2)
                    all_contour_points.append(contour_points)

        if all_contour_points:
            all_contour_points = np.vstack(all_contour_points)
            measurement_time = Time()
            header, mapped_points = self.ipm.map_points(
                self.plane,
                all_contour_points,
                measurement_time,
                plane_frame_id='map',
                output_frame_id='map'
            )
            self.initial_transformed_points = (header, np.array(mapped_points))

    def add_noise(self, points):
        noise = np.random.normal(0, self.noise_std_dev, points.shape)
        return points + noise

    def convert_pointcloud_to_laserscan(self, points, header):
        num_scan_points = int((self.scan_angle_max - self.scan_angle_min) / self.scan_angle_increment)
        ranges = [self.scan_range_max] * num_scan_points

        for point in points:
            x, y, z = point
            distance = np.sqrt(x ** 2 + y ** 2)
            angle = np.arctan2(y, x)

            if self.scan_angle_min <= angle <= self.scan_angle_max and self.scan_range_min <= distance <= self.scan_range_max:
                index = int((angle - self.scan_angle_min) / self.scan_angle_increment)
                if distance < ranges[index]:
                    ranges[index] = distance

        scan = LaserScan()
        scan.header = header
        scan.angle_min = self.scan_angle_min
        scan.angle_max = self.scan_angle_max
        scan.angle_increment = self.scan_angle_increment
        scan.range_min = self.scan_range_min
        scan.range_max = self.scan_range_max
        scan.ranges = ranges

        return scan

    def publish_scan(self):
        if self.initial_transformed_points:
            header, points = self.initial_transformed_points
            noisy_points = self.add_noise(points)
            scan = self.convert_pointcloud_to_laserscan(noisy_points, header)
            self.scan_pub.publish(scan)
            self.get_logger().info("Published LaserScan.")

    def publish_pointcloud(self):
        if self.initial_transformed_points:
            header, points = self.initial_transformed_points
            noisy_points = self.add_noise(points)
            point_cloud = create_cloud_xyz32(header, noisy_points)
            self.pointcloud_pub.publish(point_cloud)
            self.get_logger().info("Published PointCloud2.")

def main(args=None):
    rclpy.init(args=args)
    client = ImageSegmentationClient()
    rclpy.spin(client)
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
