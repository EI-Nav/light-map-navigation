import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_srvs.srv import Trigger
from std_msgs.msg import Header
from example_interfaces.action import Fibonacci
import json
import numpy as np
import tf2_ros as tf2
from builtin_interfaces.msg import Time
from ipm_library.ipm import IPM
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from sensor_msgs.msg import CameraInfo, PointCloud2, LaserScan
from shape_msgs.msg import Plane

class ImageSegmentationService(Node):
    def __init__(self):
        super().__init__('image_segmentation_service')
        self.service = self.create_service(Trigger, 'segment_and_publish', self.handle_service_request)
        self.action_client = ActionClient(self, Fibonacci, 'segment_image')
        self.segmentation_received = False
        self.timer = None  # Timer for periodic publication

        self.ipm_projection = IPMContourProjection()
        self.get_logger().info("ImageSegmentationService node initialized and service ready.")

    def handle_service_request(self, request, response):
        self.get_logger().info("Received service request.")

        # Stop any existing timer before starting a new process
        if self.timer:
            self.get_logger().info("Stopping current publishing loop.")
            self.timer.cancel()

        # Ensure the action client is ready
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server not available.")
            response.success = False
            response.message = "Action server not available."
            return response

        self.get_logger().info("Sending goal to 'segment_image' action server...")
        goal_msg = Fibonacci.Goal()  # Goal does not need specific fields in your implementation
        self._send_goal_future = self.action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(lambda f: self.process_goal_response(f, response))

        return response

    def process_goal_response(self, future, response):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Segmentation goal rejected.")
            response.success = False
            response.message = "Segmentation goal rejected."
            return

        self.get_logger().info("Segmentation goal accepted, waiting for result...")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(lambda f: self.process_result(f, response))

    def process_result(self, future, response):
        try:
            result = future.result().result.sequence
            if result[0] == 1:
                json_data_str = ''.join(chr(x) for x in result[1:])
                world_coordinates = json.loads(json_data_str)
                self.segmentation_received = True
                self.ipm_projection.load_contours_from_data(world_coordinates)

                response.success = True
                response.message = "Received data and starting to publish."
                self.get_logger().info("Segmentation completed successfully.")
                self.start_publishing_loop()
            else:
                self.get_logger().error("Segmentation failed.")
                response.success = False
                response.message = "Segmentation failed."
        except Exception as e:
            self.get_logger().error(f"Error processing result: {str(e)}")
            response.success = False
            response.message = "Error processing segmentation result."

    def start_publishing_loop(self):
        self.get_logger().info("Starting publishing loop.")

        if self.timer is not None:
            self.timer.cancel()  # Cancel any existing timer before starting a new one

        def publish_timer_callback():
            if self.segmentation_received:
                self.ipm_projection.publish_scan()
                self.ipm_projection.publish_pointcloud()
            else:
                self.get_logger().warn("Segmentation data not available for publishing.")

        # Set up a timer to publish data at regular intervals (e.g., 0.5 seconds)
        self.timer = self.create_timer(0.5, publish_timer_callback)

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
            d=[0.0, 0.0, 0.0, 0.0, 0.0]
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
        self.get_logger().info("Loading contours from received data.")
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
            self.get_logger().info("Contours loaded and transformed.")

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
    service_node = ImageSegmentationService()
    rclpy.spin(service_node)
    service_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
