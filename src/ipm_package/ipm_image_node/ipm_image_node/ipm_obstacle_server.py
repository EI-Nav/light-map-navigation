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

camera_class = 1 # 选择相机种类，0代表RGB单目，1是深度相机

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
        
        goal_msg = Fibonacci.Goal()  
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

        if camera_class == 0:
            self.camera_info = CameraInfo(
                header=Header(frame_id='base_link'),
                width=640,
                height=480,
                k=[381.36246688113556, 0.0, 320.5, 0.0, 381.36246688113556, 240.5, 0.0, 0.0, 1.0],
                d=[0.0, 0.0, 0.0, 0.0, 0.0]
            )
            print("use No.1 camera")

        elif camera_class == 1:
            self.camera_info = CameraInfo(
                header=Header(frame_id='d435_color_optical_frame'),
                width=640,
                height=480,
                k=[462.1379699707031, 0.0, 320.0, 0.0, 462.1379699707031, 240.0, 0.0, 0.0, 1.0],
                d=[0.0, 0.0, 0.0, 0.0, 0.0]
            )
            print("use No.2 camera")


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
        """Load contours, filter by robot's vicinity, and store for publishing."""
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

            # 获取机器人当前位置
            try:
                transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                robot_position = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y
                ])
                self.get_logger().info(f"Robot position for filtering: {robot_position}")
            except Exception as e:
                self.get_logger().error(f"Failed to get robot position: {e}")
                return

            print(mapped_points)
            # 过滤 7 米以内的点云
            filtered_points = []
            for point in mapped_points:
                distance = np.linalg.norm(point[:2] - robot_position)
                if distance <= 7.0:
                    filtered_points.append(point)
                else:
                    self.get_logger().debug(f"Filtered out point: {point}, distance: {distance}")

            # 如果过滤后没有点，记录警告并退出
            if not filtered_points:
                self.get_logger().warn("No valid points within 7 meters. Skipping this data.")
                self.previous_transformed_points = self.initial_transformed_points
                self.initial_transformed_points = None
                return

            # 更新点云数据，保存上一次点云
            self.previous_transformed_points = self.initial_transformed_points
            self.initial_transformed_points = (header, np.array(filtered_points))
            self.get_logger().info(f"Filtered and loaded {len(filtered_points)} points.")

    
    def merge_points(self):
        """Merge the current and previous transformed points."""
        if not self.previous_transformed_points and not self.initial_transformed_points:
            self.get_logger().warn("No data available for merging.")
            return None

        if not self.previous_transformed_points:
            return self.initial_transformed_points

        if not self.initial_transformed_points:
            return self.previous_transformed_points

        # 合并当前点云和上一次点云
        header = self.initial_transformed_points[0]  # 使用当前点云的 Header
        current_points = self.initial_transformed_points[1]
        previous_points = self.previous_transformed_points[1]
        merged_points = np.vstack((current_points, previous_points))  # 合并点云
        self.get_logger().info(f"Merged points: current={len(current_points)}, previous={len(previous_points)}, total={len(merged_points)}.")
        return header, merged_points



    def add_noise(self, points):
        noise = np.random.normal(0, self.noise_std_dev, points.shape)
        return points + noise


    def convert_pointcloud_to_laserscan(self, points, header):
        """Convert point cloud to LaserScan message."""
        num_scan_points = int((self.scan_angle_max - self.scan_angle_min) / self.scan_angle_increment)
        ranges = [self.scan_range_max] * num_scan_points  # 初始化为最大范围值

        for point in points:
            x, y, z = point
            distance = np.sqrt(x**2 + y**2)  # 计算平面距离
            angle = np.arctan2(y, x)  # 计算点的角度

            # 检查点是否在激光扫描的范围内
            if self.scan_angle_min <= angle <= self.scan_angle_max and self.scan_range_min <= distance <= self.scan_range_max:
                index = int((angle - self.scan_angle_min) / self.scan_angle_increment)
                # 更新范围，保留更近的点
                if distance < ranges[index]:
                    ranges[index] = distance

        # 构造 LaserScan 消息
        scan = LaserScan()
        scan.header = header
        scan.angle_min = self.scan_angle_min
        scan.angle_max = self.scan_angle_max
        scan.angle_increment = self.scan_angle_increment
        scan.range_min = self.scan_range_min
        scan.range_max = self.scan_range_max
        scan.ranges = ranges

        return scan
    
    def filter_points_by_robot(self, points):
        """Filter points to only include those within 7 meters of the robot."""
        # 获取机器人当前位置
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            robot_position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y
            ])
            self.get_logger().info(f"Robot position: {robot_position}")
        except Exception as e:
            self.get_logger().error(f"Failed to get robot position: {e}")
            return np.array([])

        # 计算点云中每个点相对于机器人的距离，并过滤
        filtered_points = []
        for point in points:
            distance = np.linalg.norm(point[:2] - robot_position)
            if distance <= 7.0:
                filtered_points.append(point)
            else:
                self.get_logger().debug(f"Filtered out point: {point}, distance: {distance}")

        return np.array(filtered_points)

    def publish_scan(self):
        """Publish LaserScan message."""
        merged_data = self.merge_points()
        if not merged_data:
            self.get_logger().warn("No data available for LaserScan publication.")
            return

        header, points = merged_data

        # 获取机器人当前位置
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            robot_position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y
            ])
        except Exception as e:
            self.get_logger().error(f"Failed to get robot position: {e}")
            return

        # 过滤 7 米以内的点云
        filtered_points = []
        for point in points:
            distance = np.linalg.norm(point[:2] - robot_position)
            if distance <= 7.0:
                filtered_points.append(point)

        if not filtered_points:
            self.get_logger().warn("No points within 7 meters for LaserScan publication.")
            return
        
        # 将 filtered_points 转换为 NumPy 数组
        filtered_points = np.array(filtered_points)

        if filtered_points.size == 0:
            self.get_logger().warn("No points within 7 meters for LaserScan publication.")
            return

        noisy_points = self.add_noise(filtered_points)

        scan = self.convert_pointcloud_to_laserscan(noisy_points, header)
        self.scan_pub.publish(scan)
        self.get_logger().info(f"Published merged LaserScan with {len(noisy_points)} points.")

    def publish_pointcloud(self):
        """Publish PointCloud2 message."""
        merged_data = self.merge_points()
        if not merged_data:
            self.get_logger().warn("No data available for PointCloud2 publication.")
            return

        header, points = merged_data

        # 获取机器人当前位置
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            robot_position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y
            ])
        except Exception as e:
            self.get_logger().error(f"Failed to get robot position: {e}")
            return

        # 过滤 7 米以内的点云
        filtered_points = []
        for point in points:
            distance = np.linalg.norm(point[:2] - robot_position)
            if distance <= 7.0:
                filtered_points.append(point)

        if not filtered_points:
            self.get_logger().warn("No points within 7 meters for PointCloud2 publication.")
            return
        
        # 将 filtered_points 转换为 NumPy 数组
        filtered_points = np.array(filtered_points)

        if filtered_points.size == 0:
            self.get_logger().warn("No points within 7 meters for LaserScan publication.")
            return

        noisy_points = self.add_noise(filtered_points)

        point_cloud = create_cloud_xyz32(header, noisy_points)
        self.pointcloud_pub.publish(point_cloud)
        self.get_logger().info(f"Published merged PointCloud2 with {len(noisy_points)} points.")


def main(args=None):
    rclpy.init(args=args)
    service_node = ImageSegmentationService()
    rclpy.spin(service_node)
    service_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()