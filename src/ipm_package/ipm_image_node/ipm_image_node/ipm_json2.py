import json
import numpy as np
import rclpy
import tf2_ros as tf2
import time
from builtin_interfaces.msg import Time
from ipm_library.ipm import IPM
from rclpy.node import Node
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from sensor_msgs.msg import CameraInfo, PointCloud2
from shape_msgs.msg import Plane
from std_msgs.msg import Header

class IPMContourProjection(Node):
    def __init__(self, json_file, max_distance=10.0):
        super().__init__('ipm_contour_projection')

        # Load pixel data from the JSON file
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        # Set maximum projection distance for filtering points
        self.max_distance = max_distance

        # Camera intrinsics setup for IPM projection
        self.camera_info = CameraInfo(
            header=Header(frame_id='d435_color_optical_frame'),
            width=640,
            height=480,
            k=[462.1379699707031, 0., 320.0, 0., 462.1379699707031, 240.0, 0., 0., 1.],
            d=[0., 0., 0., 0., 0.]
        )

        # Point cloud publisher
        self.point_cloud_pub = self.create_publisher(PointCloud2, 'ipm_contour_points', 10)

        # TF2 buffer and listener for coordinate transformation
        self.tf_buffer = tf2.Buffer()
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self, spin_thread=True)

        # Initialize IPM projection object
        self.ipm = IPM(self.tf_buffer, self.camera_info, distortion=True)

        # Define projection plane
        self.plane = Plane()
        self.plane.coef[2] = 1.0  # Plane normal vector in the Z direction

        # Placeholder to store transformed, fixed points
        self.initial_transformed_points = None

    def project_contours_to_plane_once(self):
        all_pixel_points = []

        # Extract non-background pixel points from JSON data
        for obj in self.data:
            if 'pixels' in obj:
                pixels = obj['pixels']
                for pixel in pixels:
                    all_pixel_points.append([pixel[1], pixel[0]])

        if not all_pixel_points:
            self.get_logger().info("No non-background pixels found in the provided JSON file.")
            return False

        # Convert list to numpy array
        all_pixel_points = np.array(all_pixel_points)
        measurement_time = Time()

        # Project pixel points to 3D using IPM
        header, mapped_points = self.ipm.map_points(
            self.plane,
            all_pixel_points,
            measurement_time,
            plane_frame_id='map',
            output_frame_id='map'
        )

        # Filter points based on max_distance
        filtered_points = [
            point for point in mapped_points
            if np.linalg.norm(point) <= self.max_distance
        ]

        if not filtered_points:
            self.get_logger().info("No points found within the distance threshold.")
            return False

        # Store transformed points and header for future publications
        self.initial_transformed_points = (header, np.array(filtered_points))
        return True

    def publish_fixed_point_cloud(self):
        if not self.initial_transformed_points:
            return

        header, points = self.initial_transformed_points
        # Publish fixed point cloud
        point_cloud = create_cloud_xyz32(header, points)
        self.point_cloud_pub.publish(point_cloud)
        self.get_logger().info(f"Published {len(points)} points within {self.max_distance} meters.")

    def main(self):
        # Project contours only once and check if it succeeded
        if not self.project_contours_to_plane_once():
            return

        # Continue publishing the same point cloud at intervals
        while rclpy.ok():
            self.publish_fixed_point_cloud()
            time.sleep(1)  # Set interval for publishing

def main():
    json_file_path = "/workspaces/light-map-navigation1/src/ipm/non_background_pixels.json"
    rclpy.init()
    node = IPMContourProjection(json_file_path, max_distance=7.0)
    node.main()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
