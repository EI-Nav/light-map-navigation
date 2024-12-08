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
    def __init__(self, json_file):
        super().__init__('ipm_contour_projection')

        # Load contour data from the JSON file
        with open(json_file, 'r') as f:
            self.contours_data = json.load(f)

        # Set up camera intrinsics for IPM projection
        self.camera_info = CameraInfo(
            header=Header(frame_id='d435_color_optical_frame'),
            width=640,
            height=480,
            k=[462.1379699707031, 0., 320.0, 0., 462.1379699707031, 240.0, 0., 0., 1.],
            d=[0., 0., 0., 0., 0.]  # Assuming no distortion
        )

        # Publisher for point cloud
        self.point_cloud_pub = self.create_publisher(PointCloud2, 'ipm_contour_points', 10)

        # TF2 buffer and listener for coordinate transformation
        self.tf_buffer = tf2.Buffer()
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self, spin_thread=True)

        # Initialize IPM object for projection
        self.ipm = IPM(self.tf_buffer, self.camera_info, distortion=True)

        # Define projection plane
        self.plane = Plane()
        self.plane.coef[2] = 1.0  # Z-direction normal vector for plane

        # Placeholder for transformed fixed points
        self.initial_transformed_points = None

    def project_contours_to_plane_once(self):
        all_contour_points = []

        # Extract contours from JSON data
        for obj in self.contours_data:
            if 'contour' in obj:
                contours = obj['contour']
                for contour in contours:
                    contour_points = np.array(contour).reshape(-1, 2)  # Keep x, y pixel coordinates
                    all_contour_points.append(contour_points)

        # Combine all contour points into one array
        if all_contour_points:
            all_contour_points = np.vstack(all_contour_points)
        else:
            self.get_logger().info("No contours found in the provided JSON file.")
            return False

        measurement_time = Time()

        # Project points to 3D plane using IPM
        header, mapped_points = self.ipm.map_points(
            self.plane,
            all_contour_points,
            measurement_time,
            plane_frame_id='map',
            output_frame_id='map'
        )

        # Store the header and points to use for repeated publishing
        self.initial_transformed_points = (header, np.array(mapped_points))
        return True

    def publish_fixed_point_cloud(self):
        # Ensure transformed points are available before publishing
        if not self.initial_transformed_points:
            return

        header, points = self.initial_transformed_points
        point_cloud = create_cloud_xyz32(header, points)
        self.point_cloud_pub.publish(point_cloud)
        self.get_logger().info("Published fixed PointCloud2 based on initial contours.")

    def main(self):
        if not self.project_contours_to_plane_once():
            return

        # Publish the same point cloud at intervals
        while rclpy.ok():
            self.publish_fixed_point_cloud()
            time.sleep(1)  # Set publishing interval

def main():
    # Path to JSON file containing contour data
    json_file_path = "/workspaces/light-map-navigation1/src/ipm/mask.json"

    rclpy.init()
    node = IPMContourProjection(json_file_path)
    node.main()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
