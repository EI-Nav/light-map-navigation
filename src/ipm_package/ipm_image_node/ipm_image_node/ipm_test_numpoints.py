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


class IPMExample(Node):
    def __init__(self):
        # Let's initialize our node
        super().__init__('ipm_example')

        # We will need to provide the camera's intrinsic parameters to perform the projection
        # In a real scenario, this would be provided by the camera driver on a topic
        # If you don't know the intrinsic parameters of your camera,
        # you can use the camera_calibration ROS package to calibrate your camera
        self.camera_info = CameraInfo(
            header=Header(
                # This defines where the camera is located on the robot
                frame_id='camera_optical_frame',
            ),
            width=2048,
            height=1536,
            k=[1338.64532, 0., 1026.12387, 0., 1337.89746, 748.42213, 0., 0., 1.],
            d=[0., 0., 0., 0., 0.] # The distortion coefficients are optional
        )

        self.point_cloud_pub = self.create_publisher(PointCloud2, 'ipm_points', 10)

        self.tf_buffer = tf2.Buffer()
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self, spin_thread=True)

        self.ipm = IPM(self.tf_buffer, self.camera_info, distortion=True)

        self.plane = Plane()
        self.plane.coef[2] = 1.0  # Normal in z-direction

    def main(self):
        while rclpy.ok():
            points = np.meshgrid(np.arange(0, self.camera_info.width, 10), np.arange(0, self.camera_info.height, 10))
            points = np.stack(points, axis=-1).reshape(-1, 2)

            measurement_time = Time()

            # We will now project the pixel onto the plane using our library
            header, mapped_points = self.ipm.map_points(
                self.plane,
                points,
                measurement_time,
                plane_frame_id='map', # We defined a transform from the map to the camera earlier
                output_frame_id='map' # We want the output to be in the same frame as the plane
            )

            point_cloud = create_cloud_xyz32(header, mapped_points)

            self.point_cloud_pub.publish(point_cloud)

            time.sleep(0.1)

def main():
    rclpy.init()
    ipm_example = IPMExample()
    ipm_example.main()
    rclpy.shutdown()


if __name__ == '__main__':
    rclpy.init()
    ipm_example = IPMExample()
    ipm_example.main()
    rclpy.shutdown()