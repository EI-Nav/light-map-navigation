import rclpy
import tf2_ros as tf2
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PointStamped
from ipm_library.exceptions import NoIntersectionError
from ipm_library.ipm import IPM
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from shape_msgs.msg import Plane
from std_msgs.msg import Header
from vision_msgs.msg import Point2D


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

        self.point_pub = self.create_publisher(PointStamped, 'ipm_point', 10)

        self.tf_buffer = tf2.Buffer()
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self, spin_thread=True)

        self.ipm = IPM(self.tf_buffer, self.camera_info, distortion=True)

        self.plane = Plane()
        self.plane.coef[2] = 1.0  # Normal in z direction

    def main(self):
        while rclpy.ok():
            # We will ask the user for a pixel to project
            point = Point2D(
                x = float(input('Enter pixel x: ')),
                y = float(input('Enter pixel y: '))
            )
            time = Time()

            try:
                point = self.ipm.map_point(
                    self.plane,
                    point,
                    time,
                    plane_frame_id='map', # We defined a transform from the map to the camera earlier
                    output_frame_id='map' # We want the output to be in the same frame as the plane
                )

                # Print the result
                print(f'Projected point: {point.point.x}, {point.point.y}, {point.point.z}')

                # Now we will publish the projected point on a topic so we can visualize it in RViz
                self.point_pub.publish(point)
            except NoIntersectionError:
                print('No intersection found')


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