# Copyright (c) 2022 Hamburg Bit-Bots
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from ipm_image_node.ipm import IPMImageNode
import numpy as np
from numpy.lib import recfunctions as rfn
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py.point_cloud2 import read_points, read_points_numpy
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage

cv_bridge = CvBridge()

# Dummy CameraInfo Message
camera_info = CameraInfo(
        header=Header(
            frame_id='camera_optical_frame',
        ),
        width=2048,
        height=1536,
        binning_x=4,
        binning_y=4,
        k=[1338.64532, 0., 1024.0, 0., 1337.89746, 768.0, 0., 0., 1.])

img_center_x = camera_info.width / camera_info.binning_x // 2
img_center_y = camera_info.height / camera_info.binning_y // 2


def standard_ipm_image_test_case(
        input_topic: str,
        input_msg: Image,
        output_topic: str,
        mode: str = 'mask') -> Tuple[PointCloud2, Image]:
    # Init ros
    rclpy.init()
    # Create IPM node
    node = IPMImageNode()
    # Create test node which comunicates with the IPM node
    test_node = Node('test_handler')
    # Create publishers to send data to the IPM node
    ball_pub = test_node.create_publisher(
        Image, input_topic, 10)
    camera_info_pub = test_node.create_publisher(
        CameraInfo, 'camera_info', 10)
    tf_pub = test_node.create_publisher(
        TFMessage, 'tf', 10)

    # Create a shared reference to the recived message in the local scope
    received_msg: List[Optional[PointCloud2]] = [None]

    # Create a callback with sets this reference
    def callback(msg):
        received_msg[0] = msg

    # Subscribe to IPM results
    test_node.create_subscription(
        PointCloud2, output_topic, callback, 10)

    # Create header message for the current time stamp in the camera frame
    header = Header(
        stamp=node.get_clock().now().to_msg(),
        frame_id='camera_optical_frame')

    # Create a dummy transform from the camera to the base_footprint frame
    tf = TransformStamped(
        header=header,
        child_frame_id='base_footprint',
    )
    tf.transform.translation.z = 1.0
    tf.transform.rotation.x = 0.0
    tf.transform.rotation.w = 1.0

    # Publish the dummy transform
    tf_pub.publish(TFMessage(
        transforms=[tf]
    ))
    # Spin the ipm to process the new data
    rclpy.spin_once(node, timeout_sec=0.1)

    node.set_parameters([Parameter('type', value=mode)])

    # Send camera info message to the IPM
    camera_info.header.stamp = header.stamp
    camera_info_pub.publish(camera_info)
    # Spin the IPM to process the new data
    rclpy.spin_once(node, timeout_sec=0.1)

    # Send image space detection
    input_msg.header = header
    ball_pub.publish(input_msg)
    # Spin the IPM to process the new data
    rclpy.spin_once(node, timeout_sec=0.1)

    # Spin the test__node to recive the results from the IPM
    rclpy.spin_once(test_node, timeout_sec=0.1)

    # Assert that we recived a message
    assert received_msg[0] is not None

    # Clean shutdown of the nodes
    rclpy.shutdown()
    node.destroy_node()
    test_node.destroy_node()

    return received_msg[0], input_msg


def test_ipm_mask():
    # Create image detection
    image_np = np.zeros(
        (
            camera_info.height // camera_info.binning_y,
            camera_info.width // camera_info.binning_x,
            1
        ),
        dtype=np.uint8)
    image_np[int(img_center_y), int(img_center_x)] = 1
    image_np[int(img_center_y) + 1, int(img_center_x) + 1] = 255
    image = cv_bridge.cv2_to_imgmsg(image_np, '8UC1')

    out, inp = standard_ipm_image_test_case(
        'input',
        image,
        'projected_point_cloud')

    # Convert point cloud to numpy
    output_np = read_points_numpy(out)

    # Assert that we recived the correct message
    assert len(output_np) == 2, 'Wrong number of pixels'
    assert out.header.stamp == inp.header.stamp, 'Time stamp got changed by the ipm'
    assert out.header.frame_id == 'base_footprint', \
        'Output frame is not "base_footprint"'
    np.testing.assert_allclose(
        output_np[0],
        [0.0, 0.0, 0.0])


def test_ipm_image():
    # Create image detection
    image_np = np.empty(
        (
            camera_info.height // camera_info.binning_y,
            camera_info.width // camera_info.binning_x,
            3
        ),
        dtype=np.uint8)
    image_np[..., 0] = 0
    image_np[..., 1] = 10
    image_np[..., 2] = 100
    image_np[int(img_center_y), int(img_center_x), :] = 255
    image = cv_bridge.cv2_to_imgmsg(image_np, '8UC3')

    # Run ipm
    out, inp = standard_ipm_image_test_case(
        'input',
        image,
        'projected_point_cloud',
        mode='rgb_image')

    # Convert point cloud to numpy
    output_np = read_points(out)

    output_np = output_np.reshape(
        camera_info.height // camera_info.binning_y,
        camera_info.width // camera_info.binning_x)

    # Get center pixel
    center_output = output_np[int(img_center_y), int(img_center_x)]

    # Assert that we recived the correct message
    assert image_np.shape[0] * image_np.shape[1] == image_np.shape[0] * image_np.shape[1], \
        'Wrong number of pixels'
    assert out.header.stamp == inp.header.stamp, 'Time stamp got changed by the ipm'
    assert out.header.frame_id == 'base_footprint', \
        'Output frame is not "base_footprint"'
    # Check if the center pixel is projected to (0, 0, 0)
    np.testing.assert_allclose(
        rfn.structured_to_unstructured(center_output[['x', 'y', 'z']]),
        [0.0, 0.0, 0.0])
    # Check if all values are 255 as set earlier, aka if all bits are one
    assert center_output['rgb'] == 2**32 - 1, 'RGB values of the center point changed'
    # Calculate pointcloud binary representation of the default image rgb value
    rgb_default_value = \
        255 << 24 | image_np[0, 0, 2] << 16 | image_np[0, 0, 1] << 8 | image_np[0, 0, 0]
    # Check if the default value exists in all pixels except the center pixel
    number_of_default_color_points = np.sum(output_np['rgb'] == rgb_default_value)
    number_of_default_color_pixels = image_np.shape[0] * image_np.shape[1] - 1
    assert number_of_default_color_pixels == number_of_default_color_points
