import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import llm_exploration_py.Tongyi_VLM as VLM
from custom_interfaces.srv import GetUnitNum

class CheckUnitNumberService(Node):

    def __init__(self):
        super().__init__('check_unit_number_service')
        self.srv = self.create_service(GetUnitNum, 'check_unit_number', self.check_unit_number_callback)
        self.image_sub = self.create_subscription(Image, '/camera_sensor/image_raw', self.image_callback, 10)

        self.bridge = CvBridge()
        self.latest_image = None
        self.desire_unit_num = None
        self.is_task_success = False
        self.temp_image_path = '/tmp/temp_image.jpg'

    def check_unit_number_callback(self, request, response):
        self.desire_unit_num = request.desire_num
        self.get_logger().info(f'Received request for unit number: {self.desire_unit_num}')

        if self.latest_image is not None:
            cv2.imwrite(self.temp_image_path, self.latest_image)
            try:
                self.is_task_success = self.check_unit(self.temp_image_path, self.desire_unit_num)
                response.result = self.is_task_success
            except Exception as e:
                self.get_logger().error(f"Error checking unit number: {e}")
                response.result = False
        else:
            self.get_logger().info('No image received yet.')
            response.result = False

        return response

    def image_callback(self, msg):
        # self.get_logger().info("Image received.")
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except cv2.error as e:
            self.get_logger().error(f"Error converting image: {e}")
            self.latest_image = None

    def check_unit(self, image_path, desire_unit_num):
        try:
            return VLM.check_unit(image_path, desire_unit_num)
        except Exception as e:
            self.get_logger().error(f"Error using VLM to check unit number: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    node = CheckUnitNumberService()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()