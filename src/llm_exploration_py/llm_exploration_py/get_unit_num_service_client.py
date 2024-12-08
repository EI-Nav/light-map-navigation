# 文件路径：src/my_service_pkg/my_service_pkg/check_unit_number_client.py

import rclpy
from rclpy.node import Node
from custom_interfaces.srv import GetUnitNum

class CheckUnitNumberClient(Node):

    def __init__(self):
        super().__init__('check_unit_number_client')
        self.cli = self.create_client(GetUnitNum, 'check_unit_number')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = GetUnitNum.Request()

    def send_request(self, desire_num):
        self.req.desire_num = desire_num
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    client = CheckUnitNumberClient()
    
    # 这里指定目标单元楼号进行测试
    desire_num = '2'
    response = client.send_request(desire_num)
    
    client.get_logger().info(f'Result: {response.result}')
    
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
