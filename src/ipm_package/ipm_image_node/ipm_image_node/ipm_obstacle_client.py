import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class SegmentAndPublishClient(Node):
    def __init__(self):
        super().__init__('segment_and_publish_client')
        self.client = self.create_client(Trigger, '/segment_and_publish')
        
        # 等待服务可用
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /segment_and_publish service...')
        
        self.send_request()

    def send_request(self):
        # 创建服务请求
        request = Trigger.Request()
        
        # 发送请求并等待响应
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        # 处理响应
        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'Success: {response.success}, Message: "{response.message}"')
        else:
            self.get_logger().error('Failed to call service /segment_and_publish')

def main(args=None):
    rclpy.init(args=args)
    client = SegmentAndPublishClient()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
