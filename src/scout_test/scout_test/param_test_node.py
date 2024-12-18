import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

class ParameterTestNode(Node):
    def __init__(self):
        super().__init__('parameter_test_node')

        # 声明参数
        self.declare_parameter('position', None)
        self.declare_parameter('rotation', None)
        self.declare_parameter('rgb', None)
        self.declare_parameter('depth', None)

        # 启动后立即获取并打印参数值
        self.get_logger().info("Getting parameters...")

        # 获取参数值
        self.position = self.get_parameter('position').value
        self.rotation = self.get_parameter('rotation').value
        self.rgb = self.get_parameter('rgb').value
        self.depth = self.get_parameter('depth').value

        # 打印获取到的参数
        self.print_parameters()

    def print_parameters(self):
        """打印获取到的参数信息"""
        self.get_logger().info(f"Position: {self.position}")
        self.get_logger().info(f"Rotation: {self.rotation}")
        self.get_logger().info(f"RGB Image: {self.rgb}")
        self.get_logger().info(f"Depth Image: {self.depth}")


def main(args=None):
    rclpy.init(args=args)
    test_node = ParameterTestNode()
    rclpy.spin(test_node)
    test_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()