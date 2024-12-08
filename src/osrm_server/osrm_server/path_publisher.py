import rclpy
from rclpy.node import Node
from custom_interfaces.msg import Path  # Replace 'your_package_name' with the name of your package

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        self.publisher_ = self.create_publisher(Path, 'path_topic', 10)
        timer_period = 2.0  # seconds
        self.timer = self.create_timer(timer_period, self.publish_path)

    def publish_path(self):
        msg = Path()
        msg.start = '10.511509522,-0.000094233'
        msg.end = '10.511010327,0.000235218'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published Path: start={msg.start}, end={msg.end}')

def main(args=None):
    rclpy.init(args=args)
    path_publisher = PathPublisher()
    rclpy.spin(path_publisher)

    path_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()