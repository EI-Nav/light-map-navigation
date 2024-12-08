import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
from custom_interfaces.srv import TaskRecord
import csv
import os
import time
import math
import argparse


class OdomLogger(Node):
    """A ROS2 node that logs odometry data to a CSV file based on service requests."""

    def __init__(self, filename):
        """Initialize the OdomLogger node."""
        super().__init__('odom_logger')
        self.subscription = None
        self.is_logging = False
        self.current_status = None
        self.current_address = None
        self.status_flag = None
        self.log_frequency = 1.0  # Default logging frequency in Hz
        self.last_log_time = time.time()
        self.filename = filename

        # Distance threshold for logging (in meters)
        self.distance_threshold = 0.5
        self.last_position = None

        # Create custom service
        self.task_record_service = self.create_service(
            TaskRecord, 'task_record', self.task_record_callback
        )

        self.file = None
        self.csv_writer = None

    def task_record_callback(self, request, response):
        """Handle requests to start or stop logging odometry data."""
        if request.status == 'start':
            if not self.is_logging:
                self.start_logging(request.address)
                response.success = True
            else:
                self.get_logger().info('Logging is already started.')
                response.success = False
        elif request.status == 'end':
            if self.is_logging:
                self.stop_logging()
                response.success = True
            else:
                self.get_logger().info('Logging is already stopped.')
                response.success = False
        else:
            self.get_logger().warn('Unknown status received. Use "start" or "end".')
            response.success = False

        return response

    def start_logging(self, address):
        """Start logging odometry data."""
        self.current_status = 'start'
        self.current_address = address
        self.status_flag = True

        # Open file in append mode and prepare to write
        file_exists = os.path.isfile(self.filename)
        self.file = open(self.filename, 'a', newline='')

        self.csv_writer = csv.writer(self.file)
        # Write the header only if the file is new
        if not file_exists:
            self.csv_writer.writerow(['timestamp', 'status', 'address', 'robot_x', 'robot_y', 'robot_z'])

        # Subscribe to odometry topic
        qos_profile = QoSProfile(depth=10)
        self.subscription = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile
        )

        self.is_logging = True
        self.get_logger().info(f'Started logging odometry data to {self.filename}.')

    def stop_logging(self):
        """Stop logging odometry data."""
        self.current_status = 'end'
        if self.subscription:
            self.destroy_subscription(self.subscription)
        if self.file:
            self.file.close()

        self.is_logging = False
        self.get_logger().info('Stopped logging odometry data.')

    def odom_callback(self, msg):
        """Handle incoming odometry messages."""
        if self.is_logging and time.time() - self.last_log_time >= 1.0 / self.log_frequency:
            current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)

            if self.last_position is None:
                self.last_position = current_position

            # Calculate the distance moved since the last log
            distance_moved = math.sqrt(
                (current_position[0] - self.last_position[0]) ** 2 +
                (current_position[1] - self.last_position[1]) ** 2 +
                (current_position[2] - self.last_position[2]) ** 2
            )

            if distance_moved >= self.distance_threshold:
                if self.status_flag:
                    self.status_flag = False  # Update status after the first log
                else:
                    self.current_status = 'running'

                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                # Write data to the CSV file
                self.csv_writer.writerow([timestamp, self.current_status, self.current_address,
                                          current_position[0], current_position[1], current_position[2]])
                self.get_logger().info(f'Logged odometry data at {timestamp}: x={current_position[0]}, '
                                       f'y={current_position[1]}, z={current_position[2]}')
                self.last_log_time = time.time()
                self.last_position = current_position

    def destroy_node(self):
        """Ensure resources are properly cleaned up when node is destroyed."""
        if self.is_logging and self.file:
            self.file.close()
        super().destroy_node()


def main(args=None):
    """Main entry point for the node."""
    rclpy.init(args=args)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='OdomLogger Node')
    parser.add_argument('--filename', type=str, default='trajectory.csv',
                        help='The filename to log odometry data')
    args = parser.parse_args()

    odom_logger = OdomLogger(filename=args.filename)

    try:
        rclpy.spin(odom_logger)
    except KeyboardInterrupt:
        pass
    finally:
        odom_logger.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()