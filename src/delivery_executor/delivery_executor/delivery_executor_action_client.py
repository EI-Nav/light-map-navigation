import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from custom_interfaces.action import DeliveryTask
from rclpy.callback_groups import ReentrantCallbackGroup
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile
import signal
import sys
import csv
import os
import time
import threading
import argparse
from datetime import datetime

class DeliveryExecutorActionClient(Node):
    """Action client node for initiating and monitoring delivery tasks."""

    def __init__(self, log_filename=None):
        super().__init__('delivery_executor_action_client')
        self._is_finished = False
        self._goal_handle = None
        
        callback_group = ReentrantCallbackGroup()
        
        self._action_client = ActionClient(
            self,
            DeliveryTask,
            'execute_delivery',
            callback_group=callback_group
        )
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.get_logger().info('Delivery Executor Action Client started')
        self._send_goal_future = None
        self._get_result_future = None
        
        # Add logging-related attributes
        self.is_logging = False
        self.log_filename = log_filename or f'delivery_feedback_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self.file = None
        self.csv_writer = None
        self.lock = threading.Lock()  # For thread-safe file access
        self.task_start_time = None
        self.user_destination = None
        
        # Robot position tracking
        self.current_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.last_position = None
        self.distance_threshold = 0.5  # Distance threshold for logging (in meters)
        
        # Subscribe to odometry topic
        qos_profile = QoSProfile(depth=10)
        self.odom_subscription = self.create_subscription(
            Odometry, 
            '/odom', 
            self.odom_callback, 
            qos_profile
        )
        self.get_logger().info('Subscribed to /odom topic')
        
        # Current task status
        self.current_status = 'idle'
        self.last_logged_status = None
        self.has_valid_position = False
        
        # Terminal task status
        self.terminal_task_status = ""

    def odom_callback(self, msg):
        """Handle incoming odometry messages to track robot position."""
        try:
            # Update current position
            self.current_position = {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'z': msg.pose.pose.position.z
            }
            
            # Mark that we have a valid position
            if not self.has_valid_position and (
                self.current_position['x'] != 0.0 or 
                self.current_position['y'] != 0.0 or 
                self.current_position['z'] != 0.0
            ):
                self.has_valid_position = True
                
                # If we're already logging, log the initial position
                if self.is_logging and self.last_logged_status is None:
                    self.log_position()
            
            # Log position if we're logging and position has changed significantly
            if self.is_logging and self.has_valid_position:
                if self.last_position is None:
                    self.last_position = self.current_position.copy()
                else:
                    # Calculate distance moved
                    distance_moved = ((self.current_position['x'] - self.last_position['x']) ** 2 +
                                     (self.current_position['y'] - self.last_position['y']) ** 2 +
                                     (self.current_position['z'] - self.last_position['z']) ** 2) ** 0.5
                    
                    if distance_moved >= self.distance_threshold:
                        self.log_position()
                        self.last_position = self.current_position.copy()
                        
        except Exception as e:
            self.get_logger().error(f'Error in odom_callback: {str(e)}')

    def log_position(self):
        """Log the current robot position with task status."""
        if not self.is_logging or not self.has_valid_position:
            return
            
        try:
            with self.lock:
                if self.file and self.csv_writer:
                    # Only log if status has changed or this is a position update during 'running' state
                    if (self.current_status != self.last_logged_status or 
                        (self.current_status == 'running' and self.last_logged_status == 'running')):
                        
                        # Format timestamp to 6 decimal places and coordinates to 3 decimal places
                        timestamp = f"{time.time():.6f}"
                        x = f"{self.current_position['x']:.3f}"
                        y = f"{self.current_position['y']:.3f}"
                        z = f"{self.current_position['z']:.3f}"
                        
                        self.csv_writer.writerow([
                            timestamp,
                            self.current_status,
                            self.terminal_task_status,
                            self.user_destination,
                            x,
                            y,
                            z
                        ])
                        self.file.flush()  # Ensure data is written to file immediately
                        self.last_logged_status = self.current_status
        except Exception as e:
            self.get_logger().error(f'Error logging position: {str(e)}')

    def _signal_handler(self, signum, frame):
        self.get_logger().info('Received shutdown signal, canceling task...')
        try:
            if self._goal_handle is not None:
                cancel_future = self._goal_handle.cancel_goal_async()
                if not rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=1.0):
                    self.get_logger().warn('Cancel goal timed out, forcing exit...')
            
            self.stop_logging('interrupted')
            self.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f'Error during shutdown: {str(e)}')
        finally:
            sys.exit(1)

    def start_logging(self, user_input):
        """Start recording navigation feedback information"""
        try:
            with self.lock:
                self.task_start_time = time.time()
                self.user_destination = user_input
                self.current_status = 'start'
                self.last_logged_status = None
                
                file_exists = os.path.isfile(self.log_filename)
                self.file = open(self.log_filename, 'a', newline='')
                self.csv_writer = csv.writer(self.file)
                
                if not file_exists:
                    self.csv_writer.writerow(['timestamp', 'system_status', 'task_status', 'instruction', 'robot_x', 'robot_y', 'robot_z'])
                
                # Only log initial position if we have valid data
                if self.has_valid_position:
                    self.log_position()
                
            self.is_logging = True
            self.get_logger().info(f'Started logging trajectory data to {self.log_filename}')
        except Exception as e:
            self.get_logger().error(f'Error starting logging: {str(e)}')

    def update_status(self, status):
        """Update the current task status and log the position."""
        # Only update if status has changed
        if status != self.current_status:
            self.current_status = status
            if self.has_valid_position:
                self.log_position()

    def update_terminal_task_status(self, task_status):
        """Update the terminal task status."""
        if task_status and task_status != self.terminal_task_status:
            self.terminal_task_status = task_status
            # Log position if we have valid position data and are already logging
            if self.is_logging and self.has_valid_position:
                self.log_position()

    def stop_logging(self, final_status='completed'):
        """Stop recording feedback information"""
        if not self.is_logging:
            return
            
        try:
            # Record final position with end status if it's different from the last logged status
            if final_status != self.last_logged_status and self.has_valid_position:
                self.current_status = final_status
                self.log_position()
            
            with self.lock:
                if self.file:
                    self.file.close()
                    self.file = None
                    
            self.is_logging = False
            self.get_logger().info(f'Stopped logging trajectory data. Final status: {final_status}')
        except Exception as e:
            self.get_logger().error(f'Error stopping logging: {str(e)}')

    def cancel_goal(self):
        """Cancel current task with timeout mechanism"""
        if self._goal_handle is not None:
            self.get_logger().info('Canceling current task...')
            try:
                cancel_future = self._goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
                self.get_logger().info('Task canceled')
                self.stop_logging('canceled')
            except Exception as e:
                self.get_logger().error(f'Error canceling task: {str(e)}')
        else:
            self.get_logger().warn('No active task to cancel')

    def send_goal(self, user_input):
        """
        Send a delivery task goal to the action server.
        
        Args:
            user_input (str): The delivery task details provided by the user
            
        Returns:
            bool: False if input validation fails
        """
        self.get_logger().info('Waiting for Action Server...')
        
        while not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Action Server not available, waiting...')
        
        if not user_input or not isinstance(user_input, str):
            self.get_logger().error('Invalid user input')
            return False
        
        # Start logging
        self.start_logging(user_input)
        
        goal_msg = DeliveryTask.Goal()
        goal_msg.user_input = user_input
        
        self.get_logger().info(f'Sending delivery task goal: {user_input}')
        self.update_status('sending_goal')
        
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Handle the response to our goal request.
        
        Args:
            future: Future object containing the goal response
        """
        try:
            goal_handle = future.result()
            
            if not goal_handle.accepted:
                self.get_logger().info('Task rejected')
                self.update_status('rejected')
                self.stop_logging('rejected')
                self._is_finished = True
                return
            
            self.get_logger().info('Task accepted')
            self.update_status('accepted')
            self._goal_handle = goal_handle
            
            self._get_result_future = goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.get_result_callback)
            
        except Exception as e:
            error_msg = f'Error handling task response: {str(e)}'
            self.get_logger().error(error_msg)
            self.update_status('error')
            self.stop_logging('error')
            self._is_finished = True

    def get_result_callback(self, future):
        """
        Process the final result of the delivery task.
        
        Args:
            future: Future object containing the task result
        """
        status = future.result().status
        result = future.result().result
        result_message = result.result.message
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            status_str = 'succeeded'
            self.get_logger().info(f'Task completed successfully: {result_message}')
        elif status == GoalStatus.STATUS_CANCELED:
            status_str = 'canceled'
            self.get_logger().info(f'Task canceled: {result_message}')
        else:
            status_str = 'failed'
            self.get_logger().info(f'Task failed: {result_message}')
        
        # Record final result
        self.update_status(status_str)
        self.stop_logging(status_str)
        
        self.get_logger().info('Task processing complete, preparing to exit...')
        self._goal_handle = None
        self._is_finished = True
        
        self.destroy_node()
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        """
        Handle feedback updates from the action server.
        
        Args:
            feedback_msg: Contains the DeliveryFeedback message with progress information
        """
        status = feedback_msg.feedback.feedback.status
        self.get_logger().info(f'Received feedback:\n{status}')
        
        # Extract and update terminal task status if present in the feedback
        if status:
            # Look for task status patterns like "Starting task 2/2: EXPLORATION - building1:unit1"
            import re
            task_status_match = re.search(r'Starting task \d+/\d+: ([A-Z]+ - .+)', status)
            if task_status_match:
                task_status = task_status_match.group(1)
                self.update_terminal_task_status(task_status)
        
        # Update status to running and log position
        self.update_status('running')

    def destroy_node(self):
        """
        Clean up resources before node shutdown.
        """
        # Ensure log file is closed
        if self.is_logging:
            self.stop_logging('terminated')
            
        if self._action_client:
            self._action_client.destroy()
        super().destroy_node()

def main():
    """Main function to run the action client."""
    try:
        rclpy.init()
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Delivery Executor Action Client')
        parser.add_argument('--log-file', type=str, default=None,
                            help='The filename to log trajectory data (default: delivery_feedback_TIMESTAMP.csv)')
        parser.add_argument('--instruction', type=str, default=None,
                            help='The delivery instruction')
        parser.add_argument('--distance-threshold', type=float, default=0.5,
                            help='Minimum distance (meters) the robot must move to log a new position')
        
        args = parser.parse_args(rclpy.utilities.remove_ros_args(args=sys.argv)[1:])
        
        action_client = DeliveryExecutorActionClient(log_filename=args.log_file)
        
        if args.distance_threshold:
            action_client.distance_threshold = args.distance_threshold
        
        if args.instruction:
            user_input = args.instruction
        elif len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
            user_input = sys.argv[1]
        else:
            print("Please enter delivery instruction (e.g., 'Please deliver this apple to unit 1 of building 1'):")
            user_input = input("> ")
            
            if not user_input.strip():
                print("No instruction provided, using default instruction")
                user_input = "Please deliver this apple to unit 1 of building 1"
        
        action_client.send_goal(user_input)
        
        rclpy.spin(action_client)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        if rclpy.ok():
            try:
                if hasattr(action_client, '_goal_handle') and action_client._goal_handle is not None:
                    action_client.cancel_goal()
                action_client.destroy_node()
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")
            finally:
                rclpy.shutdown()

if __name__ == '__main__':
    main() 