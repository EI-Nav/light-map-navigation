import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
import geometry_msgs.msg
from math import radians, cos, sin, atan2, sqrt
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import time
from tf_transformations import euler_from_quaternion, quaternion_from_euler

def euler_to_quaternion(yaw):
    q = geometry_msgs.msg.Quaternion()
    q.w = cos(yaw / 2)
    q.x = 0.0
    q.y = 0.0
    q.z = sin(yaw / 2)
    return q

def main(args=None):
    rclpy.init(args=args)
    node = Node('waypoint_navigation_node')
    navigator = BasicNavigator()

    # Define the waypoints
    waypoints = [
        (7.04, -0.13, 0.0),
        (9.5, -0.1, 0.0)
    ]

    for index, waypoint in enumerate(waypoints):
        # Create a PoseStamped message for each waypoint
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = node.get_clock().now().to_msg()
        goal.pose.position.x = waypoint[0]
        goal.pose.position.y = waypoint[1]
        goal.pose.position.z = 0.0
        q = quaternion_from_euler(0,0,waypoint[2])
        goal.pose.orientation.x = q[0]
        goal.pose.orientation.y = q[1]
        goal.pose.orientation.z = q[2]
        goal.pose.orientation.w = q[3]

        # Send the goal to the navigator
        navigator.goToPose(goal)
        node.get_logger().info(f"Sent waypoint {index + 1}: ({waypoint[0]}, {waypoint[1]})")

        # Wait for the navigator to complete the task
        while not navigator.isTaskComplete():
            feedback = navigator.getFeedback()
            # if feedback:
            #     node.get_logger().info(f"Feedback: {feedback}")
        
        # Get the result of the navigation
        result = navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            node.get_logger().info(f"Arrived at waypoint {index}")
            # time.sleep(3)
            index = index + 1
        else:
            node.get_logger().error(f"Failed to reach waypoint {index + 1}")
            break  # Stop navigation if a waypoint is not reached successfully

    node.destroy_node()
    rclpy.shutdown()
