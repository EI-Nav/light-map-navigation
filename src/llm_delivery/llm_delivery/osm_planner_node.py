import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from geometry_msgs.msg import PolygonStamped
from geometry_msgs.msg import PoseStamped, Pose
from visualization_msgs.msg import Marker 
from std_srvs.srv import Trigger

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

from llm_delivery.llm_delivery_node_sem import DeliveryActionClientRunner

import numpy as np
import json



class WaypointNavigator(Node):
    def __init__(self, waypoints):
        super().__init__('waypoint_navigator')
        self.waypoints = waypoints
        self.current_wp_index = 0  # Index of the current waypoint
        self.odom_subscription = self.create_subscription(Odometry, '/Odometry', self.odometry_callback, 10)
        self.concave_hull_subscription = self.create_subscription(PolygonStamped, '/concave_hull', self.concave_hull_callback, 10)
        self.line_segment_subscription = self.create_subscription(PolygonStamped, '/line_segment', self.line_segment_callback, 10)

        self.is_line_reveive = False

        self.robot_position = None
        self.line_segment_points = None
        self.concave_hull_points = []
        self.wp_real = None  # The actual next waypoint to navigate to
        
        # Initialize a marker publisher
        self.marker_publisher = self.create_publisher(Marker, '/wp_real_marker', 10)

        self.real_navigator = BasicNavigator() 

        
    def odometry_callback(self, msg):
        # Get the robot's position from the Odometry message
        self.robot_position = msg.pose.pose.position
        self.check_reach_waypoint()

    def concave_hull_callback(self, msg):
        # Store the vertices of the concave hull
        self.concave_hull_points = msg.polygon.points

    def line_segment_callback(self, msg):
        # Assuming the line segment is represented by two points in the PolygonStamped
        print(msg.polygon.points, len(msg.polygon.points))
        if len(msg.polygon.points) >= 2:
            self.is_line_reveive = True
            self.line_segment_points = (msg.polygon.points[0], msg.polygon.points[1])

    def check_reach_waypoint(self):
        if self.robot_position is None or self.current_wp_index >= len(self.waypoints):
            return
        if not self.is_line_reveive:
            return

        waypoint = self.waypoints[self.current_wp_index]

        # Check if the robot has reached the waypoint
        distance = np.sqrt((self.robot_position.x - waypoint.pose.position.x) ** 2 + 
                           (self.robot_position.y - waypoint.pose.position.y) ** 2)
        # self.get_logger().info(f"Dis {distance}") # 距离下一个点的wp

        if distance < 0.6: # 小于0.6认为直接到达
            self.get_logger().info(f"Reached waypoint {self.current_wp_index + 1}: {waypoint.pose.position.x}, {waypoint.pose.position.y}")
            self.current_wp_index += 1  # Move to the next waypoint

            if self.current_wp_index < len(self.waypoints):
                wp = self.calculate_next_waypoint() # 计算实际执行的点
                self.get_logger().info(f"real wp exec: {wp}")
                # 下一步将算出来的wp发送到BasicNavigator进行执行
                if wp: # 如果wp中有真实的数据：
                    is_nav_success = self.send_goal_to_basic_navigator(wp) # 将一个点发送到导航器执行
                    if is_nav_success: # 导航执行成功，通过client调用分割算法进行分割，重新计算可通行区域
                        self.get_logger().info("Calling segmentation client for updated navigable area...")
                        self.segmentation_client.send_request() # 调用分割算法，返回最新的可通行区域

            else:
                self.get_logger().info("All waypoints reached.")

    def calculate_next_waypoint(self):
        if self.current_wp_index >= len(self.waypoints):
            return

        if self.line_segment_points is None:
            self.get_logger().warn("Line segment data is not available.")
            return

        current_waypoint = self.waypoints[self.current_wp_index - 1]  # The previous waypoint
        next_waypoint = self.waypoints[self.current_wp_index]

        # Calculate slope between current and next waypoints
        slope = (next_waypoint.pose.position.y - current_waypoint.pose.position.y) / \
                (next_waypoint.pose.position.x - current_waypoint.pose.position.x) \
                if (next_waypoint.pose.position.x - current_waypoint.pose.position.x) != 0 else float('inf')

        # Get line segment points
        line_start, line_end = self.line_segment_points

        # Calculate the intersection point
        intersection_point = self.calculate_line_intersection(
            (self.robot_position.x, self.robot_position.y),
            slope,
            (line_start.x, line_start.y),
            (line_end.x, line_end.y)
        )

        if intersection_point:
            wp_exec = Point(x=intersection_point[0], y=intersection_point[1], z=0.0)

            # Check if the calculated waypoint is inside the concave hull
            if self.is_point_in_polygon(wp_exec, self.concave_hull_points):
                self.get_logger().info(f"Calculated next real waypoint: {wp_exec.x}, {wp_exec.y}")
                self.publish_wp_real_marker(wp_exec)  # Visualize wp_real
                
                # Convert Point to PoseStamped and return it
                wp_exec_pose_stamped = self.convert_point_to_pose_stamped(wp_exec)
                return wp_exec_pose_stamped  # Returning PoseStamped for BasicNavigator
                
            else:
                self.get_logger().warn(f"Calculated waypoint {wp_exec.x}, {wp_exec.y} is outside the concave hull.")
        else:
            self.get_logger().warn("No valid intersection point found.")

    def convert_point_to_pose_stamped(self, point):
        """Convert a Point message to a PoseStamped message."""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"  
        pose_stamped.header.stamp = self.get_clock().now().to_msg()  # Current timestamp

        pose_stamped.pose.position = point
        pose_stamped.pose.orientation.w = 1.0  # Neutral orientation (no rotation)

        return pose_stamped

    def publish_wp_real_marker(self, wp):
        marker = Marker()
        marker.header.frame_id = "map"  # Set to your relevant frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "waypoints"
        marker.id = self.current_wp_index
        marker.type = Marker.SPHERE  # You can use other types like POINTS, CUBE, etc.
        marker.action = Marker.ADD

        # Set the position and scale of the marker
        marker.pose.position = wp
        marker.scale.x = 0.2  # Sphere radius
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0  # Alpha (transparency)
        marker.color.r = 0.0  # Red
        marker.color.g = 1.0  # Green
        marker.color.b = 0.0  # Blue

        self.marker_publisher.publish(marker)  # Publish the marker

    def calculate_line_intersection(self, robot_pos, slope, line_start, line_end):
        # Extract coordinates
        x1, y1 = line_start
        x2, y2 = line_end

        # Line equations:
        # Robot line: y - robot_pos[1] = slope * (x - robot_pos[0])
        # Line segment: y = y1 + (y2 - y1)/(x2 - x1) * (x - x1)

        # Handle vertical line case for line segment
        if x2 - x1 == 0:  # Line segment is vertical
            x_intersection = x2
            y_intersection = slope * (x_intersection - robot_pos[0]) + robot_pos[1]
            if min(y1, y2) <= y_intersection <= max(y1, y2):
                return (x_intersection, y_intersection)
            else:
                return None

        # Calculate the slope of the line segment
        line_slope = (y2 - y1) / (x2 - x1)

        # Calculate intersection point
        if slope != line_slope:
            # Solve for x
            x_intersection = (line_slope * x1 - slope * robot_pos[0] + robot_pos[1] - y1) / (line_slope - slope)
            y_intersection = slope * (x_intersection - robot_pos[0]) + robot_pos[1]

            if (min(x1, x2) <= x_intersection <= max(x1, x2)):
                return (x_intersection, y_intersection)
        return None

    def is_point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using the ray-casting algorithm."""
        n = len(polygon)
        inside = False

        x_intercept = point.x
        y_intercept = point.y

        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            if y_intercept > min(p1y, p2y):
                if y_intercept <= max(p1y, p2y):
                    if x_intercept <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y_intercept - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x_intercept <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def send_goal_to_basic_navigator(self, waypoint_pose_stamped):
        self.real_navigator.goToPose(waypoint_pose_stamped)
        # 等待导航完成
        while not self.real_navigator.isTaskComplete():
            feedback = self.real_navigator.getFeedback()

        # 获取导航结果
        result = self.real_navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("Arrived at waypoint")
            return True
        else:
            self.get_logger().error("Failed to reach waypoint")

def main():
    # 从OSM中获取初步规划的waypoints
    runner = DeliveryActionClientRunner()
    waypoints = runner.run_delivery_client("send this box to building13 unit2")

    if waypoints is not None:
        print("Waypoints received:", waypoints)
        navigator = WaypointNavigator(waypoints) # 传入navigator进行分步执行
        
        # Spin the node to process callbacks
        rclpy.spin(navigator)
    else:
        print("Failed to generate waypoints.")

    # Shutdown
    runner.shutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
