import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Time
import math
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import std_msgs.msg
import random
from scipy.spatial import KDTree
import numpy as np


from llm_delivery.llm_delivery_node_sem import DeliveryActionClientRunner


debug_mode = True

class WaypointPlanner(Node):
    def __init__(self):
        super().__init__('waypoint_planner')
        self.runner = DeliveryActionClientRunner()

        self.is_nav_run = False
        self.is_map_receive = False
        self.is_position_receive = False
        self.is_first_resume = True
        self.is_seg_image = False
        self.is_last_wp = False
        
        self.costmap = None
        self.current_position = None
        self.previous_position = None  
        self.accumulated_distance = 0  
        self.obstacle_tree = None
        self.real_navigator = BasicNavigator()

        self.latest_wp = None

        # 从OSM路由中获取路径
        self.waypoints = self.runner.run_delivery_client("send this box to building7 unit2")
        if not self.waypoints:
            self.get_logger().error("Failed to generate initial waypoints")
            return
        
        # 对初始的wp施加整体平移，测试OSM不准确的情况
        if debug_mode:
            # 对 waypoints 施加整体平移（测试功能）
            self.translate_waypoints(dx=2.0, dy=3.0)  # 平移测试，dx 和 dy 可以调整
            self.get_logger().info("Waypoints translated by dx=2.0, dy=3.0 for testing")

        self.current_waypoint_index = 0  # Track the current waypoint index

        # Subscriptions and publishers
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap', self.costmap_callback, 10
        )
        self.robot_position_sub = self.create_subscription(
            Pose, '/robot_position', self.robot_position_callback, 10
        )

        self.marker_pub = self.create_publisher(Marker, '/waypoints_marker', 10)

        # Timer for non-blocking execution
        self.timer = self.create_timer(0.5, self.execute_callback)

        # Service client
        self.segmentation_client = self.create_client(Trigger, '/segment_and_publish')

        # Initialize variables for delay and service
        self.start_time = None
        self.delay_duration = 8.0  # seconds
        self.elapsed_time = 0.0

        self.publish_waypoints_marker()

        # Call service and set up delay handling
        self.call_segment_and_publish_service()

    def translate_waypoints(self, dx, dy):
        """Translate all waypoints by a given offset."""
        for waypoint in self.waypoints:
            waypoint.pose.position.x += dx
            waypoint.pose.position.y += dy
        self.get_logger().info(f"Translated all waypoints by dx={dx}, dy={dy}")

    def costmap_callback(self, msg):
        """ Callback for receiving the costmap. """
        self.costmap = msg
        self.is_map_receive = True
        obstacles = self.get_obstacles_from_costmap(msg)
        if obstacles:
            self.obstacle_tree = KDTree(np.array(obstacles))
            self.get_logger().info("KDTree for obstacles created")
        else:
            self.get_logger().warn("No obstacles found in the costmap")

    def robot_position_callback(self, msg):
        """ Callback for receiving robot position. """
        if self.current_position:
            distance = self.distance(
                self.current_position.x, self.current_position.y,
                msg.position.x, msg.position.y
            )
            self.accumulated_distance += distance

            if self.accumulated_distance >= 5.0:
                self.accumulated_distance = 0  # 重置累计距离
                self.get_logger().info("Reached 5 meters, calling /segment_and_publish service")
                self.real_navigator.cancelTask()
                self.get_logger().info("Navigation canceled")
                self.is_nav_run = False
                self.call_segment_and_publish_service()

        self.current_position = msg.position
        self.is_position_receive = True
        # self.get_logger().info("Received current position")
        # self.update_current_waypoint_index_based_on_position()113213

    def call_segment_and_publish_service(self):
        """ Call segmentation service and wait for completion. """
        request = Trigger.Request()
        future = self.segmentation_client.call_async(request)
        self.get_logger().info("Segment service called.")

        # Start tracking time after calling the service
        self.start_time = self.get_clock().now()
        self.is_seg_image = True

        # Create a timer to check elapsed time and resume navigation after the delay
        self.create_timer(0.1, self.check_elapsed_time)

    def check_elapsed_time(self):
        """ Check elapsed time and resume navigation when the delay has passed. """
        if  not self.is_seg_image:
            return 
        
        if self.start_time is not None:
            self.elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9  # seconds
            self.get_logger().info(f"Elapsed time: {self.elapsed_time}")

            if self.elapsed_time >= self.delay_duration:
                self.resume_navigation()
                self.get_logger().info(f"Waited {self.delay_duration} seconds, resuming navigation.")
                # Stop the timer after resuming navigation


    def resume_navigation(self):
        """ Resume the navigation to the next waypoint. """
        self.is_nav_run = True
        self.is_seg_image = False
        if self.current_waypoint_index < len(self.waypoints) - 1:
            next_waypoint = self.waypoints[self.current_waypoint_index]
            self.get_logger().info("Resuming navigation to the next waypoint")
            self.is_position_receive = False
            if self.is_first_resume:
                self.real_navigator.goToPose(next_waypoint)
                self.latest_wp = next_waypoint
                self.is_first_resume = False
            else:
                self.real_navigator.goToPose(self.latest_wp)
                
        
    def execute_callback(self):
        self.publish_waypoints_marker()

        """ Main execution logic in a non-blocking manner. """
        if not (self.is_map_receive and self.is_position_receive):
            self.get_logger().info("Waiting for costmap and position data")
            return

        if not self.is_nav_run:
            self.get_logger().info("Navigation is stopped, waiting for services")
            return
        
        if self.is_last_wp:
            if self.distance(self.latest_wp.pose.position.x, self.latest_wp.pose.position.y,
                             self.current_position.x, self.current_position.y) < 0.2: # 判断到达
                self.current_waypoint_index += 1
                self.is_last_wp = False
                self.get_logger().warn(f"Waypoint {self.current_waypoint_index} reached. Moving to the next.")
            else:
                self.get_logger().info("Waiting arrive last waypoint")
            return 

        if self.current_waypoint_index < len(self.waypoints) - 1:
            start = self.waypoints[self.current_waypoint_index].pose.position
            end = self.waypoints[self.current_waypoint_index + 1].pose.position
            
            # 如果优化出的点和target_position的距离接近
            if self.check_wp_reached(end):
                self.is_last_wp = True # 开始执行最后一步是否到每到达
                # Check if the waypoint `end` is feasible
                if self.is_feasible(end):
                    self.get_logger().info("Waypoint is feasible, proceeding with OSM waypoint.")
                    self.real_navigator.goToPose(self.waypoints[self.current_waypoint_index + 1])
                    self.latest_wp = self.waypoints[self.current_waypoint_index + 1]
                else:
                    # If `end` is not feasible, use `latest_wp` and adjust subsequent waypoints
                    self.get_logger().warn("Waypoint not feasible, using last optimized waypoint.")
                    self.real_navigator.goToPose(self.latest_wp)
                    self.adjust_waypoints_based_on_last_wp()
                
            else:
                # Optimize and navigate to the next waypoint
                if self.is_position_receive:
                    self.get_logger().info("Navigating to next waypoint")
                    self.navigate_to_waypoint(start, end)

    def adjust_waypoints_based_on_last_wp(self):
        """ Adjusts waypoints based on the last valid waypoint. """
        if self.current_waypoint_index < len(self.waypoints) - 1:
            last_valid_position = self.latest_wp.pose.position
            original_osm_position = self.waypoints[self.current_waypoint_index + 1].pose.position
            
            # Calculate translation vector to shift OSM waypoints
            translation_vector = (
                last_valid_position.x - original_osm_position.x,
                last_valid_position.y - original_osm_position.y
            )
            
            # Apply the translation to all subsequent waypoints
            for i in range(self.current_waypoint_index + 1, len(self.waypoints)):
                wp = self.waypoints[i].pose.position
                wp.x += translation_vector[0]
                wp.y += translation_vector[1]

            self.get_logger().info("Adjusted subsequent OSM waypoints based on feasibility.")


    def navigate_to_waypoint(self, start, end):
        """ Use optimized waypoint navigation to send goal. """
        optimized_wp = self.optimize_waypoint(start, end)
        if optimized_wp:
            waypoint_pose_stamped = PoseStamped(
                header=self.create_header(),
                pose=Pose(
                    position=Point(x=optimized_wp[0], y=optimized_wp[1], z=0.0),
                    orientation=self.get_orientation_towards(end)
                )
            )
            self.latest_wp = waypoint_pose_stamped
            self.real_navigator.goToPose(waypoint_pose_stamped)

    # check_wp_reached1：检查OSM最后一个点和规划出路径的最后一个点之间的距离
    # def check_wp_reached(self, target_position):
    #     if self.current_position:
    #         distance = self.distance(
    #             self.latest_wp.pose.position.x,self.latest_wp.pose.position.y,
    #             target_position.x, target_position.y
    #         )
    #         return distance <= 3.0
    #     return False

    # check_wp_reached2：检查OSM最后一个点和规划出路径、机器人当前位置的距离
    def check_wp_reached(self, target_position):
        if self.current_position is None or self.latest_wp is None:
            self.get_logger().warn("Current position or latest waypoint is not available.")
            return False

        # Distance from the latest waypoint to the target position
        distance_from_latest_wp = self.distance(
            self.latest_wp.pose.position.x, self.latest_wp.pose.position.y,
            target_position.x, target_position.y
        )

        # Distance from the current position to the target position
        distance_from_current_position = self.distance(
            self.current_position.x, self.current_position.y,
            target_position.x, target_position.y
        )

        # Threshold for waypoint reach (can be adjusted as needed)
        threshold = 3.0

        # Log distances for debugging
        self.get_logger().info(
            f"Distance from latest waypoint: {distance_from_latest_wp}, "
            f"Distance from current position: {distance_from_current_position}, "
            f"Threshold: {threshold}"
        )

        # Return True if either condition is satisfied
        return (distance_from_latest_wp <= threshold or 
                distance_from_current_position <= threshold)

    # check_wp_reached3：检查OSM最后一个点和规划出路径、机器人当前位置的距离
    # def check_wp_reached(self):
    #     """
    #     Dynamically adjust the `self.current_waypoint_index` based on the robot's current position.
    #     The closest waypoint with an index greater than `self.current_waypoint_index` is selected.
    #     """
    #     if self.current_position is None or not self.waypoints:
    #         self.get_logger().warn("Current position or waypoints not available.")
    #         return False

    #     min_distance = float('inf')
    #     closest_index = self.current_waypoint_index  # Default to current index

    #     # Only consider waypoints with index greater than the current waypoint index
    #     for i in range(self.current_waypoint_index + 1, len(self.waypoints)):
    #         wp_position = self.waypoints[i].pose.position
    #         distance = self.distance(
    #             self.current_position.x, self.current_position.y,
    #             wp_position.x, wp_position.y
    #         )

    #         if distance < min_distance:
    #             min_distance = distance
    #             closest_index = i

    #     # Update the current waypoint index if a closer waypoint is found
    #     if closest_index != self.current_waypoint_index:
    #         self.get_logger().info(
    #             f"Waypoint index updated from {self.current_waypoint_index} to {closest_index}. "
    #             f"Closest waypoint distance: {min_distance:.2f}."
    #         )
    #         self.current_waypoint_index = closest_index
    #         return True

    #     return False

    def distance_to_obstacle(self, x, y):
        """ Calculate the distance to the nearest obstacle using KDTree. """
        if self.obstacle_tree is None:
            self.get_logger().warn("Obstacle KDTree not initialized")
            return float('inf')

        distance, _ = self.obstacle_tree.query([x, y])
        return distance

    def get_obstacles_from_costmap(self, msg):
        """Extract obstacles from the costmap and return a list of obstacle positions."""
        resolution = msg.info.resolution  # 获取地图分辨率
        origin_x = msg.info.origin.position.x  # 获取地图原点x坐标
        origin_y = msg.info.origin.position.y  # 获取地图原点y坐标
        width = msg.info.width  # 地图宽度（网格个数）
        height = msg.info.height  # 地图高度（网格个数）

        obstacles = []  # 存储障碍物的坐标

        # 遍历所有网格，检查是否为障碍物（值>=50表示障碍物）
        for i in range(height):
            for j in range(width):
                index = i * width + j  # 当前网格在数据中的索引
                if msg.data[index] >= 50:  # 如果该位置是障碍物
                    # 根据分辨率和原点位置计算障碍物的坐标
                    x = origin_x + j * resolution
                    y = origin_y + i * resolution
                    obstacles.append([x, y])  # 将障碍物坐标添加到列表

        return obstacles  # 返回障碍物坐标列表
    
    # def is_feasible(self, point):
    #     """ Check if a given point is feasible (e.g., not in an obstacle). """
    #     return self.distance_to_obstacle(point.x, point.y) > 0.5

    def is_feasible(self, end):
        """ 检查是否机器人与waypoint之间的连线中间有costmap障碍物返回True或者False """
        if self.costmap is None:
            self.get_logger().warn("Costmap not received yet")
            return True  # Assume feasible if costmap is not available

        resolution = self.costmap.info.resolution
        origin_x = self.costmap.info.origin.position.x
        origin_y = self.costmap.info.origin.position.y
        width = self.costmap.info.width
        height = self.costmap.info.height

        # Get robot's current position and end position
        start = self.current_position
        if start is None:
            self.get_logger().warn("Current position not received yet")
            return True  # Assume feasible if position is not available

        # Generate points along the line from start to end
        points = self.get_line_points(start.x, start.y, end.x, end.y, resolution)

        # Check each point against the costmap for obstacles
        for px, py in points:
            # Convert point (px, py) to costmap grid index
            grid_x = int((px - origin_x) / resolution)
            grid_y = int((py - origin_y) / resolution)

            if 0 <= grid_x < width and 0 <= grid_y < height:
                index = grid_y * width + grid_x
                if self.costmap.data[index] >= 50:  # Threshold for obstacles in the costmap
                    self.get_logger().info(f"Obstacle found at ({px}, {py}), grid index ({grid_x}, {grid_y})")
                    return False
            else:
                self.get_logger().warn(f"Point ({px}, {py}) is out of costmap bounds.")

        return True  
    
    def get_line_points(self, x1, y1, x2, y2, resolution):
        """Generate points along a line from (x1, y1) to (x2, y2) with a given resolution."""
        distance = self.distance(x1, y1, x2, y2)
        num_points = max(2, int(distance / resolution))  # At least two points (start and end)
        points = [
            (x1 + i * (x2 - x1) / (num_points - 1), y1 + i * (y2 - y1) / (num_points - 1))
            for i in range(num_points)
        ]
        return points

    def update_current_waypoint_index_based_on_position(self):
        if self.current_position is None or not self.waypoints:
            self.get_logger().warn("Current position or waypoints not available for waypoint update.")
            return

        # Ensure we have at least two waypoints for the logic
        if self.current_waypoint_index >= len(self.waypoints) - 1:
            self.get_logger().info("Robot is at or beyond the final waypoint.")
            return

        # Current waypoint and the next waypoint
        wp1 = self.waypoints[self.current_waypoint_index].pose.position
        wp2 = self.waypoints[self.current_waypoint_index + 1].pose.position

        # Vector from wp1 to wp2 and from wp1 to robot
        vector_wp1_wp2 = np.array([wp2.x - wp1.x, wp2.y - wp1.y])
        vector_wp1_robot = np.array([self.current_position.x - wp1.x, self.current_position.y - wp1.y])

        # Calculate dot product to determine the robot's position relative to wp1 and wp2
        dot_product = np.dot(vector_wp1_wp2, vector_wp1_robot)
        wp1_to_wp2_length_squared = np.dot(vector_wp1_wp2, vector_wp1_wp2)

        # Check if the robot has moved past wp2
        if dot_product >= wp1_to_wp2_length_squared:
            self.get_logger().info(f"Robot has moved beyond waypoint {self.current_waypoint_index + 1}. Updating index.")
            self.current_waypoint_index += 1
        elif dot_product < 0:  # Robot is before wp1
            self.get_logger().info(f"Robot is before waypoint {self.current_waypoint_index}. Keeping index.")
        else:
            # Robot is between wp1 and wp2
            self.get_logger().info(f"Robot is between waypoint {self.current_waypoint_index} and waypoint {self.current_waypoint_index + 1}. Keeping index.")



    def optimize_waypoint(self, start, end):
        """Optimize and find the best waypoint with normalized cost components."""
        best_wp = None
        best_score = float('inf')

        dx = end.x - start.x
        dy = end.y - start.y
        target_angle = math.atan2(dy, dx)

        cost_slope_values = []
        cost_dis_obs_values = []
        cost_dis_xy_values = []

        # First pass to gather cost statistics
        for _ in range(50):
            angle = target_angle + random.uniform(-math.pi / 4, math.pi / 4)
            distance = random.uniform(0, 3)
            x = self.current_position.x + distance * math.cos(angle)
            y = self.current_position.y + distance * math.sin(angle)

            cost_slope = self.slope_diff(x, y, start, end)
            cost_dis_obs = self.distance_to_obstacle(x, y)
            cost_dis_xy = self.distance(self.current_position.x, self.current_position.y, x, y)

            # Store for normalization
            cost_slope_values.append(cost_slope)
            cost_dis_obs_values.append(cost_dis_obs)
            cost_dis_xy_values.append(cost_dis_xy)

        # Find min and max for normalization
        min_cost_slope = min(cost_slope_values)
        max_cost_slope = max(cost_slope_values)
        min_cost_dis_obs = min(cost_dis_obs_values)
        max_cost_dis_obs = max(cost_dis_obs_values)
        min_cost_dis_xy = min(cost_dis_xy_values)
        max_cost_dis_xy = max(cost_dis_xy_values)

        # Normalize costs and calculate the best score
        for i in range(50):
            angle = target_angle + random.uniform(-math.pi / 4, math.pi / 4)
            distance = random.uniform(0, 3)
            x = self.current_position.x + distance * math.cos(angle)
            y = self.current_position.y + distance * math.sin(angle)

            cost_slope = self.slope_diff(x, y, start, end)
            cost_dis_obs = self.distance_to_obstacle(x, y)
            cost_dis_xy = self.distance(self.current_position.x, self.current_position.y, x, y)

            # Normalize costs
            normalized_cost_slope = (cost_slope - min_cost_slope) / (max_cost_slope - min_cost_slope + 1e-6)
            normalized_cost_dis_obs = (cost_dis_obs - min_cost_dis_obs) / (max_cost_dis_obs - min_cost_dis_obs + 1e-6)
            normalized_cost_dis_xy = (cost_dis_xy - min_cost_dis_xy) / (max_cost_dis_xy - min_cost_dis_xy + 1e-6)

            # self.get_logger().warn(f"normalized_cost_slope:{normalized_cost_slope}, normalized_cost_dis_obs:{normalized_cost_dis_obs}, normalized_cost_dis_xy:{normalized_cost_dis_xy}")
            
            # Calculate the score using normalized costs
            score = (
                0.5 * normalized_cost_slope - 0.3 * normalized_cost_dis_obs - 0.2 * normalized_cost_dis_xy
            )

            if score < best_score:
                best_score = score
                best_wp = (x, y)

        return best_wp

    def slope_diff(self, x, y, start, end):
        """ Calculate the difference in slope between the candidate point and the path from start to end. """
        candidate_point = Point(x=x, y=y, z=0.0)
        candidate_slope = self.compute_slope(self.current_position, candidate_point)
        path_slope = self.compute_slope(start, end)
        
        if candidate_slope is None or path_slope is None:
            return float('inf') if candidate_slope != path_slope else 0
        
        return abs(candidate_slope - path_slope)

    def compute_slope(self, start, end):
        """ Compute the slope of a path between two points. """
        dx = end.x - start.x
        dy = end.y - start.y
        if dx == 0:
            return None  # Vertical line, slope is undefined
        return dy / dx

    def distance(self, x1, y1, x2, y2):
        """ Calculate the Euclidean distance between two points. """
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def create_header(self):
        """ Create a header for the pose message. """
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        return header

    def get_orientation_towards(self, target_position):
        """ Get the orientation (quaternion) towards the target position. """
        dx = target_position.x - self.current_position.x
        dy = target_position.y - self.current_position.y
        theta = math.atan2(dy, dx)
        return Quaternion(w=math.cos(theta / 2), z=math.sin(theta / 2))
    
    def publish_waypoints_marker(self):
        """ Publish waypoints as markers in RViz. """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        # Marker properties
        marker.scale.x = 0.2  # Size of the points
        marker.scale.y = 0.2
        marker.color.a = 1.0  # Opacity
        marker.color.r = 1.0  # Red color for visibility

        # Add points for each waypoint
        for waypoint in self.waypoints:
            point = Point()
            point.x = waypoint.pose.position.x
            point.y = waypoint.pose.position.y
            point.z = 0.0
            marker.points.append(point)

        self.marker_pub.publish(marker)
        self.get_logger().info("Published waypoints marker for visualization.")


def main(args=None):
    rclpy.init(args=args)
    planner = WaypointPlanner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
