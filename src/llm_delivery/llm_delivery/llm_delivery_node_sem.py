import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from pyproj import CRS, Transformer
import llm_delivery.llm_agent as LLMAgent
import llm_delivery.osm_route as OsmRoute
import numpy as np
import sys

real_world_flag = False

class DeliveryActionClient(Node):
    def __init__(self):
        super().__init__('delivery_action_client')

        # Initialize coordinate systems
        self.wgs84 = CRS.from_epsg(4326)
        if real_world_flag:
            self.utm = CRS.from_epsg(32650)  # 50n for real-world
        else:
            self.utm = CRS.from_epsg(32633)  # 33n for simulation
        self.transformer = Transformer.from_crs(self.utm, self.wgs84, always_xy=True)

        # Robot position subscriber
        self.position_subscription = self.create_subscription(
            Pose,
            '/robot_position',
            self.position_callback,
            20
        )

        # Publisher for individual waypoints
        self.waypoint_publisher = self.create_publisher(PoseStamped, '/vis_waypoint', 10)
        self.robot_position = None

    def position_callback(self, msg):
        if real_world_flag:
            transform_matrix = np.array([[1.0, 0.0, -449920.549610], [0.0, 1.0, -4424638.431542], [0.0, 0.0, 1.0]])
            transform_matrix = np.linalg.inv(transform_matrix)
            x = msg.position.x
            y = msg.position.y
            point = np.array([x, y, 1])
            trans_point = np.dot(transform_matrix, point)
            self.robot_position = [trans_point[0], trans_point[1]]
        else:
            self.robot_position = [msg.position.x, msg.position.y]

    def get_robot_position(self):
        return self.robot_position

    def publish_individual_waypoints(self, waypoints):
        for i, waypoint in enumerate(waypoints):
            waypoint.header.frame_id = 'map'
            waypoint.header.stamp = self.get_clock().now().to_msg()
            self.waypoint_publisher.publish(waypoint)
            self.get_logger().info(f"Published waypoint {i+1} at x: {waypoint.pose.position.x}, y: {waypoint.pose.position.y}")

class DeliveryActionClientRunner:
    def __init__(self, args=None):
        # rclpy.init(args=args)
        self.delivery_client = DeliveryActionClient()

    def run_delivery_client(self, user_input=None):
        if user_input is None:
            user_input = input("\nHi! I am XiaoZhi~ Do you need any delivery?\n")

        response = LLMAgent.call_llm(user_input)
        building_ids, unit_ids, building_coords, unit_coords = LLMAgent.extract_coordinates(response)

        for i in range(len(building_coords)):
            exploration_flag = False
            if unit_coords[i]:
                position = unit_coords[i]
            else:
                exploration_flag = True
                position = building_coords[i]

            try:
                while rclpy.ok() and self.delivery_client.get_robot_position() is None:
                    rclpy.spin_once(self.delivery_client, timeout_sec=1.0)

                robot_position = self.delivery_client.get_robot_position()
                if robot_position is None:
                    self.delivery_client.get_logger().error("Robot position is not available.")
                    continue

                curr_robot_position = list(self.delivery_client.transformer.transform(robot_position[0], robot_position[1]))

                start_position = f"{curr_robot_position[0]:.9f},{curr_robot_position[1]:.9f}"
                end_position = f"{position[0]:.9f},{position[1]:.9f}"

                # 获取由 OSM 路径规划生成的 waypoints
                waypoints = OsmRoute.get_route(start_position, end_position) 

                if waypoints:
                    # 可选：发布可视化 waypoints 到 RViz
                    self.delivery_client.publish_individual_waypoints(waypoints)
                    self.delivery_client.get_logger().info("Published all waypoints individually for RViz visualization.")
                    return waypoints  # 返回生成的 waypoints 列表
                else:
                    self.delivery_client.get_logger().info(f"Unable to get a valid navigation path from {start_position} to {end_position}")
                    return None  # 如果未生成有效路径，返回 None

            except Exception as e:
                self.delivery_client.get_logger().error(f"Error occurred: {e}")
                return None  # 发生异常时返回 None

    def shutdown(self):
        if rclpy.ok():
            rclpy.shutdown()
