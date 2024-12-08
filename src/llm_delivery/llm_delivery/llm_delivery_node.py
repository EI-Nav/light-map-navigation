import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_simple_commander.robot_navigator import BasicNavigator
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Pose
from pyproj import CRS, Transformer
import llm_delivery.llm_agent as LLMAgent
import llm_delivery.osm_route as OsmRoute
from tf2_ros import TransformListener, Buffer
from custom_interfaces.action import FindUnit
from custom_interfaces.srv import TaskRecord
import numpy as np
import sys

real_world_flag = False

class DeliveryActionClient(Node):
    def __init__(self):
        super().__init__('delivery_action_client')

        self.navigation_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.exploration_action_client = ActionClient(self, FindUnit, 'find_unit')
        self.navigator = BasicNavigator()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.wgs84 = CRS.from_epsg(4326)
        if real_world_flag is True:
            self.utm = CRS.from_epsg(32650)  #50n
        else:
            self.utm = CRS.from_epsg(32633)  #33n
        self.transformer = Transformer.from_crs(self.utm, self.wgs84, always_xy=True)
        self.current_waypoint_index = 0
        self.waypoints = []

        self.robot_position = None
        self.position_subscription = self.create_subscription(
            Pose,
            '/robot_position',
            self.position_callback,
            20
        )

        # Service client for TaskRecord
        self.task_record_client = self.create_client(TaskRecord, 'task_record')
        while not self.task_record_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('TaskRecord service not available, waiting...')

    def position_callback(self, msg):
        if real_world_flag is True:
            transform_matrix = np.array([[1.0, 0.0, -449920.549610], [0.0, 1.0, -4424638.431542], [0.000000, 0.000000, 1.000000]])
            transform_matrix = np.linalg.inv(transform_matrix)
            x = msg.position.x
            y = msg.position.y
            point = np.array([x, y, 1])
            trans_point = np.dot(transform_matrix, point)
            self.robot_position = [trans_point[0], trans_point[1]]
        else:
            self.robot_position = [msg.position.x, msg.position.y]
        # self.get_logger().info(f"Received robot position: x={msg.position.x}, y={msg.position.y}")

    def get_robot_position(self):
        return self.robot_position
    
    def send_task_record_request(self, status, address):
        request = TaskRecord.Request()
        request.status = status
        request.address = address

        future = self.task_record_client.call_async(request)
        future.add_done_callback(self.task_record_response_callback)

    def task_record_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('TaskRecord service call successful.')
            else:
                self.get_logger().info('TaskRecord service call failed.')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    # navigation action service
    def nav_send_goal(self, waypoint):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = waypoint
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        self.navigation_action_client.wait_for_server()
        self._send_goal_future = self.navigation_action_client.send_goal_async(goal_msg, feedback_callback=self.nav_feedback_callback)
        self._send_goal_future.add_done_callback(self.nav_goal_response_callback)

    def nav_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # self.get_logger().info(f'Received feedback: {feedback}')

    def nav_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result}')

        self.current_waypoint_index += 1
        if self.current_waypoint_index < len(self.waypoints):
            next_waypoint = self.waypoints[self.current_waypoint_index]
            self.nav_send_goal(next_waypoint)
        else:
            self.get_logger().info("All waypoints completed.")
            rclpy.shutdown()

    # exploration action service
    def expl_send_goal(self, building_id, unit_number, pos_x, pos_y):
        goal_msg = FindUnit.Goal()
        goal_msg.building_id = building_id
        goal_msg.unit_number = unit_number
        goal_msg.cur_x = pos_x
        goal_msg.cur_y = pos_y

        self.exploration_action_client.wait_for_server()
        self._send_goal_future = self.exploration_action_client.send_goal_async(goal_msg, feedback_callback=self.expl_feedback_callback)
        self._send_goal_future.add_done_callback(self.expl_goal_response_callback)

    def expl_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def expl_get_result_callback(self, future):
        try:
            result = future.result().result
            self.get_logger().info(f'Result_success {result.success}, Message: {result.message}')
        except Exception as e:
            self.get_logger().error(f'Error in get_result_callback: {e}')

    def expl_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: Status: {feedback.status}')

def main(user_input=None, args=None):
    if user_input is None and len(sys.argv) > 1:
        user_input = sys.argv[1]
    elif user_input is None:
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

        print(exploration_flag)
        print(position)

        rclpy.init(args=args)
        delivery_client = DeliveryActionClient()

        # Start recording task
        delivery_client.send_task_record_request('start', f'{unit_ids[i]}{building_ids[i]}')

        delivery_client.get_logger().info(f"Delivering to location {i}, please wait patiently!\n")

        delivery_client.get_logger().info("#################### Navigation Step ####################")

        try:
            while rclpy.ok() and delivery_client.get_robot_position() is None:
                rclpy.spin_once(delivery_client, timeout_sec=1.0)
            robot_position = delivery_client.get_robot_position()

            if robot_position is None:
                delivery_client.get_logger().error("Robot position is not available.")
                continue

            curr_robot_position = list(delivery_client.transformer.transform(robot_position[0], robot_position[1]))

            start_position = f"{curr_robot_position[0]:.9f},{curr_robot_position[1]:.9f}"
            end_position = f"{position[0]:.9f},{position[1]:.9f}"

            waypoints = OsmRoute.get_route(start_position, end_position)
            print(waypoints)
            
            if waypoints:
                delivery_client.waypoints = waypoints
                delivery_client.nav_send_goal(delivery_client.waypoints[0])
                rclpy.spin(delivery_client)
            else:
                delivery_client.get_logger().info(f"Unable to get a valid navigation path from {start_position} to {end_position}\n")

        except Exception as e:
            delivery_client.get_logger().error(f"Error occurred: {e}")

        finally:
            if rclpy.ok():
                rclpy.shutdown()

        delivery_client.get_logger().info("##########################################################\n")

        rclpy.init(args=args)
        delivery_client = DeliveryActionClient()

        delivery_client.get_logger().info("#################### Exploration Step ####################")

        if exploration_flag:
            try:
                while rclpy.ok() and delivery_client.get_robot_position() is None:
                    rclpy.spin_once(delivery_client, timeout_sec=1.0)
                robot_position = delivery_client.get_robot_position()

                if robot_position is None:
                    delivery_client.get_logger().error("Robot position is not available.")
                    continue

                curr_robot_position = list(delivery_client.transformer.transform(robot_position[0], robot_position[1]))

                print(building_ids[i])
                print(unit_ids[i])

                delivery_client.expl_send_goal(building_ids[i][0], unit_ids[i][0][-1:], curr_robot_position[0], curr_robot_position[1])
                rclpy.spin(delivery_client)

            except Exception as e:
                delivery_client.get_logger().error(f"Error occurred: {e}")

            finally:
                delivery_client.send_task_record_request('end', f'location_{i}')
                if rclpy.ok():
                    rclpy.shutdown()
        
        delivery_client.get_logger().info("##########################################################")

        delivery_client.get_logger().info(f"Task completed!\n")

        # End recording task
        # rclpy.init(args=args)
        # delivery_client = DeliveryActionClient()
        # delivery_client.send_task_record_request('end', f'location_{i}')
        # rclpy.spin_once(delivery_client, timeout_sec=1.0)

if __name__ == '__main__':
    main()