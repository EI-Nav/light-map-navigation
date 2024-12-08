import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from custom_interfaces.action import FindUnit
from custom_interfaces.srv import GetUnitNum
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import llm_exploration_py.exploration_route as expRoute
import numpy as np

class LlmExplorationNode(Node):
    def __init__(self):
        super().__init__('llm_exploration_node')
        self.get_logger().info('Initializing LlmExplorationNode...')

        # Service client setup
        self.get_unit_cli = self.create_client(GetUnitNum, 'check_unit_number')
        while not self.get_unit_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service "check_unit_number"...')

        self.navigator = BasicNavigator()
        self._reset_state()

        # Action server setup
        self._action_server = ActionServer(
            self,
            FindUnit,
            'find_unit',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

    def _reset_state(self):
        self.cur_position = None
        self.latest_image = None
        self.is_task_success = False
        self.is_odom_receive = False
        self.is_first_get_wp = False
        self.all_wp_send = False
        self.nav_result = False
        self.waypoint_ = []
        self.current_waypoint_index = 0
        self.building_id = ''
        self.desire_unit_num = ''
        self._cancel_requested = False
        self._service_call_future = None

    async def get_unit_send_request(self, desire_num):
        req = GetUnitNum.Request()
        req.desire_num = desire_num
        future = self.get_unit_cli.call_async(req)
        try:
            response = await future
            return response
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')
            return None

    def goal_callback(self, goal_handle):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        self._cancel_requested = True
        return CancelResponse.ACCEPT

    async def exec_navigation(self, goal_handle):
        try:
            while self.current_waypoint_index < len(self.waypoint_) and not self._cancel_requested:
                feedback_msg = FindUnit.Feedback()
                feedback_msg.status = f"Exploring {self.current_waypoint_index}-th waypoint!"
                goal_handle.publish_feedback(feedback_msg)

                waypoint = self.waypoint_[self.current_waypoint_index]
                goal = PoseStamped()
                goal.header.frame_id = 'map'
                goal.header.stamp = self.get_clock().now().to_msg()
                goal.pose.position.x = waypoint[0]
                goal.pose.position.y = waypoint[1]
                goal.pose.position.z = 0.0
                goal.pose.orientation = expRoute.euler_to_quaternion(waypoint[2])

                self.navigator.goToPose(goal)
                self.get_logger().info(f"Sent waypoint {self.current_waypoint_index + 1}: ({waypoint[0]}, {waypoint[1]})")

                while not self.navigator.isTaskComplete():
                    if self._cancel_requested:
                        self.get_logger().info('Navigation task was cancelled.')
                        goal_handle.canceled()
                        return FindUnit.Result(success=False)
                    feedback = self.navigator.getFeedback()
                    # if feedback:
                    #     self.get_logger().info(f'Current progress: {feedback.progress}')
                
                self.nav_result = self.navigator.getResult()

                if self.nav_result == TaskResult.SUCCEEDED:
                    self.get_logger().info("Arrived at target waypoint")
                    response = await self.get_unit_send_request(self.desire_unit_num)
                    if response and response.result:
                        self.is_task_success = True
                        break
                    else:
                        self.current_waypoint_index += 1
                        self.get_logger().info('Not the desired unit number, moving to next waypoint.')
                else:
                    self.get_logger().error(f'Failed to reach waypoint: {self.nav_result}')
                    self.current_waypoint_index += 1

            self.all_wp_send = True
            self.get_logger().info("All waypoints have been sent.")
            if self.is_task_success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
        except Exception as e:
            self.get_logger().error(f"Exception in exec_navigation: {str(e)}")
            goal_handle.abort()

        return FindUnit.Result(success=self.is_task_success)

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        self._reset_state()

        self.building_id = goal_handle.request.building_id
        self.desire_unit_num = goal_handle.request.unit_number
        self.cur_position = (goal_handle.request.cur_x, goal_handle.request.cur_y)
        self.get_logger().info(f"Exploration target: {self.building_id}, {self.desire_unit_num}, {self.cur_position}")

        try:
            self.waypoint_ = expRoute.execute_exploration(
                '/workspaces/light-map-navigation/src/llm_exploration_py/OSM/real1.osm',
                self.building_id, 6, 5, self.cur_position
            )
        except Exception as e:
            self.get_logger().error(f"Failed to get waypoints: {str(e)}")
            goal_handle.abort()
            return FindUnit.Result(success=False)

        return await self.exec_navigation(goal_handle)

def main(args=None):
    rclpy.init(args=args)
    llm_exploration_node = LlmExplorationNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(llm_exploration_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        llm_exploration_node.get_logger().info("Node interrupted by user")
    except Exception as e:
        llm_exploration_node.get_logger().error(f"Exception in main: {str(e)}")
    finally:
        llm_exploration_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()