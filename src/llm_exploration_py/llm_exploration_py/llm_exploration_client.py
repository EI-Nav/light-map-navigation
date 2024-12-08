import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from custom_interfaces.action import FindUnit

class FindUnitClient(Node):
    def __init__(self):
        super().__init__('find_unit_client')
        self._action_client = ActionClient(self, FindUnit, 'find_unit')

    def send_goal(self, building_id, unit_number, pos_x, pos_y):
        goal_msg = FindUnit.Goal()
        goal_msg.building_id = building_id
        goal_msg.unit_number = unit_number
        goal_msg.cur_x = pos_x
        goal_msg.cur_y = pos_y


        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        try:
            result = future.result().result
            self.get_logger().info(f'Result success: {result.success}, Message: {result.message}')
        except Exception as e:
            self.get_logger().error(f'Error in get_result_callback: {e}')
            rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: Progress: {feedback.progress:.2%}, Status: {feedback.status}')

def main(args=None):
    rclpy.init(args=args)
    client = FindUnitClient()
    target_building = 'building10'
    target_unit = '2'
    cur_x = 0.0
    cur_y = 0.0
    client.send_goal(target_building, target_unit, cur_x ,cur_y)
    rclpy.spin(client)

if __name__ == '__main__':
    main()
