import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from custom_interfaces.srv import Wpcalcu, GetIPMContourPoints, GetConcaveHull
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
import numpy as np

class ImageSegmentationCoordinator(Node):
    def __init__(self):
        super().__init__('image_segmentation_coordinator')
        
        # Initialize position storage
        self.robot_position = None
        
        # Create service clients
        self.segment_client = self.create_client(Trigger, 'segment_image')
        self.ipm_client = self.create_client(GetIPMContourPoints, 'get_ipm_contour_points')
        self.concave_hull_client = self.create_client(GetConcaveHull, 'compute_concave_hull')
        
        # Create a service for coordinate segmentation
        self.srv = self.create_service(Wpcalcu, 'coordinate_segmentation', self.handle_segmentation_request)
        
        # Odometry subscriber to get the robot's position
        self.odom_subscription = self.create_subscription(Odometry, '/Odometry', self.odometry_callback, 10)

        # Wait for services to be available
        self.wait_for_services()

    def wait_for_services(self):
        # Waiting for segment_image, IPM, and ConcaveHull services
        while not self.segment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for segment_image service...')
        while not self.ipm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for IPM service...')
        while not self.concave_hull_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for ConcaveHull service...')

    def odometry_callback(self, msg):
        """Callback function for odometry updates to store the robot's current position."""
        self.robot_position = msg.pose.pose.position
        self.get_logger().info(f"Updated robot position: {self.robot_position.x}, {self.robot_position.y}")

    def handle_segmentation_request(self, request, response):
        """Handles requests to this service by calling required services and calculating waypoint."""
        self.get_logger().info('Received segmentation request.')
        
        # Step 1: Call segmentation service
        segmentation_request = Trigger.Request()
        future = self.segment_client.call_async(segmentation_request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None and future.result().success:
            self.get_logger().info("Segmentation service call was successful.")
            
            # Step 2: Call IPM service to get point cloud
            ipm_request = GetIPMContourPoints.Request()
            ipm_future = self.ipm_client.call_async(ipm_request)
            rclpy.spin_until_future_complete(self, ipm_future)
            
            if ipm_future.result() is not None:
                ipm_point_cloud = ipm_future.result().point_cloud
                self.get_logger().info("Received PointCloud2 data from IPM service.")

                # Step 3: Call ConcaveHull service
                concave_hull_request = GetConcaveHull.Request()
                concave_hull_request.point_cloud = ipm_point_cloud
                concave_hull_future = self.concave_hull_client.call_async(concave_hull_request)
                rclpy.spin_until_future_complete(self, concave_hull_future)
                
                if concave_hull_future.result() is not None:
                    concave_hull = concave_hull_future.result().concave_hull
                    line_segment = concave_hull_future.result().line_segment
                    self.get_logger().info("Received ConcaveHull data.")
                    
                    # Step 4: Calculate waypoint based on concave hull and line segment
                    waypoint = self.calculate_next_waypoint(concave_hull, line_segment)
                    if waypoint:
                        response.wp = waypoint  # Set calculated waypoint in response
                    else:
                        self.get_logger().error("Failed to calculate a valid waypoint.")
                else:
                    self.get_logger().error("Failed to get response from ConcaveHull service.")
            else:
                self.get_logger().error("Failed to get response from IPM service.")
        else:
            self.get_logger().error("Segmentation service call failed.")
        
        return response

    def calculate_next_waypoint(self, concave_hull, line_segment):
        """Calculate next waypoint based on the robot's position, concave hull, and line segment."""
        if self.robot_position is None:
            self.get_logger().warn("Robot position is not available yet.")
            return None
        
        if not line_segment or len(line_segment.points) < 2:
            self.get_logger().warn("Line segment data is invalid.")
            return None
        
        # Use the start and end points of the line segment
        line_start, line_end = line_segment.points[0], line_segment.points[1]
        
        # Calculate slope between robot position and target point
        slope = (line_end.y - line_start.y) / (line_end.x - line_start.x) if line_end.x - line_start.x != 0 else float('inf')
        
        # Calculate intersection or closest point based on the line segment and concave hull
        intersection_point = self.calculate_line_intersection((self.robot_position.x, self.robot_position.y), slope, (line_start.x, line_start.y), (line_end.x, line_end.y))
        
        if intersection_point:
            wp_point = Point(x=intersection_point[0], y=intersection_point[1], z=0.0)
            
            # Verify if the point is inside the concave hull polygon
            if self.is_point_in_polygon(wp_point, concave_hull.points):
                return self.convert_point_to_pose_stamped(wp_point)
        
        self.get_logger().warn("No valid waypoint calculated.")
        return None

    def calculate_line_intersection(self, robot_pos, slope, line_start, line_end):
        """Calculate intersection point between the robot's direction line and line segment."""
        x1, y1 = line_start
        x2, y2 = line_end
        if x2 - x1 == 0:
            x_intersection = x2
            y_intersection = slope * (x_intersection - robot_pos[0]) + robot_pos[1]
            return (x_intersection, y_intersection) if min(y1, y2) <= y_intersection <= max(y1, y2) else None
        line_slope = (y2 - y1) / (x2 - x1)
        if slope != line_slope:
            x_intersection = (line_slope * x1 - slope * robot_pos[0] + robot_pos[1] - y1) / (line_slope - slope)
            y_intersection = slope * (x_intersection - robot_pos[0]) + robot_pos[1]
            if min(x1, x2) <= x_intersection <= max(x1, x2):
                return (x_intersection, y_intersection)
        return None

    def is_point_in_polygon(self, point, polygon):
        """Determine if a point is within a polygon."""
        n = len(polygon)
        inside = False
        xinters = 0
        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            if point.y > min(p1y, p2y):
                if point.y <= max(p1y, p2y):
                    if point.x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point.y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point.x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def convert_point_to_pose_stamped(self, point):
        """Convert Point to PoseStamped."""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position = point
        pose_stamped.pose.orientation.w = 1.0
        return pose_stamped

def main(args=None):
    rclpy.init(args=args)
    segmentation_coordinator = ImageSegmentationCoordinator()
    rclpy.spin(segmentation_coordinator)
    segmentation_coordinator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
