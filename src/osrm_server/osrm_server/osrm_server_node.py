#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from custom_interfaces.srv import GetWaypoints
import requests
from math import radians, cos, sin, atan2
import geometry_msgs.msg
from pyproj import CRS, Transformer


# 将 Transformer 提取到全局变量中
wgs84 = CRS.from_epsg(4326)
utm = CRS.from_epsg(32633)  # 33n
#utm = CRS.from_epsg(32650) # 50n
transformer = Transformer.from_crs(wgs84, utm, always_xy=True)


def get_route(start, end):
    """Fetch route from OSRM and convert to waypoints in UTM Zone 33N."""

    osrm_url = "http://waytous.tajiyu.com:5001/route/v1/driving/"

    # Construct request URL for OSRM
    request_url = f"{osrm_url}{start};{end}"

    # Request parameters
    params = {
        "overview": "full",  # Return full path
        "geometries": "geojson"  # Return path in GeoJSON format
    }

    try:
        # Send request to OSRM
        response = requests.get(request_url, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")
        return []

    waypoints = []

    # Convert start and end points from WGS84 to UTM Zone 33N
    start_lon, start_lat = map(float, start.split(","))
    end_lon, end_lat = map(float, end.split(","))
    start_x, start_y = wgs84_to_utm(start_lon, start_lat)
    end_x, end_y = wgs84_to_utm(end_lon, end_lat)

    # Add route waypoints
    if data.get('routes'):
        route = data['routes'][0]
        coordinates = route['geometry']['coordinates']
        coordinates.append([end_lon, end_lat])

        for i in range(len(coordinates) - 1):
            lon1, lat1 = coordinates[i]
            lon2, lat2 = coordinates[i + 1]

            utm_x1, utm_y1 = wgs84_to_utm(lon1, lat1)
            utm_x2, utm_y2 = wgs84_to_utm(lon2, lat2)

            print(f"{utm_x1} {utm_y1} {3} 0")

            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = utm_x1
            pose.pose.position.y = utm_y1
            pose.pose.position.z = 0.0

            # Calculate orientation
            yaw = atan2(utm_y2 - utm_y1, utm_x2 - utm_x1)
            q = euler_to_quaternion(yaw)
            pose.pose.orientation = q

            waypoints.append(pose)

    print(f"{end_x} {end_y} {3} 0")

    if not waypoints:
        print("No valid navigation path found")

    return waypoints


def euler_to_quaternion(yaw):
    """Convert yaw angle to quaternion."""

    q = geometry_msgs.msg.Quaternion()
    q.w = cos(yaw / 2)
    q.x = 0.0
    q.y = 0.0
    q.z = sin(yaw / 2)
    return q


def create_pose_stamped(navigator, position_x, position_y, orientation_z):
    """Create a PoseStamped message for the given position and orientation."""

    q = euler_to_quaternion(radians(orientation_z))
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = position_x
    pose.pose.position.y = position_y
    pose.pose.position.z = 0.0
    pose.pose.orientation = q
    return pose


def wgs84_to_utm(lon, lat):
    """Convert WGS84 coordinates to UTM Zone 33N."""

    x, y = transformer.transform(lon, lat)
    return x, y


class WaypointService(Node):
    def __init__(self):
        super().__init__('waypoint_service')
        self.srv = self.create_service(GetWaypoints, '~/get_waypoints', self.get_waypoints_callback)

    def get_waypoints_callback(self, request, response):
        start = request.start
        end = request.end
        waypoints = get_route(start, end)

        if waypoints:
            response.waypoints = waypoints
            response.success = True
        else:
            response.success = False

        return response


def main():
    rclpy.init()
    node = WaypointService()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()