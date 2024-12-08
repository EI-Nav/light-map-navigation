#!/usr/bin/env python3

import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped, Quaternion
import requests
import tf2_ros
from math import radians, cos, sin, atan2
import geometry_msgs.msg
from pyproj import CRS, Transformer
# import matplotlib.pyplot as plt
import numpy as np

real_world_flag = False

def get_route(start, end, plot_route=False):
    """Fetch route from OSRM and convert to waypoints in UTM Zone. Optionally plot the route."""
    
    if real_world_flag is False:
        # for large
        osrm_url = "http://101.200.33.217:30447/route/v1/driving/"
        # for medium
        # osrm_url = "http://101.200.33.217:30457/route/v1/driving/"
    else:
        osrm_url = "http://101.200.33.217:30437/route/v1/driving/"
    
    # Construct request URL for OSRM
    request_url = f"{osrm_url}{start};{end}"
    
    # Request parameters
    params = {
        "overview": "full",  # Return full path
        "geometries": "geojson"  # Return path in GeoJSON format
    }
    
    # Send request to OSRM
    response = requests.get(request_url, params=params)
    data = response.json()
    
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
        
        if plot_route:
            plot_path_in_utm(coordinates)  # Plot the path in UTM coordinates if requested
        
        for i in range(len(coordinates) - 1):
            lon1, lat1 = coordinates[i]
            lon2, lat2 = coordinates[i + 1]
            
            utm_x1, utm_y1 = wgs84_to_utm(lon1, lat1)
            utm_x2, utm_y2 = wgs84_to_utm(lon2, lat2)

            if real_world_flag is True:
                transform_matrix = np.array([[1.0, 0.0, -449920.549610], [0.0, 1.0, -4424638.431542], [0.000000, 0.000000, 1.000000]])
                point1 = np.array([utm_x1, utm_y1, 1])
                point2 = np.array([utm_x2, utm_y2, 1])
                point1 = np.dot(transform_matrix, point1)
                point2 = np.dot(transform_matrix, point2)
                utm_x1, utm_y1 = point1[:2]
                utm_x2, utm_y2 = point2[:2]
            
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
    
    if not waypoints:
        print("No valid navigation path found")
    
    return waypoints

def plot_path_in_utm(coordinates):
    """Plot the path on a 2D plot using UTM coordinates."""
    utm_x, utm_y = [], []
    
    for lon, lat in coordinates:
        x, y = wgs84_to_utm(lon, lat)

        transform_matrix = np.array([[1.0, 0.0, -449920.549610], [0.0, 1.0, -4424638.431542], [0.000000, 0.000000, 1.000000]])
        point1 = np.array([x, y, 1])
        point1 = np.dot(transform_matrix, point1)
        x, y = point1[:2]

        utm_x.append(x)
        utm_y.append(y)
    
    # plt.figure(figsize=(10, 8))
    # plt.plot(utm_x, utm_y, marker='o', color='blue', linewidth=2, markersize=5)
    # plt.title("Navigation Path in UTM Coordinates")
    # plt.xlabel("UTM X (meters)")
    # plt.ylabel("UTM Y (meters)")
    # plt.grid(True)
    # plt.show()

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
    
    wgs84 = CRS.from_epsg(4326)
    if real_world_flag is False:
        utm = CRS.from_epsg(32633)  # UTM Zone 33N
    else:
        utm = CRS.from_epsg(32650)  # UTM Zone 50N
    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

def main():
    """Main function to initialize ROS2, get route, and follow waypoints."""
    
    rclpy.init()
    navigator = BasicNavigator()

    start = "10.510999655,0.000264656"
    end = "10.511513114,-0.000091084"
    waypoints = get_route(start, end, plot_route=True)  # Enable route plotting

    print(f"Number of waypoints: {len(waypoints)}")

    if waypoints:
        navigator.followWaypoints(waypoints)
        while not navigator.isTaskComplete():
            feedback = navigator.getFeedback()
            # Uncomment to print feedback
            # print(feedback)

        print(navigator.getResult())
    else:
        print("Unable to get a valid navigation path")

    rclpy.shutdown()

if __name__ == '__main__':
    main()