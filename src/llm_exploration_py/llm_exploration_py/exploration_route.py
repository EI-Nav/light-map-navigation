#!/usr/bin/env python3

import osmium as osm
import sys
from pyproj import CRS, Transformer
import pyclipper
import numpy as np
from math import radians, cos, sin, atan2, sqrt
import geometry_msgs.msg

real_world_flag = False

class MapHandler(osm.SimpleHandler):
    def __init__(self, tags, transformer, target_name):
        osm.SimpleHandler.__init__(self)
        self.nodes = {}
        self.ways = []
        self.tags = tags
        self.transformer = transformer
        self.target_name = target_name
        self.target_nodes = []

    def node(self, n):
        lon, lat = n.location.lon, n.location.lat
        x, y = self.transformer.transform(lon, lat)
        self.nodes[n.id] = (x, y)

    def way(self, w):
        if any(tag in w.tags for tag in self.tags):
            way_name = w.tags.get('name', 'Unnamed Way')
            valid_nodes = [node.ref for node in w.nodes if node.ref in self.nodes]
            if len(valid_nodes) > 1:
                self.ways.append((way_name, valid_nodes))
                if way_name == self.target_name:
                    self.target_nodes = valid_nodes

def inflate_building(coordinates, offset_distance):
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(coordinates, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    inflated_polygon = pco.Execute(offset_distance)
    return inflated_polygon[0] if inflated_polygon else coordinates

def sample_waypoints_evenly(coordinates, distance):
    total_length = 0
    num_points = len(coordinates)
    for i in range(num_points):
        start = np.array(coordinates[i])
        end = np.array(coordinates[(i + 1) % num_points])
        total_length += np.linalg.norm(end - start)

    sampled_waypoints = []
    accumulated_distance = 0

    for i in range(num_points):
        start = np.array(coordinates[i])
        end = np.array(coordinates[(i + 1) % num_points])
        segment_length = np.linalg.norm(end - start)

        while accumulated_distance + segment_length >= len(sampled_waypoints) * distance:
            ratio = (len(sampled_waypoints) * distance - accumulated_distance) / segment_length
            new_point = start + ratio * (end - start)
            sampled_waypoints.append(tuple(new_point))

        accumulated_distance += segment_length

    return sampled_waypoints

def calculate_yaw(waypoint, target):
    vector_to_target = np.array(target) - np.array(waypoint)
    yaw = atan2(vector_to_target[1], vector_to_target[0])
    return yaw

def calculate_orientations(all_waypoints, center):
    orientations = []
    for wp in all_waypoints:
        yaw = calculate_yaw(wp, center)
        orientations.append(yaw)
    return orientations

def get_closest_waypoint_index(waypoints, robot_position):
    distances = [sqrt((x - robot_position[0])**2 + (y - robot_position[1])**2) for x, y, yaw in waypoints]
    closest_index = distances.index(min(distances))
    return closest_index

def reorder_waypoints(waypoints, start_index):
    return waypoints[start_index:] + waypoints[:start_index]

def print_ways(osm_file, tags, transformer, target_name, offset_distance, additional_distance, robot_position):
    handler = MapHandler(tags, transformer, target_name)
    handler.apply_file(osm_file)
    if handler.target_nodes:
        coordinates = []
        for node_id in handler.target_nodes:
            x, y = handler.nodes[node_id]
            if real_world_flag is True:
                transform_matrix = np.array([[1.0, 0.0, -449920.549610], [0.0, 1.0, -4424638.431542], [0.000000, 0.000000, 1.000000]])
                point = np.array([x, y, 1])
                trans_point = np.dot(transform_matrix, point)
                x, y = trans_point[:2]
            coordinates.append((x, y))
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])
        inflated_building = inflate_building(coordinates, offset_distance)
        evenly_sampled_waypoints = sample_waypoints_evenly(inflated_building, additional_distance)
        center = np.mean(coordinates, axis=0)
        orientations = calculate_orientations(evenly_sampled_waypoints, center)
        waypoints_with_yaw = [(x, y, yaw) for (x, y), yaw in zip(evenly_sampled_waypoints, orientations)]
        
        closest_index = get_closest_waypoint_index(waypoints_with_yaw, robot_position)
        waypoints_with_yaw = reorder_waypoints(waypoints_with_yaw, closest_index)
        
        return waypoints_with_yaw
    else:
        print(f"No building named '{target_name}' found.")
        return []

def execute_exploration(osm_file, target_name, offset_distance, additional_distance, robot_position):
    if real_world_flag is True:
        utm = CRS.from_epsg(32650) # 50n
    else:
        utm = CRS.from_epsg(32633)  # 33n
    wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)

    waypoints_with_yaw = print_ways(osm_file, ["building"], transformer, target_name, offset_distance, additional_distance, robot_position)

    if not waypoints_with_yaw:
        print("No valid waypoints found.")
        return []

    return waypoints_with_yaw

def euler_to_quaternion(yaw):
    q = geometry_msgs.msg.Quaternion()
    q.w = cos(yaw / 2)
    q.x = 0.0
    q.y = 0.0
    q.z = sin(yaw / 2)
    return q
    

if __name__ == '__main__':
    osm_file = sys.argv[1]
    target_name = sys.argv[2]
    offset_distance = float(sys.argv[3])
    additional_distance = float(sys.argv[4])
    robot_position = (float(sys.argv[5]), float(sys.argv[6]))
    
    waypoints = execute_exploration(osm_file, target_name, offset_distance, additional_distance, robot_position)
    
    if waypoints:
        print("\nWaypoints with yaw:")
        for i, (x, y, yaw) in enumerate(waypoints):
            print(f"Waypoint {i + 1}: ({x:.9f}, {y:.9f}, {yaw:.9f})")
