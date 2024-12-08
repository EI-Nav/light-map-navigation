import osmium as osm
import sys
from pyproj import CRS, Transformer
import xml.etree.ElementTree as ET
from collections import defaultdict
import math
import re

class MapHandler(osm.SimpleHandler):
    def __init__(self, tags, transformer):
        osm.SimpleHandler.__init__(self)
        self.nodes = {}
        self.ways = []
        self.tags = tags
        self.transformer = transformer

    def node(self, n):
        lon, lat = n.location.lon, n.location.lat
        x, y = self.transformer.transform(lon, lat)
        self.nodes[n.id] = (x, y)

    def way(self, w):
        if any(tag in w.tags for tag in self.tags):
            valid_nodes = [node.ref for node in w.nodes if node.ref in self.nodes]
            if len(valid_nodes) > 1:
                way_name = w.tags.get('name', f'building{w.id}')
                self.ways.append((way_name, valid_nodes))

def get_all_matching_model_positions(sdf_file, text_model):
    tree = ET.parse(sdf_file)
    root = tree.getroot()
    matching_models = []

    for model in root.findall(".//model"):
        model_name = model.get('name')
        if text_model in model_name:
            pose = model.find('pose')
            if pose is not None:
                position = pose.text.strip().split()
                x, y, z = float(position[0]), float(position[1]), float(position[2])

                if z > 0:
                    matching_models.append((model_name, (x, y, z)))

    return matching_models

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_centroid(building_nodes, nodes):
    x_coords = [nodes[node_id][0] for node_id in building_nodes]
    y_coords = [nodes[node_id][1] for node_id in building_nodes]
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    return centroid_x, centroid_y

def assign_models_to_buildings(buildings, models_positions, handler):
    building_assignments = defaultdict(list)
    building_centroids = {}

    # Calculate centroids for all buildings
    for building_name, building_nodes in buildings:
        centroid_x, centroid_y = calculate_centroid(building_nodes, handler.nodes)
        building_centroids[building_name] = (centroid_x, centroid_y)

    # Assign models to the nearest building based on centroid distance
    for model_name, (model_x, model_y, _) in models_positions:
        min_distance = float('inf')
        closest_building = None

        for building_name, (centroid_x, centroid_y) in building_centroids.items():
            distance = calculate_distance(model_x, model_y, centroid_x, centroid_y)

            if distance < min_distance:
                min_distance = distance
                closest_building = building_name

        if closest_building:
            building_assignments[closest_building].append((model_name, (model_x, model_y)))

    return building_assignments

def replace_model_names(models):
    replaced_models = []
    for model_name, _ in models:
        if 'text_model1' in model_name:
            replaced_models.append('unit1')
        elif 'text_model2' in model_name:
            replaced_models.append('unit2')
        elif 'text_model3' in model_name:
            replaced_models.append('unit3')
        else:
            replaced_models.append(model_name)
    return replaced_models

def print_building_assignments(building_assignments):
    sorted_buildings = sorted(building_assignments.items(), key=lambda x: int(re.search(r'\d+', x[0]).group()))

    for building, models in sorted_buildings:
        replaced_models = replace_model_names([(name, pos) for name, pos in models])
        model_positions = [f"({pos[0]:.2f}, {pos[1]:.2f})" for _, pos in models]
        print(f'"{building}": {replaced_models},')
    
    for building, models in sorted_buildings:
        replaced_models = replace_model_names([(name, pos) for name, pos in models])
        model_positions = [f"({pos[0]:.2f}, {pos[1]:.2f})" for _, pos in models]
        print(f'"{building}_coords": {model_positions},')

if __name__ == "__main__":
    osm_file = sys.argv[1]  # Input OSM file
    sdf_file = sys.argv[2]  # Input SDF file
    text_model = sys.argv[3]  # Text to match model names

    tags = ["building"]  # List of tags to process

    wgs84 = CRS.from_epsg(4326)
    utm = CRS.from_epsg(32633)  # 33n
    #utm = CRS.from_epsg(32650) # 50n
    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)

    handler = MapHandler(tags, transformer)
    handler.apply_file(osm_file)

    matching_models_positions = get_all_matching_model_positions(sdf_file, text_model)

    if matching_models_positions:
        building_assignments = assign_models_to_buildings(handler.ways, matching_models_positions, handler)
        print_building_assignments(building_assignments)
    else:
        print(f"No models found containing '{text_model}' with z > 0")