import osmium as osm
import sys
from pyproj import CRS, Transformer
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

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
                way_name = w.tags.get('name', 'Unnamed Way')
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

def plot_buildings_and_models(buildings, models_positions, transformer):
    plt.figure(figsize=(8, 6))

    for _, building_nodes in buildings:
        x_coords = [transformer.nodes[node_id][0] for node_id in building_nodes]
        y_coords = [transformer.nodes[node_id][1] for node_id in building_nodes]
        plt.plot(x_coords, y_coords, 'r-', linewidth=2)

    x_coords_models = [position[0] for _, position in models_positions]
    y_coords_models = [position[1] for _, position in models_positions]

    plt.scatter(x_coords_models, y_coords_models, c='blue', marker='o')

    for model_name, (x, y, _) in models_positions:
        plt.text(x, y, model_name, fontsize=9, ha='right')

    plt.title("Building Boundaries and Model Positions")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

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
        plot_buildings_and_models(handler.ways, matching_models_positions, handler)
    else:
        print(f"No models found containing '{text_model}' with z > 0")