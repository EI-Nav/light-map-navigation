import osmium as osm
import math
import sys
from pyproj import CRS, Transformer

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

def calculate_centroid(nodes, handler):
    total_x = 0
    total_y = 0
    num_nodes = len(nodes)

    for node_id in nodes:
        x, y = handler.nodes[node_id]
        total_x += x
        total_y += y

    centroid_x = total_x / num_nodes
    centroid_y = total_y / num_nodes
    return centroid_x, centroid_y

def print_ways(osm_file, tags, transformer):
    handler = MapHandler(tags, transformer)
    handler.apply_file(osm_file)

    # Define the inverse transformer for converting back to latitude and longitude
    inverse_transformer = Transformer.from_crs(transformer.target_crs, transformer.source_crs, always_xy=True)

    for idx, (way_name, way_nodes) in enumerate(handler.ways):
        centroid_x, centroid_y = calculate_centroid(way_nodes, handler)
        lon, lat = inverse_transformer.transform(centroid_x, centroid_y)
        print(f"{way_name}: ({lon:.9f},{lat:.9f})")

if __name__ == "__main__":
    osm_file = sys.argv[1]  # Input OSM file
    tags = ["building"]  # List of tags to process

    # Define coordinate reference systems
    wgs84 = CRS.from_epsg(4326)
    utm = CRS.from_epsg(32633)  # 33n
    #utm = CRS.from_epsg(32650) # 50n
    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)

    print_ways(osm_file, tags, transformer)