import osmium as osm
from PIL import Image, ImageDraw
import math
import sys
import yaml
from pyproj import Proj
import os
import argparse


class CoordinateTransformer:
    @staticmethod
    def get_utm_epsg(lon, lat):
        zone_number = int((lon + 180) / 6) + 1
        epsg = f"326{zone_number}" if lat >= 0 else f"327{zone_number}"
        return epsg

    @staticmethod
    def wgs84_to_utm(lon, lat):
        utm_proj = Proj(proj='utm', zone=int((lon + 180) / 6) + 1, ellps='WGS84', south=lat < 0)
        x, y = utm_proj(lon, lat)
        return x, y, 0


class MapHandler(osm.SimpleHandler):
    def __init__(self, tags):
        osm.SimpleHandler.__init__(self)
        self.nodes = {}
        self.ways = []
        self.tags = tags
        self.min_x = self.min_y = float('inf')
        self.max_x = self.max_y = float('-inf')
        self.lats = []
        self.lons = []

    def node(self, n):
        try:
            lon, lat = n.location.lon, n.location.lat
            self.lats.append(lat)
            self.lons.append(lon)
            self.nodes[n.id] = (lon, lat)
        except Exception as e:
            print(f"Error processing node {n.id}: {e}")

    def way(self, w):
        if any(w.tags.get(tag) for tag in self.tags):
            valid_nodes = [node.ref for node in w.nodes if node.ref in self.nodes]
            if len(valid_nodes) > 1:
                self.ways.append(valid_nodes)

    def finalize(self):
        if not self.lats or not self.lons:
            print("Error: No valid nodes found.")
            sys.exit(1)

        center_lon = sum(self.lons) / len(self.lons)
        center_lat = sum(self.lats) / len(self.lats)
        epsg = CoordinateTransformer.get_utm_epsg(center_lon, center_lat)
        print(f"Map center: ({center_lon:.6f}, {center_lat:.6f})")
        print(f"UTM EPSG code: {epsg}")

        print("\nCoordinate bounds (WGS84):")
        print(f"Longitude: {min(self.lons):.6f} to {max(self.lons):.6f}")
        print(f"Latitude: {min(self.lats):.6f} to {max(self.lats):.6f}")

        for node_id, (lon, lat) in self.nodes.items():
            x, y, _ = CoordinateTransformer.wgs84_to_utm(lon, lat)
            self.nodes[node_id] = (x, y)
            self.min_x = min(self.min_x, x)
            self.min_y = min(self.min_y, y)
            self.max_x = max(self.max_x, x)
            self.max_y = max(self.max_y, y)

        print("\nCoordinate bounds (UTM):")
        print(f"X: {self.min_x:.2f}m to {self.max_x:.2f}m")
        print(f"Y: {self.min_y:.2f}m to {self.max_y:.2f}m")
        print(f"Map size: {self.max_x - self.min_x:.2f}m x {self.max_y - self.min_y:.2f}m")


def create_pgm_image(osm_file, pgm_file, meters_per_pixel, tags, padding_meters=10.0):
    """
    Convert OSM file to PGM image.
    
    Args:
        osm_file: Path to OSM file
        pgm_file: Output PGM file path
        meters_per_pixel: Resolution in meters per pixel
        tags: List of OSM tags to include (e.g. ['building', 'footway'])
        padding_meters: Padding in meters
        
    Returns:
        Tuple of (origin_x, origin_y, resolution)
    """
    handler = MapHandler(tags)
    handler.apply_file(osm_file)
    handler.finalize()

    # Calculate range with padding
    x_range_meters = (handler.max_x - handler.min_x) + (2 * padding_meters)
    y_range_meters = (handler.max_y - handler.min_y) + (2 * padding_meters)
    width = int(x_range_meters / meters_per_pixel)
    height = int(y_range_meters / meters_per_pixel)

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}. Check meters_per_pixel value.")

    print(f"Image size with padding: {width}x{height} pixels")
    image = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(image)

    # Convert padding distance to pixels
    padding_pixels = int(padding_meters / meters_per_pixel)

    for way in handler.ways:
        pixels = []
        for node_id in way:
            x, y = handler.nodes[node_id]
            # Add padding offset when calculating pixel coordinates
            px = ((x - handler.min_x) / meters_per_pixel) + padding_pixels
            py = (y_range_meters - (y - handler.min_y)) / meters_per_pixel - padding_pixels
            pixels.append((int(px), int(py)))

        if len(pixels) > 1:
            draw.line(pixels, fill=0)

    image.save(pgm_file)
    # Return new origin coordinates (considering padding)
    return handler.min_x - padding_meters, handler.min_y - padding_meters, meters_per_pixel


def create_yaml_file(yaml_file, pgm_file, origin, resolution):
    """
    Create YAML metadata file for PGM map.
    
    Args:
        yaml_file: Output YAML file path
        pgm_file: Path to PGM file
        origin: Tuple of (origin_x, origin_y)
        resolution: Map resolution in meters per pixel
    """
    data = {
        'image': os.path.basename(pgm_file),
        'resolution': resolution,
        'origin': [origin[0], origin[1], 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.25,
    }
    with open(yaml_file, 'w') as f:
        yaml.dump(data, f) 