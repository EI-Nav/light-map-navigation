import osmium
from typing import Dict, List, Tuple, Optional
from pyproj import Transformer
import math

class OSMHandler(osmium.SimpleHandler):
    """
    A handler for processing OpenStreetMap data.
    Provides functionality to extract and analyze nodes, ways, and their properties.
    """
    def __init__(self, way_types: Dict[str, str] = None):
        """
        Initialize the OSM handler.
        
        Args:
            way_types: Dictionary mapping OSM tags to custom type names.
                      e.g., {'building': 'building', 'highway': 'road'}
        """
        super(OSMHandler, self).__init__()
        self.nodes = 0
        self.ways = 0
        self.way_types = way_types or {'building': 'building', 'highway': 'road'}
        self.way_counts = {type_name: 0 for type_name in self.way_types.values()}
        
        self.nodes_locations: Dict[int, Tuple[float, float]] = {}
        self.tagged_nodes: List[Dict] = []
        self.ways_info: List[Dict] = []

    def node(self, n):
        """Process each OSM node."""
        self.nodes += 1
        self.nodes_locations[n.id] = (n.location.lat, n.location.lon)
        
        if len(n.tags) > 0:
            node_info = {
                'id': n.id,
                'lat': n.location.lat,
                'lon': n.location.lon,
                'tags': dict(n.tags)
            }
            self.tagged_nodes.append(node_info)

    def way(self, w):
        """Process each OSM way."""
        self.ways += 1
        way_info = {
            'id': w.id,
            'tags': dict(w.tags),
            'nodes': [n.ref for n in w.nodes]
        }
        
        for tag_key, type_name in self.way_types.items():
            if tag_key in w.tags:
                self.way_counts[type_name] += 1
                way_info['type'] = type_name
                
        self.ways_info.append(way_info)

    def get_node_location(self, node_id: int) -> Optional[Tuple[float, float]]:
        """
        Get the location of a node by its ID.
        
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        return self.nodes_locations.get(node_id)

    def find_nodes_by_tag(self, tag_key: str, tag_value: str) -> List[Dict]:
        """
        Find all nodes that match a specific tag key-value pair.
        
        Returns:
            List of matching node dictionaries
        """
        return [
            node for node in self.tagged_nodes
            if tag_key in node['tags'] and node['tags'][tag_key] == tag_value
        ]

    def get_way_nodes_locations(self, way_id: int) -> List[Tuple[float, float]]:
        """
        Get all node locations for a specific way.
        
        Returns:
            List of (latitude, longitude) tuples
        """
        way = next((w for w in self.ways_info if w['id'] == way_id), None)
        if not way:
            return []
        
        return [
            self.nodes_locations[node_id]
            for node_id in way['nodes']
            if node_id in self.nodes_locations
        ]

    def get_way_center(self, way_id: int) -> Optional[Tuple[float, float]]:
        """
        Calculate the center point of a way (average of all nodes).
        
        Returns:
            Tuple of (latitude, longitude) or None if way not found
        """
        locations = self.get_way_nodes_locations(way_id)
        if not locations:
            return None
        
        lat_sum = sum(lat for lat, _ in locations)
        lon_sum = sum(lon for _, lon in locations)
        count = len(locations)
        
        return (lat_sum / count, lon_sum / count)

    def get_way_center_by_name(self, way_name: str) -> List[Tuple[float, float]]:
        """
        Find center points of all ways matching a specific name.
        
        Returns:
            List of (latitude, longitude) tuples for matching ways
        """
        centers = []
        for way in self.ways_info:
            if 'name' in way['tags'] and way['tags']['name'] == way_name:
                center = self.get_way_center(way['id'])
                if center:
                    centers.append(center)
        return centers

    def get_node_location_by_name(self, node_name: str) -> List[Tuple[float, float]]:
        """
        Find locations of all nodes matching a specific name.
        
        Returns:
            List of (latitude, longitude) tuples for matching nodes
        """
        locations = []
        for node in self.tagged_nodes:
            if 'name' in node['tags'] and node['tags']['name'] == node_name:
                locations.append((node['lat'], node['lon']))
        return locations