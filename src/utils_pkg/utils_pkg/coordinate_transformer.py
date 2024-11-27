import math
from typing import Tuple
from pyproj import Transformer

class CoordinateTransformer:
    """Utility class for coordinate transformation between WGS84 and UTM coordinate systems"""
    
    @staticmethod
    def get_utm_epsg(lon: float, lat: float) -> int:
        """
        Calculate the EPSG code for UTM projection based on latitude and longitude
        
        Args:
            lon (float): Longitude in degrees
            lat (float): Latitude in degrees
            
        Returns:
            int: EPSG code for UTM projection
        """
        zone = math.floor((lon + 180) / 6) + 1
        base = 32600 if lat >= 0 else 32700
        return base + zone

    @staticmethod
    def wgs84_to_utm(lon: float, lat: float) -> Tuple[float, float, int]:
        """
        Convert WGS84 coordinates (longitude, latitude) to UTM coordinates (easting, northing)
        
        Args:
            lon (float): Longitude in degrees
            lat (float): Latitude in degrees
            
        Returns:
            Tuple[float, float, int]: (easting, northing, EPSG code) where coordinates are in meters
        """
        epsg = CoordinateTransformer.get_utm_epsg(lat, lon)
        epsg = 32633 # TODO: remove hardcode
        transformer = Transformer.from_proj('EPSG:4326', f'EPSG:{epsg}')
        easting, northing = transformer.transform(lat, lon)
        return (easting, northing, epsg)

    @staticmethod
    def utm_to_wgs84(easting: float, northing: float, epsg: int) -> Tuple[float, float]:
        """
        Convert UTM coordinates to WGS84 coordinates
        
        Args:
            easting (float): UTM easting coordinate in meters
            northing (float): UTM northing coordinate in meters
            epsg (int): EPSG code for UTM projection
            
        Returns:
            Tuple[float, float]: (longitude, latitude) in degrees
        """
        transformer = Transformer.from_proj(f'EPSG:{epsg}', 'EPSG:4326')
        lat, lon = transformer.transform(easting, northing)
        return (lon, lat)