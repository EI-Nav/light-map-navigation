import argparse
import os
from typing import Tuple, List, Optional, Union

import numpy as np
import open3d as o3d
from PIL import Image


def pcd_to_pgm(
    pcd_file_path: str,
    pgm_file_path: str,
    yaml_file_path: str,
    resolution: float = 0.05,
    margins: Tuple[float, float, float, float] = (10.0, 10.0, 10.0, 10.0),
    occupied_threshold: int = 200,
    invert: bool = False,
    height_slice: Tuple[Optional[float], Optional[float]] = (None, None)
) -> Tuple[float, float]:
    """Convert PCD point cloud file to PGM map file.
    
    Args:
        pcd_file_path: Path to the PCD file
        pgm_file_path: Path to output PGM file
        yaml_file_path: Path to output YAML metadata file
        resolution: Map resolution in meters/pixel
        margins: Map margins [left, right, top, bottom] in meters
        occupied_threshold: Threshold for determining if a grid cell is occupied
        invert: Whether to invert map colors (swap black and white)
        height_slice: Optional height slice (min_z, max_z) to filter the point cloud
    
    Returns:
        Tuple of (origin_x, origin_y): Map origin coordinates in meters
    """
    # Read PCD file
    print(f"Reading PCD file: {pcd_file_path}")
    try:
        pcd = o3d.io.read_point_cloud(pcd_file_path)
        points = np.asarray(pcd.points)
    except Exception as e:
        print(f"Error reading PCD file: {e}")
        raise
    
    if len(points) == 0:
        raise ValueError("No points found in PCD file")
    
    # Apply height slice filtering
    points = _filter_by_height(points, height_slice)
    
    # Calculate map boundaries
    min_bounds, max_bounds = _calculate_point_cloud_bounds(points)
    min_x, min_y, min_z = min_bounds
    max_x, max_y, max_z = max_bounds
    
    # Calculate map dimensions with margins
    left_margin, right_margin, top_margin, bottom_margin = margins
    
    # Calculate map size (including margins)
    world_width = (max_x - min_x) + left_margin + right_margin
    world_height = (max_y - min_y) + top_margin + bottom_margin
    
    # Calculate map origin (bottom-left corner, including margins)
    origin_x = min_x - left_margin
    origin_y = min_y - bottom_margin
    
    # Calculate map dimensions (pixels)
    width_pixels = int(world_width / resolution)
    height_pixels = int(world_height / resolution)
    
    print(f"Map dimensions (pixels): {width_pixels}x{height_pixels}")
    print(f"Map resolution: {resolution} meters/pixel")
    
    # Create occupancy grid
    grid = _create_occupancy_grid(
        points, 
        origin_x, 
        origin_y, 
        width_pixels, 
        height_pixels, 
        resolution
    )
    
    # Convert grid to image
    pgm_image = _grid_to_image(grid, height_pixels, width_pixels, occupied_threshold, invert)
    
    # Save PGM image
    print(f"Saving PGM file: {pgm_file_path}")
    image = Image.fromarray(pgm_image)
    image.save(pgm_file_path)
    
    # Create and save YAML file
    _save_yaml_metadata(
        yaml_file_path, 
        pgm_file_path, 
        resolution, 
        origin_x, 
        origin_y
    )
    
    return origin_x, origin_y


def _filter_by_height(
    points: np.ndarray, 
    height_slice: Tuple[Optional[float], Optional[float]]
) -> np.ndarray:
    """Filter point cloud by height.
    
    Args:
        points: Point cloud as numpy array
        height_slice: Tuple of (min_z, max_z) height limits
        
    Returns:
        Filtered point cloud
    """
    min_z, max_z = height_slice
    
    if min_z is not None or max_z is not None:
        min_z = min_z if min_z is not None else float('-inf')
        max_z = max_z if max_z is not None else float('inf')
        
        mask = (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
        filtered_points = points[mask]
        print(f"After height slicing, {len(filtered_points)} points remain")
        
        if len(filtered_points) == 0:
            raise ValueError("No points remain after height slicing")
        
        return filtered_points
    
    return points


def _calculate_point_cloud_bounds(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate min and max bounds of point cloud.
    
    Args:
        points: Point cloud as numpy array
        
    Returns:
        Tuple of (min_bounds, max_bounds)
    """
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    
    min_x, min_y, min_z = min_bounds
    max_x, max_y, max_z = max_bounds
    
    print(f"Point cloud boundaries (meters):")
    print(f"X: {min_x:.3f} to {max_x:.3f}")
    print(f"Y: {min_y:.3f} to {max_y:.3f}")
    print(f"Z: {min_z:.3f} to {max_z:.3f}")
    
    return min_bounds, max_bounds


def _create_occupancy_grid(
    points: np.ndarray,
    origin_x: float,
    origin_y: float,
    width_pixels: int,
    height_pixels: int,
    resolution: float
) -> np.ndarray:
    """Create occupancy grid from point cloud.
    
    Args:
        points: Point cloud as numpy array
        origin_x: X-coordinate of map origin
        origin_y: Y-coordinate of map origin
        width_pixels: Width of map in pixels
        height_pixels: Height of map in pixels
        resolution: Map resolution in meters/pixel
        
    Returns:
        Occupancy grid as numpy array
    """
    grid = np.zeros((height_pixels, width_pixels), dtype=np.int32)
    
    # Map 3D points to 2D grid
    for point in points:
        # Convert to pixel coordinates
        px = int((point[0] - origin_x) / resolution)
        py = int((point[1] - origin_y) / resolution)
        
        # Check if point is within map bounds
        if 0 <= px < width_pixels and 0 <= py < height_pixels:
            # Increment count for this cell
            grid[py, px] += 1
    
    return grid


def _grid_to_image(
    grid: np.ndarray,
    height_pixels: int,
    width_pixels: int,
    occupied_threshold: int,
    invert: bool
) -> np.ndarray:
    """Convert occupancy grid to PGM image.
    
    Args:
        grid: Occupancy grid as numpy array
        height_pixels: Height of map in pixels
        width_pixels: Width of map in pixels
        occupied_threshold: Threshold for determining if a grid cell is occupied
        invert: Whether to invert map colors
        
    Returns:
        PGM image as numpy array
    """
    pgm_image = np.zeros((height_pixels, width_pixels), dtype=np.uint8)
    
    # Determine occupied cells based on threshold
    occupied = grid >= occupied_threshold
    
    # Set colors for occupied and free areas
    if invert:
        # Occupied areas are white (255), free areas are black (0)
        pgm_image[occupied] = 0
        pgm_image[~occupied] = 255
    else:
        # Occupied areas are black (0), free areas are white (255)
        pgm_image[occupied] = 0
        pgm_image[~occupied] = 255
    
    # In PGM coordinate system, origin is at top-left, but we need to flip
    # the map to have the origin at bottom-left
    pgm_image = np.flipud(pgm_image)
    
    return pgm_image


def _save_yaml_metadata(
    yaml_file_path: str,
    pgm_file_path: str,
    resolution: float,
    origin_x: float,
    origin_y: float
) -> None:
    """Save YAML metadata file for the map.
    
    Args:
        yaml_file_path: Path to output YAML file
        pgm_file_path: Path to PGM file
        resolution: Map resolution in meters/pixel
        origin_x: X-coordinate of map origin
        origin_y: Y-coordinate of map origin
    """
    print(f"Saving YAML file: {yaml_file_path}")
    
    with open(yaml_file_path, 'w') as f:
        # Write YAML content in fixed format
        f.write(f"free_thresh: 0.25\n")
        f.write(f"image: {os.path.basename(pgm_file_path)}\n")
        f.write(f"negate: 0\n")
        f.write(f"occupied_thresh: 0.65\n")
        f.write(f"origin: \n")
        f.write(f"- {origin_x}\n")
        f.write(f"- {origin_y}\n")
        f.write(f"- 0.0\n")
        f.write(f"resolution: {resolution}\n") 