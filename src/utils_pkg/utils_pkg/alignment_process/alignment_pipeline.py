#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import yaml
import subprocess
import tempfile
import datetime
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

# 修改导入方式为绝对导入
try:
    # 先尝试相对导入 (当作为包的一部分导入时)
    from .osm_converter import create_pgm_image, create_yaml_file
    from .pcd_converter import pcd_to_pgm
    from .pgm_aligner import PGMCoordinateAligner
    from .transform import rigid_transform_from_matrix
except ImportError:
    # 如果相对导入失败，尝试绝对导入 (当直接运行脚本时)
    from src.utils_pkg.utils_pkg.alignment_process.osm_converter import create_pgm_image, create_yaml_file
    from src.utils_pkg.utils_pkg.alignment_process.pcd_converter import pcd_to_pgm
    from src.utils_pkg.utils_pkg.alignment_process.pgm_aligner import PGMCoordinateAligner
    from src.utils_pkg.utils_pkg.alignment_process.transform import rigid_transform_from_matrix


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Integrate OSM and PCD map files')
    parser.add_argument('--osm_file', required=True, help='Input OSM file path')
    parser.add_argument('--pcd_file', required=True, help='Input PCD file path')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--resolution', type=float, default=0.2, help='Map resolution (meters/pixel)')
    parser.add_argument('--tags', default='building,footway', help='OSM tags, comma-separated')
    parser.add_argument('--padding', type=float, default=10.0, help='Map padding (meters)')
    parser.add_argument('--min_z', type=float, default=None, help='PCD point cloud minimum height filter (meters)')
    parser.add_argument('--max_z', type=float, default=None, help='PCD point cloud maximum height filter (meters)')
    parser.add_argument('--margins', type=float, nargs=4, default=[10.0, 10.0, 10.0, 10.0], 
                        help='Transformed map margins [left right top bottom] in meters')
    parser.add_argument('--interactive', action='store_true', help='Use interactive mode to align maps')
    parser.add_argument('--skip_alignment', action='store_true', help='Skip alignment step, use existing transformation')
    
    return parser.parse_args()


def setup_output_dirs(output_dir: str) -> Tuple[str, str]:
    """Create output directory structure.
    
    Args:
        output_dir: Base output directory path
        
    Returns:
        Tuple containing paths to temporary and final output directories
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create intermediate files and final output directories
    temp_dir = os.path.join(output_dir, 'temp')
    final_dir = os.path.join(output_dir, 'final')
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    return temp_dir, final_dir


def convert_osm(
    osm_file: str, 
    output_dir: str, 
    resolution: float, 
    tags: str, 
    padding: float
) -> Tuple[str, str]:
    """Convert OSM file to PGM format.
    
    Args:
        osm_file: Path to input OSM file
        output_dir: Directory to save output files
        resolution: Map resolution in meters/pixel
        tags: Comma-separated OSM tags to extract
        padding: Map padding in meters
        
    Returns:
        Tuple of (pgm_file_path, yaml_file_path)
    """
    print(f"Converting OSM file: {osm_file}")
    
    # Set output file paths
    pgm_file = os.path.join(output_dir, 'osm_map.pgm')
    yaml_file = os.path.join(output_dir, 'osm_map.yaml')
    
    # Convert tags string to list
    tag_list = tags.split(',')
    
    # Call OSM conversion function
    origin_x, origin_y, res = create_pgm_image(
        osm_file, 
        pgm_file, 
        resolution, 
        tag_list, 
        padding
    )
    
    # Create YAML metadata file
    create_yaml_file(yaml_file, pgm_file, (origin_x, origin_y), resolution)
    
    print(f"OSM conversion completed: {pgm_file}")
    return pgm_file, yaml_file


def convert_pcd(
    pcd_file: str, 
    output_dir: str, 
    resolution: float, 
    min_z: Optional[float], 
    max_z: Optional[float], 
    margins: List[float]
) -> Tuple[str, str]:
    """Convert PCD file to PGM format.
    
    Args:
        pcd_file: Path to input PCD file
        output_dir: Directory to save output files
        resolution: Map resolution in meters/pixel
        min_z: Minimum height filter (meters), None to disable
        max_z: Maximum height filter (meters), None to disable
        margins: Map margins [left, right, top, bottom] in meters
        
    Returns:
        Tuple of (pgm_file_path, yaml_file_path)
    """
    print(f"Converting PCD file: {pcd_file}")
    
    # Set output file paths
    pgm_file = os.path.join(output_dir, 'pcd_map.pgm')
    yaml_file = os.path.join(output_dir, 'pcd_map.yaml')
    
    # Height slice settings
    height_slice = (min_z, max_z)
    
    # Call PCD conversion function
    pcd_to_pgm(
        pcd_file,
        pgm_file,
        yaml_file,
        resolution=resolution,
        margins=margins,
        occupied_threshold=1,
        invert=False,
        height_slice=height_slice
    )
    
    print(f"PCD conversion completed: {pgm_file}")
    return pgm_file, yaml_file


def align_maps(
    osm_pgm: str, 
    osm_yaml: str, 
    pcd_pgm: str, 
    pcd_yaml: str, 
    interactive: bool = True
) -> np.ndarray:
    """Align OSM and PCD maps.
    
    Args:
        osm_pgm: Path to OSM PGM file
        osm_yaml: Path to OSM YAML file
        pcd_pgm: Path to PCD PGM file
        pcd_yaml: Path to PCD YAML file
        interactive: Whether to use interactive mode for alignment
        
    Returns:
        Transformation matrix as numpy array
        
    Raises:
        ValueError: If alignment fails
        NotImplementedError: If non-interactive mode is requested (not implemented)
    """
    print("Aligning OSM and PCD maps...")
    
    # Create aligner instance
    aligner = PGMCoordinateAligner(osm_pgm, osm_yaml, pcd_pgm, pcd_yaml)
    
    if interactive:
        # In interactive mode, user manually selects corresponding points
        aligner.select_points()
        
        # Calculate alignment transformation
        if not aligner.compute_alignment():
            raise ValueError("Map alignment failed, please ensure enough corresponding points are selected")
        
        # Display alignment results
        aligner.display_alignment_results()
    else:
        # In non-interactive mode, we would need pre-selected points, omitted here
        # In a real application, we might load pre-defined corresponding points from a file
        raise NotImplementedError("Non-interactive mode is not implemented yet, please use --interactive option")
    
    # Return transformation matrix
    return aligner.transform


def transform_osm_map(
    osm_pgm: str, 
    osm_yaml: str, 
    output_dir: str, 
    transform_matrix: np.ndarray, 
    margins: List[float]
) -> Tuple[str, str]:
    """Apply transformation matrix to OSM map.
    
    Args:
        osm_pgm: Path to OSM PGM file
        osm_yaml: Path to OSM YAML file
        output_dir: Directory to save transformed map
        transform_matrix: 3x3 transformation matrix
        margins: Map margins [left, right, top, bottom] in meters
        
    Returns:
        Tuple of (transformed_pgm_file_path, transformed_yaml_file_path)
    """
    print("Applying transformation matrix to OSM map...")
    
    # Set output file paths
    output_pgm = os.path.join(output_dir, 'transformed_osm_map.pgm')
    output_yaml = os.path.join(output_dir, 'transformed_osm_map.yaml')
    
    # Apply rigid transformation
    rigid_transform_from_matrix(
        image_path=osm_pgm,
        output_image_path=output_pgm,
        yaml_path=osm_yaml,
        output_yaml_path=output_yaml,
        margins=margins,
        transform_matrix=transform_matrix
    )
    
    print(f"OSM map transformation completed: {output_pgm}")
    return output_pgm, output_yaml


def save_transform_matrix(
    transform_matrix: np.ndarray, 
    file_path: str,
    source_name: str = "OSM",
    target_name: str = "PCD"
) -> None:
    """Save transformation matrix to a txt file in ROS format.
    
    Args:
        transform_matrix: 3x3 transformation matrix
        file_path: Path to save the txt file
        source_name: Name of the source coordinate frame
        target_name: Name of the target coordinate frame
    """
    # Change file extension to txt
    file_path = file_path.replace('.yaml', '.txt')
    
    # Calculate inverse transformation
    inverse_transform = np.linalg.inv(transform_matrix)
    
    # Get current datetime
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save both forward and inverse transformations to the same file
    with open(file_path, 'w') as f:
        # Add forward transformation description and data
        f.write(f"# Transformation from {source_name} to {target_name} coordinate system\n")
        f.write(f"# Format: 3x3 matrix, row-major order\n")
        f.write(f"# Date: {current_time}\n\n")
        
        for i in range(3):
            row = ' '.join([f"{value:.6f}" for value in transform_matrix[i]])
            f.write(f"{row}\n")
        
        f.write("\n\n")
        
        # Add inverse transformation description and data
        f.write(f"# Inverse transformation from {target_name} to {source_name} coordinate system\n")
        f.write(f"# Format: 3x3 matrix, row-major order\n\n")
        
        for i in range(3):
            row = ' '.join([f"{value:.6f}" for value in inverse_transform[i]])
            f.write(f"{row}\n")
    
    print(f"Forward and inverse transformation matrices saved to: {file_path}")


def copy_pcd_output(
    pcd_pgm: str, 
    pcd_yaml: str, 
    final_dir: str
) -> Tuple[str, str]:
    """Copy PCD map files to final directory.
    
    Args:
        pcd_pgm: Source PCD PGM file path
        pcd_yaml: Source PCD YAML file path
        final_dir: Final output directory
        
    Returns:
        Tuple of (destination_pgm_file_path, destination_yaml_file_path)
    """
    import shutil
    
    pcd_pgm_final = os.path.join(final_dir, 'pcd_map.pgm')
    pcd_yaml_final = os.path.join(final_dir, 'pcd_map.yaml')
    
    shutil.copy2(pcd_pgm, pcd_pgm_final)
    shutil.copy2(pcd_yaml, pcd_yaml_final)
    
    return pcd_pgm_final, pcd_yaml_final


def main() -> int:
    """Main function.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Parse command-line arguments
    args = parse_args()
    
    try:
        # Set up output directories
        temp_dir, final_dir = setup_output_dirs(args.output_dir)
        
        # Step 1: Convert OSM file to PGM
        osm_pgm, osm_yaml = convert_osm(
            args.osm_file, 
            temp_dir, 
            args.resolution, 
            args.tags, 
            args.padding
        )
        
        # Step 2: Convert PCD file to PGM
        pcd_pgm, pcd_yaml = convert_pcd(
            args.pcd_file, 
            temp_dir, 
            args.resolution, 
            args.min_z, 
            args.max_z, 
            args.margins
        )
        
        # Step 3: Align maps
        if not args.skip_alignment:
            transform_matrix = align_maps(
                osm_pgm, 
                osm_yaml, 
                pcd_pgm, 
                pcd_yaml, 
                args.interactive
            )
        else:
            # Use default transformation matrix or load from config file
            print("Skipping alignment step, using default transformation matrix")
            transform_matrix = np.eye(3)  # Default to identity transformation
        
        # Step 4: Apply transformation matrix to OSM map
        transformed_osm_pgm, transformed_osm_yaml = transform_osm_map(
            osm_pgm, 
            osm_yaml, 
            final_dir, 
            transform_matrix, 
            args.margins
        )
        
        # Step 5: Copy PCD map to final directory
        pcd_pgm_final, pcd_yaml_final = copy_pcd_output(
            pcd_pgm, 
            pcd_yaml, 
            final_dir
        )
        
        # Save transformation matrix to file for future use
        transform_file = os.path.join(final_dir, 'transform_matrix.txt')
        save_transform_matrix(
            transform_matrix, 
            transform_file,
            source_name="OSM",
            target_name="PCD"
        )
        
        print("=== Processing completed ===")
        print(f"OSM map: {transformed_osm_pgm}")
        print(f"PCD map: {pcd_pgm_final}")
        print(f"Transformation matrices (OSM→PCD and PCD→OSM): {transform_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 