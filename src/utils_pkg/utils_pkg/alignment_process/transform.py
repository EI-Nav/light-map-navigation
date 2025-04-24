import os
from typing import List, Tuple
import cv2
import numpy as np
import yaml

def pixel_to_physical(x, y, resolution, origin):
    """
    Converts pixel coordinates to physical coordinates using resolution and origin.
    Adjusted to correct coordinate system alignment.
    """
    px = x * resolution + origin[0]
    py = (y * resolution) + origin[1]
    return np.array([px, py, 1])

def physical_to_pixel(px, py, resolution, origin):
    """
    Converts physical coordinates to pixel coordinates using resolution and origin.
    Adjusted to align with image coordinates.
    """
    x = int((px - origin[0]) / resolution)
    y = int((py - origin[1]) / resolution)
    return (x, y)

def transform_points(image, transform_matrix, resolution, origin, threshold=50):
    """
    Transforms only pixels that are close to black using the transformation matrix.
    Returns the transformed points as physical coordinates.
    """
    points = []
    h, w = image.shape
    
    print(f"Image shape: {h}x{w}")

    for y in range(h):
        for x in range(w):
            if image[y, x] < threshold: 
                phys_coords = pixel_to_physical(x, h - y, resolution, origin)
                transformed_phys_coords = np.dot(transform_matrix, phys_coords)
                points.append(transformed_phys_coords[:2])

    return np.array(points)

def generate_new_pgm(points, resolution, output_image_path, margins):
    """Generate new PGM image with controllable margins
    
    Args:
        points: Transformed point set
        resolution: Resolution in meters/pixel
        output_image_path: Output image path
        margins: Margin parameters [left, right, top, bottom] in meters
    """
    if len(points) == 0:
        raise ValueError("No points to transform. Image may not contain any black pixels.")

    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    # Calculate margins in pixels
    left_margin = int(margins[0] / resolution)
    right_margin = int(margins[1] / resolution)
    top_margin = int(margins[2] / resolution)
    bottom_margin = int(margins[3] / resolution)

    points_width = int((max_x - min_x) / resolution)
    points_height = int((max_y - min_y) / resolution)

    new_width = points_width + left_margin + right_margin
    new_height = points_height + top_margin + bottom_margin

    new_origin = [
        min_x - (left_margin * resolution),
        min_y - (bottom_margin * resolution)
    ]

    new_image = np.full((new_height, new_width), 255, dtype=np.uint8)

    for point in points:
        px, py = physical_to_pixel(point[0], point[1], resolution, new_origin)
        if 0 <= px < new_width and 0 <= py < new_height:
            new_image[new_height - 1 - py, px] = 0

    cv2.imwrite(output_image_path, new_image)
    return new_origin

def update_yaml(yaml_data, new_image_path, new_origin, transform_matrix):
    """Updates YAML data with new parameters
    
    Args:
        yaml_data: Original YAML data
        new_image_path: Path to the transformed image
        new_origin: New origin coordinates
        transform_matrix: Applied transformation matrix
    """
    yaml_data['image'] = new_image_path.split('/')[-1]
    yaml_data['origin'] = [float(new_origin[0]), float(new_origin[1]), yaml_data['origin'][2]]
    yaml_data['transform_matrix'] = {
        'row1': transform_matrix[0].tolist(),
        'row2': transform_matrix[1].tolist(),
        'row3': transform_matrix[2].tolist()
    }
    return yaml_data

def rigid_transform_from_matrix(image_path, output_image_path, yaml_path, output_yaml_path, margins=[10, 10, 10, 10], transform_matrix=None):
    """Transform PGM image using transformation matrix
    
    Args:
        image_path: Path to input PGM image
        output_image_path: Path to output transformed PGM image
        yaml_path: Path to input YAML metadata
        output_yaml_path: Path to output YAML metadata
        margins: Margin parameters [left, right, top, bottom] in meters
        transform_matrix: 3x3 transformation matrix to apply
    """
    if transform_matrix is None:
        # Use identity matrix if none provided
        transform_matrix = np.eye(3)
    
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")
    
    # Load the YAML data
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    resolution = yaml_data['resolution']
    origin = yaml_data['origin']

    # Transform the points
    transformed_points = transform_points(image, transform_matrix, resolution, origin)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate new PGM file
    new_origin = generate_new_pgm(transformed_points, resolution, output_image_path, margins)

    # Update and save YAML metadata
    updated_yaml_data = update_yaml(yaml_data, output_image_path, new_origin, transform_matrix)
    with open(output_yaml_path, 'w') as file:
        yaml.dump(updated_yaml_data, file, default_flow_style=False, sort_keys=False)
    
    print(f"Transformed image saved to: {output_image_path}")
    print(f"Transformed YAML saved to: {output_yaml_path}") 