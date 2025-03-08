import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# ===== File Reading and Data Processing Functions =====

def read_csv_file(file_path):
    """Read CSV file"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def read_and_print_csv(file_name):
    """Read CSV file and print its basic information"""
    try:
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Build the full path of the CSV file
        file_path = os.path.join(current_dir, file_name)
        
        # Read the CSV file
        df = read_csv_file(file_path)
        if df is not None:
            print_data_info(df, file_name)
        
        return df
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def print_data_info(df, file_name):
    """Print basic information of the dataset"""
    print(f"File '{file_name}' contains {len(df)} rows and {len(df.columns)} columns")
    print("\nColumn names:")
    print(df.columns.tolist())
    
    print("\nFirst 5 rows:")
    print(df.head())

def read_ground_truth_points(gt_file_path):
    """Read ground truth point coordinates"""
    try:
        gt_points = []
        with open(gt_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '(' in line and ')' in line:
                    # Extract coordinates inside parentheses
                    coords = line.split('(')[1].split(')')[0]
                    # Split X and Y coordinates
                    x, y = map(float, coords.split(','))
                    # Assume Z coordinate is 0
                    gt_points.append(np.array([x, y, 0.0]))
        
        print(f"Read {len(gt_points)} ground truth points from file {gt_file_path}")
        for i, point in enumerate(gt_points):
            print(f"Ground truth point {i+1}: X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f}")
        
        return gt_points
    except Exception as e:
        print(f"Error reading ground truth file: {e}")
        return []

# ===== Exploration Process Identification and Data Segmentation Functions =====

def identify_exploration_statuses(df, status_column='task_status'):
    """Identify exploration statuses in the data"""
    if status_column not in df.columns:
        print(f"Warning: Column '{status_column}' not found, cannot identify task status changes")
        return []
    
    # Find possible values in the status column to determine the correct status name
    unique_statuses = df[status_column].unique()
    print(f"Task status values in the data: {unique_statuses}")
    
    # Only identify exploration statuses
    exploration_keywords = ['exploration']
    
    exploration_statuses = []
    
    for status in unique_statuses:
        status_lower = str(status).lower()
        if any(keyword in status_lower for keyword in exploration_keywords):
            exploration_statuses.append(status)
    
    print(f"Identified exploration statuses: {exploration_statuses}")
    
    return exploration_statuses

def find_exploration_end_indices(df, status_column='task_status'):
    """Find the last row index of each continuous exploration process"""
    # Identify exploration statuses
    exploration_statuses = identify_exploration_statuses(df, status_column)
    
    # If statuses cannot be automatically identified, prompt the user
    if not exploration_statuses:
        print("Cannot automatically identify exploration statuses, please check the data and adjust the code")
        return []
    
    # Create a boolean sequence indicating whether each row is in exploration status
    is_exploration = df[status_column].isin(exploration_statuses)
    
    # Find continuous blocks of exploration status
    # When the status changes from exploration to non-exploration or vice versa, it indicates the start or end of a block
    status_changes = (is_exploration != is_exploration.shift(1)).cumsum()
    
    # Find the last row index of each exploration block
    end_indices = []
    
    # For each status change block
    for block_id in status_changes.unique():
        # Get all rows in this block
        block_mask = status_changes == block_id
        # If this block is in exploration status
        if is_exploration[block_mask].any():
            # Only consider rows in exploration status
            exploration_rows = block_mask & is_exploration
            if exploration_rows.any():
                # Find the index of the last row in exploration status
                last_idx = df.index[exploration_rows][-1]
                end_indices.append(last_idx)
    
    return sorted(end_indices)

def split_data_by_exploration(df, end_indices):
    """Split data into multiple sub-datasets based on the end indices of exploration processes"""
    if not end_indices:
        print("No end indices of exploration processes found, cannot split data")
        return []
    
    # Add start indices
    start_indices = [0] + [idx + 1 for idx in end_indices[:-1]]
    
    # Split data
    split_data = []
    for start_idx, end_idx in zip(start_indices, end_indices):
        # Extract data from start index to end index
        segment = df.iloc[start_idx:end_idx+1].copy()
        split_data.append(segment)
    
    # If the last end index is not the last row of the data, add the remaining data
    if end_indices[-1] < len(df) - 1:
        remaining = df.iloc[end_indices[-1]+1:].copy()
        split_data.append(remaining)
    
    return split_data

def extract_xyz_coordinates(segments):
    """Extract XYZ coordinate data from each sub-dataset"""
    xyz_data = []
    
    for i, segment in enumerate(segments):
        # Check if XYZ coordinate columns exist
        if 'robot_x' in segment.columns and 'robot_y' in segment.columns and 'robot_z' in segment.columns:
            # Extract XYZ coordinates
            x = segment['robot_x'].values
            y = segment['robot_y'].values
            z = segment['robot_z'].values
            
            # Create coordinate array
            coords = np.column_stack((x, y, z))
            
            # Get the coordinates of the last point
            last_point = coords[-1]
            
            # Add to result list
            xyz_data.append({
                'segment_id': i + 1,
                'coordinates': coords,
                'last_point': last_point,
                'instruction': segment['instruction'].iloc[0] if 'instruction' in segment.columns else 'Unknown'
            })
            
            print(f"XYZ coordinates of sub-dataset {i+1} extracted, total {len(coords)} points")
            print(f"  Last point coordinates: X={last_point[0]:.3f}, Y={last_point[1]:.3f}, Z={last_point[2]:.3f}")
        else:
            print(f"Warning: Sub-dataset {i+1} is missing XYZ coordinate columns")
    
    return xyz_data

# ===== Evaluation Metric Calculation Functions =====

def calculate_sr(distances, threshold):
    """
    Calculate SR (Success Rate)
    
    Args:
        distances (list): List of distances from trajectory endpoints to ground truth points
        threshold (float): Distance threshold for success determination
        
    Returns:
        tuple: (success rate percentage, success count, total count)
    """
    if not distances:
        return 0.0, 0, 0
    
    success_count = sum(1 for d in distances if d <= threshold)
    total_count = len(distances)
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0.0
    
    return success_rate, success_count, total_count

def calculate_spl(traj_lengths, origin_distances, distances, threshold):
    """
    Calculate SPL (Success weighted by Path Length)
    
    Args:
        traj_lengths (list): List of lengths for each trajectory
        origin_distances (list): List of distances from origin to each ground truth point
        distances (list): List of distances from trajectory endpoints to ground truth points
        threshold (float): Distance threshold for success determination
        
    Returns:
        float: SPL value (percentage)
    """
    if not traj_lengths or not origin_distances or not distances:
        return 0.0
    
    N = len(traj_lengths)
    spl_sum = 0.0
    
    for i in range(N):
        # Success flag: 1 if distance is less than threshold, 0 otherwise
        S_i = 1 if distances[i] <= threshold else 0
        
        # Trajectory length
        p_i = traj_lengths[i]
        
        # Straight-line distance from origin to target
        l_i = origin_distances[i]
        
        # Calculate ratio, avoid division by zero
        ratio = l_i / max(p_i, l_i) if max(p_i, l_i) > 0 else 0
        
        # Accumulate SPL
        spl_sum += S_i * ratio
    
    # Calculate average SPL
    spl = spl_sum / N
    return spl * 100  # Convert to percentage

def calculate_lsr(distances, threshold=1.0, r=0.9):
    """
    Calculate LSR (Length-weighted Success Rate)
    
    Args:
        distances (list): List of distances from trajectory endpoints to ground truth points
        threshold (float): Distance threshold for success determination
        r (float): Weight decay factor
        
    Returns:
        float: LSR value (percentage)
    """
    n = len(distances)
    if n == 0:
        return 0.0
        
    weighted_success_rate = 0.0
    total_weight = 0.0

    for i in range(1, n + 1):
        S_i = 1 if distances[i-1] <= threshold else 0
        c_i = (r ** (i - 1) * (1 - r)) / (1 - r ** n)
        weighted_success_rate += c_i * S_i
        total_weight += c_i

    # Normalization
    if total_weight > 0:
        lsr = weighted_success_rate / total_weight
    else:
        lsr = 0  # If total weight is 0, return 0

    return lsr * 100  # Convert to percentage

def calculate_lspl(traj_lengths, sequential_distances, distances, threshold=1.0, r=0.9):
    """
    Calculate LSPL (Length-weighted Success rate weighted by Path Length)
    
    Args:
        traj_lengths (list): List of lengths for each trajectory
        sequential_distances (list): List of sequential distances
        distances (list): List of distances from trajectory endpoints to ground truth points
        threshold (float): Distance threshold for success determination
        r (float): Weight decay factor
        
    Returns:
        float: LSPL value (percentage)
    """
    n = len(traj_lengths)
    if n == 0:
        return 0.0
        
    lspl_sum = 0.0
    total_weight = 0.0

    for i in range(1, n + 1):
        S_i = 1 if distances[i-1] <= threshold else 0
        c_i = (r ** (i - 1) * (1 - r)) / (1 - r ** n)
        l_i = sequential_distances[i-1]
        p_i = traj_lengths[i-1]
        lspl_sum += c_i * S_i * (l_i / max(p_i, l_i))
        total_weight += c_i

    # Normalization
    if total_weight > 0:
        lspl = lspl_sum / total_weight
    else:
        lspl = 0  # If total weight is 0, return 0

    return lspl * 100  # Convert to percentage

# ===== Distance Calculation Functions =====

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points (XY plane)"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_optimal_distances(gt_points):
    """
    Calculate straight-line distances from origin (0,0,0) to each ground truth point
    
    Args:
        gt_points (list): List of ground truth point coordinates
        
    Returns:
        list: Optimal distances for each ground truth point
    """
    origin = np.array([0.0, 0.0, 0.0])
    optimal_distances = []
    
    for point in gt_points:
        # Calculate straight-line distance from origin to ground truth point
        distance = np.sqrt(np.sum((point - origin)**2))
        optimal_distances.append(distance)
    
    return optimal_distances

def calculate_sequential_distances(gt_points):
    """
    Calculate sequential distances: origin to first point, first point to second point, and so on
    
    Args:
        gt_points (list): List of ground truth point coordinates
        
    Returns:
        list: List of sequential distances
    """
    if not gt_points or len(gt_points) == 0:
        return []
    
    sequential_distances = []
    # Calculate distance from origin (0,0,0) to first point
    origin = np.array([0.0, 0.0, 0.0])
    first_point = gt_points[0]
    first_distance = np.sqrt(np.sum((first_point - origin)**2))
    sequential_distances.append(first_distance)
    
    # Calculate distances between each pair of adjacent points
    for i in range(1, len(gt_points)):
        point1 = gt_points[i-1]
        point2 = gt_points[i]
        distance = np.sqrt(np.sum((point2 - point1)**2))
        sequential_distances.append(distance)
    
    return sequential_distances

def calculate_trajectory_length(coordinates):
    """
    Calculate the total length of a trajectory
    
    Args:
        coordinates (ndarray): Coordinate array, shape (n, 3), each row contains (x, y, z) coordinates
        
    Returns:
        float: Total length of the trajectory (meters)
    """
    if len(coordinates) < 2:
        return 0.0
    
    # Calculate Euclidean distances between adjacent points
    distances = np.sqrt(np.sum(np.diff(coordinates, axis=0)**2, axis=1))
    
    # Return total length
    return np.sum(distances)

# ===== Evaluation Result Analysis Functions =====

def evaluate_success_rate(xyz_data, gt_points, distance_threshold=1.0):
    """
    Evaluate distances between trajectory endpoints and ground truth points, calculate success rate
    
    Args:
        xyz_data (list): List containing multiple coordinate arrays
        gt_points (list): List of ground truth point coordinates
        distance_threshold (float): Distance threshold for success determination
        
    Returns:
        tuple: (success count, total count, success rate, results list)
    """
    if not xyz_data or not gt_points:
        print("Cannot evaluate success rate: missing trajectory data or ground truth points")
        return 0, 0, 0.0, []
    
    # Ensure the number of ground truth points matches the number of trajectories
    if len(gt_points) < len(xyz_data):
        print(f"Warning: Number of ground truth points ({len(gt_points)}) is less than number of trajectories ({len(xyz_data)})")
        print(f"Will only evaluate the first {len(gt_points)} trajectories")
    
    results = []
    total_count = min(len(xyz_data), len(gt_points))
    
    for i in range(total_count):
        data = xyz_data[i]
        gt_point = gt_points[i]
        last_point = data['last_point']
        
        # Calculate distance
        distance = calculate_distance(last_point, gt_point)
        
        # Determine if successful
        is_success = distance < distance_threshold
        
        # Record result
        results.append({
            'segment_id': data['segment_id'],
            'last_point': last_point,
            'gt_point': gt_point,
            'distance': distance,
            'is_success': is_success
        })
    
    # Calculate success rate
    success_count = sum(1 for r in results if r['is_success'])
    success_rate = success_count / total_count if total_count > 0 else 0.0
    
    return success_count, total_count, success_rate, results

def print_metrics_summary(xyz_data, gt_points, results, distance_threshold):
    """
    Print summary of all evaluation metrics
    
    Args:
        xyz_data (list): List containing multiple coordinate arrays
        gt_points (list): List of ground truth point coordinates
        results (list): List of evaluation results
        distance_threshold (float): Distance threshold for success determination
    """
    if not xyz_data or not gt_points or not results:
        print("Cannot calculate evaluation metrics: missing trajectory data, ground truth points, or evaluation results")
        return
    
    # Calculate trajectory lengths
    traj_lengths = [calculate_trajectory_length(data['coordinates']) for data in xyz_data]
    
    # Calculate distances from origin to ground truth points
    origin_distances = calculate_optimal_distances(gt_points)
    
    # Extract distances from endpoints to ground truth points
    distances = [result['distance'] for result in results]
    
    # Calculate sequential distances
    sequential_distances = calculate_sequential_distances(gt_points)
    
    # Calculate SR (Success Rate)
    sr, success_count, total_count = calculate_sr(distances, distance_threshold)
    
    # Calculate SPL (Success weighted by Path Length)
    spl = calculate_spl(traj_lengths, origin_distances, distances, distance_threshold)
    
    # Calculate LSR (Length-weighted Success Rate)
    lsr = calculate_lsr(distances, distance_threshold, r=0.9)
    
    # Calculate LSPL (Length-weighted Success rate weighted by Path Length)
    lspl = calculate_lspl(traj_lengths, sequential_distances, distances, distance_threshold, r=0.9)
    
    print("\nEvaluation Metrics Summary:")
    print("=" * 80)
    print(f"Distance Threshold: {distance_threshold} meters")
    print("-" * 80)
    print(f"SR  (Success Rate)                                : {sr:.2f}% ({success_count}/{total_count})")
    print(f"SPL (Success weighted by Path Length)             : {spl:.2f}%")
    print(f"LSR (Length-weighted Success Rate)                : {lsr:.2f}%")
    print(f"LSPL (Length-weighted Success rate by Path Length): {lspl:.2f}%")
    print("=" * 80)
    
    # Print sequential distances
    print("\nSequential Distances (from origin to first point, then point to point):")
    for i, dist in enumerate(sequential_distances):
        if i == 0:
            print(f"  Origin -> Point 1: {dist:.2f}m")
        else:
            print(f"  Point {i} -> Point {i+1}: {dist:.2f}m")
    
    # Print detailed SPL calculation process
    print("\nSPL Calculation Details:")
    print("-" * 100)
    print(f"{'Trajectory ID':<10}{'Traj Length(p_i)':<15}{'Shortest Dist(l_i)':<15}{'End Distance':<15}{'Success(S_i)':<15}{'Ratio(l_i/max(p_i,l_i))':<20}{'S_i * Ratio':<15}")
    print("-" * 100)
    
    spl_sum = 0.0
    for i in range(min(len(traj_lengths), len(origin_distances), len(distances))):
        p_i = traj_lengths[i]
        l_i = origin_distances[i]
        dist = distances[i]
        S_i = 1 if dist <= distance_threshold else 0
        ratio = l_i / max(p_i, l_i) if max(p_i, l_i) > 0 else 0
        contribution = S_i * ratio
        spl_sum += contribution
        
        print(f"{i+1:<10}{p_i:<15.2f}{l_i:<15.2f}{dist:<15.2f}{S_i:<15}{ratio:<20.4f}{contribution:<15.4f}")
    
    print("-" * 100)
    print(f"SPL Sum: {spl_sum:.4f}, Trajectory Count: {len(traj_lengths)}, SPL: {spl_sum/len(traj_lengths)*100:.2f}%")

def print_evaluation_results(results, success_rate, distance_threshold):
    """
    Print evaluation results
    
    Args:
        results (list): List of evaluation results
        success_rate (float): Success rate
        distance_threshold (float): Distance threshold for success determination
    """
    print("\nDistance Evaluation Results between Trajectory Endpoints and Ground Truth Points:")
    print("-" * 100)
    print(f"{'Exploration ID':<12}{'End X':<10}{'End Y':<10}{'End Z':<10}{'GT X':<10}{'GT Y':<10}{'GT Z':<10}{'Distance':<10}{'Success':<10}")
    print("-" * 100)
    
    for result in results:
        segment_id = result['segment_id']
        last_point = result['last_point']
        gt_point = result['gt_point']
        distance = result['distance']
        is_success = result['is_success']
        
        print(f"{segment_id:<12}{last_point[0]:<10.2f}{last_point[1]:<10.2f}{last_point[2]:<10.2f}"
              f"{gt_point[0]:<10.2f}{gt_point[1]:<10.2f}{gt_point[2]:<10.2f}"
              f"{distance:<10.2f}{'Yes' if is_success else 'No':<10}")
    
    print("-" * 100)
    print(f"Distance Threshold: {distance_threshold} meters")
    print(f"SR (Success Rate): {success_rate:.2%} ({sum(r['is_success'] for r in results)}/{len(results)})")

def print_trajectory_efficiency(xyz_data, gt_points):
    """
    Print trajectory efficiency analysis (comparison between actual trajectory length and shortest distance)
    
    Args:
        xyz_data (list): List containing multiple coordinate arrays
        gt_points (list): List of ground truth point coordinates
    """
    if not xyz_data or not gt_points:
        print("Cannot analyze trajectory efficiency: missing trajectory data or ground truth points")
        return
    
    # Calculate optimal distances (straight-line distances from origin to ground truth points)
    optimal_distances = calculate_optimal_distances(gt_points)
    
    print("\nTrajectory Efficiency Analysis (Comparison between Actual Trajectory Length and Shortest Distance):")
    print("-" * 100)
    print(f"{'Exploration ID':<12}{'Actual Length(m)':<20}{'Shortest Distance(m)':<20}{'Efficiency Ratio':<15}{'Instruction':<30}")
    print("-" * 100)
    
    total_actual_length = 0.0
    total_optimal_length = 0.0
    
    # Ensure only processing trajectories with corresponding ground truth points
    count = min(len(xyz_data), len(gt_points))
    
    for i in range(count):
        data = xyz_data[i]
        segment_id = data['segment_id']
        coords = data['coordinates']
        instruction = data['instruction']
        optimal_distance = optimal_distances[i]
        
        # Calculate trajectory length
        actual_length = calculate_trajectory_length(coords)
        
        # Calculate efficiency ratio (shortest distance/actual length)
        efficiency_ratio = optimal_distance / actual_length if actual_length > 0 else 0.0
        
        # Truncate long instructions
        short_instruction = instruction[:27] + "..." if len(instruction) > 30 else instruction
        
        print(f"{segment_id:<12}{actual_length:<20.3f}{optimal_distance:<20.3f}{efficiency_ratio:<15.3f}{short_instruction:<30}")
        
        total_actual_length += actual_length
        total_optimal_length += optimal_distance
    
    # Calculate overall efficiency ratio
    overall_efficiency = total_optimal_length / total_actual_length if total_actual_length > 0 else 0.0
    
    print("-" * 100)
    print(f"Total: {total_actual_length:.3f} meters vs Shortest Total Distance: {total_optimal_length:.3f} meters")
    print(f"Overall Efficiency Ratio: {overall_efficiency:.3f} (Value closer to 1 indicates trajectory closer to optimal)")
    print("-" * 100)

def evaluate_with_ground_truth(xyz_data, gt_file_path, distance_threshold, output_dir, save_plot, show_plot):
    """
    Evaluate trajectories using ground truth
    
    Args:
        xyz_data (list): List containing multiple coordinate arrays
        gt_file_path (str): Path to ground truth file
        distance_threshold (float): Distance threshold for success determination
        output_dir (str): Path to output directory
        save_plot (bool): Whether to save evaluation result plot
        show_plot (bool): Whether to display evaluation result plot
    """
    # Read ground truth points
    gt_points = read_ground_truth_points(gt_file_path)
    
    if not gt_points:
        print("Failed to read ground truth points, cannot perform evaluation")
        return
    
    # Evaluate success rate
    success_count, total_count, success_rate, results = evaluate_success_rate(
        xyz_data, gt_points, distance_threshold)
    
    # Print evaluation results
    print_evaluation_results(results, success_rate, distance_threshold)
    
    # Analyze trajectory efficiency (comparison between actual trajectory and shortest distance)
    print_trajectory_efficiency(xyz_data, gt_points)
    
    # Calculate and print all evaluation metrics
    print_metrics_summary(xyz_data, gt_points, results, distance_threshold)
    
    # Create evaluation result visualization
    if show_plot or save_plot:
        # Create output directory (if need to save image)
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            
        create_evaluation_plot(
            xyz_data, gt_points, results, 
            output_dir, 
            save_plot=save_plot,
            show_plot=show_plot
        )

# ===== Visualization and Output Functions =====

def create_evaluation_plot(xyz_data, gt_points, results, output_dir, save_plot=True, show_plot=False):
    """Create evaluation result visualization plot"""
    # Create 2D plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot XY projection of each trajectory
    for i, data in enumerate(xyz_data):
        if i >= len(results):
            continue
            
        coords = data['coordinates']
        last_point = data['last_point']
        gt_point = gt_points[i]
        is_success = results[i]['is_success']
        
        # Plot trajectory
        ax.plot(coords[:, 0], coords[:, 1], '-', linewidth=1, alpha=0.7, label=f'Trajectory {i+1}')
        
        # Mark start point
        ax.scatter(coords[0, 0], coords[0, 1], c='g', marker='o', s=80)
        
        # Mark end point, using different colors based on success
        marker_color = 'blue' if is_success else 'red'
        ax.scatter(last_point[0], last_point[1], c=marker_color, marker='x', s=100)
        
        # Mark ground truth point
        ax.scatter(gt_point[0], gt_point[1], c='purple', marker='*', s=150)
        
        # Connect end point and ground truth point
        ax.plot([last_point[0], gt_point[0]], [last_point[1], gt_point[1]], 
                'k--', linewidth=1, alpha=0.5)
        
        # Add labels
        ax.annotate(f"{i+1}", (last_point[0], last_point[1]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax.annotate(f"GT{i+1}", (gt_point[0], gt_point[1]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Set legend and labels
    ax.set_title('Trajectory Endpoint and Ground Truth Distance Evaluation')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    custom_lines = [
        plt.Line2D([0], [0], color='g', marker='o', linestyle='', markersize=8),
        plt.Line2D([0], [0], color='blue', marker='x', linestyle='', markersize=8),
        plt.Line2D([0], [0], color='red', marker='x', linestyle='', markersize=8),
        plt.Line2D([0], [0], color='purple', marker='*', linestyle='', markersize=10)
    ]
    ax.legend(custom_lines, ['Start Point', 'Success Endpoint', 'Failure Endpoint', 'Ground Truth Point'], 
              loc='upper right', framealpha=0.9)
    
    # Save image
    if save_plot:
        plt.tight_layout()
        plt.savefig(f'{output_dir}/evaluation_results.png', dpi=300)
        print(f"Evaluation result plot saved to {output_dir}/evaluation_results.png")
    
    # Display image
    if show_plot:
        plt.show()
    
    plt.close()

# ===== Information Printing Functions =====

def print_exploration_indices(df, end_indices):
    """Print information about the end indices of exploration processes"""
    print("\nLast row indices of each continuous exploration process:")
    for i, idx in enumerate(end_indices):
        print(f"Exploration process {i+1} ends at index {idx}, corresponding row data:")
        print(df.iloc[idx])
    
    print(f"\nTotal {len(end_indices)} exploration processes found")

def print_segments_info(segments):
    """Print information about split sub-datasets"""
    print(f"\nData split into {len(segments)} sub-datasets")
    for i, segment in enumerate(segments):
        print(f"\nSub-dataset {i+1} contains {len(segment)} rows")
        print(f"Start row:")
        print(segment.iloc[0])
        print(f"End row:")
        print(segment.iloc[-1])

def print_last_points_summary(xyz_data):
    """Print summary of coordinates of last points in all datasets"""
    print("\nSummary of coordinates of last points in all exploration processes:")
    print("-" * 80)
    print(f"{'Exploration ID':<12}{'X Coordinate':<15}{'Y Coordinate':<15}{'Z Coordinate':<15}{'Instruction':<30}")
    print("-" * 80)
    
    for data in xyz_data:
        segment_id = data['segment_id']
        last_point = data['last_point']
        instruction = data['instruction']
        
        # Truncate long instructions
        short_instruction = instruction[:27] + "..." if len(instruction) > 30 else instruction
        
        print(f"{segment_id:<12}{last_point[0]:<15.3f}{last_point[1]:<15.3f}{last_point[2]:<15.3f}{short_instruction:<30}")
    
    print("-" * 80)

def print_trajectory_lengths(xyz_data):
    """
    Print information about trajectory lengths in all exploration processes
    
    Args:
        xyz_data (list): List containing multiple coordinate arrays
    """
    print("\nTrajectory lengths in all exploration processes:")
    print("-" * 80)
    print(f"{'Exploration ID':<12}{'Trajectory Length(m)':<15}{'Point Count':<15}{'Average Point Distance(m)':<20}{'Instruction':<30}")
    print("-" * 80)
    
    total_length = 0.0
    
    for data in xyz_data:
        segment_id = data['segment_id']
        coords = data['coordinates']
        instruction = data['instruction']
        
        # Calculate trajectory length
        length = calculate_trajectory_length(coords)
        
        # Calculate average point distance
        avg_distance = length / (len(coords) - 1) if len(coords) > 1 else 0.0
        
        # Truncate long instructions
        short_instruction = instruction[:27] + "..." if len(instruction) > 30 else instruction
        
        print(f"{segment_id:<12}{length:<15.3f}{len(coords):<15}{avg_distance:<20.3f}{short_instruction:<30}")
        
        total_length += length
    
    print("-" * 80)
    print(f"Total trajectory length: {total_length:.3f} meters")
    print(f"Average trajectory length: {total_length/len(xyz_data):.3f} meters (Total {len(xyz_data)} trajectories)")
    print("-" * 80)

# ===== Command Line Argument Processing =====

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process exploration data in CSV file')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('--gt-file', help='Path to ground truth file')
    parser.add_argument('--distance-threshold', type=float, default=1.0, 
                        help='Distance threshold for success determination (meters)')
    parser.add_argument('--no-save-plots', action='store_true', help='Do not save evaluation result plots')
    parser.add_argument('--show-plots', action='store_true', help='Display evaluation result plots on screen')
    parser.add_argument('--output-dir', default='result/plots', help='Path to output directory')
    
    return parser.parse_args()

# ===== Main Function =====

def process_exploration_data(data):
    """Main logic for processing exploration data"""
    # Find the last row index of each continuous exploration process
    end_indices = find_exploration_end_indices(data)
    
    if not end_indices:
        print("\nNo end points found for any exploration process, please check data format and status column name")
        return None
    
    # Print information about the end indices of exploration processes
    print_exploration_indices(data, end_indices)
    
    # Split data based on the end indices of exploration processes
    split_data = split_data_by_exploration(data, end_indices)
    
    # Print information about split sub-datasets
    print_segments_info(split_data)
    
    # Extract XYZ coordinates from each sub-dataset
    xyz_data = extract_xyz_coordinates(split_data)
    
    if not xyz_data:
        print("Cannot extract any XYZ coordinate data")
        return None
    
    # Print summary of coordinates of last points in all datasets
    print_last_points_summary(xyz_data)
    
    # Calculate and print trajectory lengths
    print_trajectory_lengths(xyz_data)
    
    return xyz_data

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Read and print CSV file
    data = read_and_print_csv(args.csv_file)
    
    if data is None:
        return
    
    # Process exploration data
    xyz_data = process_exploration_data(data)
    
    if xyz_data is None:
        return
    
    # If ground truth file is provided, perform evaluation
    if args.gt_file:
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Build the full path of the ground truth file
        gt_file_path = os.path.join(current_dir, args.gt_file)
        
        evaluate_with_ground_truth(
            xyz_data, 
            gt_file_path, 
            args.distance_threshold, 
            args.output_dir, 
            not args.no_save_plots, 
            args.show_plots
        )

if __name__ == "__main__":
    main()
