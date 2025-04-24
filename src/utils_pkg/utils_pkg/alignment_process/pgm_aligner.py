#!/usr/bin/env python3
import numpy as np
import yaml
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.spatial.transform import Rotation
import matplotlib.patches as patches

class PGMCoordinateAligner:
    def __init__(self, pgm1_path, yaml1_path, pgm2_path, yaml2_path):
        """Initialize PGM coordinate aligner with two PGM maps and their YAML metadata files"""
        # Read the first PGM and its YAML
        self.pgm1 = cv2.imread(pgm1_path, cv2.IMREAD_GRAYSCALE)
        if self.pgm1 is None:
            raise ValueError(f"Unable to read PGM file: {pgm1_path}")
        
        with open(yaml1_path, 'r') as f:
            self.yaml1 = yaml.safe_load(f)
        
        # Read the second PGM and its YAML
        self.pgm2 = cv2.imread(pgm2_path, cv2.IMREAD_GRAYSCALE)
        if self.pgm2 is None:
            raise ValueError(f"Unable to read PGM file: {pgm2_path}")
        
        with open(yaml2_path, 'r') as f:
            self.yaml2 = yaml.safe_load(f)
        
        # Initialize point lists
        self.points1 = []  # Points on the first image (image coordinates)
        self.points2 = []  # Points on the second image (image coordinates)
        
        # Currently selected image (0: first, 1: second)
        self.current_img = 0
        
        # Track point selection state
        self.current_point_index = 0  # Used to track current point pair
        self.waiting_for_second_point = False  # Flag to indicate if waiting for point on map 2
        
        # Calculation results
        self.transform = None  # Transformation matrix
        self.translation = None  # Translation vector
        self.rotation = None  # Rotation matrix
        self.rotation_angle = None  # Rotation angle (radians)
        
        # Colors for point visualization
        self.colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        
        # Status text for coordinate display
        self.status_text = None
        self.pending_point = None
        
        # Store world coordinates
        self.world_points1 = []
        self.world_points2 = []
        
        # Debug flag
        self.debug = True
    
    def image_to_world(self, img_point, img_idx):
        """Convert image coordinates to world coordinates"""
        if img_idx == 0:
            yaml_data = self.yaml1
            img_height = self.pgm1.shape[0]
        else:
            yaml_data = self.yaml2
            img_height = self.pgm2.shape[0]
        
        resolution = yaml_data['resolution']
        origin = yaml_data['origin']
        
        # Note that PGM coordinate system origin is at the top-left corner,
        # while the map origin is at the bottom-left corner
        # Need to flip the y coordinate
        x_img, y_img = img_point
        y_img = img_height - y_img - 1  
        
        # Convert to world coordinates
        x_world = origin[0] + x_img * resolution
        y_world = origin[1] + y_img * resolution
        
        return [x_world, y_world]
    
    def compute_alignment(self):
        """Calculate coordinate transformation based on selected corresponding points"""
        if len(self.points1) < 2 or len(self.points1) != len(self.points2):
            print("At least two pairs of corresponding points are needed")
            return False
        
        # Convert to world coordinates (if not already done)
        if not self.world_points1:
            self.world_points1 = [self.image_to_world(p, 0) for p in self.points1]
        if not self.world_points2:
            self.world_points2 = [self.image_to_world(p, 1) for p in self.points2]
        
        world_points1 = np.array(self.world_points1)
        world_points2 = np.array(self.world_points2)
        
        # Calculate centroids
        centroid1 = np.mean(world_points1, axis=0)
        centroid2 = np.mean(world_points2, axis=0)
        
        # De-center the points
        centered_points1 = world_points1 - centroid1
        centered_points2 = world_points2 - centroid2
        
        # Calculate H matrix (covariance matrix)
        H = centered_points1.T @ centered_points2
        
        # Singular value decomposition
        U, _, Vt = np.linalg.svd(H)
        
        # Calculate rotation matrix
        R = Vt.T @ U.T
        
        # Ensure it's a proper rotation matrix (determinant = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Calculate translation vector
        t = centroid2 - R @ centroid1
        
        # Calculate transformation matrix
        transform = np.eye(3)
        transform[:2, :2] = R
        transform[:2, 2] = t
        
        # Save results
        self.transform = transform
        self.translation = t
        self.rotation = R
        
        # Calculate rotation angle (assuming 2D rotation)
        self.rotation_angle = np.arctan2(R[1, 0], R[0, 0])
        
        return True
    
    def display_alignment_results(self):
        """Display transformation results"""
        if self.transform is None:
            print("Please calculate the transformation first")
            return
        
        # Map1 to Map2 transformation (the one we calculated)
        print("\nTransformation Results (Map1 to Map2):")
        print(f"Rotation Matrix:\n{self.rotation}")
        print(f"Rotation Angle: {np.degrees(self.rotation_angle):.4f} degrees")
        print(f"Translation Vector: {self.translation}")
        print(f"Transformation Matrix:\n{self.transform}")
        
        # Convert 2D rotation matrix to 3D rotation matrix for scipy.spatial.transform
        rotation_3d = np.eye(3)
        rotation_3d[:2, :2] = self.rotation
        
        # Convert transformation matrix to ROS format output
        r = Rotation.from_matrix(rotation_3d)
        roll, pitch, yaw = r.as_euler('xyz')
        
        print("\nROS Format Output (Map1 to Map2):")
        print(f"position:")
        print(f"  x: {self.translation[0]:.6f}")
        print(f"  y: {self.translation[1]:.6f}")
        print(f"  z: 0.000000")
        print(f"orientation:")
        print(f"  x: 0.000000")
        print(f"  y: 0.000000")
        print(f"  z: {np.sin(yaw/2):.6f}")
        print(f"  w: {np.cos(yaw/2):.6f}")
        
        # Map2 to Map1 transformation (inverse of the one we calculated)
        inv_transform = np.linalg.inv(self.transform)
        inv_rotation = inv_transform[:2, :2]
        inv_translation = inv_transform[:2, 2]
        inv_rotation_angle = np.arctan2(inv_rotation[1, 0], inv_rotation[0, 0])
        
        print("\nTransformation Results (Map2 to Map1):")
        print(f"Rotation Matrix:\n{inv_rotation}")
        print(f"Rotation Angle: {np.degrees(inv_rotation_angle):.4f} degrees")
        print(f"Translation Vector: {inv_translation}")
        print(f"Transformation Matrix:\n{inv_transform}")
        
        # Convert 2D rotation matrix to 3D rotation matrix for scipy.spatial.transform
        inv_rotation_3d = np.eye(3)
        inv_rotation_3d[:2, :2] = inv_rotation
        
        # Convert transformation matrix to ROS format output
        inv_r = Rotation.from_matrix(inv_rotation_3d)
        inv_roll, inv_pitch, inv_yaw = inv_r.as_euler('xyz')
        
        print("\nROS Format Output (Map2 to Map1):")
        print(f"position:")
        print(f"  x: {inv_translation[0]:.6f}")
        print(f"  y: {inv_translation[1]:.6f}")
        print(f"  z: 0.000000")
        print(f"orientation:")
        print(f"  x: 0.000000")
        print(f"  y: 0.000000")
        print(f"  z: {np.sin(inv_yaw/2):.6f}")
        print(f"  w: {np.cos(inv_yaw/2):.6f}")
        
        # Print matching point pairs for reference
        print("\nMatching Point Pairs (World Coordinates):")
        for i, (p1, p2) in enumerate(zip(self.world_points1, self.world_points2)):
            print(f"Pair {i+1}:")
            print(f"  Map 1: ({p1[0]:.4f}, {p1[1]:.4f})")
            print(f"  Map 2: ({p2[0]:.4f}, {p2[1]:.4f})")
    
    def visualize_aligned_points(self):
        """Visualize the transformation result by showing point clouds in world coordinates"""
        if self.transform is None:
            print("Please calculate the transformation first")
            return
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(1, 2, width_ratios=[1, 1])
        plt.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.95)  # Maximize display area
        
        # First subplot: Original map
        ax1 = plt.subplot(gs[0])
        ax1.imshow(self.pgm1, cmap='gray')
        ax1.set_title('Original Map 1')
        
        # Second subplot: Point cloud in world coordinates
        ax2 = plt.subplot(gs[1])
        ax2.set_title('Point Cloud (World Coordinates)')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        
        try:
            # Get parameters for both maps
            h1, w1 = self.pgm1.shape
            h2, w2 = self.pgm2.shape
            resolution1 = self.yaml1['resolution']
            resolution2 = self.yaml2['resolution']
            origin1 = self.yaml1['origin']
            origin2 = self.yaml2['origin']
            
            # Extract point clouds from maps
            progress_text = plt.figtext(0.5, 0.01, "Processing point cloud data...", ha='center', va='center', fontsize=12)
            plt.pause(0.1)  # Allow progress to display
            
            # Extract all points from map 1
            points1_x = []
            points1_y = []
            for y in range(0, h1):
                for x in range(0, w1):
                    # If it's an occupied area (black or gray, value less than threshold)
                    if self.pgm1[y, x] < 200:
                        # Convert pixel coordinates to world coordinates
                        # Note the y-coordinate needs to be flipped
                        wx = origin1[0] + x * resolution1
                        wy = origin1[1] + (h1 - y - 1) * resolution1
                        
                        points1_x.append(wx)
                        points1_y.append(wy)
            
            # Extract all points from map 2
            points2_x = []
            points2_y = []
            for y in range(0, h2):
                for x in range(0, w2):
                    # If it's an occupied area (black or gray, value less than threshold)
                    if self.pgm2[y, x] < 200:
                        # Convert pixel coordinates to world coordinates
                        wx = origin2[0] + x * resolution2
                        wy = origin2[1] + (h2 - y - 1) * resolution2
                        
                        # Note: self.transform is the transformation from map1 to map2 coordinate system
                        # Therefore, we need to use the inverse matrix to transform map2 points to map1 coordinate system
                        inv_transform = np.linalg.inv(self.transform)
                        wx_transformed, wy_transformed, _ = inv_transform @ np.array([wx, wy, 1])
                        
                        points2_x.append(wx_transformed)
                        points2_y.append(wy_transformed)
            
            # Display the point clouds
            ax2.scatter(points1_x, points1_y, c='blue', s=0.5, alpha=0.5, label='Map 1 Points')
            ax2.scatter(points2_x, points2_y, c='red', s=0.5, alpha=0.5, label='Map 2 Points (Transformed)')
            
            # Get overall coordinate range
            x_min = min(min(points1_x), min(points2_x))
            x_max = max(max(points1_x), max(points2_x))
            y_min = min(min(points1_y), min(points2_y))
            y_max = max(max(points1_y), max(points2_y))
            
            # Set consistent axis ranges
            margin = 1.0  # Margin in meters
            ax2.set_xlim(x_min - margin, x_max + margin)
            ax2.set_ylim(y_min - margin, y_max + margin)
            
            # Show control points on original maps
            for i, (p1, p2) in enumerate(zip(self.points1, self.points2)):
                color = self.colors[i % len(self.colors)]
                
                # Draw points on the first map
                ax1.plot(p1[0], p1[1], marker='o', markersize=8, 
                       color=color, markeredgecolor='black')
                ax1.annotate(f"{i+1}", (p1[0]+5, p1[1]+5), color=color, 
                           fontsize=10, fontweight='bold', backgroundcolor='white', alpha=0.8)
                
                # Get world coordinates for control points
                wx1, wy1 = self.world_points1[i]
                wx2, wy2 = self.world_points2[i]
                
                # Transform map2 control points to map1 coordinate system
                inv_transform = np.linalg.inv(self.transform)
                transformed_x, transformed_y, _ = inv_transform @ np.array([wx2, wy2, 1])
                
                # Show control point pairs on point cloud
                ax2.plot(wx1, wy1, marker='o', markersize=8, 
                       color=color, markeredgecolor='black')
                ax2.annotate(f"{i+1}", (wx1+0.1, wy1+0.1), color=color, 
                           fontsize=10, fontweight='bold', backgroundcolor='white', alpha=0.8)
                
                ax2.plot(transformed_x, transformed_y, marker='x', markersize=8, 
                       color=color, markeredgewidth=2)
            
            # Add legend
            ax2.legend(loc='upper right')
            
            # Set equal axis aspect ratio
            ax2.set_aspect('equal')
            
            # Update progress
            progress_text.set_text("Point cloud processing complete")
            
            # Add save button
            ax_save = plt.axes([0.45, 0.01, 0.1, 0.04])
            button_save = Button(ax_save, 'Save Result')
            
            def save_result(event):
                # Get filename without extension
                map1_name = os.path.splitext(os.path.basename(self.yaml1['image']))[0]
                map2_name = os.path.splitext(os.path.basename(self.yaml2['image']))[0]
                
                # Create output filename
                output_path = f"{map1_name}_points_with_{map2_name}.png"
                
                # Save the current figure
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Point cloud image saved as: {output_path}")
            
            # Register callback
            button_save.on_clicked(save_result)
            
        except Exception as e:
            print(f"Error during point cloud visualization: {e}")
            import traceback
            traceback.print_exc()
        
        # Display the figure
        plt.tight_layout()
        plt.show()
    
    def select_points(self):
        """Interactive selection of corresponding points"""
        # Create graphics window
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        plt.subplots_adjust(bottom=0.2, top=0.95, left=0.05, right=0.95)  # Maximize map area
        
        # Create a text area for displaying coordinates
        ax_status = plt.axes([0.1, 0.15, 0.8, 0.05])
        ax_status.axis('off')
        self.status_text = ax_status.text(0.5, 0.5, "Hover over Map 1 and click to select a point", 
                                         ha='center', va='center', fontsize=12)
        
        # Display two images
        ax1.imshow(self.pgm1, cmap='gray')
        ax1.set_title('First Map (Click to select points)', color='red')
        ax2.imshow(self.pgm2, cmap='gray')
        ax2.set_title('Second Map')
        
        # Create buttons
        ax_switch = plt.axes([0.3, 0.05, 0.15, 0.075])
        button_switch = Button(ax_switch, 'Reset Current Point')
        
        ax_compute = plt.axes([0.5, 0.05, 0.15, 0.075])
        button_compute = Button(ax_compute, 'Calculate Transform')
        
        ax_visualize = plt.axes([0.7, 0.05, 0.15, 0.075])
        button_visualize = Button(ax_visualize, 'Visualize Result')
        
        # For displaying pending point
        self.pending_point_artist = None
        self.pending_crosshair_h = None
        self.pending_crosshair_v = None
        
        # Store artists for points to ensure they remain visible
        self.point_artists1 = []
        self.point_artists2 = []
        
        # Update plot function to display selected points
        def update_plot():
            # Clear existing artists
            for artist_list in [self.point_artists1, self.point_artists2]:
                for artist in artist_list:
                    if artist is not None:
                        try:
                            artist.remove()
                        except:
                            pass
            self.point_artists1 = []
            self.point_artists2 = []
            
            # Redraw the base images
            ax1.clear()
            ax2.clear()
            ax1.imshow(self.pgm1, cmap='gray')
            ax2.imshow(self.pgm2, cmap='gray')
            
            # Hide axis ticks to maximize display area
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            # Highlight the active map based on selection state
            if self.waiting_for_second_point:
                ax1.set_title('First Map')
                ax2.set_title('Second Map (Click to select matching point)', color='red')
            else:
                ax1.set_title('First Map (Click to select point)', color='red')
                ax2.set_title('Second Map')
            
            # Draw selected points with better visibility
            for i, p1 in enumerate(self.points1):
                # Use different colors for different point pairs
                color = self.colors[i % len(self.colors)]
                
                # Ensure points are within image bounds
                if not (0 <= p1[0] < self.pgm1.shape[1] and 0 <= p1[1] < self.pgm1.shape[0]):
                    if self.debug:
                        print(f"Warning: Point {i+1} on Map 1 is out of bounds: {p1}")
                    continue
                
                # Get world coordinates for this point (used for internal calculations, not displayed)
                world_p1 = self.world_points1[i] if i < len(self.world_points1) else self.image_to_world(p1, 0)
                
                # Draw point marker
                point_artist = ax1.plot(p1[0], p1[1], marker='o', markersize=10, 
                                      color=color, markeredgecolor='black')[0]
                self.point_artists1.append(point_artist)
                
                # Draw ID number only
                text_artist = ax1.annotate(f"{i+1}", (p1[0]+5, p1[1]+5), color=color, 
                                        fontsize=10, fontweight='bold', backgroundcolor='white', alpha=0.8)
                self.point_artists1.append(text_artist)
            
            for i, p2 in enumerate(self.points2):
                # Use same color for corresponding point pairs
                color = self.colors[i % len(self.colors)]
                
                # Ensure points are within image bounds
                if not (0 <= p2[0] < self.pgm2.shape[1] and 0 <= p2[1] < self.pgm2.shape[0]):
                    if self.debug:
                        print(f"Warning: Point {i+1} on Map 2 is out of bounds: {p2}")
                    continue
                
                # Get world coordinates for this point (used for internal calculations, not displayed)
                world_p2 = self.world_points2[i] if i < len(self.world_points2) else self.image_to_world(p2, 1)
                
                # Draw point marker
                point_artist = ax2.plot(p2[0], p2[1], marker='o', markersize=10, 
                                      color=color, markeredgecolor='black')[0]
                self.point_artists2.append(point_artist)
                
                # Draw ID number only
                text_artist = ax2.annotate(f"{i+1}", (p2[0]+5, p2[1]+5), color=color, 
                                        fontsize=10, fontweight='bold', backgroundcolor='white', alpha=0.8)
                self.point_artists2.append(text_artist)
            
            fig.canvas.draw_idle()
        
        def reset_current_point(event):
            """Reset the current point selection state"""
            # If waiting for second point, go back to first map
            if self.waiting_for_second_point:
                self.waiting_for_second_point = False
                # If there's a point already selected on map 1, remove it
                if len(self.points1) > len(self.points2):
                    self.points1.pop()
                    if self.world_points1 and len(self.world_points1) > len(self.world_points2):
                        self.world_points1.pop()
            
            # Remove pending point visual
            if self.pending_point_artist is not None:
                self.pending_point_artist.remove()
                self.pending_point_artist = None
            if self.pending_crosshair_h is not None:
                self.pending_crosshair_h.remove()
                self.pending_crosshair_h = None
            if self.pending_crosshair_v is not None:
                self.pending_crosshair_v.remove()
                self.pending_crosshair_v = None
            
            # Reset pending point
            self.pending_point = None
            
            # Update status text
            self.status_text.set_text("Selection reset. Click on Map 1 to select a new point.")
            
            # Update the display
            update_plot()
        
        def compute_callback(event):
            """Compute the alignment transformation"""
            if self.compute_alignment():
                self.display_alignment_results()
                self.status_text.set_text("Transformation calculated. Use 'Visualize Result' to see the alignment.")
                fig.canvas.draw_idle()
            else:
                self.status_text.set_text("Failed to calculate transformation. Need at least 2 point pairs.")
                fig.canvas.draw_idle()
        
        def visualize_callback(event):
            """Visualize the alignment results"""
            if self.transform is not None:
                plt.close(fig)  # Close the current figure before opening a new one
                self.visualize_aligned_points()
            else:
                self.status_text.set_text("Please calculate transformation first using 'Calculate Transform' button.")
                fig.canvas.draw_idle()
        
        def onmotion(event):
            """Handle mouse movement events"""
            # Clear previous pending point if it exists
            if self.pending_point_artist is not None:
                self.pending_point_artist.remove()
                self.pending_point_artist = None
            if self.pending_crosshair_h is not None:
                self.pending_crosshair_h.remove()
                self.pending_crosshair_h = None
            if self.pending_crosshair_v is not None:
                self.pending_crosshair_v.remove()
                self.pending_crosshair_v = None
            
            # Get active axes based on current selection state
            ax = ax1 if not self.waiting_for_second_point else ax2
            
            if event.inaxes == ax:
                x, y = int(event.xdata), int(event.ydata)
                
                # Display image coordinates
                img_coords = f"Image coordinates: ({x}, {y})"
                
                # Update status text
                self.status_text.set_text(img_coords)
                
                # Show crosshair
                self.pending_crosshair_h = ax.axhline(y=y, color='yellow', alpha=0.5, linestyle='--')
                self.pending_crosshair_v = ax.axvline(x=x, color='yellow', alpha=0.5, linestyle='--')
                
                # Draw pending point
                self.pending_point_artist = ax.plot(x, y, marker='o', markersize=5, 
                                                 color='yellow', alpha=0.7)[0]
                
                # Save pending point coordinates
                self.pending_point = [x, y]
                
                # Redraw canvas
                fig.canvas.draw_idle()
        
        def onclick(event):
            """Handle mouse click events"""
            # Check if click is in the appropriate map based on selection state
            if (not self.waiting_for_second_point and event.inaxes == ax1) or \
               (self.waiting_for_second_point and event.inaxes == ax2):
                
                if event.button == 1 and self.pending_point is not None:  # Left click
                    x, y = self.pending_point
                    
                    if not self.waiting_for_second_point:
                        # First map - add point to the first map
                        self.points1.append([x, y])
                        self.world_points1.append(self.image_to_world([x, y], 0))
                        
                        # Toggle to waiting for second point
                        self.waiting_for_second_point = True
                        self.status_text.set_text(f"Point {len(self.points1)} selected on Map 1. Now select matching point on Map 2.")
                    else:
                        # Second map - add point to the second map
                        self.points2.append([x, y])
                        self.world_points2.append(self.image_to_world([x, y], 1))
                        
                        # Toggle back to first map for next point
                        self.waiting_for_second_point = False
                        self.current_point_index += 1
                        
                        # Update status for next point
                        self.status_text.set_text(f"Point pair {len(self.points2)} completed. Click on Map 1 to select next point.")
                    
                    # Clear pending point
                    self.pending_point = None
                    
                    # Update the display
                    update_plot()
        
        # Register callbacks
        button_switch.on_clicked(reset_current_point)
        button_compute.on_clicked(compute_callback)
        button_visualize.on_clicked(visualize_callback)
        fig.canvas.mpl_connect('motion_notify_event', onmotion)
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        # Display initial plot
        update_plot()
        
        # Show figure and wait for interaction
        plt.show() 