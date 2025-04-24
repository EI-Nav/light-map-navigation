#!/usr/bin/env python3
"""
Script to launch the map alignment tool.
Resolves relative import issues. Simply execute this script to start the alignment process.
"""
import sys
import os
import argparse
import subprocess

def main():
    # Get project root directory (assuming current script is in src/utils_pkg)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))  # Go back to project root
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Launch OSM and PCD map alignment tool')
    parser.add_argument('--osm_file', required=True, help='OSM file path')
    parser.add_argument('--pcd_file', required=True, help='PCD file path')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--resolution', type=float, default=0.2, help='Map resolution (meters/pixel)')
    parser.add_argument('--tags', default='building,footway', help='OSM tags, comma separated')
    parser.add_argument('--padding', type=float, default=10.0, help='Map padding (meters)')
    parser.add_argument('--min_z', type=float, default=None, help='PCD point cloud minimum height filter (meters)')
    parser.add_argument('--max_z', type=float, default=None, help='PCD point cloud maximum height filter (meters)')
    parser.add_argument('--margins', type=float, nargs=4, default=[10.0, 10.0, 10.0, 10.0], 
                        help='Transformation map margins [left right top bottom] in meters')
    parser.add_argument('--interactive', action='store_true', help='Use interactive mode to align maps')
    parser.add_argument('--skip_alignment', action='store_true', help='Skip alignment step, use existing transformation')
    
    args = parser.parse_args()
    
    # Prepare to run alignment_pipeline.py file directly
    alignment_script = os.path.join(project_root, 'src', 'utils_pkg', 'utils_pkg', 'alignment_process', 'alignment_pipeline.py')
    
    # Ensure script exists
    if not os.path.exists(alignment_script):
        print(f"Error: Alignment script not found: {alignment_script}")
        sys.exit(1)
        
    # Build command
    cmd = [sys.executable, alignment_script]
    
    # Add all arguments
    if args.osm_file:
        cmd.extend(["--osm_file", args.osm_file])
    if args.pcd_file:
        cmd.extend(["--pcd_file", args.pcd_file])
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    if args.resolution:
        cmd.extend(["--resolution", str(args.resolution)])
    if args.tags:
        cmd.extend(["--tags", args.tags])
    if args.padding:
        cmd.extend(["--padding", str(args.padding)])
    if args.min_z is not None:
        cmd.extend(["--min_z", str(args.min_z)])
    if args.max_z is not None:
        cmd.extend(["--max_z", str(args.max_z)])
    if args.margins:
        cmd.extend(["--margins"] + [str(m) for m in args.margins])
    if args.interactive:
        cmd.append("--interactive")
    if args.skip_alignment:
        cmd.append("--skip_alignment")
    
    # Set environment variables, add project root to PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + ":" + env.get('PYTHONPATH', '')
    
    # Use subprocess instead of execvp to pass environment variables
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running alignment script: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nUser interrupted")
        sys.exit(0)

if __name__ == "__main__":
    main() 