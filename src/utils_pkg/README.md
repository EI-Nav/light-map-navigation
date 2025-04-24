# Map Alignment Process

This module provides tools for aligning OpenStreetMap (OSM) data with Point Cloud Data (PCD) to create maps for navigation.

## Overview

The alignment process consists of the following steps:
1. Convert OSM data to PGM format
2. Convert PCD data to PGM format
3. Interactively align the two maps
4. Apply the resulting transformation matrix
5. Generate the final aligned maps

## Components

* `osm_converter.py`: Converts OSM files to PGM format
* `pcd_converter.py`: Converts PCD point clouds to PGM format
* `pgm_aligner.py`: Interactive tool for aligning PGM maps
* `transform.py`: Applies transformations to PGM maps
* `alignment_pipeline.py`: Main pipeline integrating all steps

## Usage

```bash
python -m utils_pkg.alignment_process.alignment_pipeline \
    --osm_file /path/to/map.osm \
    --pcd_file /path/to/pointcloud.pcd \
    --output_dir /path/to/output \
    --resolution 0.2 \
    --tags building,footway \
    --min_z 0.5 \
    --max_z 2.5 \
    --interactive
```

### Parameters

* `--osm_file`: Path to input OSM file (required)
* `--pcd_file`: Path to input PCD file (required)
* `--output_dir`: Output directory for aligned maps (required)
* `--resolution`: Map resolution in meters/pixel (default: 0.2)
* `--tags`: OSM tags to extract, comma-separated (default: "building,footway")
* `--padding`: Map padding in meters (default: 10.0)
* `--min_z`: Minimum height for PCD filtering (default: None)
* `--max_z`: Maximum height for PCD filtering (default: None)
* `--margins`: Margins for transformed map [left right top bottom] in meters (default: [10.0, 10.0, 10.0, 10.0])
* `--interactive`: Enable interactive alignment mode (recommended)
* `--skip_alignment`: Skip alignment step and use identity transformation

## Output Files

The alignment process generates the following files in the output directory:

* `temp/osm_map.pgm` and `temp/osm_map.yaml`: Intermediate OSM map
* `temp/pcd_map.pgm` and `temp/pcd_map.yaml`: Intermediate PCD map
* `final/transformed_osm_map.pgm` and `final/transformed_osm_map.yaml`: Transformed OSM map
* `final/pcd_map.pgm` and `final/pcd_map.yaml`: Final PCD map
* `final/transform_matrix.txt`: Transformation matrices in both directions

## Interactive Alignment Usage

When running in interactive mode:

1. Click points on the first map (OSM)
2. The system will automatically switch to the second map (PCD)
3. Click the corresponding point on the second map
4. Repeat for at least 2-3 pairs of points for accurate alignment
5. Click "Calculate Transform" to compute the transformation
6. Click "Visualize Result" to see the alignment result

## Example

1. **Install Open3D**

```bash
pip install open3d
```

2. **Run the alignment process**

```bash
python3 run_alignment.py --osm_file example/BUCT/BUCT.osm --pcd_file example/BUCT/BUCT.pcd --output_dir example/ --min_z 7.0 --max_z 10.0 --interactive
```

![Initial alignment interface](../../doc/Map_Alignment/image-20250424141929730.png)

3. **Select at least three pairs of corresponding points, then click "Calculate Transform"**

![Points selection](../../doc/Map_Alignment/image-20250424142023642.png)

4. **Click "Visualize Result" to check the alignment (it may take a few seconds to render)**

![Alignment visualization](../../doc/Map_Alignment/image-20250424142416825.png)

5. **If the alignment is satisfactory, close the program interface and the results will be automatically saved (wait for the terminal to complete)**

![Terminal output](../../doc/Map_Alignment/image-20240718102111552.png)

6. **Pay attention to these three files - the first two will be used as the final OSM PGM files**

![Output files](../../doc/Map_Alignment/image-20250424144421735.png)

7. **Copy the resulting transformation matrix to the necessary configuration files**

![Transformation matrix](../../doc/Map_Alignment/image-20250424144702605.png)

Currently, there are two places where you need to update the transformation matrix:

1) In `/workspaces/light-map-navigation/src/delivery_bringup/config/delivery_bringup_sim.yaml`:
   - Location 1: Use OSM-->PCD transformation
   - Location 2: Use PCD-->OSM transformation

![Configuration file](../../doc/Map_Alignment/image-20250424150332555.png)

2) When drawing OSM in OPEN-MIND, use the PCD-->OSM transformation matrix