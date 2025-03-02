# OPEN: Openstreetmap-enhanced oPen-air sEmantic Navigation

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2502.09238)
[![arXiv](https://img.shields.io/badge/arXiv-2502.09238-b31b1b)](http://arxiv.org/abs/2502.09238)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://ei-nav.github.io/OpenBench/)

OPEN is a system designed for efficient last-mile delivery using autonomous robots. It leverages OpenStreetMap (OSM) for scalable map representation, Large Language Models (LLMs) for understanding delivery instructions, and Vision-Language Models (VLMs) for localization, map updates, and house number recognition.

## Installation

The current development environment is Ubuntu 22.04, ROS2 Humble, and Gazebo Classic 11.

### Using Docker

This project includes a basic Dockerfile and can be used with [Dev Container](https://containers.dev/) for simulation testing and development.

1. Install Docker and NVIDIA Container Toolkit and configure them on the host machine:

    ```bash
    # Docker installation
    https://docs.docker.com/engine/install/ubuntu/
    wget http://fishros.com/install -O fishros (optional)
    
    # NVIDIA Container Toolkit installation
    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

    # nvidia-docker installation
    sudo apt install nvidia-docker2
    
    # Configure Docker container to use GPU
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

2. Download and install the image:

    ```sh
    # Download from Baidu Cloud
    Link: https://pan.baidu.com/s/1xSDHBTYh8PTt4HNBLy7FWA?pwd=emgo 
    Password: emgo 
    
    # Load the image
    docker load -i open.tar
    ```
    
3. Clone the repository:
    ```sh
    git clone https://github.com/EI-Nav/light-map-navigation.git --depth=1
    git submodule init
    git submodule update
    
    cd src/classic_localization/FAST_LIO/
    git submodule init
    git submodule update
    ```

4. Download Gazebo models and extract them to the `~/.gazebo/` directory:
    ```sh
    # Download from Baidu Cloud Drive (Choose one of the following):
    
    # Option 1: Complete model library (includes unused models)
    Link: https://pan.baidu.com/s/1Mkn4BHXaeWyvjxCMvw-gZQ
    Password: pp8a
    
    # Option 2: Minimal model library (recommended, contains only required models)
    Link: https://pan.baidu.com/s/1CojOxnFrJjSLM_jpmkK_KA
    Password: 49xs
    
    # After extraction, place files in
    ~/.gazebo/
    ```

5. Install the `ms-vscode-remote.remote-containers` extension in VSCode on the host machine.

6. Open this project in VSCode, press `Ctrl+Shift+P`, type and select `Dev Containers: Rebuild and Reopen in Container`.

7. Install dependencies and compile:
    ```sh
    # Set up your proxy
    export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
    
    # Install dependencies
    ./build_dependencies.sh
    
    # Compile
    ./build_project.sh
    ```

### Source Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/EI-Nav/light-map-navigation.git --depth=1
    git submodule init
    git submodule update

    cd src/classic_localization/FAST_LIO/
    git submodule init
    git submodule update
    ```

2. Install [Livox SDK2](https://github.com/Livox-SDK/Livox-SDK2):

    ```sh
    sudo apt install cmake
    ```

    ```sh
    git clone https://github.com/Livox-SDK/Livox-SDK2.git
    cd ./Livox-SDK2/
    mkdir build
    cd build
    cmake .. && make -j
    sudo make install
    ```

3. Set up the conda environment:

    ```sh
    conda create --name dl_env python=3.10
    conda activate dl_env
    
    conda install -c conda-forge libstdcxx-ng
    pip install "numpy<2.0" pandas matplotlib scikit-learn jupyterlab
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install opencv-python pycocotools matplotlib onnxruntime
    pip install git+https://github.com/facebookresearch/sam2.git
    pip install git+https://github.com/IDEA-Research/GroundingDINO.git
    ```

4. Install dependencies for ros package:

    ```sh
    cd light-map-navigation
    
    rosdep install -r --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y
    
    pip3 install -r requirements.txt
    ```

4. Install other dependencies according to the [build_dependencies.sh](./build_dependencies.sh).

4. Compile:

    ```sh
    ./build.sh
    ```



## Configuration

To use this repository, you'll need to configure the following credentials:

### VLM API

```yaml
# /workspace/light-map-navigation/src/building_entrance_recognition/config/api_config.yaml
vlm_api:
  key: "your_key"
  base_url: "your_base_url"
  model_name: "your_model_name" 
```

### LLM API

```yaml
# /workspace/light-map-navigation/src/task_planning/config/api_config.yaml
llm_api:
  key: "your_key"
  base_url: "your_base_url"
  model_name: "your_model_name" 
```


### GSAM2 Config

```yaml
grounded_sam2_node:
  ros__parameters:
    gsam2:
      scale: 1.0
      do_sharpen: false
      base_path: "/workspaces/light-map-navigation/src/grounded_sam2/grounded_sam2/grounded_sam2/"
      text_prompt: "ground . car . building"
      sam2_checkpoint: "./checkpoints/sam2.1_hiera_small.pt"
      sam2_model_config: "configs/sam2.1/sam2.1_hiera_s.yaml"
      grounding_dino_config: "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
      grounding_dino_checkpoint: "gdino_checkpoints/groundingdino_swint_ogc.pth"
      box_threshold: 0.3
      text_threshold: 0.25
      area_threshold: 80
      processing_rate: 10.0
```



## Run

```sh
# Allow the container to access the host's X server
xhost +
```

### Simulation Mode Example

```sh
# Run the robot's navigation system and simulation environment
ros2 launch classic_nav_bringup bringup_sim.launch.py \
world:=MEDIUM_OSM \
mode:=nav \
lio:=fastlio \
localization:=icp \
lio_rviz:=False \
nav_rviz:=True \
use_sim_time:=True

# Run delivery-related nodes
ros2 launch delivery_bringup delivery_system_sim.launch.py

# Run client to send delivery requests
ros2 run delivery_executor delivery_executor_action_client
```

### GSAM2 Node
If you want to use Grounded-SAM-2 for open-vocabulary instance segmentation, you can run the following commands:
```sh
# Set up your proxy
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890

# Download SAM2 checkpoints
cd src/grounded_sam2/grounded_sam2/grounded_sam2/checkpoints
bash download_ckpts.sh

# Download GroundingDINO checkpoints
cd src/grounded_sam2/grounded_sam2/grounded_sam2/gdino_checkpoints
bash download_ckpts.sh

# Run Grounded-SAM-2 server
cd /workspaces/light-map-navigation
ros2 launch grounded_sam2 gsam2_node.launch.py
```



## Robot Model

Four-wheeled vehicle using a differential drive model.



## Known Issues

- There are coordinate system issues, with the vehicle appearing below the map.



## TODO

- Delete old delivery code, models, and tests.
- Improve documentation.



## Acknowledgments

This project builds upon and is inspired by several excellent works:

- [PB-RM-Simulation](https://github.com/LihanChen2004/pb_rm_simulation)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-SAM-2)
- [OpenStreetMap](https://www.openstreetmap.org)

We thank all the contributors of these open-source projects for their valuable work.


## Citation

If you find our work useful for your research, please consider giving this repository a star ⭐ and citing our paper as follows:

```bibtex
@misc{wang2025openbenchnewbenchmarkbaseline,
      title={OpenBench: A New Benchmark and Baseline for Semantic Navigation in Smart Logistics}, 
      author={Junhui Wang and Dongjie Huo and Zehui Xu and Yongliang Shi and Yimin Yan and Yuanxin Wang and Chao Gao and Yan Qiao and Guyue Zhou},
      year={2025},
      eprint={2502.09238},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2502.09238}, 
}
```