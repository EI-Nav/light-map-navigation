## 依赖及权重文件安装过程
***核心***：控制`numpy==1.23.5`和`numba`的版本
```bash
pip install numba numpy==1.23.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
1. 安装`pytorch`
    ```bash
    conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
2. 安装`Detectron2`
    ```bash
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron2/
    pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```
3. 安装`Deformable-DETR`
    * 注意安装这个库的时候，每一次都需要重新编译，而且要先确定好`torch`和`numpy`的版本
    ```bash
    git clone https://github.com/fundamentalvision/Deformable-DETR.git
    cd Deformable-DETR/models/ops/
    sh ./make.sh
    ```
4. 安装其他库
    * 法1：
    `pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
    * 法2：
    ```bash
    pip install pathfinding==1.0.9 -i https://pypi.tuna.tsinghua.edu.cn/simple

    pip install salesforce_lavis==1.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

    pip install xformers==0.0.28.post1 -i https://pypi.tuna.tsinghua.edu.cn/simple

    pip install apex==0.9.10dev -i https://pypi.tuna.tsinghua.edu.cn/simple

    pip install onnxscript -i https://pypi.tuna.tsinghua.edu.cn/simple

    pip install open3d==0.18.0 openai==1.45.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

    pip install opencv_python -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```
5. `docker`里面的`conda`环境中使用`rclpy`
    ```bash
    conda install -c conda-forge libstdcxx-ng
    ```

6. `conda`虚拟环境中使用`cv_bridge`(可写入bashrc)
    ```bash
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5 
    ```

7. 安装模型权重文件
    * 安装GLEE的权重文件 -> 下载地址 `Hugging Face hub`中
    ```bash
    https://huggingface.co/spaces/Junfeng5/GLEE_demo/tree/main
    下载: GLEE_SwinL_Scaleup10m.pth
    存放路径：instructnav/thirdparty/GLEE/
    ```
    * `CLIP`权重文件安装 -> 下载地址 `Hugging Face hub`中
    ```
    1. 先克隆小文件
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/openai/clip-vit-base-patch32
    2. 去网站下载大的权重文件
    https://huggingface.co/openai/clip-vit-base-patch32/tree/main
    3. 存放路径：instructnav/checkpoints
    ```
8. 配置权重文件
* `clip`配置：`instructnav/thirdparty/GLEE/glee/models/glee_model.py`中的`self.tokenizer`和`self.text_encoder`,配置的路径是`clip-vit-base-patch32`的对应路径
* 修改`instructnav/constants.py`中的`GLEE_CONFIG_PATH`和`GLEE_CHECKPOINT_PATH`

## 使用流程
```bash
0. 在urdf中添加摄像头
1. 启动仿真环境
2. python src/llm_obj_nav/llm_obj_nav/instructnav/objnav_agent.py
```