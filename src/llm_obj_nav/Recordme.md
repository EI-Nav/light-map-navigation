# InstructNav迁移工作日志
## 12-11
* 跨文件import python库，可以动态指定库的路径,放在需要导入的库前面
    ```python
    import os
    import sys
    sys.path.append(os.path.join(os.getcwd(), "src/llm_obj_nav/llm_obj_nav"))
    ```
* 调通`GLEE`分割算法，现在可以拿回来分割的`class`和`mask`
* 问题记录：遇到cv_bridge无法使用的情况，具体报错
```bash
ImportError: /lib/libgdal.so.30: undefined symbol: TIFFReadRGBATileExt, version LIBTIFF_4.0

解决方案：export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5
```

## 12-12
* 修改`objnav_agent.py`中的东西，核心是将`habitat`的部分改成`Gazebo`的部分
* `self.obs`的文件构成：由RGB图像,深度图像,机器人位置、相机位置组成
* todo: 
    * 还需要修改`objnav_agent.py`中的：需要验证这些方法或者参数究竟是不是必须的
    ```bash
    self.env.episode_over
    self.env.current_episode.object_category
    self.env.get_metrics
    ```
    * 下一步需要对`objnav_benchmark.py`进行修改，核心的任务是对`objnav_agent`进行一个调用和功能测试

## 12-16
* 继续修改`objnav_agent.py`，完成对于不必要参数的去除
* 添加了一个机器人旋转功能(有待测试一下)
* todo: 根据`objnav_benchmark.py`编写一个测试的main_node

## 12-17
* 增加机器人从多个相机获取图片的功能
* 将`obj_agent`封装成一个ros节点,其中里面使用到的相机参数、机器人位置都通过一个话题进行获取
* 现存问题：运行代码的时候出现报错
    ```bash
    self.segmentation_trajectory.append(self.mapper.segmentation)
    AttributeError: 'Instruct_Mapper' object has no attribute 'segmentation'
    ```
    * 原因：由于现在仿真环境中直接获取深度相机的深度图，没有获取根据RGB图对应的深度图，因此需要做一个深度图和RGB图的对齐，根据深度图发出一个和RGB图大小一致的深度图。
    * 具体思路：通过不同大小的深度图和RGB图对齐，最终输出一个和RGB图大小一致的深度信息，对应RGB图上每一个像素点的深度信息。

## 12-18
* 基本迁移完成，但是需要修改很多东西，目前有许多参数需要进行调整
    * Instruct_Mapper初始化参数中的`self.floor_height`等参数
    * 深度图像需要将深度图进行单位转换，从米转换到厘米