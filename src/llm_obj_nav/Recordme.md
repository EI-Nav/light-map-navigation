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

## 12-19
* 修复bug，现在发现的一个问题：机器人的位置在`objnav_agent.py`中不随着代码运行更新，程序被后续的东西阻塞住了。
面临这问题是：程序运行过程中没有进话题的回调函数，需要找一个方法进行解决(多线程？)

## 12-20 
* 利用`threading`库开启多线程，但是现在有一个问题是`mapper`中的机器人位置没有更新，这个需要去看一下
* 多线程解决思路及注意点：
    * 有关`ros`的回调必须都放在一个线程中，其他的线程中不能出现`ros`相关内容（包括`ros`的导航器）
* todo: 
    * 目前GPT4V的`Prompt`还没有调好，有时候可以分析出正确的方向但是无法输出成正确的格式，导致后续无法解析从而乱生成方向，需要进行修改。
    * 还需要研究一下wp的生成方式
    * 需要对`mapper`中的相关内容进行更新

## 12-23 
* 发现一个问题，不能让机器人阻塞式去走，阻塞进程的话会影响后续的所有过程。

## 12-24
* 修改`obj_agent`中的主函数部分，现在机器人可以连续进行行走
* 修复了一个函数`bug`,使得`self.obs`不会存在没有数的情况

## 12-26
* 目前`GPT4V`解析出来的方向是正确的，但是生成的`waypoints`是不对的，这个需要进行修改,需要看生成的`value map`是什么样子的（思路是可以在`rviz`中进行可视化操作）
* 现在需要在`rviz`中可视化的东西有：
    * 通过`pathfinding`搜索出来的路径
    * ~~ 生成的`.ply`的点云文件(各个`value map`)以点云的形式发布 ~~
* 目前首次loop中的`value map`生成有问题
* 目前需要修复的Bug
    - [] 现在由于`mapper`的初始化操作，只能在gazebo的原点进行启动，不能在任意位置启动，需要修改`mapper`的初始化逻辑，可以赋值为当前位置，而不是0. 
    - [] 代价地图生成的是长条状，按道理应该生成类似于矩形的东西，导致后续选点的时候局限性很大。现在所有问题就聚焦到了代价地图的生成上
    - [] 生成的`waypoint`为什么会在地图下面？？？坐标系问题？?
    - ~~[X] 需要修改相机图片的坐标系(期望坐标系是，实际坐标系是[右、下、前])~~
    - ~~[X] 在获取全景图像的时候，由于原来的代码是机器人旋转360度获取，我现在改成了通过读取六个摄像头，所以在`mapper.update`中，需要传入的就应该是六个方向每一个的`rotation`,现在只是获取0号相机的`rotation`~~

* 总结：调整了`habitat`与`gazebo`之间的坐标转换，现在value map的生成效果比原来好很多，现存问题:
    * 初始化的时候第一次加载`mapper`加载的不是很对
    * 最终机器人停止或者重新进行`make_plan`函数执行的逻辑需要更正，一种现象是由于move_distance距离过远，导致一直无法进行make_plan，出现机器人原地转圈的情况，这种情况需要改进一下逻辑。   


## 12-27
* 关于坐标系转换的一个思路：把所有的东西（图像、位置）都模拟成habitat坐标系输入，然后把输出转换为从habitat坐标系转换到gazebo坐标系，中间的**可视化**也需要转换成gazebo坐标系
    * step1： 将图片从**右下前**转换为**右上后**