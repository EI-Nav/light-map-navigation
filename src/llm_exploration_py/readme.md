## 任务需求
* 在完成llm_navigation包后，机器人已经走到目标building附近，此时需要对目标building进行探索
## 实现效果
* 输入：building号和单元号
* 实现效果：去探索对应单元，中间不断进行反馈
* 返回：是否找到对应的单元号 success
## 注意事项
* 小车需要选择距离自身位置最近的点开始探索，在每到达一个waypoints的时候需要对楼进行拍照，上传至大模型进行询问
* 小车的位置通过`/Odometry`进行获取

## 代码说明:
* 启动服务端
    ``` bash
    ros2 run llm_exploration_py find_unit_server
    ```
* 客户端测试
    * 客户端参数: 楼号：`target_building = 'building13'` 单元号：`target_unit = '3'`
    ```bash
    ros2 run llm_exploration_py find_unit_client
    ```

## 现存问题
1. action部分的反馈没有做好，`client`能发送，但是收不到任务完成进度，判断任务完成的标志位在`server`端中有，但目前没有反馈回`client`端
