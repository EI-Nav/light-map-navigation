# LLM Navigation

运行环境：推荐使用提供的docker进行开发，节约环境对齐的时间。[主文档](../../README.md)



目前项目中包含三个脚本，其功能如下：

- Tongyi_LLM_Nav.py: 基于通义千问实现导航助理，分析要去的地点，并通过osm_route获取导航waypoints.

  ```bash
  # 主要关注这个
  python3 Tongyi_LLM_Nav.py
  ```

  

- osm_route.py: 输入起始点和终止点，从OSRM服务器上获取路线.

  ```bash
  python3 osm_route.py
  ```

  

- osm_handler.py：获取指定类型元素的位置.

  ```bash
  # 可暂时忽略这个脚本
  python3 osm_handler.py osm_world.osm
  ```