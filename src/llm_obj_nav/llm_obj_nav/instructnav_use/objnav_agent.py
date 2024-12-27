# ros interfaces
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped, PoseStamped, Quaternion
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

import math
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import quaternion

import numpy as np
import cv2
import ast
import open3d as o3d

from threading import Thread

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "src/llm_obj_nav/llm_obj_nav/instructnav"))
from mapping_utils.geometry import *
from mapping_utils.projection import *
from mapping_utils.path_planning import *
from mapping_utils.transform import gazebo_camera_intrinsic
from mapper import Instruct_Mapper
from llm_utils.nav_prompt import CHAINON_PROMPT,GPT4V_PROMPT
# from llm_utils.get_request_gpt import gpt_response,gptv_response
from llm_utils.gpt_request import gpt_response,gptv_response



class CameraPosition:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"CameraPosition(x={self.x}, y={self.y}, z={self.z})"

user_input = "Please approach the car parked on the roadside"

class HM3D_Objnav_Agent(Node):
    def __init__(self,mapper:Instruct_Mapper):

        self.mapper = mapper
        self.episode_samples = 0

        super().__init__('HM3D_Objnav_Agent')

        # 初始化变量
        self.position = [0.0, 0.0, 0.0]  # [x, y, z] 坐标
        self.rotation = [0.0, 0.0, 0.0, 0.0]  # 四元数 [x, y, z, w]
        self.rgb = None  # 存储 RGB 图像
        self.depth = None  # 存储深度图像
        self.action_wp = [None]

        self.camera_position = CameraPosition()  # 相机的 [x, y, z] 坐标
        self.camera_rotation = quaternion.quaternion(1,0,0,0)  # 相机的四元数 [x, y, z, w]

        self.camera_positions = [CameraPosition()] *6  # 相机的 [x, y, z] 坐标
        self.camera_rotations = [quaternion.quaternion(1,0,0,0)] * 6   # 相机的四元数 [x, y, z, w]

        # 用于临时存储接收到的最新数据
        self._latest_rgb = [None] * 6  # 存储6个相机的RGB图像
        self._latest_depth = [None] * 6  # 存储6个相机的深度图像
        self._latest_position = [0.0, 0.0, 0.0]
        self._latest_rotation = [0.0, 0.0, 0.0, 0.0]

        # 创建 CvBridge 对象用于图像转换
        self.bridge = CvBridge()

        # TF2 相关
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.navigator = BasicNavigator()

        # 标志位
        self.is_rgb_receive = False
        self.is_depth_receive = False
        self.is_odom_receive = False
        self.is_plan_init = False
        self.is_nav_process = False # 如果导航任务正在进行（小车正在运动，则为True，否则为Fasle）
        self.is_nav_action = False # 是否开始进行导航任务


        # 订阅6个相机的RGB图像
        self.rgb_subscribers = []
        for i in range(6):
            topic = f'/d435_{i}/color/image_raw'
            self.rgb_subscribers.append(
                self.create_subscription(
                    Image,
                    topic,
                    lambda msg, idx=i: self.rgb_callback(msg, idx),
                    10
                )
            )

        # 订阅6个相机的深度图像
        self.depth_subscribers = []
        for i in range(6):
            topic = f'/d435_{i}/depth/image_raw'
            self.depth_subscribers.append(
                self.create_subscription(
                    Image,
                    topic,
                    lambda msg, idx=i: self.depth_callback(msg, idx),
                    10
                )
            )

        # 订阅里程计信息
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/Odometry',  # 替换为实际的里程计话题
            self.odom_callback,
            10
        )

        # value map可视化
        self.affordance_publisher = self.create_publisher(PointCloud2, 'affordance_pointcloud', 10)
        self.semantic_publisher = self.create_publisher(PointCloud2, 'semantic_pointcloud', 10)
        self.history_publisher = self.create_publisher(PointCloud2, 'history_pointcloud', 10)
        self.action_publisher = self.create_publisher(PointCloud2, 'action_pointcloud', 10)
        self.gpt4v_publisher = self.create_publisher(PointCloud2, 'gpt4v_pointcloud', 10)
        self.obstacle_publisher = self.create_publisher(PointCloud2, 'obstacle_pointcloud', 10)

        self.thread1 = Thread(target=self.run)
        self.thread1.daemon = True  # 守护线程，程序退出时自动结束
        self.thread1.start()

        self.create_timer(0.5, self.nav_executor)

        # self.reset() # 初始化各个列表

        self.get_logger().info("Agent Init Successful !!!!!")

    def move_next(self, wp):
        self.action_wp = wp
        self.is_nav_action = True
        # 等待导航任务完成
        while self.is_nav_action:
            pass
    
    def nav_executor(self):
        if not self.is_nav_action:
            return

        # 目标点
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = self.action_wp[0]
        goal.pose.position.y = self.action_wp[1]
        goal.pose.position.z = 0.0

        # 计算到目标点的偏航角（yaw），但不会强制机器人调整到这个方向
        dx = self.action_wp[0] - self._latest_position[0]
        dy = self.action_wp[1] - self._latest_position[1]
        yaw = math.atan2(dy, dx)

        # 将偏航角转为四元数
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        # 发送目标点
        self.navigator.goToPose(goal)

        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                # self.get_logger().info(f"Feedback: {feedback}")
                self.position = [feedback.current_pose.pose.position.x,
                                 feedback.current_pose.pose.position.y,
                                 feedback.current_pose.pose.position.z]
                
                # print(self.position)
        
        # Get the result of the navigation
        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info(f"success go to next action")
            # 任务完成
            self.env_step()
            self.is_rgb_receive = False
            self.is_depth_receive = False
            self.is_odom_receive = False
            self.is_nav_action = False


    def odom_callback(self, msg):
        """处理里程计信息"""
        # 提取位置信息并存储到最新数据
        self._latest_position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ]

        # 提取旋转的四元数
        self._latest_rotation = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        self.is_odom_receive = True
        # self.get_logger().info("Received Odometry Information.")

    def transform_rgb_image(self, rgb_image):
        """
        将 RGB 图像的像素坐标从 Gazebo 的左下前坐标系转换到目标右上前坐标系。
        """
        # return cv2.flip(rgb_image, 0)  # 垂直翻转图像
        return rgb_image

    
    def transform_depth_image(self, depth_image):
        """
        将深度图像从 Gazebo 的左下前坐标系转换到目标右上前坐标系。
        """
        flipped_depth = cv2.flip(depth_image, 0)  # 垂直翻转图像
        # return flipped_depth  # 深度值取反
        return depth_image

    def rgb_callback(self, msg, camera_id):
        """处理 RGB 图像消息"""
        try:
            # 将 ROS 图像消息转换为 OpenCV 格式并存储
            raw_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._latest_rgb[camera_id] = self.transform_rgb_image(raw_rgb)
            # self.get_logger().info(f"Received RGB image from camera {camera_id}.")
            self.is_rgb_receive = True

        except Exception as e:
            self.get_logger().error(f"Failed to process RGB image from camera {camera_id}: {e}")

    # def depth_callback(self, msg, camera_id):
    #     """处理深度图像消息"""
    #     try:
    #         depth_image = [None] * 6  # 存储6个相机的深度图像
    #         # 将 ROS 图像消息转换为 OpenCV 格式并存储
    #         depth_image[camera_id] = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
    #         self._latest_depth[camera_id] = depth_image[camera_id].astype(np.float32) / 1000.0
    #         # self.get_logger().info(f"Received Depth image from camera {camera_id}.")
    #         self.is_depth_receive = True
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to process Depth image from camera {camera_id}: {e}")

    def depth_callback(self, msg, camera_id):
        try:
            depth_image_raw = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
            # 转换深度值到浮点数（以米为单位）
            depth_image_float = depth_image_raw.astype(np.float32) / 1000.0
            # 应用坐标变换
            self._latest_depth[camera_id] = self.transform_depth_image(depth_image_float)
            self.is_depth_receive = True
        except Exception as e:
            self.get_logger().error(f"Failed to process Depth image from camera {camera_id}: {e}")


    def get_panorama(self):
        """返回每个相机的RGB图像和深度图像"""
        panorama_rgb = {i: self._latest_rgb[i] for i in range(6)}
        panorama_depth = {i: self._latest_depth[i] for i in range(6)}

        return {"rgb": panorama_rgb, "depth": panorama_depth}

    def env_step(self):
        # 更新位置信息和旋转信息
        self.position = self._latest_position[:]
        self.rotation = self._latest_rotation[:]

        # 更新图像数据
        self.rgb = self._latest_rgb[0]
        self.depth = self._latest_depth[0]

        # 更新相机的位姿
        # self.update_camera_pose()

        self.update_camera_poses_all()

        self.get_logger().info(
            f"Step executed. Position: {self.position}, Rotation: {self.rotation}"
        )

        if self.rgb is not None:
            self.get_logger().info("RGB image updated.")
        else:
            self.get_logger().warn("RGB image not received yet.")

        if self.depth is not None:
            self.get_logger().info("Depth image updated.")
        else:
            self.get_logger().warn("Depth image not received yet.")

        self.obs = {
            "rgb": self.rgb,
            "depth": self.depth
        }

    def env_reset(self):
        # 重置位置和旋转信息
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0, 0.0]

        # 清空图像数据
        self.rgb = None
        self.depth = None

        # 重置相机信息
        self.camera_position = CameraPosition()
        self.camera_rotation = quaternion.quaternion(1,0,0,0)

        self.get_logger().info("Node state has been reset.")
        self.obs = {
            "rgb": self.rgb,
            "depth": self.depth
        }

    def update_camera_pose(self):
        try:
            # 查询 lidar_odom 到 d435_0_link 的坐标变换
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'lidar_odom',  # 父坐标系
                'd435_0_depth_optical_frame',  # 子坐标系
                rclpy.time.Time()  # 查询最新时间的变换
            )

            # 提取平移和旋转信息
            self.camera_position.x = transform.transform.translation.x
            self.camera_position.y = transform.transform.translation.y
            self.camera_position.z = transform.transform.translation.z
            
            self.camera_rotation = quaternion.from_float_array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])

            self.get_logger().info(
                f"Camera pose updated: Position = {self.camera_position}, Rotation = {self.camera_rotation}"
            )

        except Exception as e:
            self.get_logger().warn(f"Failed to lookup transform: {e}")

    def update_camera_poses_all(self):
        """
        更新所有相机的位置信息和旋转信息。
        """
        try:
            for i in range(6):
                camera_frame = f'd435_{i}_depth_optical_frame'  # 动态生成相机 frame 名称

                # 查询坐标变换
                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    'map',  
                    camera_frame, 
                    rclpy.time.Time()  
                )

                # 提取平移信息并存储为 CameraPosition
                position = CameraPosition(
                    # x=transform.transform.translation.x,
                    # y=transform.transform.translation.y,
                    # z=transform.transform.translation.z
                    x=transform.transform.translation.x,
                    y=transform.transform.translation.z,
                    z=transform.transform.translation.y
                )

                # 提取旋转信息
                rotation = quaternion.from_float_array([
                    # transform.transform.rotation.x,
                    # transform.transform.rotation.y,
                    # transform.transform.rotation.z,
                    # transform.transform.rotation.w
                    transform.transform.rotation.x,
                    transform.transform.rotation.z,
                    transform.transform.rotation.y,
                    transform.transform.rotation.w
                ])

                # 更新位置信息和旋转信息到列表中
                if i < len(self.camera_positions):
                    self.camera_positions[i] = position
                    self.camera_rotations[i] = rotation
                else:
                    self.camera_positions.append(position)
                    self.camera_rotations.append(rotation)

            self.get_logger().info(f"Updated positions: {self.camera_positions}")

        except Exception as e:
            self.get_logger().warn(f"Failed to lookup transform for camera {i}: {e}")

    # 将某些物体的名称翻译成更具体，更符合模型需求的术语
    def translate_objnav(self,object_goal):
        if object_goal.lower() == 'plant':
            return "Find the <%s>."%"potted_plant"
        elif object_goal.lower() == "tv_monitor":
            return "Find the <%s>."%"television_set"
        else:
            return "Find the <%s>."%object_goal
    
    def reset_debug_probes(self):
        self.rgb_trajectory = []
        self.depth_trajectory = []
        self.topdown_trajectory = []
        self.segmentation_trajectory = []

        self.gpt_trajectory = []
        self.gptv_trajectory = []
        self.panoramic_trajectory = []
        
        self.obstacle_affordance_trajectory = []
        self.semantic_affordance_trajectory = []
        self.history_affordance_trajectory = []
        self.action_affordance_trajectory = []
        self.gpt4v_affordance_trajectory = []
        self.affordance_trajectory = []

    def reset(self):
        self.episode_samples += 1
        self.episode_steps = 0
        # self.obs = self.env_reset() #！修改完成
        self.env_step() # 更新为机器人当前的最新位置
        # todo 现在这一步给出来的`self.camera_position`是在Gazebo坐标系下，可能需要换一下
        self.mapper.reset(self.camera_positions[0],self.camera_rotations[0]) 
        # 获取user的输入，这个可以手动给
        self.instruct_goal = user_input
        self.trajectory_summary = ""
        self.reset_debug_probes()     
    
    def rotate_panoramic(self,rotate_times = 6):
        """旋转环境，获取全景图像和点云，输入的是旋转次数"""
        self.temporary_pcd = []
        self.temporary_images = []
        panorama = self.get_panorama() # 获取机器人全景图
        self.env_step() # 更新当前的图像和各个相机位置
        for i in range(rotate_times):
            self.rgb_trajectory.append(cv2.cvtColor(panorama['rgb'][i],cv2.COLOR_BGR2RGB))
            self.depth_trajectory.append((panorama['depth'][i]/5.0 * 255.0).astype(np.uint8))

            # 更新地图，输入当前的RGB图像，深度图像，相机的位姿信息
            self.mapper.update(self.rgb_trajectory[-1],panorama['depth'][i],self.camera_positions[i],self.camera_rotations[i])
            # 获取当前的语义分割图像
            self.segmentation_trajectory.append(self.mapper.segmentation)
            # 获取当前环境中目标检测到的物体
            self.observed_objects = self.mapper.get_appeared_objects()

            self.temporary_pcd.append(self.mapper.current_pcd) # 记录每一个视角对应的PCD 
            self.temporary_images.append(self.rgb_trajectory[-1]) # 获取当前的RGB图像，循环了rotate_times次，所以最终这个数组存储的是rotate_times个RGB图像

            
    def concat_panoramic(self,images):
        try:
            height,width = images[0].shape[0],images[0].shape[1]
        except:
            height,width = 480,640
        background_image = np.zeros((2*height + 3*10, 3*width + 4*10, 3),np.uint8)
        copy_images = np.array(images,dtype=np.uint8)
        print("******************************************")
        print("len of copy_image:",len(copy_images))
        print("******************************************")
        for i in range(6):  # 假设 len(copy_images) = 6
            row = (i // 3)  # 计算行号：每行 3 个图像
            col = (i % 3)  # 计算列号：每列最多 3 个图像
            
            # 为当前图像添加文本
            copy_images[i] = cv2.putText(
                copy_images[i], 
                "Direction %d" % i,  # 使用 i 作为编号
                (100, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2, 
                (255, 0, 0), 
                6, 
                cv2.LINE_AA
            )
            
            # 将当前图像放置到背景图的正确位置
            background_image[
                10 * (row + 1) + row * height : 10 * (row + 1) + row * height + height,
                col * width + col * 10 : col * width + col * 10 + width, 
                :] = copy_images[i]
        
        return background_image
    
    def update_trajectory(self):
        """更新地图,获取当前的RGB图像,深度图像,语义分割结果,目标检测结果,以及相机的位姿信息"""
        while(not self.is_rgb_receive and not self.is_depth_receive and not self.is_odom_receive):
            print("In update loop waitting for sensor information")
        self.env_step()

        self.episode_steps += 1
        # self.metrics = self.env.get_metrics() #todo 还未修改，是否需要使用需要研究一下
        # 添加当前的RGB图像，深度图像
        self.rgb_trajectory.append(cv2.cvtColor(self.obs['rgb'],cv2.COLOR_BGR2RGB))
        self.depth_trajectory.append((self.obs['depth']/5.0 * 255.0).astype(np.uint8))

        # 更新地图，输入当前的RGB图像，深度图像，相机的位姿信息
        self.mapper.update(self.rgb_trajectory[-1],self.obs['depth'],self.camera_positions[0],self.camera_rotations[0])
        # 获取当前的语义分割图像
        self.segmentation_trajectory.append(self.mapper.segmentation)
        # 获取当前环境中目标检测到的物体
        self.observed_objects = self.mapper.get_appeared_objects()

        # 显示当前的RGB图像，深度图像，语义分割图像作为实时监控
        cv2.imwrite("monitor-rgb.jpg",self.rgb_trajectory[-1])
        cv2.imwrite("monitor-depth.jpg",self.depth_trajectory[-1])
        cv2.imwrite("monitor-segmentation.jpg",self.segmentation_trajectory[-1])

    def publish_valuemap(self, points, publisher, frame_id="map"):
        if points is not None:
            # 检查数据类型并提取 NumPy 数组
            if isinstance(points, o3d.cuda.pybind.geometry.PointCloud) or isinstance(points, o3d.geometry.PointCloud):
                points = np.asarray(points.points)  # 提取点云为 NumPy 数组

            if not isinstance(points, np.ndarray):
                self.get_logger().error("Invalid point cloud data format")
                return

            # 确保点云有正确的形状
            if points.shape[1] != 3:
                self.get_logger().error("Point cloud must have shape (N, 3)")
                return

            # 转换为 ROS2 消息并发布
            cloud_msg = convert_cloud_to_ros_msg(points, frame_id)
            publisher.publish(cloud_msg)
            self.get_logger().info(f"Published point cloud to {publisher.topic_name}")

    
    def vis_valuemap_rviz(self):
        for index, (afford, safford, hafford, cafford, gafford, oafford) in enumerate(
            zip(self.affordance_trajectory, self.semantic_affordance_trajectory,
                self.history_affordance_trajectory, self.action_affordance_trajectory,
                self.gpt4v_affordance_trajectory, self.obstacle_affordance_trajectory)
        ):
            self.publish_valuemap(afford, self.affordance_publisher, frame_id="map")
            self.publish_valuemap(safford, self.semantic_publisher, frame_id="map")
            self.publish_valuemap(hafford, self.history_publisher, frame_id="map")
            self.publish_valuemap(cafford, self.action_publisher, frame_id="map")
            self.publish_valuemap(gafford, self.gpt4v_publisher, frame_id="map")
            self.publish_valuemap(oafford, self.obstacle_publisher, frame_id="map")

    def save_trajectory(self,dir="./tmp_objnav/"):
        """保存导航的轨迹，包括RGB图像，深度图像，语义分割图像，全景图像，以及导航链，GPT4和"""
        import imageio
        import os
        os.makedirs(dir)

        self.mapper.save_pointcloud_debug(dir) 
        # fps_writer = imageio.get_writer(dir+"fps.mp4", fps=4)
        # dps_writer = imageio.get_writer(dir+"depth.mp4", fps=4)
        # seg_writer = imageio.get_writer(dir+"segmentation.mp4", fps=4)
        # metric_writer = imageio.get_writer(dir+"metrics.mp4",fps=4)
        # for i,img,dep,seg,met in zip(np.arange(len(self.rgb_trajectory)),self.rgb_trajectory,self.depth_trajectory,self.segmentation_trajectory,self.topdown_trajectory):
        #     fps_writer.append_data(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        #     dps_writer.append_data(dep)
        #     seg_writer.append_data(cv2.cvtColor(seg,cv2.COLOR_BGR2RGB))
        #     metric_writer.append_data(cv2.cvtColor(met,cv2.COLOR_BGR2RGB))

        for index,pano_img in enumerate(self.panoramic_trajectory):
            cv2.imwrite(dir+"%d-pano.jpg"%index,pano_img)
        with open(dir+"gpt4_history.txt",'w') as file:
            file.write("".join(self.gpt_trajectory))
        with open(dir+"gpt4v_history.txt",'w') as file:
            file.write("".join(self.gptv_trajectory))

        for i,afford,safford,hafford,cafford,gafford,oafford in zip(np.arange(len(self.affordance_trajectory)),self.affordance_trajectory,self.semantic_affordance_trajectory,self.history_affordance_trajectory,self.action_affordance_trajectory,self.gpt4v_affordance_trajectory,self.obstacle_affordance_trajectory):
            o3d.io.write_point_cloud(dir+"afford-%d-plan.ply"%i,afford)
            o3d.io.write_point_cloud(dir+"semantic-afford-%d-plan.ply"%i,safford)
            o3d.io.write_point_cloud(dir+"history-afford-%d-plan.ply"%i,hafford)
            o3d.io.write_point_cloud(dir+"action-afford-%d-plan.ply"%i,cafford)
            o3d.io.write_point_cloud(dir+"gpt4v-afford-%d-plan.ply"%i,gafford)
            o3d.io.write_point_cloud(dir+"obstacle-afford-%d-plan.ply"%i,oafford)

        # fps_writer.close()
        # dps_writer.close()
        # seg_writer.close()
        # metric_writer.close()

    def query_chainon(self):
        """创新点1:动态导航链生成"""
        semantic_clue = {'observed object':self.observed_objects} # 存储现在环境中观测到的物体信息
        # 构建GPT的输入字符串，主要包括导航指令，之前的动作序列(action-landmark形式)，以及环境中的观测信息
        query_content = "<Navigation Instruction>:{}, <Previous Plan>:{}, <Semantic Clue>:{}".format(self.instruct_goal,"{" + self.trajectory_summary + "}",semantic_clue)
        self.gpt_trajectory.append("Input:\n%s \n"%query_content)
        # 尝试询问10次GPT
        for i in range(10):
            try:
                raw_answer = gpt_response(query_content,CHAINON_PROMPT) # 调用LLM，传入输入和模型的prompt
                # print("GPT-4 Output Response: %s"%raw_answer)
                answer = raw_answer.replace(" ","")
                answer = answer[answer.index("{"):answer.index("}")+1]
                answer = ast.literal_eval(answer)

                # 正确返回Action, Landmark, Flag之后才能退出，相当于构建了一个导航链
                if 'Action' in answer.keys() and 'Landmark' in answer.keys() and 'Flag' in answer.keys():
                    break
            except:
                continue
            
        self.gpt_trajectory.append("\nGPT-4 Answer:\n%s"%raw_answer)
        # 记录已经生成过的action和landmark,构建导航链(具体在于真正形成一个链状)
        if self.trajectory_summary == "":
            self.trajectory_summary = self.trajectory_summary + str(answer['Action']) + '-' + str(answer['Landmark'])
        else:
            self.trajectory_summary = self.trajectory_summary + '-' + str(answer['Action']) + '-' + str(answer['Landmark'])
        return answer
    
    def query_gpt4v(self):
        images = self.temporary_images # 当前RGB图像（rotate_times个图像组成，分别对应机器人旋转360的不同视角）
        print("********************************************")
        print("len of temporary_images:",len(self.temporary_images))
        print("********************************************")

        inference_image = self.concat_panoramic(images) # 将多个图像拼接成一个全景图像
        cv2.imwrite("monitor-panoramic.jpg",inference_image)
        print("inference_image write successful !!!!!!!!")

        # 输入用户指令和导航链生成的Action和Landmark，根据机器人生成的全景图，调用一个多模态大模型，输入RGB图像和自然语言，判断往哪一个方向走可以帮助机器人完成导航任务
        text_content = "<Navigation Instruction>:{}\n <Sub Instruction>:{}".format(self.instruct_goal,self.trajectory_summary.split("-")[-2] + "-" + self.trajectory_summary.split("-")[-1])
        self.gptv_trajectory.append("\nInput:\n%s \n"%text_content)
        print("MLLM prompt:",text_content)
        for i in range(10):
            # try:
            #     raw_answer = gptv_response(text_content,inference_image,GPT4V_PROMPT)
            #     # raw_answer = 0
            #     print("************loop gpt4v output****************")
            #     print("GPT-4V Output Response_a: %s"%raw_answer)
            #     answer = raw_answer[raw_answer.index("Judgement: Direction"):]
            #     print("Raw answer close!!!!!!!!")
            #     print(11111)
            #     answer = answer.replace(" ","")
            #     print(22222)
            #     answer = int(answer.split("Direction")[-1])
            #     print(33333)
            #     print("************************************")
            #     print("GPT-4V answer in loop!:",answer)
            #     print("************************************")
            #     break
            # except:
            #     continue

            try:
                # # 调用 GPT-4V 并获取原始答案
                # raw_answer = gptv_response(text_content, inference_image, GPT4V_PROMPT)
                # print("GPT-4V Output Response_a: %s"%raw_answer)

                # # 查找 "<Judgement>: Direction" 的位置
                # start_index = raw_answer.index("<Judgement>: Direction")
                
                # # 提取从 "<Judgement>" 开始的部分
                # judgment_text = raw_answer[start_index:].strip()

                # # 提取 "Direction" 后的数字
                # if "Direction" in judgment_text:
                #     answer = int(judgment_text.split("Direction")[-1].split()[0])  # 提取第一个数字
                #     print("Parsed Direction:", answer)
                #     break
                # else:
                #     raise ValueError("No 'Direction' found in judgement text")
                answer = 0
                    
            except ValueError as e:
                print("Failed to parse direction due to value error:", e)
                continue
            except IndexError as e:
                print("Failed to parse due to missing fields:", e)
                continue

        # self.gptv_trajectory.append("GPT-4V Answer:\n%s"%raw_answer) # 记录GPT-4V的输出结果
        self.panoramic_trajectory.append(inference_image) # 保存全景图像
        try:
            return answer
        except:
            print("GPT4V return randint!!!!!!!!!!!!!!! maybe exploration")
            return np.random.randint(0,6)
    
    def make_plan(self,rotate=True,failed=False):
        while(not self.is_rgb_receive and not self.is_depth_receive and not self.is_odom_receive):
            print("Waitting for Gazebo Simulator Get Robot Information")

        if rotate == True: # 如果开启360度旋转模式，则获取全景图像和点云
            self.rotate_panoramic()
        
        self.chainon_answer = self.query_chainon() # 调用LLM获取一个初始的action和landmark
        self.gpt4v_answer = self.query_gpt4v()     # 调用MLM，结合刚才获得的action和landmark，寻找一个最有可能完成任务的方向

        # self.gpt4v_answer = 0
        print("***********************************")
        print("GPT4 - chainon_answer:", self.chainon_answer)
        print("***********************************")
        print("GPT4v answer:",self.gpt4v_answer)
        print("***********************************")

        self.gpt4v_pcd = o3d.t.geometry.PointCloud(self.mapper.pcd_device) # 初始化一个空的点云对象
        # print("*******************************")
        # print(self.gpt4v_pcd)
        # print("*******************************")

        self.gpt4v_pcd = gpu_merge_pointcloud(self.gpt4v_pcd,self.temporary_pcd[self.gpt4v_answer]) # 向继续前进方向的PCD

        self.found_goal = bool(self.chainon_answer['Flag']) # 判断是否完成任务

        # 返回融合后的value_map和可视化的value_map
        self.affordance_pcd,self.colored_affordance_pcd = self.mapper.get_objnav_affordance_map(self.chainon_answer['Action'],self.chainon_answer['Landmark'],self.gpt4v_pcd,self.chainon_answer['Flag'],failure_mode=failed)

        # 输出调试信息
        self.semantic_afford,self.history_afford,self.action_afford,self.gpt4v_afford,self.obs_afford = self.mapper.get_debug_affordance_map(self.chainon_answer['Action'],self.chainon_answer['Landmark'],self.gpt4v_pcd)

        # 判断value_map是否全为0，如果全为0，则重新生成
        if self.affordance_pcd.max() == 0:
            self.affordance_pcd,self.colored_affordance_pcd = self.mapper.get_objnav_affordance_map(self.chainon_answer['Action'],self.chainon_answer['Landmark'],self.gpt4v_pcd,False,failure_mode=failed)
            self.found_goal = False
        
        # 将value_map从点云形式转换为栅格地图(2D)形式,用来做后续的规划,生成2D的costmap
        self.affordance_map,self.colored_affordance_map = project_costmap(self.mapper.navigable_pcd,self.affordance_pcd,self.mapper.grid_resolution)

        # 令最终的目标点是value_map中最大值对应的点
        self.target_point = self.mapper.navigable_pcd.point.positions[self.affordance_pcd.argmax()].cpu().numpy()

        # 获取当前机器人的位置
        self.plan_position = self.mapper.current_position.copy()
        # 根据起始点和终点获取在2D栅格地图中的索引，用于路径搜索
        target_index = translate_point_to_grid(self.mapper.navigable_pcd,self.target_point,self.mapper.grid_resolution)
        start_index = translate_point_to_grid(self.mapper.navigable_pcd,self.mapper.current_position,self.mapper.grid_resolution)
        # 调用A*算法搜索出路径
        self.path = path_planning(self.affordance_map,start_index,target_index)
        self.path = [translate_grid_to_point(self.mapper.navigable_pcd,np.array([[waypoint.y,waypoint.x,0]]),self.mapper.grid_resolution)[0] for waypoint in self.path]

        # print("Using pathplanning lib to generate path:",self.path)
        # visualized_path = visualize_path(self.affordance_map, self.path)
        # print(type(visualized_path))
        # cv2.imshow("Costmap with Path", visualized_path)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 根据规划出来的路径选择下一步的waypoint
        if len(self.path) == 0: # 如果路径为空，则在可通行区域中选择一个具有最大
            self.waypoint = self.mapper.navigable_pcd.point.positions.cpu().numpy()[np.argmax(self.affordance_pcd)]
            self.waypoint[2] = self.mapper.current_position[2] # 确保z坐标高度一致
        elif len(self.path) < 5: 
            self.waypoint = self.path[-1]
            self.waypoint[2] = self.mapper.current_position[2]
        else:
            self.waypoint = self.path[4] # 路径长度大于5则只选择第5个点作为下一个目标点
            self.waypoint[2] = self.mapper.current_position[2]
        

        self.affordance_trajectory.append(self.colored_affordance_pcd)
        self.obstacle_affordance_trajectory.append(self.obs_afford)
        self.semantic_affordance_trajectory.append(self.semantic_afford)
        self.history_affordance_trajectory.append(self.history_afford)
        self.action_affordance_trajectory.append(self.action_afford)
        self.gpt4v_affordance_trajectory.append(self.gpt4v_afford)

        print("************************************")
        print("save_trajcetory")
        print("************************************")
        # self.save_trajectory()
        self.vis_valuemap_rviz()
        # print("ok")
    def step(self):
        if not self.is_rgb_receive and not self.is_depth_receive and not self.is_odom_receive:
            print("Waitting for Gazebo Simulator Get Robot Information")
            return 
        print("****************************************************")
        print(self.is_rgb_receive,self.is_depth_receive,self.is_odom_receive)
        print("****************************************************")
        # 计算当前位置到目标点的距离
        to_target_distance = np.sqrt(np.sum(np.square(self.mapper.current_position - self.waypoint)))
        if to_target_distance < 0.6 and len(self.path) > 0:
            self.path = self.path[min(5,len(self.path)-1):]
            if len(self.path) < 3:
                self.waypoint = self.path[-1]
                self.waypoint[2] = self.mapper.current_position[2]
            else:
                self.waypoint = self.path[2]
                self.waypoint[2] = self.mapper.current_position[2]

        pid_waypoint = self.waypoint + self.mapper.initial_position # 得到计算出来目标点相对于初始点的增加一个偏移量，waypoint是相对于机器人自身的坐标，这一步相当于转换到全局坐标系
        # todo 这一步为什么要这么写？？？
        # todo 检查这一步的坐标系设置-> habitat坐标系和普通的世界坐标系不同
        # pid_waypoint = np.array([pid_waypoint[0],self.position[1],pid_waypoint[1]])
        pid_waypoint = np.array([-pid_waypoint[1],pid_waypoint[0],self.position[1]])
        
        # todo 增加机器人的replan环节
        move_distance =  np.sqrt(np.sum(np.square(self.mapper.current_position - self.plan_position)))
        # 如果机器人没有到达目标点，或者走过的距离大于3米，则重新规划路径
        if (move_distance > 3.0) and not self.found_goal:
            self.make_plan(rotate=True)
            pid_waypoint = self.waypoint + self.mapper.initial_position
            # pid_waypoint = np.array([pid_waypoint[0],self.position[1],pid_waypoint[1]])
            pid_waypoint = np.array([-pid_waypoint[1],pid_waypoint[0],self.position[1]])


        if not self.found_goal:
            self.make_plan(False,True)
            pid_waypoint = self.waypoint + self.mapper.initial_position
            # pid_waypoint = np.array([pid_waypoint[0],self.position[1],pid_waypoint[1]])
            pid_waypoint = np.array([-pid_waypoint[1],pid_waypoint[0],self.position[1]])


        # 这里才执行导航操作，前面都是根据假设看看走的和不合适
        self.move_next(pid_waypoint)
        self.env_step()
        self.update_trajectory()

    def run(self):
        while(not self.is_rgb_receive or not self.is_depth_receive or not self.is_odom_receive):
        # if not self.is_rgb_receive or not self.is_depth_receive or not self.is_odom_receive:
            self.get_logger().info("Waitting for Gazebo Simulator Get Robot Information ")
            time.sleep(1) # 等待1s获取
            
        if not self.is_plan_init:
            self.reset()
            self.make_plan()
            self.is_plan_init = True
            # self.plan完了应该一直运行self.plan，不应该重新reset 

        while(True):
            self.step()
            # print("************************************")
            # print("save_trajcetory")
            # print("************************************")
            # self.save_trajectory()


def main(args=None):
    rclpy.init(args=args)
    gazebo_mapper = Instruct_Mapper(gazebo_camera_intrinsic(),
                                            pcd_resolution = 0.05,
                                            grid_resolution=0.2,
                                            grid_size=5
                                            )
            
    obj_agent = HM3D_Objnav_Agent(gazebo_mapper)

    rclpy.spin(obj_agent)
    
    rclpy.shutdown()
   

if __name__=="__main__":
    main()
       
