import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformException

from matplotlib import colormaps
from constants import *
import open3d as o3d
from lavis.models import load_model_and_preprocess
from PIL import Image
import cv2

sys.path.append(os.path.join(os.getcwd(), "src/llm_obj_nav/llm_obj_nav/instructnav"))
from mapping_utils.geometry import *
from mapping_utils.preprocess import *
from mapping_utils.projection import *
from mapping_utils.transform import *
from mapping_utils.path_planning import *
from cv_utils.image_percevior import GLEE_Percevior

d3_40_colors_rgb: np.ndarray = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)


class Instruct_Mapper(Node):
    def __init__(self,
                 camera_intrinsic,
                 pcd_resolution=0.05,
                 grid_resolution=0.1,
                 grid_size=5,
                 floor_height=-0.5,
                 ceiling_height=0.8,
                 translation_func=gazebo_translation,
                 rotation_func=gazebo_rotation,
                 rotate_axis=[0,1,0],
                 device='cuda:0'):
        
        super().__init__('Instruct_Mapper')

        self.device = device
        self.camera_intrinsic = camera_intrinsic
        self.pcd_resolution = pcd_resolution
        self.grid_resolution = grid_resolution
        self.grid_size = grid_size
        self.floor_height = floor_height
        self.ceiling_height = ceiling_height
        self.translation_func = translation_func
        self.rotation_func = rotation_func
        self.rotate_axis = np.array(rotate_axis)
        self.object_percevior = GLEE_Percevior(device=device)
        self.pcd_device = o3d.core.Device(device.upper())

        # Camera information vis
        self.worldPcd_pub = self.create_publisher(PointCloud2, 'world_points', 10)
        self.cameraPoint_pub = self.create_publisher(PointCloud2, 'camera_points', 10)
        self.scenePoint_pub = self.create_publisher(PointCloud2, 'scene_points', 10)

        # value map information vis
        self.obsmap_pub = self.create_publisher(PointCloud2, 'obsmap', 10)
        self.frontiermap_pub = self.create_publisher(PointCloud2, 'frontiermap', 10)
        self.semanticmap_pub = self.create_publisher(PointCloud2, 'semanticmap', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


    def publish_pointcloud(self, points, publisher, frame_id="map"):
        if points is not None:
            # Debug: Log the type and shape of points
            self.get_logger().info(f"Type of points: {type(points)}")
            
            # Check if the point cloud is on the GPU (CUDA-based PointCloud)
            if isinstance(points, o3d.cuda.pybind.t.geometry.PointCloud):
                self.get_logger().info("Point cloud is on GPU. Transferring to CPU...")
                points = points.to_legacy()  # Transfer to CPU as a legacy PointCloud object
            
            # Now points should be a regular Open3D point cloud (CPU-based)
            if isinstance(points, o3d.geometry.PointCloud):
                # Extract points as a NumPy array from the Open3D point cloud (CPU-based)
                points = np.asarray(points.points)
            
            # If points is not a NumPy array at this point, return an error
            if not isinstance(points, np.ndarray):
                self.get_logger().error("Invalid point cloud data format")
                return

            # Ensure the point cloud has the correct shape (N, 3)
            if points.shape[1] != 3:
                self.get_logger().error("Point cloud must have shape (N, 3)")
                return

            # Convert to ROS2 message and publish
            cloud_msg = convert_cloud_to_ros_msg(points, frame_id)
            publisher.publish(cloud_msg)
            self.get_logger().info(f"Published point cloud to {publisher.topic_name}")


    
    def init_map(self,position,rotation):
        self.update_iterations = 0
        self.initial_position = self.translation_func(position)
        self.current_position = self.translation_func(position) - self.initial_position
        self.current_rotation = self.rotation_func(rotation)
        self.scene_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.object_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.object_entities = []
        self.trajectory_position = []
    
    def reset(self,position,rotation):
        self.current_position = self.translation_func(position) - self.initial_position
        self.current_rotation = self.rotation_func(rotation)
        self.scene_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.object_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.object_entities = []


    def visualize_masks(self, image, pred_masks):
        # Convert image to numpy array if needed
        image = np.array(image)
        
        # Ensure image is 3D (H,W,3)
        if len(image.shape) != 3:
            raise ValueError("Input image must be 3-channel RGB")
            
        # Create visualization image
        vis_image = image.copy()
        
        # Define colors if not already defined
        if not hasattr(self, 'colors'):
            self.colors = [(255,0,0), (0,255,0), (0,0,255)]  # RGB colors
        
        # 叠加每个mask
        for i, mask in enumerate(pred_masks):
            # Ensure mask is 2D
            mask = np.array(mask).astype(bool)
            if len(mask.shape) != 2:
                continue
                
            # Create color mask with same shape as image
            color_mask = np.zeros_like(image)
            color_mask[mask] = self.colors[i % len(self.colors)]
            
            # Blend with original image
            alpha = 0.5
            vis_image = cv2.addWeighted(vis_image, 1, color_mask, alpha, 0)
            
            # 添加轮廓
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.imwrite("vis mask.jpg", vis_image)
    
    def update(self,rgb,depth,position,rotation):
        # 更新位姿
        self.current_position = self.translation_func(position) - self.initial_position
        self.current_rotation = self.rotation_func(rotation)
        # 获取当前帧的点云和RGB
        self.current_depth = preprocess_depth(depth)
        # print(self.current_depth)
        self.current_rgb = preprocess_image(rgb)
        self.trajectory_position.append(self.current_position)
        # to avoid there is no valid depth value (especially in real-world)
        if np.sum(self.current_depth) > 0:
            # todo 这一步生成的没问题
            camera_points,camera_colors = get_pointcloud_from_depth(self.current_rgb,self.current_depth,self.camera_intrinsic)
            # self.publish_pointcloud(camera_points, self.cameraPoint_pub, "d435_0_color_optical_frame")
            world_points = translate_to_world(camera_points,self.current_position,self.current_rotation)
            # self.publish_pointcloud(world_points, self.worldPcd_pub, "map")

            self.current_pcd = gpu_pointcloud_from_array(world_points,camera_colors,self.pcd_device).voxel_down_sample(self.pcd_resolution)
            # self.publish_pointcloud(self.current_pcd, self.cameraPoint_pub, "map")
        else:
            # print(2)
            return
        
        # 利用GLEE模型对机器人的观测进行实例分割和目标检测
        classes,masks,confidences,visualization = self.object_percevior.perceive(self.current_rgb)
        print("-----------------------------------")
        print(classes)
        print("-----------------------------------")
        self.visualize_masks(self.current_rgb, masks)

        self.segmentation = visualization[0]
        # 根据分割结果，提取每个目标的3D点云，位置信息
        current_object_entities = self.get_object_entities(self.current_depth,classes,masks,confidences)
        # 在多帧中对目标进行关联，如果目标在多帧中出现，将其合并
        self.object_entities = self.associate_object_entities(self.object_entities,current_object_entities)
        self.object_pcd = self.update_object_pcd()
        # self.publish_pointcloud(self.object_pcd, self.frontiermap_pub, "map")

        # pointcloud update
        # 更新场景中的点云
        self.scene_pcd = gpu_merge_pointcloud(self.current_pcd,self.scene_pcd).voxel_down_sample(self.pcd_resolution)

        # self.publish_pointcloud(self.scene_pcd, self.scenePoint_pub, "map")

        # 对scene_pcd进行高度过滤，卡在地板高度和最高高度之间
        self.scene_pcd = self.scene_pcd.select_by_index((self.scene_pcd.point.positions[:,2]>self.floor_height).nonzero()[0])
        self.useful_pcd = self.scene_pcd.select_by_index((self.scene_pcd.point.positions[:,2]<self.ceiling_height).nonzero()[0])

        # 这里把所有的楼梯都看成了可通行区域
        # for entity in current_object_entities:
        #     if entity['class'] == 'stairs':
        #         self.navigable_pcd = gpu_merge_pointcloud(self.navigable_pcd,entity['pcd'])


        # geometry 
        current_navigable_point = self.current_pcd.select_by_index((self.current_pcd.point.positions[:,2]<self.floor_height).nonzero()[0])

        # self.publish_pointcloud(current_navigable_point, self.scenePoint_pub, "map")

        current_navigable_position = current_navigable_point.point.positions.cpu().numpy()

        # 计算机器人现在的位置
        standing_position = np.array([self.current_position[0],self.current_position[1],current_navigable_position[:,2].mean()])

        interpolate_points = np.linspace(np.ones_like(current_navigable_position)*standing_position,current_navigable_position,25).reshape(-1,3)

        interpolate_points = interpolate_points[(interpolate_points[:,2] > self.floor_height-0.2) & (interpolate_points[:,2] < self.floor_height+0.2)]
        interpolate_colors = np.ones_like(interpolate_points) * 100
        try:
            current_navigable_pcd = gpu_pointcloud_from_array(interpolate_points,interpolate_colors,self.pcd_device).voxel_down_sample(self.grid_resolution)
            # print("curr pcd:",current_navigable_pcd)
            self.navigable_pcd = gpu_merge_pointcloud(self.navigable_pcd,current_navigable_pcd).voxel_down_sample(self.pcd_resolution)
            # self.publish_pointcloud(self.navigable_pcd, self.scenePoint_pub, "map")

        except:
            self.navigable_pcd = self.useful_pcd.select_by_index((self.useful_pcd.point.positions[:,2]<self.floor_height).nonzero()[0])
            # print("except pcd:",current_navigable_pcd)
            
        # 过滤障碍物点云
        # self.obstacle_pcd = self.useful_pcd.select_by_index((self.useful_pcd.point.positions[:,2]>self.floor_height+0.1).nonzero()[0])
        try:
            # 获取高于地面的点
            obstacle_indices = (self.useful_pcd.point.positions[:,2]>self.floor_height+0.1).nonzero()[0]
            
            # 检查是否存在障碍物点
            if len(obstacle_indices) > 0:
                self.obstacle_pcd = self.useful_pcd.select_by_index(obstacle_indices)
            else:
                # 如果没有障碍物点,创建一个空的点云
                self.obstacle_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        except Exception as e:
            # 出错时创建空点云
            self.get_logger().warning(f"Failed to filter obstacle points: {e}")
            self.obstacle_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        # self.publish_pointcloud(self.obstacle_pcd, self.obsmap_pub, "map")

        self.trajectory_pcd = gpu_pointcloud_from_array(np.array(self.trajectory_position),np.zeros((len(self.trajectory_position),3)),self.pcd_device)
        self.frontier_pcd = project_frontier(self.obstacle_pcd,self.navigable_pcd,self.floor_height+0.2,self.grid_resolution)
        self.frontier_pcd[:,2] = self.navigable_pcd.point.positions.cpu().numpy()[:,2].mean()
        self.frontier_pcd = gpu_pointcloud_from_array(self.frontier_pcd,np.ones((self.frontier_pcd.shape[0],3))*np.array([[255,0,0]]),self.pcd_device)
        # self.publish_pointcloud(self.frontier_pcd, self.frontiermap_pub, "map")

        self.update_iterations += 1
    
    def update_object_pcd(self):
        object_pcd = o3d.geometry.PointCloud()
        for entity in self.object_entities:
            points = entity['pcd'].point.positions.cpu().numpy()
            colors = entity['pcd'].point.colors.cpu().numpy()
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            new_pcd.colors = o3d.utility.Vector3dVector(colors)
            object_pcd = object_pcd + new_pcd
        try:
            return gpu_pointcloud(object_pcd,self.pcd_device)
        except:
            return self.scene_pcd
    
    def get_view_pointcloud(self,rgb,depth,translation,rotation):
        current_position = self.translation_func(translation) - self.initial_position
        current_rotation = self.rotation_func(rotation)
        current_depth = preprocess_depth(depth)
        current_rgb = preprocess_image(rgb)
        camera_points,camera_colors = get_pointcloud_from_depth(current_rgb,current_depth,self.camera_intrinsic)
        world_points = translate_to_world(camera_points,current_position,current_rotation)
        current_pcd = gpu_pointcloud_from_array(world_points,camera_colors,self.pcd_device).voxel_down_sample(self.pcd_resolution)
        return current_pcd
    
    def get_object_entities(self,depth,classes,masks,confidences):
        entities = []
        exist_objects = np.unique([ent['class'] for ent in self.object_entities]).tolist()
        for cls,mask,score in zip(classes,masks,confidences):
            if depth[mask>0].min() < 1.0 and score < 0.45: # object 初步过滤
                continue
            if cls not in exist_objects:
                exist_objects.append(cls)
            camera_points = get_pointcloud_from_depth_mask(depth,mask,self.camera_intrinsic)
            # self.publish_pointcloud(camera_points, self.cameraPoint_pub, "d435_0_color_optical_frame")

            world_points = translate_to_world(camera_points,self.current_position,self.current_rotation)
            # self.publish_pointcloud(world_points, self.worldPcd_pub, "map")

            point_colors = np.array([d3_40_colors_rgb[exist_objects.index(cls)%40]]*world_points.shape[0])
            if world_points.shape[0] < 10:
                continue
            object_pcd = gpu_pointcloud_from_array(world_points,point_colors,self.pcd_device).voxel_down_sample(self.pcd_resolution)
            object_pcd = gpu_cluster_filter(object_pcd)
            if object_pcd.point.positions.shape[0] < 10:
                continue
            entity = {'class':cls,'pcd':object_pcd,'confidence':score}
            entities.append(entity)
        return entities
    
    def associate_object_entities(self,ref_entities,eval_entities):        
        for entity in eval_entities:
            if len(ref_entities) == 0:
                ref_entities.append(entity)
                continue
            overlap_score = []
            eval_pcd = entity['pcd']
            for ref_entity in ref_entities:
                if eval_pcd.point.positions.shape[0] == 0:
                    break
                cdist = pointcloud_distance(eval_pcd,ref_entity['pcd'])
                overlap_condition = (cdist < 0.1)
                nonoverlap_condition = overlap_condition.logical_not()
                eval_pcd = eval_pcd.select_by_index(o3d.core.Tensor(nonoverlap_condition.cpu().numpy(),device=self.pcd_device).nonzero()[0])
                overlap_score.append((overlap_condition.sum()/(overlap_condition.shape[0]+1e-6)).cpu().numpy())
            max_overlap_score = np.max(overlap_score)
            arg_overlap_index = np.argmax(overlap_score)
            if max_overlap_score < 0.25:
                entity['pcd'] = eval_pcd
                ref_entities.append(entity)
            else:
                argmax_entity = ref_entities[arg_overlap_index]
                argmax_entity['pcd'] = gpu_merge_pointcloud(argmax_entity['pcd'],eval_pcd)
                if argmax_entity['pcd'].point.positions.shape[0] < entity['pcd'].point.positions.shape[0] or entity['class'] in INTEREST_OBJECTS:
                    argmax_entity['class'] = entity['class']
                ref_entities[arg_overlap_index] = argmax_entity
        return ref_entities
    
    def get_obstacle_affordance(self):
        try:
            distance = pointcloud_distance(self.navigable_pcd,self.obstacle_pcd)
            # print("obs affordance:",distance)
            affordance = (distance - distance.min())/(distance.max() - distance.min() + 1e-6)
            affordance[distance < 0.25] = 0
            return affordance.cpu().numpy()
        except:
            # print("nothing ")
            return np.zeros((self.navigable_pcd.point.positions.shape[0],),dtype=np.float32)
    
    def get_trajectory_affordance(self):
        try:
            distance = pointcloud_distance(self.navigable_pcd,self.trajectory_pcd)
            affordance = (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            return affordance.cpu().numpy()
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],),dtype=np.float32)
    
    def get_semantic_affordance(self,target_class,threshold=0.1):
        semantic_pointcloud = o3d.t.geometry.PointCloud()
        for entity in self.object_entities:
            if entity['class'] in target_class:
                semantic_pointcloud = gpu_merge_pointcloud(semantic_pointcloud,entity['pcd'])
                # 可视化semantic_pointcloud
                # self.publish_pointcloud(semantic_pointcloud, self.semanticmap_pub, "map")
        try:
            distance = pointcloud_2d_distance(self.navigable_pcd,semantic_pointcloud) 
            affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            affordance[distance > threshold] = 0
            affordance = affordance.cpu().numpy()
            return affordance
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],),dtype=np.float32)
    
    def get_gpt4v_affordance(self,gpt4v_pcd):
        try:
            distance = pointcloud_distance(self.navigable_pcd,gpt4v_pcd)
            affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            affordance[distance > 0.1] = 0
            return affordance.cpu().numpy()
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],),dtype=np.float32)
    
    def get_action_affordance(self,action):
        try:
            if action == 'Explore':
                distance = pointcloud_2d_distance(self.navigable_pcd,self.frontier_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.2] = 0
                return affordance.cpu().numpy()
            elif action == 'Move_Forward':
                pixel_x,pixel_z,depth_values = project_to_camera(self.navigable_pcd,self.camera_intrinsic,self.current_position,self.current_rotation)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2]*2) & (pixel_z >= 0) & (pixel_z < self.camera_intrinsic[1][2]*2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(o3d.core.Tensor(np.where(filter_condition==1)[0],device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd,filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Turn_Around':
                R = np.array([np.pi,np.pi,np.pi]) * self.rotate_axis
                turn_extrinsic = np.matmul(self.current_rotation,quaternion.as_rotation_matrix(quaternion.from_euler_angles(R)))
                pixel_x,pixel_z,depth_values = project_to_camera(self.navigable_pcd,self.camera_intrinsic,self.current_position,turn_extrinsic)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2]*2) & (pixel_z >= 0) & (pixel_z < self.camera_intrinsic[1][2]*2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(o3d.core.Tensor(np.where(filter_condition==1)[0],device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd,filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Turn_Left':
                R = np.array([np.pi/2,np.pi/2,np.pi/2]) * self.rotate_axis
                turn_extrinsic = np.matmul(self.current_rotation,quaternion.as_rotation_matrix(quaternion.from_euler_angles(R)))
                pixel_x,pixel_z,depth_values = project_to_camera(self.navigable_pcd,self.camera_intrinsic,self.current_position,turn_extrinsic)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2]*2) & (pixel_z >= 0) & (pixel_z < self.camera_intrinsic[1][2]*2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(o3d.core.Tensor(np.where(filter_condition==1)[0],device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd,filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Turn_Right':
                R = np.array([-np.pi/2,-np.pi/2,-np.pi/2]) * self.rotate_axis
                turn_extrinsic = np.matmul(self.current_rotation,quaternion.as_rotation_matrix(quaternion.from_euler_angles(R)))
                pixel_x,pixel_z,depth_values = project_to_camera(self.navigable_pcd,self.camera_intrinsic,self.current_position,turn_extrinsic)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2]*2) & (pixel_z >= 0) & (pixel_z < self.camera_intrinsic[1][2]*2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(o3d.core.Tensor(np.where(filter_condition==1)[0],device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd,filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Enter':
                return self.get_semantic_affordance(['doorway','door','entrance','exit'])
            elif action == 'Exit':
                return self.get_semantic_affordance(['doorway','door','entrance','exit'])
            else:
                return np.zeros((self.navigable_pcd.point.positions.shape[0],),dtype=np.float32) 
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],),dtype=np.float32) 

    def get_objnav_affordance_map(self,action,target_class,gpt4v_pcd,complete_flag=False,failure_mode=False):
        if failure_mode:
            obstacle_affordance = self.get_obstacle_affordance() # 获取可通行区域的栅格地图
            affordance = self.get_action_affordance('Explore')
            affordance = np.clip(affordance,0.1,1.0)
            affordance[obstacle_affordance == 0] = 0
            return affordance,self.visualize_affordance(affordance)
        elif complete_flag:
            affordance = self.get_semantic_affordance([target_class],threshold=0.1)
            return affordance,self.visualize_affordance(affordance)
        else:
            # 这里相当于计算value map，分别计算四种的value map然后进行加权平均
            obstacle_affordance = self.get_obstacle_affordance()
            semantic_affordance = self.get_semantic_affordance([target_class],threshold=1.5)
            action_affordance = self.get_action_affordance(action)
            gpt4v_affordance = self.get_gpt4v_affordance(gpt4v_pcd)
            history_affordance = self.get_trajectory_affordance()
            affordance = 0.25*semantic_affordance + 0.25*action_affordance + 0.25*gpt4v_affordance + 0.25*history_affordance
            affordance = np.clip(affordance,0.1,1.0)
            affordance[obstacle_affordance == 0] = 0 # 不能通过的区域设置为0，相当于考虑可通行区域因素进行避障
            return affordance,self.visualize_affordance(affordance/(affordance.max()+1e-6))

    def get_debug_affordance_map(self,action,target_class,gpt4v_pcd):
        obstacle_affordance = self.get_obstacle_affordance()
        semantic_affordance = self.get_semantic_affordance([target_class],threshold=1.5)
        action_affordance = self.get_action_affordance(action)
        gpt4v_affordance = self.get_gpt4v_affordance(gpt4v_pcd)
        history_affordance = self.get_trajectory_affordance()
        return self.visualize_affordance(semantic_affordance/(semantic_affordance.max()+1e-6)),\
               self.visualize_affordance(history_affordance/(history_affordance.max()+1e-6)),\
               self.visualize_affordance(action_affordance/(action_affordance.max()+1e-6)),\
               self.visualize_affordance(gpt4v_affordance/(gpt4v_affordance.max()+1e-6)),\
               self.visualize_affordance(obstacle_affordance/(obstacle_affordance.max()+1e-6))

    def visualize_affordance(self,affordance):
        cmap = colormaps.get('jet')
        color_affordance = cmap(affordance)[:,0:3]
        color_affordance = cpu_pointcloud_from_array(self.navigable_pcd.point.positions.cpu().numpy(),color_affordance)
        return color_affordance
    
    def get_appeared_objects(self):
        return [entity['class'] for entity in self.object_entities]

    def save_pointcloud_debug(self,path="./"):
        save_pcd = o3d.geometry.PointCloud()
        try:
            assert self.useful_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(self.useful_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.useful_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "scene.ply",save_pcd)
        except:
            pass
        try:
            assert self.navigable_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(self.navigable_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.navigable_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "navigable.ply",save_pcd)
        except:
            pass
        try:
            assert self.obstacle_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(self.obstacle_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.obstacle_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "obstacle.ply",save_pcd)
        except:
            pass
        
        object_pcd = o3d.geometry.PointCloud()
        for entity in self.object_entities:
            points = entity['pcd'].point.positions.cpu().numpy()
            colors = entity['pcd'].point.colors.cpu().numpy()
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            new_pcd.colors = o3d.utility.Vector3dVector(colors)
            object_pcd = object_pcd + new_pcd
        if len(object_pcd.points) > 0:
            o3d.io.write_point_cloud(path + "object.ply",object_pcd)