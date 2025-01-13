import numpy as np
import open3d as o3d
import quaternion
import time
import torch
import cv2
from sensor_msgs.msg import PointCloud2, PointField
import rclpy
from std_msgs.msg import Header 
import numpy as np
import geometry_msgs.msg
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformException
import math


# 通过深度相机的深度图和RGB图像以及相机内参获取点云，将像素坐标映射为3D空间坐标(pixel,depth->pcd)
def get_pointcloud_from_depth(rgb:np.ndarray,depth:np.ndarray,intrinsic:np.ndarray):
    if len(depth.shape) == 3:
        depth = depth[:,:,0]
   
    # 获取图像尺寸
    height, width = depth.shape
   
    # 创建网格坐标
    x, y = np.meshgrid(np.arange(width), np.arange(height))
   
    # 计算3D点坐标
    z = depth
    x = (x - intrinsic[0][2]) * z / intrinsic[0][0]  # (u - cx) * z / fx
    y = (y - intrinsic[1][2]) * z / intrinsic[1][1]  # (v - cy) * z / fy
   
    # 过滤有效深度值
    valid_mask = depth > 0
   
    # 将点云数据重组为(N,3)的形状
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
   
    # 使用有效深度掩码过滤点云
    valid_points = points[valid_mask.flatten()]
    color_values = rgb.reshape(-1, rgb.shape[-1])[valid_mask.flatten()]
   
    # 移除NaN和Inf值
    valid_indices = ~(np.isnan(valid_points).any(axis=1) | np.isinf(valid_points).any(axis=1))
    valid_points = valid_points[valid_indices]
    color_values = color_values[valid_indices]
   
    return valid_points, color_values

# 通过深度相机的深度图和mask图像以及相机内参获取点云，将像素坐标映射为3D空间坐标，其中利用了mask进行过滤，即考虑了分割信息
# def get_pointcloud_from_depth_mask(depth:np.ndarray,mask:np.ndarray,intrinsic:np.ndarray):
#     if len(depth.shape) == 3:
#         depth = depth[:,:,0]
#     if len(mask.shape) == 3:
#         mask = mask[:,:,0]
       
#     # 获取图像尺寸
#     height, width = depth.shape
   
#     # 创建网格坐标
#     x, y = np.meshgrid(np.arange(width), np.arange(height))
   
#     # 计算3D点坐标，注意保持原有的坐标系转换
#     z = depth
#     x = (x - intrinsic[0][2]) * z / intrinsic[0][0]  # (u - cx) * z / fx
#     y = (depth.shape[0] - 1 - y - intrinsic[1][2]) * z / intrinsic[1][1]  # (height-1-v - cy) * z / fy
   
#     # 过滤有效深度值和mask
#     valid_mask = (depth > 0) & (mask > 0)
   
#     # 将点云数据重组为(N,3)的形状，注意坐标轴的对应关系
#     points = np.stack((x[valid_mask], -y[valid_mask], z[valid_mask]), axis=-1)
   
#     return points

def get_pointcloud_from_depth_mask(depth: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray):
    if len(depth.shape) == 3:
        depth = depth[:,:,0]
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
   
    # 获取图像尺寸
    height, width = depth.shape
   
    # 创建网格坐标
    x, y = np.meshgrid(np.arange(width), np.arange(height))
   
    # 计算3D点坐标，注意保持原有的坐标系转换
    z = depth
    x = (x - intrinsic[0][2]) * z / intrinsic[0][0]  # (u - cx) * z / fx
    y = (y - intrinsic[1][2]) * z / intrinsic[1][1]  # (v - cy) * z / fy
   
    # 过滤有效深度值和mask
    valid_mask = (depth > 0) & (mask > 0)
   
    # 将点云数据重组为(N,3)的形状，注意坐标轴的对应关系
    points = np.stack((x[valid_mask], -y[valid_mask], z[valid_mask]), axis=-1)
   
    return points


# 坐标系转换，从相机坐标系转换到世界坐标系
import numpy as np

def translate_to_world(pointcloud, position, rotation):
    # 创建一个4x4的单位矩阵
    extrinsic = np.eye(4)
    
    # 设置旋转部分
    extrinsic[0:3, 0:3] = rotation
    
    # 设置平移部分
    extrinsic[0:3, 3] = position
    
    # 执行变换，将点云从相机坐标系转换到世界坐标系
    world_points = np.matmul(extrinsic, np.concatenate((pointcloud, np.ones((pointcloud.shape[0], 1))), axis=-1).T).T
    
    # 返回变换后的点云
    return world_points[:, 0:3]


# 将点云投影到相机平面，生成像素坐标和深度值(pcd->pixel,depth)
def project_to_camera(pcd,intrinsic,position,rotation):
    extrinsic = np.eye(4)
    extrinsic[0:3,0:3] = rotation
    extrinsic[0:3,3] = position
    extrinsic = np.linalg.inv(extrinsic)
    try:
        camera_points = np.concatenate((pcd.point.positions.cpu().numpy(),np.ones((pcd.point.positions.shape[0],1))),axis=-1)
    except:
        camera_points = np.concatenate((pcd.points,np.ones((np.array(pcd.points).shape[0],1))),axis=-1)
    camera_points = np.matmul(extrinsic,camera_points.T).T[:,0:3]
    depth_values = -camera_points[:,2]
    filter_x = (camera_points[:,0] * intrinsic[0][0] / depth_values + intrinsic[0][2]).astype(np.int32)
    filter_z = (-camera_points[:,1] * intrinsic[1][1] / depth_values - intrinsic[1][2] + intrinsic[1][2]*2 - 1).astype(np.int32)
    return filter_x,filter_z,depth_values
    
# 计算两个点云之间的最小距离
def pointcloud_distance(pcdA,pcdB,device='cpu'):
    try:
        pointsA = torch.tensor(pcdA.point.positions.cpu().numpy(),device=device)
        pointsB = torch.tensor(pcdB.point.positions.cpu().numpy(),device=device)
    except:
        pointsA = torch.tensor(np.array(pcdA.points),device=device)
        pointsB = torch.tensor(np.array(pcdB.points),device=device)
    cdist = torch.cdist(pointsA,pointsB)
    min_distances1, _ = cdist.min(dim=1)
    return min_distances1

# 在二维平面上计算点云间的距离(忽略z坐标)
def pointcloud_2d_distance(pcdA,pcdB,device='cpu'):
    pointsA = torch.tensor(pcdA.point.positions.cpu().numpy(),device=device)
    pointsA[:,2] = 0
    pointsB = torch.tensor(pcdB.point.positions.cpu().numpy(),device=device)
    pointsB[:,2] = 0
    cdist = torch.cdist(pointsA,pointsB)
    min_distances1, _ = cdist.min(dim=1)
    return min_distances1

def cpu_pointcloud_from_array(points,colors):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)
    return pointcloud

def gpu_pointcloud_from_array(points,colors,device):
    pointcloud = o3d.t.geometry.PointCloud(device)
    pointcloud.point.positions = o3d.core.Tensor(points,dtype=o3d.core.Dtype.Float32,device=device)
    pointcloud.point.colors = o3d.core.Tensor(colors.astype(np.float32)/255.0,dtype=o3d.core.Dtype.Float32,device=device)
    return pointcloud

def gpu_pointcloud(pointcloud,device):
    new_pointcloud = o3d.t.geometry.PointCloud(device)
    new_pointcloud.point.positions = o3d.core.Tensor(np.asarray(pointcloud.points),device=device)
    new_pointcloud.point.colors = o3d.core.Tensor(np.asarray(pointcloud.colors),device=device)
    return new_pointcloud
    
def cpu_pointcloud(pointcloud):
    new_pointcloud = o3d.geometry.PointCloud()
    new_pointcloud.points = o3d.utility.Vector3dVector(pointcloud.point.positions.cpu().numpy())
    new_pointcloud.colors = o3d.utility.Vector3dVector(pointcloud.point.colors.cpu().numpy())
    return new_pointcloud

def cpu_merge_pointcloud(pcdA,pcdB):
    return pcdA + pcdB

def gpu_merge_pointcloud(pcdA,pcdB):
    if pcdA.is_empty():
        return pcdB
    if pcdB.is_empty():
        return pcdA
    return pcdA + pcdB

def gpu_cluster_filter(pointcloud,eps=0.3,min_points=20):
    labels = pointcloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    numpy_labels = labels.cpu().numpy()
    unique_labels = np.unique(numpy_labels)
    largest_cluster_label = max(unique_labels, key=lambda x: np.sum(numpy_labels == x))
    largest_cluster_pc = pointcloud.select_by_index((labels == largest_cluster_label).nonzero()[0])
    return largest_cluster_pc

def cpu_cluster_filter(pointcloud,eps=0.3,min_points=20):
    labels = pointcloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    unique_labels = np.unique(labels)
    largest_cluster_label = max(unique_labels, key=lambda x: np.sum(labels == x))
    largest_cluster_pc = pointcloud.select_by_index((labels == largest_cluster_label).nonzero()[0])
    return largest_cluster_pc

def quat2array(quat):
    return np.array([quat.w,quat.x,quat.y,quat.z],np.float32)

def quaternion_distance(quatA,quatB):
    # M*4, N*4
    dot = np.dot(quatA,quatB.T)
    dot[dot<0] = -dot[dot<0]
    angle = 2*np.arccos(dot)
    return angle/np.pi*180

# 计算两组点之间的距离
def eculidean_distance(posA,posB):
    posA_reshaped = posA[:, np.newaxis, :]
    posB_reshaped = posB[np.newaxis, :, :]
    pairwise_distance = np.sqrt(np.sum((posA_reshaped - posB_reshaped)**2, axis=2))
    return pairwise_distance


def convert_cloud_to_ros_msg(points, frame_id="map"):
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    import numpy as np

    # 检查输入是否为 NumPy 数组
    if not isinstance(points, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(points)}")

    # 确保点云形状为 (N, 3)
    if points.shape[1] != 3:
        raise ValueError(f"Points array must have shape (N, 3), got {points.shape}")

    # 定义点云字段 (x, y, z)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    # 创建 Header 对象
    header = Header()
    header.stamp = rclpy.time.Time(seconds=0).to_msg()
    header.frame_id = frame_id

    # 转换点云数据为二进制格式
    cloud_msg = PointCloud2()
    cloud_msg.header = header
    cloud_msg.height = 1
    cloud_msg.width = points.shape[0]  # NumPy 数组的行数为点的数量
    cloud_msg.is_dense = True
    cloud_msg.is_bigendian = False
    cloud_msg.fields = fields
    cloud_msg.point_step = 12  # 每个点占用的字节数 (x, y, z -> 3 * float32)
    cloud_msg.row_step = cloud_msg.point_step * points.shape[0]
    cloud_msg.data = points.astype(np.float32).tobytes()

    return cloud_msg


def quaternion_to_rotation_matrix(quaternion):
    """
    将四元数转换为旋转矩阵
    :param quaternion: 四元数，geometry_msgs.msg.Quaternion 类型
    :return: 旋转矩阵，np.array 类型
    """
    w, x, y, z = quaternion.w, quaternion.x, quaternion.y, quaternion.z
    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])
    return R

def calculate_3Dpoint_distance(point1, point2):
    """
    计算两个点之间的欧几里得距离
    """
    return math.sqrt((point2.x - point1.x) ** 2 + 
                        (point2.y - point1.y) ** 2 + 
                        (point2.z - point1.z) ** 2)