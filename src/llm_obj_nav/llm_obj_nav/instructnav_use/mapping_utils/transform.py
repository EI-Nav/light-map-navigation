import numpy as np
import quaternion
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point

# 获取相机内参
def gazebo_camera_intrinsic():

    fx = 347.99755859375  # Focal length in x
    fy = 347.99755859375  # Focal length in y
    xc = 320.0  # Principal point x (cx)
    zc = 240.0  # Principal point y (cy)

    intrinsic_matrix = np.array([[fx, 0, xc],
                                 [0, fy, zc],
                                 [0, 0, 1]], np.float32)
    return intrinsic_matrix

# Gazebo 平移坐标转换: Gazebo 的坐标系是(x(右),y(前),z(上))
# def gazebo_translation(position):
#     return np.array([-position.y, position.x, -position.z]) # 转换为[右、前、上]

# # Gazebo 旋转坐标转换
# def gazebo_rotation(rotation):
#     # 将四元数转为旋转矩阵
#     rotation_matrix = quaternion.as_rotation_matrix(rotation)
#     # 坐标系变换矩阵
#     transform_matrix = np.array([[0, -1, 0],
#                                   [1, 0, 0],
#                                   [0, 0, -1]])
#     # 应用变换
#     rotation_matrix = np.matmul(transform_matrix, rotation_matrix)
#     return rotation_matrix

def gazebo_translation(position):
    return np.array([position.x,position.y,position.z])

# habitat旋转坐标转换
def gazebo_rotation(rotation):
    # rotation_matrix = quaternion.as_rotation_matrix(rotation)
    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    transform_matrix = np.array([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
    rotation_matrix = np.matmul(transform_matrix,rotation_matrix)
    return rotation_matrix


def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    :param q: 四元数 [x, y, z, w]
    :return: 旋转矩阵 R
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R

