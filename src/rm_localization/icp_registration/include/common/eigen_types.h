/*
 * Description:
 * Created by WJH on 2023/4/24.
 */

#ifndef WTS_SENSOR_CALIB_EIGEN_TYPES_H_
#define WTS_SENSOR_CALIB_EIGEN_TYPES_H_

#include <Eigen/Eigen>

using Vec2f = Eigen::Vector2f;
using Vec3f = Eigen::Vector3f;
using Vec4f = Eigen::Vector4f;
using Vec5f = Eigen::Matrix<float, 5, 1>;
using VecXf = Eigen::Matrix<float, Eigen::Dynamic, 1>;

using Mat3f = Eigen::Matrix3f;
using Mat4f = Eigen::Matrix4f;
using MatX3f = Eigen::Matrix<float, Eigen::Dynamic, 3>;

using Mat4d = Eigen::Matrix4d;

#endif  // WTS_SENSOR_CALIB_EIGEN_TYPES_H_
