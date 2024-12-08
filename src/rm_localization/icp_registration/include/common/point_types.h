/*
 * Description:
 * Created by WJH on 2023/4/24.
 */

#ifndef WTS_SENSOR_CALIB_POINT_TYPES_H_
#define WTS_SENSOR_CALIB_POINT_TYPES_H_

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "common/eigen_types.h"

struct LidarPointXYZIRT
{
  PCL_ADD_POINT4D;
  float intensity;
  std::uint16_t ring;
  double timestamp;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    LidarPointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        std::uint16_t, ring, ring)(double, timestamp, timestamp))

struct OusterPointXYZIRT
{
  PCL_ADD_POINT4D;
  float intensity;
  std::uint32_t t;
  std::uint16_t reflectivity;
  std::uint8_t ring;
  std::uint16_t noise;
  std::uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(
    OusterPointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        std::uint32_t, t, t)(std::uint16_t, reflectivity, reflectivity)(
        std::uint8_t, ring, ring)(std::uint16_t, noise, noise)(std::uint32_t,
                                                               range, range))

namespace wts_sensor_calib {

using PointType = pcl::PointXYZI;
using PointCloudType = pcl::PointCloud<PointType>;
using CloudPtr = PointCloudType::Ptr;

using LidarPointType = LidarPointXYZIRT;
using LidarCloudType = pcl::PointCloud<LidarPointType>;
using LidarCloudPtr = LidarCloudType::Ptr;

}  // namespace wts_sensor_calib
#endif  // WTS_SENSOR_CALIB_POINT_TYPES_H_
