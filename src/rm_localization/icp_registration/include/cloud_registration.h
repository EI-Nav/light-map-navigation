/*
 * Description:
 * Created by WJH on 2023/4/25.
 */

#ifndef WTS_SENSOR_CALIB_CLOUD_REGISTRATION_H_
#define WTS_SENSOR_CALIB_CLOUD_REGISTRATION_H_

#include <iostream>
#include <omp.h>

#include <pcl/common/angles.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include "common/point_types.h"
#include "fast_gicp/gicp/fast_gicp.hpp"
#include "fast_gicp/gicp/fast_vgicp.hpp"
#include "ndt_omp/ndt_omp.h"

namespace wts_sensor_calib {

class CloudRegistration
{
 public:
  CloudRegistration();
  virtual ~CloudRegistration();

  /// 配准算法
  Mat4f FastVGICP(const CloudPtr& src_cloud, const CloudPtr& tgt_cloud,
                  const Mat4f& guess_mat, double& score);
  Mat4f FastGICP(const CloudPtr& src_cloud, const CloudPtr& tgt_cloud,
                 const Mat4f& guess_mat);
  Mat4f NDTOMP(const CloudPtr& src_cloud, const CloudPtr& tgt_cloud,
               const Mat4f& guess_mat);
  float SVDICP(const std::vector<int>& source_perm,
               const CloudPtr& source_cloud, const CloudPtr& target_cloud,
               Mat4f& calib_matrix);
  Mat4f PointToPlaneICP(const CloudPtr& src_cloud, const CloudPtr& tgt_cloud,
                       const Mat4f& guess_mat);

 private:
  void SurfAssociation(const CloudPtr& src_cloud, const CloudPtr& tgt_cloud);
  void UpdatePointAssociateToMap();
  Eigen::Isometry3d RzRyRxToMatrix(double* pose);
  void PointAssociateToMap(PointType const* const pi, PointType* const po);
  bool LMOptimization(int iter_count);

 private:
  CloudPtr cloud_ori_;
  CloudPtr coeff_sel_;

  Eigen::Matrix<float, 6, 6> matP;

  bool b_degenerate_;

  Eigen::Isometry3d transform_cloud_to_map_ = Eigen::Isometry3d::Identity();

  double transform_to_be_mapped_[6];

  int num_threads_ = 4;
};
}  // namespace wts_sensor_calib

#endif  // WTS_SENSOR_CALIB_CLOUD_REGISTRATION_H_
