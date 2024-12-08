/*
 * Description:
 * Created by WJH on 2023/4/25.
 */

#include "cloud_registration.h"
#include "nanoflann_pcl.h"

namespace wts_sensor_calib {

CloudRegistration::CloudRegistration() {}

CloudRegistration::~CloudRegistration() {}

Mat4f wts_sensor_calib::CloudRegistration::FastVGICP(const CloudPtr& src_cloud,
                                                     const CloudPtr& tgt_cloud,
                                                     const Mat4f& guess_mat,
                                                     double& score)
{
  std::cout << "Start Registration" << std::endl;

  // 降采样
  CloudPtr target_cloud_filtered(new PointCloudType);
  CloudPtr source_cloud_filtered(new PointCloudType);

  pcl::ApproximateVoxelGrid<PointType> downsample;
  downsample.setLeafSize(1.0f, 1.0f, 1.0f);
  downsample.setInputCloud(src_cloud);
  downsample.filter(*source_cloud_filtered);

  downsample.setInputCloud(tgt_cloud);
  downsample.filter(*target_cloud_filtered);

  pcl::transformPointCloud(*source_cloud_filtered, *source_cloud_filtered,
                           guess_mat);

  PointCloudType transformed_cloud;
  Mat4f temp_matrix;

  fast_gicp::FastVGICP<PointType, PointType> vgicp;
  vgicp.setResolution(1.0);
  vgicp.setNumThreads(num_threads_);
  vgicp.setMaxCorrespondenceDistance(10);
  vgicp.setTransformationEpsilon(1e-10);
  vgicp.setEuclideanFitnessEpsilon(0.001);
  vgicp.clearTarget();
  vgicp.clearSource();
  vgicp.setInputTarget(target_cloud_filtered);
  vgicp.setInputSource(source_cloud_filtered);
  vgicp.align(transformed_cloud);
  temp_matrix = vgicp.getFinalTransformation();

  score = vgicp.getFitnessScore();

  std::cout << "Finish Registration" << std::endl;

  return temp_matrix * guess_mat;
}

Mat4f CloudRegistration::FastGICP(const CloudPtr& src_cloud,
                                  const CloudPtr& tgt_cloud,
                                  const Mat4f& guess_mat)
{
  std::cout << "Start Registration" << std::endl;

  // 降采样
  CloudPtr target_cloud_filtered(new PointCloudType);
  CloudPtr source_cloud_filtered(new PointCloudType);

  pcl::ApproximateVoxelGrid<PointType> downsample;
  downsample.setLeafSize(0.2f, 0.2f, 0.2f);
  downsample.setInputCloud(src_cloud);
  downsample.filter(*source_cloud_filtered);

  downsample.setInputCloud(tgt_cloud);
  downsample.filter(*target_cloud_filtered);

  pcl::transformPointCloud(*source_cloud_filtered, *source_cloud_filtered,
                           guess_mat);

  PointCloudType transformed_cloud;
  Mat4f temp_matrix;

  fast_gicp::FastGICP<PointType, PointType> gicp;
  gicp.setNumThreads(num_threads_);
  gicp.setMaxCorrespondenceDistance(0.2);
  gicp.setTransformationEpsilon(1e-10);
  gicp.setEuclideanFitnessEpsilon(0.001);
  gicp.clearTarget();
  gicp.clearSource();
  gicp.setInputTarget(target_cloud_filtered);
  gicp.setInputSource(source_cloud_filtered);
  gicp.align(transformed_cloud);
  temp_matrix = gicp.getFinalTransformation();

  std::cout << "Finish Registration" << std::endl;

  return temp_matrix * guess_mat;
}

Mat4f CloudRegistration::NDTOMP(const CloudPtr& src_cloud,
                                const CloudPtr& tgt_cloud,
                                const Mat4f& guess_mat)
{
  std::cout << "Start Registration" << std::endl;

  // 降采样
  CloudPtr target_cloud_filtered(new PointCloudType);
  CloudPtr source_cloud_filtered(new PointCloudType);

  pcl::ApproximateVoxelGrid<PointType> downsample;
  downsample.setLeafSize(0.1f, 0.1f, 0.1f);
  downsample.setInputCloud(src_cloud);
  downsample.filter(*source_cloud_filtered);

  downsample.setInputCloud(tgt_cloud);
  downsample.filter(*target_cloud_filtered);

  pcl::transformPointCloud(*source_cloud_filtered, *source_cloud_filtered,
                           guess_mat);

  PointCloudType transformed_cloud;
  Mat4f temp_matrix;

  pclomp::NormalDistributionsTransform<PointType, PointType> ndt_omp;
  ndt_omp.setMaximumIterations(200);
  ndt_omp.setResolution(1.0);
  ndt_omp.setMaxCorrespondenceDistance(10);
  ndt_omp.setNumThreads(num_threads_);
  ndt_omp.setNeighborhoodSearchMethod(pclomp::DIRECT1);
  ndt_omp.setInputTarget(target_cloud_filtered);
  ndt_omp.setInputSource(source_cloud_filtered);
  ndt_omp.align(transformed_cloud);
  temp_matrix = ndt_omp.getFinalTransformation();

  std::cout << "Finish Registration" << std::endl;

  return temp_matrix * guess_mat;
}

float wts_sensor_calib::CloudRegistration::SVDICP(
    const std::vector<int>& source_perm, const CloudPtr& source_cloud,
    const CloudPtr& target_cloud, Mat4f& calib_matrix)
{
  if (source_cloud->size() != target_cloud->size())
  {
    std::cout << "源点云和目标点云数量需要相同" << std::endl;
    return -1;
  }

  const size_t num_of_points = source_cloud->size();
  std::vector<Vec3f> src_cloud, tgt_cloud;
  for (size_t i = 0; i < num_of_points; ++i)
  {
    auto& src_point = source_cloud->points[source_perm[i]];
    src_cloud.push_back(Vec3f(src_point.x, src_point.y, src_point.z));
    auto& tgt_point = target_cloud->points[i];
    tgt_cloud.push_back(Vec3f(tgt_point.x, tgt_point.y, tgt_point.z));
  }

  // 计算质心
  Vec3f src_center(Vec3f::Zero()), tgt_center(Vec3f::Zero());
  for (size_t i = 0; i < num_of_points; ++i)
  {
    src_center += src_cloud[i];
    tgt_center += tgt_cloud[i];
  }
  src_center /= num_of_points;
  tgt_center /= num_of_points;

  // 计算H矩阵
  Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
  for (size_t i = 0; i < num_of_points; ++i)
    H += (src_cloud[i] - src_center) * (tgt_cloud[i] - tgt_center).transpose();

  // 通过svd求解旋转矩阵
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(
      H, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::Matrix3f U = svd.matrixU();
  Eigen::Matrix3f V = svd.matrixV();
  double determinant = H.determinant();
  Eigen::Matrix3f R;
  if (determinant < 0)
    R = -V * U.transpose();
  else
    R = V * U.transpose();

  // 利用旋转矩阵求解平移
  Vec3f t = tgt_center - R * src_center;

  calib_matrix = Eigen::Matrix4f::Identity();
  Eigen::Quaternionf q_temp(R);
  q_temp.normalize();
  calib_matrix.block<3, 3>(0, 0) = q_temp.toRotationMatrix();
  calib_matrix.block<3, 1>(0, 3) = t;

  float mse = 0.0;
  for (size_t i = 0; i < num_of_points; ++i)
  {
    auto& src_p = source_cloud->points[source_perm[i]];
    Eigen::Vector4f src_p_vec(src_p.x, src_p.y, src_p.z, 1);
    auto& tgt_p = target_cloud->points[i];
    Eigen::Vector4f tgt_p_vec(tgt_p.x, tgt_p.y, tgt_p.z, 1);

    mse += (tgt_p_vec - calib_matrix * src_p_vec).squaredNorm();
  }
  mse /= num_of_points;

  return mse;
}

Mat4f CloudRegistration::PointToPlaneICP(const CloudPtr& src_cloud,
                                         const CloudPtr& tgt_cloud,
                                         const Mat4f& guess_mat)
{
  // 初始化
  cloud_ori_.reset(new PointCloudType);
  coeff_sel_.reset(new PointCloudType);

  Eigen::Vector3f euler_angles;
  euler_angles[0] = atan2(guess_mat(1, 0), guess_mat(0, 0));
  euler_angles[1] =
      atan2(-guess_mat(2, 0), sqrt(1 - guess_mat(2, 0) * guess_mat(2, 0)));
  euler_angles[2] = atan2(guess_mat(2, 1), guess_mat(2, 2));
  Eigen::Vector3f t_init_curr = guess_mat.block<3, 1>(0, 3);

  transform_to_be_mapped_[0] = euler_angles[2];
  transform_to_be_mapped_[1] = euler_angles[1];
  transform_to_be_mapped_[2] = euler_angles[0];
  transform_to_be_mapped_[3] = t_init_curr[0];
  transform_to_be_mapped_[4] = t_init_curr[1];
  transform_to_be_mapped_[5] = t_init_curr[2];

  CloudPtr source_cloud, target_cloud;
  source_cloud.reset(new PointCloudType);
  target_cloud.reset(new PointCloudType);
  *source_cloud = *src_cloud;
  *target_cloud = *tgt_cloud;

  for (int iter_count = 0; iter_count < 30; iter_count++)
  {
    cloud_ori_->clear();
    coeff_sel_->clear();

    SurfAssociation(source_cloud, target_cloud);

    if (LMOptimization(iter_count) == true) break;
  }

  return RzRyRxToMatrix(transform_to_be_mapped_).matrix().cast<float>();
}

void CloudRegistration::SurfAssociation(const CloudPtr& src_cloud,
                                        const CloudPtr& tgt_cloud)
{
  nanoflann::KdTreeFLANN<PointType> kdtree_target_cloud;
  kdtree_target_cloud.setInputCloud(tgt_cloud);

  UpdatePointAssociateToMap();

  Eigen::Matrix<float, 5, 3> matA0;
  Eigen::Matrix<float, 5, 1> matB0;
  Eigen::Vector3f matX0;
  matA0.setZero();
  matB0.fill(-1);
  matX0.setZero();
  matP.setZero();

  std::vector<int> point_search_ind;
  std::vector<float> point_search_sq_dis;

  for (int i = 0; i < src_cloud->size(); i++)
  {
    PointType point_ori, point_sel, coeff;

    point_ori = src_cloud->points[i];
    PointAssociateToMap(&point_ori, &point_sel);
    kdtree_target_cloud.nearestKSearch(point_sel, 5, point_search_ind,
                                       point_search_sq_dis);

    if (point_search_sq_dis[4] < 1.0)
    {
      for (int j = 0; j < 5; j++)
      {
        matA0(j, 0) = tgt_cloud->points[point_search_ind[j]].x;
        matA0(j, 1) = tgt_cloud->points[point_search_ind[j]].y;
        matA0(j, 2) = tgt_cloud->points[point_search_ind[j]].z;
      }
      matX0 = matA0.colPivHouseholderQr().solve(matB0);

      float pa = matX0(0, 0);
      float pb = matX0(1, 0);
      float pc = matX0(2, 0);
      float pd = 1;

      float ps = sqrt(pa * pa + pb * pb + pc * pc);
      pa /= ps;
      pb /= ps;
      pc /= ps;
      pd /= ps;

      bool planeValid = true;
      for (int j = 0; j < 5; j++)
      {
        if (fabs(pa * tgt_cloud->points[point_search_ind[j]].x +
                 pb * tgt_cloud->points[point_search_ind[j]].y +
                 pc * tgt_cloud->points[point_search_ind[j]].z + pd) > 0.2)
        {
          planeValid = false;
          break;
        }
      }

      if (planeValid)
      {
        float pd2 = pa * point_sel.x + pb * point_sel.y + pc * point_sel.z + pd;

        float s = 1 - 0.9 * fabs(pd2) /
                          sqrt(sqrt(point_sel.x * point_sel.x +
                                    point_sel.y * point_sel.y +
                                    point_sel.z * point_sel.z));

        coeff.x = s * pa;
        coeff.y = s * pb;
        coeff.z = s * pc;
        coeff.intensity = s * pd2;

        if (s > 0.1)
        {
          cloud_ori_->push_back(point_ori);
          coeff_sel_->push_back(coeff);
        }
      }
    }
  }
}

void CloudRegistration::UpdatePointAssociateToMap()
{
  transform_cloud_to_map_ = RzRyRxToMatrix(transform_to_be_mapped_);
}

Eigen::Isometry3d CloudRegistration::RzRyRxToMatrix(double* pose)
{
  Eigen::AngleAxisd roll_angle(
      Eigen::AngleAxisd(pose[0], Eigen::Vector3d::UnitX()));
  Eigen::AngleAxisd pitch_angle(
      Eigen::AngleAxisd(pose[1], Eigen::Vector3d::UnitY()));
  Eigen::AngleAxisd yaw_angle(
      Eigen::AngleAxisd(pose[2], Eigen::Vector3d::UnitZ()));
  Eigen::Quaterniond rot = yaw_angle * pitch_angle * roll_angle;
  rot.normalize();
  Eigen::Vector3d trans(pose[3], pose[4], pose[5]);
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = rot.toRotationMatrix();
  T.translation() = trans;

  return T;
}

void CloudRegistration::PointAssociateToMap(const PointType* const pi,
                                            PointType* const po)
{
  po->x = transform_cloud_to_map_(0, 0) * pi->x +
          transform_cloud_to_map_(0, 1) * pi->y +
          transform_cloud_to_map_(0, 2) * pi->z + transform_cloud_to_map_(0, 3);
  po->y = transform_cloud_to_map_(1, 0) * pi->x +
          transform_cloud_to_map_(1, 1) * pi->y +
          transform_cloud_to_map_(1, 2) * pi->z + transform_cloud_to_map_(1, 3);
  po->z = transform_cloud_to_map_(2, 0) * pi->x +
          transform_cloud_to_map_(2, 1) * pi->y +
          transform_cloud_to_map_(2, 2) * pi->z + transform_cloud_to_map_(2, 3);
  po->intensity = pi->intensity;
}

bool CloudRegistration::LMOptimization(int iter_count)
{
  float srx = sin(transform_to_be_mapped_[2]);
  float crx = cos(transform_to_be_mapped_[2]);
  float sry = sin(transform_to_be_mapped_[1]);
  float cry = cos(transform_to_be_mapped_[1]);
  float srz = sin(transform_to_be_mapped_[0]);
  float crz = cos(transform_to_be_mapped_[0]);

  int laserCloudSelNum = cloud_ori_->points.size();
  //  if (laserCloudSelNum < 50) return false;

  Eigen::Matrix<float, Eigen::Dynamic, 6> matA(laserCloudSelNum, 6);
  Eigen::Matrix<float, 6, Eigen::Dynamic> matAt(6, laserCloudSelNum);
  Eigen::Matrix<float, 6, 6> matAtA;
  Eigen::VectorXf matB(laserCloudSelNum);
  Eigen::Matrix<float, 6, 1> matAtB;
  Eigen::Matrix<float, 6, 1> matX;

  for (int i = 0; i < laserCloudSelNum; i++)
  {
    PointType point_ori, coeff;
    point_ori.x = cloud_ori_->points[i].x;
    point_ori.y = cloud_ori_->points[i].y;
    point_ori.z = cloud_ori_->points[i].z;

    coeff.x = coeff_sel_->points[i].x;
    coeff.y = coeff_sel_->points[i].y;
    coeff.z = coeff_sel_->points[i].z;
    coeff.intensity = coeff_sel_->points[i].intensity;

    float arx =
        (-srx * cry * point_ori.x -
         (srx * sry * srz + crx * crz) * point_ori.y +
         (crx * srz - srx * sry * crz) * point_ori.z) *
            coeff.x +
        (crx * cry * point_ori.x - (srx * crz - crx * sry * srz) * point_ori.y +
         (crx * sry * crz + srx * srz) * point_ori.z) *
            coeff.y;

    float ary = (-crx * sry * point_ori.x + crx * cry * srz * point_ori.y +
                 crx * cry * crz * point_ori.z) *
                    coeff.x +
                (-srx * sry * point_ori.x + srx * sry * srz * point_ori.y +
                 srx * cry * crz * point_ori.z) *
                    coeff.y +
                (-cry * point_ori.x - sry * srz * point_ori.y -
                 sry * crz * point_ori.z) *
                    coeff.z;

    float arz = ((crx * sry * crz + srx * srz) * point_ori.y +
                 (srx * crz - crx * sry * srz) * point_ori.z) *
                    coeff.x +
                ((-crx * srz + srx * sry * crz) * point_ori.y +
                 (-srx * sry * srz - crx * crz) * point_ori.z) *
                    coeff.y +
                (cry * crz * point_ori.y - cry * srz * point_ori.z) * coeff.z;

    matA(i, 0) = arz;
    matA(i, 1) = ary;
    matA(i, 2) = arx;
    matA(i, 3) = coeff.x;
    matA(i, 4) = coeff.y;
    matA(i, 5) = coeff.z;
    matB(i, 0) = -coeff.intensity;
  }
  matAt = matA.transpose();
  matAtA = matAt * matA;
  matAtB = matAt * matB;
  matX = matAtA.colPivHouseholderQr().solve(matAtB);

  if (iter_count == 0)
  {
    Eigen::Matrix<float, 1, 6> matE;
    Eigen::Matrix<float, 6, 6> matV;
    Eigen::Matrix<float, 6, 6> matV2;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> esolver(matAtA);
    matE = esolver.eigenvalues().real();  // 特征值按从小到大排序
    matV = esolver.eigenvectors()
               .real()
               .transpose();  // 转置，得到每一行是特征向量

    matV2 = matV;

    b_degenerate_ = false;
    float eignThre[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i <= 5; i++)
    {
      if (matE(0, i) < eignThre[i])
      {
        // 判断退化的维度
        float max = FLT_MIN;
        int dim = 1;
        for (int j = 0; j < 6; ++j)
        {
          if (matV2(i, j) > max)
          {
            max = matV2(i, j);
            dim = j + 1;
          }
        }

        for (int j = 0; j < 6; j++) matV2(i, j) = 0;

        b_degenerate_ = true;
      } else
      {
        break;
      }
    }

    matP = matV.inverse() * matV2;
  }

  if (b_degenerate_)
  {
    Eigen::Matrix<float, 6, 1> matX2(matX);
    matX2 = matX;
    matX = matP * matX2;
  }

  transform_to_be_mapped_[0] += matX(0, 0);
  transform_to_be_mapped_[1] += matX(1, 0);
  transform_to_be_mapped_[2] += matX(2, 0);
  transform_to_be_mapped_[3] += matX(3, 0);
  transform_to_be_mapped_[4] += matX(4, 0);
  transform_to_be_mapped_[5] += matX(5, 0);

  float deltaR =
      sqrt(pow(pcl::rad2deg(matX(0, 0)), 2) + pow(pcl::rad2deg(matX(1, 0)), 2) +
           pow(pcl::rad2deg(matX(2, 0)), 2));
  float deltaT = sqrt(pow(matX(3, 0) * 100, 2) + pow(matX(4, 0) * 100, 2) +
                      pow(matX(5, 0) * 100, 2));

  if (deltaR < 0.05 && deltaT < 0.05) return true;

  return false;
}

}  // namespace wts_sensor_calib
