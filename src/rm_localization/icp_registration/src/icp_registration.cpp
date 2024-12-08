#include "icp_registration/icp_registration.hpp"
#include <Eigen/src/Geometry/Quaternion.h>
#include <Eigen/src/Geometry/Transform.h>
#include <chrono>
#include <geometry_msgs/msg/detail/pose_with_covariance_stamped__struct.hpp>
#include <iostream>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <rclcpp/qos.hpp>
#include <stdexcept>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/create_timer_ros.h>

#include "cloud_registration.h"

namespace icp {

IcpNode::IcpNode(const rclcpp::NodeOptions &options)
    : Node("icp_registration", options), rough_iter_(10), refine_iter_(5),
      first_scan_(true) {
  is_ready_ = false;
  cloud_in_ =
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  // this->declare_parameter("use_sim_time", false);
  double rough_leaf_size = this->declare_parameter("rough_leaf_size", 0.4);
  double refine_leaf_size = this->declare_parameter("refine_leaf_size", 0.1);
  voxel_rough_filter_.setLeafSize(rough_leaf_size, rough_leaf_size,
                                  rough_leaf_size);
  voxel_refine_filter_.setLeafSize(refine_leaf_size, refine_leaf_size,
                                   refine_leaf_size);

  pcd_path_ = this->declare_parameter("pcd_path", std::string(""));
  if (!std::filesystem::exists(pcd_path_)) {
    RCLCPP_ERROR(this->get_logger(), "Invalid pcd path: %s", pcd_path_.c_str());
    throw std::runtime_error("Invalid pcd path");
  }
  // Read the pcd file
  pcl::PCDReader reader;
  map_.reset(new pcl::PointCloud<pcl::PointXYZI>);
  reader.read(pcd_path_, *map_);
  voxel_refine_filter_.setInputCloud(map_);
  voxel_refine_filter_.filter(*map_);

  // Add normal to the pointcloud
  refine_map_ = addNorm(map_);
  pcl::PointCloud<pcl::PointXYZI>::Ptr point_rough(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr filterd_point_rough(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::copyPointCloud(*refine_map_, *point_rough);
  voxel_rough_filter_.setInputCloud(point_rough);
  voxel_rough_filter_.filter(*filterd_point_rough);
  rough_map_ = addNorm(filterd_point_rough);

  RCLCPP_INFO(this->get_logger(), "pcd point size: %ld, %ld",
              refine_map_->size(), rough_map_->size());

  // Parameters
  map_frame_id_ = this->declare_parameter("map_frame_id", std::string("map"));
  odom_frame_id_ =
      this->declare_parameter("odom_frame_id", std::string("odom"));
  livox_frame_id_ = this->declare_parameter("livox_frame_id", std::string("livox_frame"));
  range_odom_frame_id_ =
      this->declare_parameter("range_odom_frame_id", std::string("lidar_odom"));
  laser_frame_id_ =
      this->declare_parameter("laser_frame_id", std::string("laser"));
  thresh_ = this->declare_parameter("thresh", 0.15);
  xy_offset_ = this->declare_parameter("xy_offset", 0.2);
  yaw_offset_ = this->declare_parameter("yaw_offset", 30.0) * M_PI / 180.0;
  yaw_resolution_ =
      this->declare_parameter("yaw_resolution", 10.0) * M_PI / 180.0;
  std::vector<double> initial_pose_vec = this->declare_parameter(
      "initial_pose", std::vector<double>{0, 0, 0, 0, 0, 0});
  try {
    initial_pose_.position.x = initial_pose_vec.at(0);
    initial_pose_.position.y = initial_pose_vec.at(1);
    initial_pose_.position.z = initial_pose_vec.at(2);
    tf2::Quaternion q;
    q.setRPY(initial_pose_vec.at(3), initial_pose_vec.at(4),
             initial_pose_vec.at(5));
  } catch (const std::out_of_range &ex) {
    RCLCPP_ERROR(this->get_logger(),
                 "initial_pose is not a vector with 6 elements, what():%s",
                 ex.what());
  }

  // Set up the pointcloud subscriber
  std::string pointcloud_topic = this->declare_parameter(
      "pointcloud_topic", std::string("/livox/lidar/pointcloud"));
  RCLCPP_INFO(this->get_logger(), "pointcloud_topic: %s",
              pointcloud_topic.c_str());
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
  pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      pointcloud_topic, qos,
      std::bind(&IcpNode::pointcloudCallback, this, std::placeholders::_1));
  // Set up the initial pose subscriber
  initial_pose_sub_ =
      create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
          "/initialpose", qos,
          [this](geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
            initialPoseCallback(msg);
          });

  // Set up the transform broadcaster
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(), this->get_node_timers_interface());
  tf_buffer_->setCreateTimerInterface(timer_interface);
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Set up the timer
  // timer_ = create_wall_timer(std::chrono::milliseconds(10), [this]() {
  //   if (is_ready_) {
  //     map_to_odom_.header.stamp = now();
  //     map_to_odom_.header.frame_id = map_frame_id_;
  //     map_to_odom_.child_frame_id = odom_frame_id_;
  //     tf_broadcaster_->sendTransform(map_to_odom_);
  //   }
  // });

  tf_publisher_thread_ = std::make_unique<std::thread>([this]() {
    rclcpp::Rate rate(100);
    while (rclcpp::ok()) {
      {
        std::lock_guard lock(mutex_);
        if (is_ready_) {
          geometry_msgs::msg::TransformStamped odom_to_livox;
          try {
              odom_to_livox = tf_buffer_->lookupTransform(
                  odom_frame_id_, livox_frame_id_, tf2::TimePointZero);
          } catch (tf2::TransformException &ex) {
              RCLCPP_WARN(this->get_logger(), "Could not transform %s to %s: %s", 
                          odom_frame_id_.c_str(), livox_frame_id_.c_str(), ex.what());
              continue;
          }

          tf2::Transform tf_map_to_odom;
          tf2::fromMsg(map_to_odom_.transform, tf_map_to_odom);

          tf2::Transform tf_odom_to_livox;
          tf2::fromMsg(odom_to_livox.transform, tf_odom_to_livox);

          tf2::Transform tf_map_to_livox = tf_map_to_odom * tf_odom_to_livox;

          tf2::Vector3 origin = tf_map_to_livox.getOrigin();
          origin.setZ(0.0);
          tf_map_to_livox.setOrigin(origin);

          tf2::Transform tf_livox_to_odom = tf_odom_to_livox.inverse();
          tf2::Transform adjusted_map_to_odom = tf_map_to_livox * tf_livox_to_odom;

          map_to_odom_.transform = tf2::toMsg(adjusted_map_to_odom);

          RCLCPP_DEBUG_STREAM(this->get_logger(),
                              "Publishing adjusted tf with offset"
                              << map_to_odom_.transform.translation.x << " "
                              << map_to_odom_.transform.translation.y << " "
                              << map_to_odom_.transform.translation.z);

          map_to_odom_.header.stamp = now();
          map_to_odom_.header.frame_id = map_frame_id_;
          map_to_odom_.child_frame_id = odom_frame_id_;
          tf_broadcaster_->sendTransform(map_to_odom_);
        }
      }
        rate.sleep();
    }
  });


  RCLCPP_INFO(this->get_logger(), "icp_registration initialized");
}

IcpNode::~IcpNode() {
  if (tf_publisher_thread_->joinable()) {
    tf_publisher_thread_->join();
  }
}

void IcpNode::pointcloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  pcl::fromROSMsg(*msg, *cloud_in_);
  if (first_scan_) {
    auto pose_msg =
        std::make_shared<geometry_msgs::msg::PoseWithCovarianceStamped>();
    pose_msg->header = msg->header;
    pose_msg->pose.pose = initial_pose_;
    initialPoseCallback(pose_msg);
    first_scan_ = false;
  }
}

void IcpNode::initialPoseCallback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {

  // Set the initial pose
  Eigen::Vector3d pos(msg->pose.pose.position.x, msg->pose.pose.position.y,
                      msg->pose.pose.position.z);
  Eigen::Quaterniond q(
      msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
      msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
  Eigen::Matrix4d initial_guess;
  initial_guess.block<3, 3>(0, 0) = q.toRotationMatrix();
  initial_guess.block<3, 1>(0, 3) = pos;
  initial_guess(3, 3) = 1;

  // radius filter
  segmented_map_.reset(new pcl::PointCloud<pcl::PointXYZI>);
  // 遍历点云中的所有点，保留在给定范围内的点
  float radius = 100.0;
  for (const auto& point : map_->points) {
      float distance = std::sqrt((point.x - pos[0]) * (point.x - pos[0]) +
                                  (point.y - pos[1]) * (point.y - pos[1]));
      if (distance <= radius) {
          segmented_map_->points.push_back(point);
      }
  }

  // Align the pointcloud
  RCLCPP_INFO(this->get_logger(), "Aligning the pointcloud");
  Eigen::Matrix4d map_to_laser = multiAlignSync(cloud_in_, initial_guess);
  // Eigen::Matrix4d result = align(cloud_in_, initial_guess);
  if (!success_) {
    map_to_laser = initial_guess;
    RCLCPP_ERROR(this->get_logger(), "ICP failed");
  }

  // Eigen::Isometry3d laser_to_odom;
  Eigen::Matrix4d laser_to_odom = Eigen::Matrix4d::Identity();
  try {
    // Get odom to laser transform
    auto transform =
        tf_buffer_->lookupTransform(laser_frame_id_, range_odom_frame_id_, now(),
                                    rclcpp::Duration::from_seconds(10));
    // RCLCPP_INFO(get_logger(), "%s", transform.header.frame_id.c_str());
    Eigen::Vector3d t(transform.transform.translation.x,
                      transform.transform.translation.y,
                      transform.transform.translation.z);
    Eigen::Quaterniond q(
        transform.transform.rotation.w, transform.transform.rotation.x,
        transform.transform.rotation.y, transform.transform.rotation.z);
    // laser_to_odom.prerotate(q);
    // laser_to_odom.translate(t);
    laser_to_odom.block<3, 3>(0, 0) = q.toRotationMatrix();
    laser_to_odom.block<3, 1>(0, 3) = t;
  } catch (tf2::TransformException &ex) {
    std::lock_guard<std::mutex> lock(mutex_);
    RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
    is_ready_ = false;
    return;
  }
  Eigen::Matrix4d result = map_to_laser * laser_to_odom.matrix().cast<double>();
  {
    std::lock_guard lock(mutex_);
    map_to_odom_.transform.translation.x = result(0, 3);
    map_to_odom_.transform.translation.y = result(1, 3);
    map_to_odom_.transform.translation.z = result(2, 3);

    Eigen::Matrix3d rotation = result.block<3, 3>(0, 0);
    q = Eigen::Quaterniond(rotation);

    map_to_odom_.transform.rotation.w = q.w();
    map_to_odom_.transform.rotation.x = q.x();
    map_to_odom_.transform.rotation.y = q.y();
    map_to_odom_.transform.rotation.z = q.z();
    is_ready_ = true;
  }
}

// Eigen::Matrix4d IcpNode::align(PointCloudXYZI::Ptr source,
//                                const Eigen::Matrix4d &init_guess) {
//   success_ = false;
//   // Eigen::Vector3d xyz = init_guess.block<3, 1>(0, 3);

//   pcl::PointCloud<pcl::PointXYZI>::Ptr rough_source(
//       new pcl::PointCloud<pcl::PointXYZI>);
//   pcl::PointCloud<pcl::PointXYZI>::Ptr refine_source(
//       new pcl::PointCloud<pcl::PointXYZI>);

//   voxel_rough_filter_.setInputCloud(source);
//   voxel_rough_filter_.filter(*rough_source);
//   voxel_refine_filter_.setInputCloud(source);
//   voxel_refine_filter_.filter(*refine_source);

//   PointCloudXYZIN::Ptr rough_source_norm = addNorm(rough_source);
//   PointCloudXYZIN::Ptr refine_source_norm = addNorm(refine_source);
//   PointCloudXYZIN::Ptr align_point(new PointCloudXYZIN);
//   auto tic = std::chrono::system_clock::now();
//   icp_rough_.setInputSource(rough_source_norm);
//   icp_rough_.align(*align_point, init_guess.cast<float>());

//   score_ = icp_rough_.getFitnessScore();
//   if (!icp_rough_.hasConverged())
//     return Eigen::Matrix4d::Zero();

//   icp_refine_.setInputSource(refine_source_norm);
//   icp_refine_.align(*align_point, icp_rough_.getFinalTransformation());
//   score_ = icp_refine_.getFitnessScore();

//   if (!icp_refine_.hasConverged())
//     return Eigen::Matrix4d::Zero();
//   if (score_ > thresh_)
//     return Eigen::Matrix4d::Zero();
//   success_ = true;
//   auto toc = std::chrono::system_clock::now();
//   std::chrono::duration<double> duration = toc - tic;
//   RCLCPP_INFO(this->get_logger(), "align used: %f ms", duration.count() *
//   1000); RCLCPP_INFO(this->get_logger(), "score: %f", score_);

//   return icp_refine_.getFinalTransformation().cast<double>();
// }

Eigen::Matrix4d IcpNode::multiAlignSync(PointCloudXYZI::Ptr source,
                                        const Eigen::Matrix4d &init_guess) {
  static auto rotate2rpy = [](Eigen::Matrix3d &rot) -> Eigen::Vector3d {
    double roll = std::atan2(rot(2, 1), rot(2, 2));
    double pitch = asin(-rot(2, 0));
    double yaw = std::atan2(rot(1, 0), rot(0, 0));
    return Eigen::Vector3d(roll, pitch, yaw);
  };

  success_ = false;
  Eigen::Vector3d xyz = init_guess.block<3, 1>(0, 3);
  Eigen::Matrix3d rotation = init_guess.block<3, 3>(0, 0);
  Eigen::Vector3d rpy = rotate2rpy(rotation);
  Eigen::AngleAxisf rollAngle(rpy(0), Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf pitchAngle(rpy(1), Eigen::Vector3f::UnitY());
  std::vector<Eigen::Matrix4f> candidates;
  Eigen::Matrix4f temp_pose;

  RCLCPP_INFO(this->get_logger(), "initial guess: %f, %f, %f, %f, %f, %f",
              xyz(0), xyz(1), xyz(2), rpy(0), rpy(1), rpy(2));

  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      for (int k = -yaw_offset_; k <= yaw_offset_; k++) {
        Eigen::Vector3f pos(xyz(0) + i * xy_offset_, xyz(1) + j * xy_offset_,
                            xyz(2));
        Eigen::AngleAxisf yawAngle(rpy(2) + k * yaw_resolution_,
                                   Eigen::Vector3f::UnitZ());
        temp_pose.setIdentity();
        temp_pose.block<3, 3>(0, 0) =
            (rollAngle * pitchAngle * yawAngle).toRotationMatrix();
        temp_pose.block<3, 1>(0, 3) = pos;
        candidates.push_back(temp_pose);
      }
    }
  }
  pcl::PointCloud<pcl::PointXYZI>::Ptr rough_source(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr refine_source(
      new pcl::PointCloud<pcl::PointXYZI>);

  voxel_rough_filter_.setInputCloud(source);
  voxel_rough_filter_.filter(*rough_source);
  voxel_refine_filter_.setInputCloud(source);
  voxel_refine_filter_.filter(*refine_source);

  auto tic = std::chrono::system_clock::now();

  // ground alignment
  auto Tg = GroundAlignment(cloud_in_, segmented_map_);

  std::cout << Tg.matrix() << std::endl;

  wts_sensor_calib::CloudRegistration cloud_reg;

  double score;
  auto T = cloud_reg.FastVGICP(rough_source, segmented_map_, init_guess.cast<float>(), score);

  auto final_T = cloud_reg.FastGICP(refine_source, segmented_map_, T);

  success_ = true;
  auto toc = std::chrono::system_clock::now();
  std::chrono::duration<double> duration = toc - tic;
  RCLCPP_INFO(this->get_logger(), "align used: %f ms", duration.count() * 1000);

  return final_T.cast<double>();
}

PointCloudXYZIN::Ptr
IcpNode::addNorm(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZI>::Ptr searchTree(
      new pcl::search::KdTree<pcl::PointXYZI>);
  searchTree->setInputCloud(cloud);

  pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> normalEstimator;
  normalEstimator.setInputCloud(cloud);
  normalEstimator.setSearchMethod(searchTree);
  normalEstimator.setKSearch(15);
  normalEstimator.compute(*normals);
  PointCloudXYZIN::Ptr out(new PointCloudXYZIN);
  pcl::concatenateFields(*cloud, *normals, *out);
  return out;
}

Eigen::Isometry3d
IcpNode::GroundAlignment(const PointCloudXYZI::Ptr &source_cloud,
                         const PointCloudXYZI::Ptr &target_cloud) {
  Eigen::Vector4d source_ground_model;
  GroundExtraction(source_cloud, source_ground_model);
  Eigen::Vector4d target_ground_model;
  GroundExtraction(target_cloud, target_ground_model);

  Eigen::Vector3d vec_before, vec_after;
  vec_before << source_ground_model[0], source_ground_model[1],
      source_ground_model[2];
  vec_after << target_ground_model[0], target_ground_model[1],
      target_ground_model[2];

  Eigen::Matrix3d rot_mat =
      Eigen::Quaterniond::FromTwoVectors(vec_before, vec_after)
          .toRotationMatrix();

  Eigen::Isometry3d T_target_source = Eigen::Isometry3d::Identity();
  T_target_source.linear() = rot_mat;
  T_target_source.translation().z() =
      source_ground_model[3] - target_ground_model[3];

  return T_target_source;
}

void IcpNode::GroundExtraction(const PointCloudXYZI::Ptr &cloud,
                               Eigen::Vector4d &ground_model) {
  PointCloudXYZI::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);

  for (size_t i = 0; i < cloud->size(); ++i) {
    const auto &p = cloud->points[i];
    double distance = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (distance > 2 && distance < 50)
      cloud_filtered->push_back(p);
  }
  // *cloud_filtered = *cloud;

  pcl::ApproximateVoxelGrid<pcl::PointXYZI> downsample;
  downsample.setLeafSize(0.2f, 0.2f, 0.2f);
  downsample.setInputCloud(cloud_filtered);
  downsample.filter(*cloud_filtered);

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

  pcl::SACSegmentation<pcl::PointXYZI> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setAxis(Eigen::Vector3f(0, 0, 1));
  seg.setEpsAngle(M_PI_4);
  seg.setDistanceThreshold(0.1);

  seg.setInputCloud(cloud_filtered);
  seg.segment(*inliers, *coefficients);

  pcl::PointCloud<pcl::PointXYZI>::Ptr ground(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::ExtractIndices<pcl::PointXYZI> extract;
  extract.setInputCloud(cloud_filtered);
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*ground);

  pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground(new pcl::PointCloud<pcl::PointXYZI>);
  extract.setInputCloud(cloud_filtered);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*non_ground);

  Eigen::Vector4f ground_centroid;
  pcl::compute3DCentroid(*ground, ground_centroid);
  Eigen::Vector4f non_ground_centroid;
  pcl::compute3DCentroid(*non_ground, non_ground_centroid);

  Eigen::Vector3d v;
  v[0] = non_ground_centroid[0] - ground_centroid[0];
  v[1] = non_ground_centroid[1] - ground_centroid[1];
  v[2] = non_ground_centroid[2] - ground_centroid[2];
  float dot_product = coefficients->values[0] * v[0] +
                      coefficients->values[1] * v[1] +
                      coefficients->values[2] * v[2];

  if (dot_product > 0)
    ground_model << coefficients->values[0], coefficients->values[1],
        coefficients->values[2], coefficients->values[3];
  else
    ground_model << -coefficients->values[0], -coefficients->values[1],
        -coefficients->values[2], -coefficients->values[3];
}

} // namespace icp

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(icp::IcpNode)
