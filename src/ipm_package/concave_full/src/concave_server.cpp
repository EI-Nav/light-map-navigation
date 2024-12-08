#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/surface/concave_hull.h>
#include <cmath>
#include <limits>
#include "custom_interfaces/srv/get_concave_hull.hpp"

double euler_from_quaternion(const geometry_msgs::msg::Quaternion& orientation) {
    tf2::Quaternion quat;
    tf2::fromMsg(orientation, quat);
    double roll, pitch, yaw;
    tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
    return yaw;
}

class ConcaveHullService : public rclcpp::Node {
public:
    ConcaveHullService() : Node("concave_hull_service") {
        // 创建 ConcaveHull 服务
        service_ = this->create_service<custom_interfaces::srv::GetConcaveHull>(
            "compute_concave_hull",
            std::bind(&ConcaveHullService::handleServiceRequest, this, std::placeholders::_1, std::placeholders::_2));

        // 订阅里程计数据
        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/Odometry", 10, std::bind(&ConcaveHullService::odometryCallback, this, std::placeholders::_1));
    }

private:
    rclcpp::Service<custom_interfaces::srv::GetConcaveHull>::SharedPtr service_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    nav_msgs::msg::Odometry::SharedPtr last_odometry_msg_;

    void handleServiceRequest(
        const std::shared_ptr<custom_interfaces::srv::GetConcaveHull::Request> request,
        std::shared_ptr<custom_interfaces::srv::GetConcaveHull::Response> response) {

        // 从请求中获取点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(request->point_cloud, *cloud);

        // 生成凹包
        pcl::ConcaveHull<pcl::PointXYZ> concave_hull;
        concave_hull.setInputCloud(cloud);
        concave_hull.setAlpha(0.1);

        std::vector<pcl::Vertices> polygons;
        pcl::PointCloud<pcl::PointXYZ>::Ptr concave_hull_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        concave_hull.reconstruct(*concave_hull_cloud, polygons);

        geometry_msgs::msg::PolygonStamped polygon_msg;
        polygon_msg.header.stamp = this->get_clock()->now();
        polygon_msg.header.frame_id = request->point_cloud.header.frame_id;

        for (const auto& polygon : polygons) {
            for (const auto& index : polygon.vertices) {
                geometry_msgs::msg::Point32 point;
                point.x = concave_hull_cloud->points[index].x;
                point.y = concave_hull_cloud->points[index].y;
                point.z = concave_hull_cloud->points[index].z;
                polygon_msg.polygon.points.push_back(point);
            }
        }

        // 设置凹包多边形到响应
        response->concave_hull = polygon_msg;

        // 如果有最近的里程计消息，则计算分割线段
        if (last_odometry_msg_) {
            response->line_segment = createLineSegment(polygon_msg);
            RCLCPP_INFO(this->get_logger(), "Concave hull and line segment computed successfully.");
        } else {
            RCLCPP_WARN(this->get_logger(), "Odometry data is not available; unable to compute line segment.");
        }
    }

    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        last_odometry_msg_ = msg;
    }

    geometry_msgs::msg::PolygonStamped createLineSegment(const geometry_msgs::msg::PolygonStamped& polygon_msg) {
        geometry_msgs::msg::PolygonStamped line_msg;
        line_msg.header.stamp = this->get_clock()->now();
        line_msg.header.frame_id = polygon_msg.header.frame_id;

        // 计算多边形的重心
        double cx = 0.0, cy = 0.0;
        int num_points = polygon_msg.polygon.points.size();
        for (const auto& point : polygon_msg.polygon.points) {
            cx += point.x;
            cy += point.y;
        }
        cx /= num_points;
        cy /= num_points;

        // 计算垂直于机器人朝向的方向
        double robot_yaw = euler_from_quaternion(last_odometry_msg_->pose.pose.orientation);
        double perpendicular_angle = robot_yaw + M_PI / 2.0;
        double dx = std::cos(perpendicular_angle);
        double dy = std::sin(perpendicular_angle);

        // 线段长度和端点计算
        double line_length = 1.0;
        geometry_msgs::msg::Point32 line_start, line_end;

        line_start.x = cx + dx * line_length;
        line_start.y = cy + dy * line_length;
        line_start.z = 0.0;

        line_end.x = cx - dx * line_length;
        line_end.y = cy - dy * line_length;
        line_end.z = 0.0;

        // 将线段端点投影到凹包边界上
        line_start = projectToConcaveHullBoundary(line_start, polygon_msg.polygon.points);
        line_end = projectToConcaveHullBoundary(line_end, polygon_msg.polygon.points);

        // 将起点和终点添加到响应中的线段消息
        line_msg.polygon.points.push_back(line_start);
        line_msg.polygon.points.push_back(line_end);

        return line_msg;
    }

    geometry_msgs::msg::Point32 projectToConcaveHullBoundary(
        const geometry_msgs::msg::Point32& point,
        const std::vector<geometry_msgs::msg::Point32>& hull_points) {
        
        geometry_msgs::msg::Point32 projected_point = point;
        double min_dist = std::numeric_limits<double>::max();

        // 找到最近的边界点作为投影点
        for (const auto& hull_point : hull_points) {
            double dist = std::sqrt(std::pow(point.x - hull_point.x, 2) + std::pow(point.y - hull_point.y, 2));
            if (dist < min_dist) {
                min_dist = dist;
                projected_point = hull_point;
            }
        }
        return projected_point;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ConcaveHullService>());
    rclcpp::shutdown();
    return 0;
}
