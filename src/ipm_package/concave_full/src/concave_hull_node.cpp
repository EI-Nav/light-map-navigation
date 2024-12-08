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
#include <iostream>
#include <limits>

double euler_from_quaternion(const geometry_msgs::msg::Quaternion& orientation) {
    tf2::Quaternion quat;
    tf2::fromMsg(orientation, quat);
    double roll, pitch, yaw;
    tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
    return yaw;
}

class ConcaveHullNode : public rclcpp::Node {
public:
    ConcaveHullNode() : Node("concave_hull_node") {
        point_cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ipm_contour_points", 10,
            std::bind(&ConcaveHullNode::pointCloudCallback, this, std::placeholders::_1));

        odometry_sub = this->create_subscription<nav_msgs::msg::Odometry>(
            "/Odometry", 10,
            std::bind(&ConcaveHullNode::odometryCallback, this, std::placeholders::_1));

        polygon_pub = this->create_publisher<geometry_msgs::msg::PolygonStamped>("concave_hull", 10);
        line_pub = this->create_publisher<geometry_msgs::msg::PolygonStamped>("line_segment", 10);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub;
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr polygon_pub;
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr line_pub;

    nav_msgs::msg::Odometry::SharedPtr last_odometry_msg_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr concave_hull_cloud_;

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*cloud_msg, *cloud);

        pcl::ConcaveHull<pcl::PointXYZ> concave_hull;
        concave_hull.setInputCloud(cloud);
        concave_hull.setAlpha(0.1);

        std::vector<pcl::Vertices> polygons;
        concave_hull_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        concave_hull.reconstruct(*concave_hull_cloud_, polygons);

        geometry_msgs::msg::PolygonStamped polygon_msg;
        polygon_msg.header.stamp = this->get_clock()->now();
        polygon_msg.header.frame_id = cloud_msg->header.frame_id;

        for (const auto& polygon : polygons) {
            for (const auto& index : polygon.vertices) {
                geometry_msgs::msg::Point32 point;
                point.x = concave_hull_cloud_->points[index].x;
                point.y = concave_hull_cloud_->points[index].y;
                point.z = concave_hull_cloud_->points[index].z;
                polygon_msg.polygon.points.push_back(point);
            }
        }
        polygon_pub->publish(polygon_msg);

        if (last_odometry_msg_) {
            drawLineSegment(polygon_msg);
        }
    }

    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        last_odometry_msg_ = msg;
        if (concave_hull_cloud_) {
            geometry_msgs::msg::PolygonStamped polygon_msg;
            polygon_msg.header.stamp = this->get_clock()->now();
            polygon_msg.header.frame_id = msg->header.frame_id;
            drawLineSegment(polygon_msg);
        }
    }

    void drawLineSegment(const geometry_msgs::msg::PolygonStamped& polygon_msg) {
        if (last_odometry_msg_ == nullptr || concave_hull_cloud_ == nullptr) {
            RCLCPP_WARN(this->get_logger(), "Odometry or concave hull data is not yet available.");
            return;
        }

        geometry_msgs::msg::Point32 centroid = calculateCentroid(polygon_msg.polygon.points);

        double robot_yaw = euler_from_quaternion(last_odometry_msg_->pose.pose.orientation);
        double perpendicular_angle = robot_yaw + M_PI / 2.0;
        double dx = std::cos(perpendicular_angle);
        double dy = std::sin(perpendicular_angle);

        double line_length = 1.0;
        geometry_msgs::msg::Point32 line_start, line_end;

        line_start.x = centroid.x + dx * line_length;
        line_start.y = centroid.y + dy * line_length;
        line_start.z = 0.0;

        line_end.x = centroid.x - dx * line_length;
        line_end.y = centroid.y - dy * line_length;
        line_end.z = 0.0;

        line_start = projectToConcaveHullBoundary(line_start, polygon_msg.polygon.points);
        line_end = projectToConcaveHullBoundary(line_end, polygon_msg.polygon.points);

        geometry_msgs::msg::PolygonStamped line_msg;
        line_msg.header.stamp = this->get_clock()->now();
        line_msg.header.frame_id = polygon_msg.header.frame_id;
        line_msg.polygon.points.push_back(line_start);
        line_msg.polygon.points.push_back(line_end);
        line_pub->publish(line_msg);

        RCLCPP_INFO(this->get_logger(), "Published perpendicular line at the centroid of the concave hull.");
    }

    geometry_msgs::msg::Point32 calculateCentroid(const std::vector<geometry_msgs::msg::Point32>& points) {
        geometry_msgs::msg::Point32 centroid;
        double sum_x = 0.0, sum_y = 0.0;

        for (const auto& point : points) {
            sum_x += point.x;
            sum_y += point.y;
        }

        centroid.x = sum_x / points.size();
        centroid.y = sum_y / points.size();
        centroid.z = 0.0;

        return centroid;
    }

    geometry_msgs::msg::Point32 projectToConcaveHullBoundary(
        const geometry_msgs::msg::Point32& point,
        const std::vector<geometry_msgs::msg::Point32>& hull_points) {
        
        geometry_msgs::msg::Point32 projected_point = point;
        double min_dist = std::numeric_limits<double>::max();

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
    rclcpp::spin(std::make_shared<ConcaveHullNode>());
    rclcpp::shutdown();
    return 0;
}
