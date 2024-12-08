// 文件路径: src/your_package_name/src/pcd_publisher.cpp

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class PcdPublisherNode : public rclcpp::Node
{
public:
    PcdPublisherNode() : Node("pcd_publisher")
    {
        // 声明并获取发布频率参数
        declare_parameter<double>("publish_frequency", 1.0);
        publish_frequency_ = this->get_parameter("publish_frequency").as_double();

        // 创建一个发布者，发布到 /point_env 话题
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/point_env", 10);

        // 读取PCD文件
        if (pcl::io::loadPCDFile<pcl::PointXYZ>("/workspaces/light-map-navigation/src/rm_nav_bringup/PCD/YOUR_MAP_NAME.pcd", cloud_) == -1) // 替换为你的PCD文件路径
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load PCD file.");
            return;
        }

        // 创建一个定时器，以设定的频率发布点云
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / publish_frequency_)),
            std::bind(&PcdPublisherNode::publishPointCloud, this)
        );
    }

private:
    void publishPointCloud()
    {
        // 将PCL点云转换为ROS2消息
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(cloud_, output);

        // 设置消息的frame_id
        output.header.frame_id = "map";
        output.header.stamp = this->now();

        // 发布消息
        publisher_->publish(output);

        RCLCPP_INFO(this->get_logger(), "PCD file published to /point_env.");
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    pcl::PointCloud<pcl::PointXYZ> cloud_;
    double publish_frequency_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PcdPublisherNode>());
    rclcpp::shutdown();
    return 0;
}
