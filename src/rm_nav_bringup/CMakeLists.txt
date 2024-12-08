cmake_minimum_required(VERSION 3.8)
project(rm_nav_bringup)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ${CMAKE_EXPORT_COMPILE_COMMANDS})
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${pcl_conversions_INCLUDE_DIRS}
)

add_executable(envpcd_pub_node src/envpcd_pub_node.cpp)
ament_target_dependencies(envpcd_pub_node rclcpp sensor_msgs)

target_link_libraries(envpcd_pub_node
    ${PCL_LIBRARIES}
    ${pcl_conversions_LIBRARIES}
)

install(TARGETS envpcd_pub_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
  map
  PCD
  rviz
  urdf
)

