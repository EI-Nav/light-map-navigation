cmake_minimum_required(VERSION 3.8)
project(concave_full)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(custom_interfaces REQUIRED)



add_executable(concave_hull_node src/concave_hull_node.cpp)
ament_target_dependencies(concave_hull_node rclcpp sensor_msgs nav_msgs pcl_conversions PCL tf2_ros tf2 tf2_geometry_msgs visualization_msgs)
install(TARGETS concave_hull_node DESTINATION lib/${PROJECT_NAME})
target_link_libraries(concave_hull_node ${PCL_LIBRARIES})


add_executable(concave_server src/concave_server.cpp)
ament_target_dependencies(concave_server rclcpp sensor_msgs nav_msgs pcl_conversions PCL tf2_ros tf2 tf2_geometry_msgs visualization_msgs custom_interfaces)
install(TARGETS concave_server DESTINATION lib/${PROJECT_NAME})
target_link_libraries(concave_server ${PCL_LIBRARIES})


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
