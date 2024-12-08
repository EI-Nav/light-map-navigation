cmake_minimum_required(VERSION 3.5)
project(llm_bt_executor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(behaviortree_ros2 REQUIRED)
find_package(btcpp_ros2_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(custom_interfaces REQUIRED)

set(THIS_PACKAGE_DEPS
    behaviortree_ros2
    std_msgs
    std_srvs
    custom_interfaces
    btcpp_ros2_interfaces )

include_directories(include)

add_library(navigation_node_dyn SHARED src/navigation_node.cpp)
target_compile_definitions(navigation_node_dyn PRIVATE  BT_PLUGIN_EXPORT )
ament_target_dependencies(navigation_node_dyn ${THIS_PACKAGE_DEPS})

add_executable(llm_bt_executor src/llm_bt_executor.cpp)
ament_target_dependencies(llm_bt_executor ${THIS_PACKAGE_DEPS})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

######################################################
# INSTALL

install(TARGETS
  navigation_node_dyn
  llm_bt_executor
  DESTINATION lib/${PROJECT_NAME}
  )

######################################################
# INSTALL plugins for other packages to load

INSTALL(TARGETS
    navigation_node_dyn
    LIBRARY DESTINATION share/${PROJECT_NAME}/bt_plugins
    ARCHIVE DESTINATION share/${PROJECT_NAME}/bt_plugins
    RUNTIME DESTINATION share/${PROJECT_NAME}/bt_plugins
)

######################################################
# INSTALL Behavior.xml's, ROS config and launch files

install(DIRECTORY
    behavior_trees
    config
    launch
    DESTINATION share/${PROJECT_NAME}/
    )

ament_export_dependencies(behaviortree_ros2 btcpp_ros2_interfaces)

ament_package()
