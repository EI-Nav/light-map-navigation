cmake_minimum_required(VERSION 3.8)
project(icp_registration)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(PCL REQUIRED QUIET)

ament_auto_find_build_dependencies()

add_library(ndt_omp SHARED
    src/ndt_omp/voxel_grid_covariance_omp.cpp
    src/ndt_omp/ndt_omp.cpp
    )
target_link_libraries(ndt_omp
    ${PCL_LIBRARIES}
    )
target_include_directories(ndt_omp PUBLIC
    include
    ${PCL_INCLUDE_DIRS}
    )

add_library(fast_gicp SHARED
    src/fast_gicp/gicp/lsq_registration.cpp
    src/fast_gicp/gicp/fast_gicp.cpp
    src/fast_gicp/gicp/fast_vgicp.cpp
    )
target_link_libraries(fast_gicp
    ${PCL_LIBRARIES}
    )
if (OPENMP_FOUND)
  if (TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(ndt_omp OpenMP::OpenMP_CXX)
    target_link_libraries(fast_gicp OpenMP::OpenMP_CXX)
  endif ()
endif ()
target_include_directories(fast_gicp PUBLIC
    include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    )

add_library(registration SHARED
    src/cloud_registration.cpp
    )
target_link_libraries(registration
    ndt_omp
    fast_gicp
    )

# add_library(icp_registration SHARED
#     src/icp_registration.cpp
#     )
# target_link_libraries(icp_registration
#     ${PCL_LIBRARIES}
#     registration
#     )

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/icp_registration.cpp
)
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} registration)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN icp::IcpNode
  EXECUTABLE ${PROJECT_NAME}_node
)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

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

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
)
