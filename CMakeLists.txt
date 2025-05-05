# Minimum version should be low enough that everyone is able to use it
cmake_minimum_required(VERSION 3.14..3.21)

project(simplellama CXX)

# Needed for fetching model & GGML
include(FetchContent)

# Some of the more popular gpu options (VULKAN is great for NON-nvidia & CUDA is almost must-have for nvidia)
option(GGML_VULKAN "Run with vulkan backend" ON)
option(GGML_CUDA "Run with CUDA backend" OFF)

# Perf options enables the profiling (tok/s) of the runs
option(GGML_PERF ON)

# This option (build_examples) gets disables automatically when imported
option(SIMPLELLAMA_BUILD_EXAMPLE "Build example" ON)

# Optionally enable ROS2 package build (Requires ROS2 jazzy to be installed!)
option(SIMPLELLAMA_ROS2_PACKAGE_BUILD "Build a ROS2 package" OFF)

find_package(SDL2 REQUIRED)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(CMAKE_CXX_STANDARD 17)
else()
  set(SIMPLELLAMA_BUILD_EXAMPLE OFF)
endif()

# Get GGML if not present already
if(TARGET ggml)
else()
  FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggerganov/ggml.git
    GIT_TAG a2af72be7baf5b1f4a33d34e77e509e5e85b7cd7
  )
  FetchContent_MakeAvailable(ggml)
endif()

# Add our fantastic llama.cpp library
add_library(llama ${CMAKE_CURRENT_LIST_DIR}/llama/llama.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-grammar.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-sampling.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-vocab.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/unicode.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/unicode-data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-adapter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-arch.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-batch.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-chat.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-context.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-cparams.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-hparams.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-impl.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-kv-cache.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-mmap.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-model-loader.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-model.cpp
  ${CMAKE_CURRENT_LIST_DIR}/llama/llama-quant.cpp)
target_include_directories(llama PUBLIC ${CMAKE_CURRENT_LIST_DIR}/llama)
# It needs GGML for inference backends
target_link_libraries(llama ggml)

# Add simplellama wrapper
add_library(simplellama ${CMAKE_CURRENT_LIST_DIR}/src/simplellama.cpp)
target_include_directories(simplellama PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(simplellama llama ggml ${SDL2_LIBRARIES})

if(SIMPLELLAMA_ROS2_PACKAGE_BUILD)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(std_srvs REQUIRED)
  add_executable(simplellama_node ${CMAKE_CURRENT_LIST_DIR}/bindings/ros2/simplellama_node.cpp)
  ament_target_dependencies(simplellama_node
    rclcpp
    std_msgs
    std_srvs
  )
  target_link_libraries(simplellama_node simplellama)
  install(TARGETS
    simplellama_node
    DESTINATION lib/${PROJECT_NAME})
endif()

if(SIMPLELLAMA_BUILD_EXAMPLE)
  if(SIMPLELLAMA_ROS2_PACKAGE_BUILD)
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(std_srvs REQUIRED)
    find_package(std_msgs REQUIRED)
    add_executable(simplellama_demo ${CMAKE_CURRENT_LIST_DIR}/example/ros2_demo/ros2_demo.cpp)
    ament_target_dependencies(simplellama_demo
      rclcpp
      std_msgs
      std_srvs
    )
    install(TARGETS
      simplellama_demo
      DESTINATION lib/${PROJECT_NAME})
    install(TARGETS
      simplellama llama
      DESTINATION lib/
    )
    install(TARGETS
      simplellama llama
      DESTINATION lib64/
    )
  else()
    add_executable(simple_text_demo ${CMAKE_CURRENT_LIST_DIR}/example/simple_text_demo/simple_text_demo.cpp)
    target_link_libraries(simple_text_demo simplellama)
  endif()
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
  endif()

endif()
if(SIMPLELLAMA_ROS2_PACKAGE_BUILD)
  ament_package()
endif()
