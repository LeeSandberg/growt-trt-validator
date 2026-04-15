cmake_minimum_required(VERSION 3.18)
project(tensorrt_growt_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(CURL REQUIRED)

set(TENSORRT_DIR "" CACHE PATH "Path to TensorRT installation")

add_library(growt_plugin SHARED
    src/growt_plugin.cpp
    src/growt_http_client.cpp
)

target_include_directories(growt_plugin PRIVATE
    ${TENSORRT_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
)

target_link_libraries(growt_plugin PRIVATE
    ${TENSORRT_DIR}/lib/libnvinfer.so
    ${CUDA_LIBRARIES}
    ${CURL_LIBRARIES}
)

set_target_properties(growt_plugin PROPERTIES
    OUTPUT_NAME "growt_plugin"
    PREFIX "lib"
)
