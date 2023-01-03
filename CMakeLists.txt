cmake_minimum_required(VERSION 3.17)
project(test_openvino)

set(CMAKE_CXX_STANDARD 17)

set(OpenVINO_DIR /Users/yang/opt/openvino_android/openvino/build_macos)
find_package(OpenVINO REQUIRED)

add_executable(test_openvino ./ultralytics_yolov3/ultralytics_yolov3.cpp)
target_link_libraries(test_openvino PRIVATE openvino::runtime)