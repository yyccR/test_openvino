cmake_minimum_required(VERSION 3.17)
project(test_openvino)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED core imgproc highgui)
include_directories(${OpenCV_INCLUDE_DIRS})

#set(OpenVINO_DIR /Users/yang/opt/openvino_android/openvino/build_macos)
#find_package(OpenVINO REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/openvino_2022.3.0/include)

add_library(OpenVINO SHARED IMPORTED)
set_target_properties(OpenVINO PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/openvino_2022.3.0/lib/intel64/Release/libopenvino.2022.3.0.dylib)
add_library(libtbb SHARED IMPORTED)
set_target_properties(libtbb PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/openvino_2022.3.0/lib/intel64/Release/libtbb.dylib)


add_executable(test_openvino ./ultralytics_yolov3/ultralytics_yolov3.cpp)
target_link_libraries(test_openvino ${OpenCV_LIBS})
#target_link_libraries(test_openvino openvino::runtime)
target_link_libraries(test_openvino OpenVINO)
target_link_libraries(test_openvino libtbb)
