cmake_minimum_required(VERSION 3.24)
project(YOLO)

set(CMAKE_CXX_STANDARD 11)
find_library( CAMERA_LIB NAMES MVSDK HINTS ./camera_lib/x86 REQUIRED)
find_package( OpenCV REQUIRED )

add_executable(YOLO src/main.cpp src/Detector.cpp src/Camera.cpp)
message(STATUS ${CAMERA_LIB})


target_include_directories(YOLO PUBLIC includes/auto_aim)
target_include_directories( YOLO PUBLIC ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(YOLO ${CAMERA_LIB} ${OpenCV_LIBS} rt pthread)

