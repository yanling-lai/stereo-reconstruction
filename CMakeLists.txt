cmake_minimum_required(VERSION 3.6)

project(stereo_reconstruction)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

set(HEADER_FILES stereo_matching.h SingleFrame.h StereoImagePairContainer.h)
set(SOURCE_FILES stereo_matching.cpp)
add_executable(stereo_reconstruction main.cpp ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(stereo_reconstruction ${OpenCV_LIBS})