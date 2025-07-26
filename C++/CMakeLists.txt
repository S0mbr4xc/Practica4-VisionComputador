cmake_minimum_required(VERSION 3.30)
project(C__)

set(OpenCV_DIR "C:/opencv")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)

add_executable(C__ main.cpp)

target_link_libraries(C__ ${OpenCV_LIBS})

