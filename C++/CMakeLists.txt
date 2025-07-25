cmake_minimum_required(VERSION 3.30)
project(Practica4_Part1C)

set(OpenCV_DIR "C:/opencv")

set(OpenCV_DIR "C:/opencv_cuda")
find_package(OpenCV REQUIRED COMPONENTS core imgproc cudaimgproc cudafilters)

# Ahora requerimos explícitamente los módulos CUDA
find_package(OpenCV REQUIRED COMPONENTS core imgproc cudaimgproc cudafilters)

add_executable(practica1C main.cpp)
target_link_libraries(practica1C PRIVATE ${OpenCV_LIBS})
