cmake_minimum_required(VERSION 3.10)

# 设置构建工具
find_program(CMAKE_MAKE_PROGRAM make REQUIRED)

# 设置编译器
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

project(onnx-ops)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 查找依赖包
find_package(Catch2 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_subdirectory(src)
add_subdirectory(test)

