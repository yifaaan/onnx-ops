cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_COMPILER "clang++")
project(scatter_elements)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(Catch2 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)


add_subdirectory(src)
add_subdirectory(test)

