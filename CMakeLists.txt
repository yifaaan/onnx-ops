cmake_minimum_required(VERSION 3.10)





project(onnx-ops)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


add_subdirectory(src)
add_subdirectory(test)

