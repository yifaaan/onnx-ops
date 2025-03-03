cmake_minimum_required(VERSION 3.10)

find_program(CMAKE_MAKE_PROGRAM make REQUIRED)

set(CMAKE_TOOLCHAIN_FILE "~/vcpkg/scripts/buildsystems/vcpkg.cmake")


project(onnx-ops)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Catch2 CONFIG REQUIRED)

add_subdirectory(src)
add_subdirectory(test)

