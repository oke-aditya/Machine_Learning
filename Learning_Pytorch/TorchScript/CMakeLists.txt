cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)
set(CMAKE_PREFIX_PATH "C:/Program Files (x86)/libtorch/libtorch_release/share/cmake/Torch")
find_package(Torch REQUIRED)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)