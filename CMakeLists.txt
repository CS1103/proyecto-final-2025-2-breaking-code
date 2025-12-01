cmake_minimum_required(VERSION 3.31)
project(epic)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)
include_directories(layers)
include_directories(activations)
include_directories(tensor)

file(GLOB SRC_FILES src/*.cpp)

add_executable(epic test.cpp)
