cmake_minimum_required(VERSION 3.1)
project(ARAP)
#include_directories(/orion/u/jingweih/arap/ceres-build/ceres-solver-1.9.0/install/include)
#link_directories(/orion/u/jingweih/arap/ceres-build/ceres-solver-1.9.0/install/lib)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG}")

find_package(ceres REQUIRED)

include_directories(/orion/u/jingweih/3rd/)
include_directories(/orion/u/jingweih/3rd/eigen3/)

include_directories(/Users/jingwei/Desktop/project/libigl/include)

set(
    deform_SRC
    src/deform.cpp
)

add_executable(
    deform
    src/main.cc
    src/deform.cc
    src/deform.h
    src/mesh.cc
    src/mesh.h
    src/uniformgrid.cc
    src/uniformgrid.h
    src/edgeloss.h
    src/types.h
)

target_link_libraries(
    deform
    ${CERES_LIBRARIES}
)
