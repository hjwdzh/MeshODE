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
#find_package(CGAL)

include_directories(/orion/u/jingweih/3rd/)
include_directories(/orion/u/jingweih/3rd/eigen3/)

include_directories(/Users/jingwei/Desktop/project/libigl/include)
include_directories(/orion/u/jingweih/3rd/cgal-install/include)

link_directories(/orion/u/jingweih/3rd/cgal-install/lib)
#include(${CGAL_USE_FILE})

add_library(
	deform SHARED
    src/deform.cc
    src/deform.h
    src/delaunay.cc
    src/delaunay.h
    src/mesh.cc
    src/mesh.h
    src/meshcover.cc
    src/meshcover.h
    src/subdivision.cc
    src/subdivision.h
    src/uniformgrid.cc
    src/uniformgrid.h
    src/edgeloss.h
    src/types.h
)

target_link_libraries(
	deform
	${CERES_LIBRARIES}
)

add_executable(
    rigid_deform
    src/rigid_deform.cc
)

target_link_libraries(
    rigid_deform
    deform
    ${CERES_LIBRARIES}
)

add_executable(
    rigid_rot_deform
    src/rigid_rot_deform.cc
)

target_link_libraries(
    rigid_rot_deform
    deform
    ${CERES_LIBRARIES}
)

add_executable(
    cad_deform
    src/cad_deform.cc
)

target_link_libraries(
    cad_deform
    deform
    ${CERES_LIBRARIES}
    gmp
    #${CGAL_LIBRARIES}
)

add_executable(
    reverse_deform
    src/reverse_deform.cc
)

target_link_libraries(
    reverse_deform
    deform
)
