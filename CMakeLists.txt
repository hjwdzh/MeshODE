cmake_minimum_required(VERSION 3.1)
project(ARAP)

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
find_package(CGAL)

include(${CGAL_USE_FILE})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/lib)
add_library(
    deform SHARED
    src/lib/callback.h
    src/lib/deformer.cc
    src/lib/deformer.h
    src/lib/delaunay.cc
    src/lib/delaunay.h
    src/lib/mesh.cc
    src/lib/mesh.h
    src/lib/meshcover.cc
    src/lib/meshcover.h
    src/lib/subdivision.cc
    src/lib/subdivision.h
    src/lib/uniformgrid.cc
    src/lib/uniformgrid.h
    src/lib/edgeloss.h
    src/lib/types.h
)

target_link_libraries(
    deform
    ${CERES_LIBRARIES}
)

add_executable(
    rigid_deform
    src/app/rigid_deform.cc
)

target_link_libraries(
    rigid_deform
    deform
    ${CERES_LIBRARIES}
)

add_executable(
    rigid_rot_deform
    src/app/rigid_rot_deform.cc
)

target_link_libraries(
    rigid_rot_deform
    deform
    ${CERES_LIBRARIES}
)

add_executable(
    cad_deform
    src/app/cad_deform.cc
)

target_link_libraries(
    cad_deform
    deform
    ${CERES_LIBRARIES}
    gmp
    ${CGAL_LIBRARIES}
)

add_executable(
    coverage_deform
    src/app/coverage_deform.cc
)

target_link_libraries(
    coverage_deform
    deform
)

add_executable(
    inverse_deform
    src/app/inverse_deform.cc
)

target_link_libraries(
    inverse_deform
    deform
)
