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

if(NOT IGL_INCLUDE_PATH)
    message('IGL_INCLUDE_PATH is set default as /usr/include/igl')
    message('Consider to use -DIGL_INCLUDE_PATH your_path to specialize your path instead.')
    set(IGL_INCLUDE_PATH "/usr/include/igl")
endif()

include_directories(IGL_INCLUDE_PATH)
add_library(
    deform_mesh
    src/lib/delaunay.cc
    src/lib/delaunay.h
    src/lib/mesh.cc
    src/lib/mesh.h
    src/lib/subdivision.cc
    src/lib/subdivision.h
    src/lib/uniformgrid.cc
    src/lib/uniformgrid.h
    src/lib/types.h
)

target_link_libraries(
    deform_mesh
    ${CGAL_LIBRARIES}
)

add_library(
    deform_optim
    src/lib/callback.h
    src/lib/linear.cc
    src/lib/linear.h
    src/lib/deformer.cc
    src/lib/deformer.h
    src/lib/edgeloss.h
)

target_link_libraries(
    deform_optim
    ${CERES_LIBRARIES}
)

add_executable(
    rigid_deform
    src/app/rigid_deform.cc
)

target_link_libraries(
    rigid_deform
    deform_mesh
    deform_optim
)

add_executable(
    rigid_rot_deform
    src/app/rigid_rot_deform.cc
)

target_link_libraries(
    rigid_rot_deform
    deform_mesh
    deform_optim
)

add_executable(
    cad_deform
    src/app/cad_deform.cc
)

target_link_libraries(
    cad_deform
    deform_mesh
    deform_optim
)

add_executable(
    coverage_deform
    src/app/coverage_deform.cc
)

target_link_libraries(
    coverage_deform
    deform_mesh
    deform_optim
)

add_executable(
    inverse_deform
    src/app/inverse_deform.cc
)

target_link_libraries(
    inverse_deform
    deform_mesh
    deform_optim
)


if(NOT TORCH_PATH)
    message('TORCH_PATH is set default as /usr/local/lib/python3.7/site-packages/torch.')
    message('Consider to use -DTORCH_PATH your_path to specialize your path instead.')
    set(TORCH_PATH "/usr/local/lib/python3.7/site-packages/torch")
endif()


find_package(pybind11)
IF (pybind11_FOUND)

include_directories(${TORCH_PATH}/include)
link_directories(${TORCH_PATH}/lib)
include_directories(${TORCH_PATH}/include/torch/csrc/api/include)


add_library(
    pyDeform SHARED
    src/interface/deform_params.h
    src/interface/deform_params.cc
    src/interface/cad_layer.h
    src/interface/cad_layer.cc
    src/interface/distance_layer.h
    src/interface/distance_layer.cc
    src/interface/graph_layer.h
    src/interface/graph_layer.cc
    src/interface/linear_layer.h
    src/interface/linear_layer.cc
    src/interface/mesh_tensor.h
    src/interface/mesh_tensor.cc
    src/interface/normalize.h
    src/interface/normalize.cc
    src/interface/rigid_layer.h
    src/interface/rigid_layer.cc
    src/interface/pydeform.cc
)

target_link_libraries(
    pyDeform
    deform_mesh
    deform_optim
    pybind11::module
    torch
)

set_target_properties(pyDeform PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")

ENDIF()
