cmake_minimum_required(VERSION 3.1)
project(ARAP)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG}")

set(CERES_INCLUDE_PATH "3rd_party/ceres-solver/include")
set(IGL_INCLUDE_PATH "3rd_party/libigl/include")
set(EIGEN_INCLUDE_PATH "3rd_party/eigen")

IF (NOT CGAL_INCLUDE_PATH)
  find_package(CGAL)
  include(${CGAL_USE_FILE})
ELSE()
  include_directories(${CGAL_INCLUDE_PATH})
endif()

include_directories(${CERES_INCLUDE_PATH})
include_directories(${IGL_INCLUDE_PATH})
include_directories(${EIGEN_INCLUDE_PATH})

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

execute_process (
    COMMAND python -c "import sys; print('.'.join(sys.version.split(' (')[0].split('.')[:2]))"
    OUTPUT_VARIABLE PY_VERSION
)
execute_process (
    COMMAND python -c "import subprocess; print('/'.join(subprocess.check_output(['which', 'python']).decode('utf-8').split('/')[:-2]))"
    OUTPUT_VARIABLE PY_BASEDIR
)
string(STRIP ${PY_VERSION} PY_VERSION)
string(STRIP ${PY_BASEDIR} PY_BASEDIR)

if(NOT TORCH_PATH)
    set(TORCH_PATH "${PY_BASEDIR}/lib/python${PY_VERSION}/site-packages/torch")
    message("TORCH_PATH is set default to ${TORCH_PATH}")
    message('Consider to use -DTORCH_PATH your_path to specialize your path instead.')
endif()

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

add_subdirectory("3rd_party/pybind11")

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

