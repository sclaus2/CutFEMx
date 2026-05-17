cmake_minimum_required(VERSION 3.16)

# Set project name and version number
project(cutfemx VERSION 0.1.0 LANGUAGES CXX C)

option(CUTFEMX_MIGRATION_BUILD
       "Build only migration-ready components while porting to current FEniCSx"
       ON)
message(STATUS "CUTFEMX_MIGRATION_BUILD: ${CUTFEMX_MIGRATION_BUILD}")

if(CUTFEMX_MIGRATION_BUILD)
  set(_CUTFEMX_LEGACY_COMPONENT_DEFAULT OFF)
else()
  set(_CUTFEMX_LEGACY_COMPONENT_DEFAULT ON)
endif()
option(CUTFEMX_BUILD_MESH_CUTTER "Build the MeshCutter component" ON)
option(CUTFEMX_BUILD_CUT_FUNCTION "Build the cut-function interpolation component" ON)
option(CUTFEMX_BUILD_LEGACY_LEVEL_SET "Build legacy level-set helpers" ${_CUTFEMX_LEGACY_COMPONENT_DEFAULT})
option(CUTFEMX_BUILD_LEGACY_QUADRATURE "Build legacy quadrature helpers" ${_CUTFEMX_LEGACY_COMPONENT_DEFAULT})
option(CUTFEMX_BUILD_LEGACY_MESH "Build legacy mesh helpers" ${_CUTFEMX_LEGACY_COMPONENT_DEFAULT})
option(CUTFEMX_BUILD_LEGACY_FEM "Build legacy FEM helpers" ${_CUTFEMX_LEGACY_COMPONENT_DEFAULT})
message(STATUS "CUTFEMX_BUILD_MESH_CUTTER: ${CUTFEMX_BUILD_MESH_CUTTER}")
message(STATUS "CUTFEMX_BUILD_CUT_FUNCTION: ${CUTFEMX_BUILD_CUT_FUNCTION}")
message(STATUS "CUTFEMX_BUILD_LEGACY_LEVEL_SET: ${CUTFEMX_BUILD_LEGACY_LEVEL_SET}")
message(STATUS "CUTFEMX_BUILD_LEGACY_QUADRATURE: ${CUTFEMX_BUILD_LEGACY_QUADRATURE}")
message(STATUS "CUTFEMX_BUILD_LEGACY_MESH: ${CUTFEMX_BUILD_LEGACY_MESH}")
message(STATUS "CUTFEMX_BUILD_LEGACY_FEM: ${CUTFEMX_BUILD_LEGACY_FEM}")

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies
find_package(MPI REQUIRED)
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind 2.0.0 REQUIRED CONFIG)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set(CUTFEMX_LOCAL_CUTCELLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../CutCells/cpp")
if(EXISTS "${CUTFEMX_LOCAL_CUTCELLS_DIR}/CMakeLists.txt")
  add_subdirectory("${CUTFEMX_LOCAL_CUTCELLS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_deps/cutcells" EXCLUDE_FROM_ALL)
  if(NOT TARGET CUTCELLS::cutcells)
    add_library(CUTCELLS::cutcells ALIAS cutcells)
  endif()
else()
  find_package(CutCells REQUIRED)
endif()
find_package(DOLFINX REQUIRED)
find_package(spdlog REQUIRED)

# Add C++ library
add_subdirectory(cpp)

# Find DOLFINX Python Wrappers (required for array.h etc included by wrappers)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "
import sys
try:
    import importlib.util, os, mpi4py, petsc4py
    spec = importlib.util.find_spec('dolfinx')
    path = os.path.dirname(spec.origin) if spec and spec.origin else ''
    candidates = [
        os.path.join(path, 'wrappers'),
        os.path.join(path, 'wrappers', 'dolfinx_wrappers'),
        os.path.join(path, 'include'),
        os.path.join(path, 'include', 'dolfinx', 'wrappers'),
        os.path.join(path, 'include', 'dolfinx', 'wrappers', 'dolfinx_wrappers'),
    ]
    wrapper_dir = ''
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, 'array.h')):
            wrapper_dir = candidate
            break
    print(wrapper_dir, end=';')
    
    # MPI4PY
    print(mpi4py.get_include(), end=';')

    # PETSC4PY
    print(petsc4py.get_include(), end='')
except ImportError as e:
    print(f'IMPORT_ERROR: {e}', end='')
except Exception as e:
    print(f'ERROR: {e}', end='')
"
  OUTPUT_VARIABLE PYTHON_INCLUDES
  ERROR_VARIABLE PYTHON_STDERR
)

if(PYTHON_INCLUDES MATCHES "IMPORT_ERROR" OR PYTHON_INCLUDES MATCHES "ERROR")
    message(WARNING "Python module detection failed: ${PYTHON_INCLUDES}")
    message(STATUS "Stderr: ${PYTHON_STDERR}")
else()
    list(LENGTH PYTHON_INCLUDES LEN)
    if(LEN EQUAL 3)
        list(GET PYTHON_INCLUDES 0 DOLFINX_WRAPPERS_DIR)
        list(GET PYTHON_INCLUDES 1 MPI4PY_INCLUDE_DIR)
        list(GET PYTHON_INCLUDES 2 PETSC4PY_INCLUDE_DIR)
    else()
        message(WARNING "Could not parse python includes. Output: ${PYTHON_INCLUDES}")
    endif()
endif()

# Fallback: Try to use environment variables or verify found paths
if(NOT DOLFINX_WRAPPERS_DIR)
  if(DEFINED ENV{DOLFINX_DIR})
     set(DOLFINX_WRAPPERS_DIR "$ENV{DOLFINX_DIR}/lib/python3.13/site-packages/dolfinx/wrappers") 
     # This is a guess, probably wrong for conda
  endif()
  message(WARNING "Could not automatically locate DOLFINX wrappers containing array.h. Build WILL FAIL if headers are not found.")
else()
  message(STATUS "Found DOLFINX wrappers: ${DOLFINX_WRAPPERS_DIR}")
endif()

message(STATUS "Found mpi4py include: ${MPI4PY_INCLUDE_DIR}")
message(STATUS "Found petsc4py include: ${PETSC4PY_INCLUDE_DIR}")

# Create Python External Module
set(WRAPPER_SOURCES "python/cutfemx/wrappers/cutfemx.cpp")
set(CUTFEMX_WRAPPER_DEFINITIONS)
if(CUTFEMX_BUILD_MESH_CUTTER)
  list(APPEND WRAPPER_SOURCES "python/cutfemx/wrappers/mesh_cutter.cpp")
  list(APPEND CUTFEMX_WRAPPER_DEFINITIONS CUTFEMX_HAS_MESH_CUTTER)
endif()
if(CUTFEMX_BUILD_CUT_FUNCTION)
  list(APPEND WRAPPER_SOURCES "python/cutfemx/wrappers/cut_function.cpp")
  list(APPEND CUTFEMX_WRAPPER_DEFINITIONS CUTFEMX_HAS_FEM_INTERPOLATION)
endif()
if(CUTFEMX_BUILD_LEGACY_LEVEL_SET)
  list(APPEND WRAPPER_SOURCES "python/cutfemx/wrappers/level_set.cpp")
  list(APPEND CUTFEMX_WRAPPER_DEFINITIONS CUTFEMX_HAS_LEVEL_SET)
endif()
if(CUTFEMX_BUILD_LEGACY_QUADRATURE)
  list(APPEND WRAPPER_SOURCES "python/cutfemx/wrappers/quadrature.cpp")
  list(APPEND CUTFEMX_WRAPPER_DEFINITIONS CUTFEMX_HAS_QUADRATURE)
endif()
if(CUTFEMX_BUILD_LEGACY_MESH)
  list(APPEND WRAPPER_SOURCES "python/cutfemx/wrappers/mesh.cpp")
  list(APPEND CUTFEMX_WRAPPER_DEFINITIONS CUTFEMX_HAS_MESH)
endif()
if(CUTFEMX_BUILD_LEGACY_FEM)
  list(APPEND WRAPPER_SOURCES "python/cutfemx/wrappers/fem.cpp")
  list(APPEND CUTFEMX_WRAPPER_DEFINITIONS CUTFEMX_HAS_FEM)
  list(APPEND WRAPPER_SOURCES "python/cutfemx/wrappers/petsc.cpp")
endif()

nanobind_add_module(cutfemx_cpp ${WRAPPER_SOURCES} NOMINSIZE)
if(CUTFEMX_MIGRATION_BUILD)
  target_compile_definitions(cutfemx_cpp PRIVATE CUTFEMX_MIGRATION_BUILD)
endif()
target_compile_definitions(cutfemx_cpp PRIVATE ${CUTFEMX_WRAPPER_DEFINITIONS})

target_include_directories(cutfemx_cpp PRIVATE 
    ${DOLFINX_WRAPPERS_DIR}
    ${MPI4PY_INCLUDE_DIR}
    ${PETSC4PY_INCLUDE_DIR}
)

target_link_libraries(cutfemx_cpp PRIVATE
    cutfemx
    MPI::MPI_CXX
    dolfinx
    CUTCELLS::cutcells
)

if(APPLE)
  set(_cutfemx_runtime_rpath "@loader_path")
else()
  set(_cutfemx_runtime_rpath "$ORIGIN")
endif()
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND _cutfemx_runtime_rpath "$ENV{CONDA_PREFIX}/lib")
endif()

set_target_properties(cutfemx_cpp PROPERTIES
  INSTALL_RPATH "${_cutfemx_runtime_rpath}"
  INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(cutfemx PROPERTIES
  INSTALL_RPATH "${_cutfemx_runtime_rpath}"
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Install the extension into the python package
install(TARGETS cutfemx_cpp DESTINATION cutfemx)
install(TARGETS cutfemx
  RUNTIME DESTINATION cutfemx
  LIBRARY DESTINATION cutfemx
  ARCHIVE DESTINATION cutfemx)
if(TARGET cutcells)
  install(TARGETS cutcells
    RUNTIME DESTINATION cutfemx
    LIBRARY DESTINATION cutfemx
    ARCHIVE DESTINATION cutfemx)
endif()
