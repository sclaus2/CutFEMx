cmake_minimum_required(VERSION 3.21)

project(cutfemx VERSION 0.2.0 LANGUAGES CXX C)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build cutfemx with shared libraries." ON)
option(CUTFEMX_USE_MULTIPRECISION_PSM
       "Use geogram.psm.MultiPrecision for robust predicates." ON)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Using Python executable: ${Python_EXECUTABLE}")

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _cutfemx_nanobind_cmake_dir
  RESULT_VARIABLE _cutfemx_nanobind_result
  ERROR_VARIABLE _cutfemx_nanobind_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _cutfemx_nanobind_result EQUAL 0)
  message(FATAL_ERROR
    "Could not query nanobind CMake directory with ${Python_EXECUTABLE}.\n"
    "${_cutfemx_nanobind_error}")
endif()
list(APPEND CMAKE_PREFIX_PATH "${_cutfemx_nanobind_cmake_dir}")

find_package(nanobind 2.0.0 REQUIRED CONFIG)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

if(NOT TARGET CUTCELLS::cutcells)
  find_package(CutCells 0.4.0 REQUIRED CONFIG)
endif()
find_package(DOLFINX REQUIRED CONFIG)
find_package(spdlog REQUIRED CONFIG)

add_subdirectory(cpp)

function(_cutfemx_python_query OUT_VAR LABEL CODE REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "${CODE}"
    OUTPUT_VARIABLE _cutfemx_query_output
    RESULT_VARIABLE _cutfemx_query_result
    ERROR_VARIABLE _cutfemx_query_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(_cutfemx_query_result EQUAL 0 AND _cutfemx_query_output)
    set(${OUT_VAR} "${_cutfemx_query_output}" PARENT_SCOPE)
    message(STATUS "Found ${LABEL}: ${_cutfemx_query_output}")
  elseif(REQUIRED)
    message(FATAL_ERROR
      "Could not find ${LABEL} using ${Python_EXECUTABLE}.\n"
      "${_cutfemx_query_error}")
  else()
    message(STATUS "${LABEL} not found: ${_cutfemx_query_error}")
  endif()
endfunction()

if(DEFINED ENV{DOLFINX_PY_WRAPPERS_DIR}
   AND EXISTS "$ENV{DOLFINX_PY_WRAPPERS_DIR}/array.h")
  set(DOLFINX_WRAPPERS_DIR "$ENV{DOLFINX_PY_WRAPPERS_DIR}")
  message(STATUS "Found DOLFINx Python wrappers: ${DOLFINX_WRAPPERS_DIR}")
else()
  _cutfemx_python_query(
    DOLFINX_WRAPPERS_DIR
    "DOLFINx Python wrappers"
    [=[
import importlib.util
import os
spec = importlib.util.find_spec("dolfinx")
if spec is None or spec.origin is None:
    raise SystemExit("dolfinx Python package not found")
path = os.path.dirname(spec.origin)
candidates = [
    os.path.join(path, "wrappers"),
    os.path.join(path, "wrappers", "dolfinx_wrappers"),
    os.path.join(path, "include"),
    os.path.join(path, "include", "dolfinx", "wrappers"),
    os.path.join(path, "include", "dolfinx", "wrappers", "dolfinx_wrappers"),
]
for candidate in candidates:
    if os.path.exists(os.path.join(candidate, "array.h")):
        print(candidate)
        break
else:
    raise SystemExit("array.h not found in dolfinx Python package")
]=]
    TRUE)
endif()

_cutfemx_python_query(
  MPI4PY_INCLUDE_DIR
  "mpi4py include directory"
  "import mpi4py; print(mpi4py.get_include())"
  TRUE)
_cutfemx_python_query(
  PETSC4PY_INCLUDE_DIR
  "petsc4py include directory"
  "import petsc4py; print(petsc4py.get_include())"
  FALSE)
_cutfemx_python_query(
  CUTFEMX_PYTHON_PREFIX
  "Python prefix"
  "import sys; print(sys.prefix)"
  TRUE)

set(WRAPPER_SOURCES
  python/cutfemx/wrappers/cutfemx.cpp
  python/cutfemx/wrappers/cut.cpp
  python/cutfemx/wrappers/distance.cpp
  python/cutfemx/wrappers/level_set.cpp
  python/cutfemx/wrappers/cut_function.cpp
  python/cutfemx/wrappers/fem.cpp
  python/cutfemx/wrappers/extensions.cpp)
set(CUTFEMX_WRAPPER_DEFINITIONS
  CUTFEMX_HAS_CUT_API
  CUTFEMX_HAS_DISTANCE
  CUTFEMX_HAS_LEVEL_SET
  CUTFEMX_HAS_FEM_INTERPOLATION
  CUTFEMX_HAS_RUNTIME_FEM
  CUTFEMX_HAS_EXTENSIONS)
if(PETSC4PY_INCLUDE_DIR)
  list(APPEND WRAPPER_SOURCES python/cutfemx/wrappers/petsc.cpp)
  list(APPEND CUTFEMX_WRAPPER_DEFINITIONS
    CUTFEMX_HAS_RUNTIME_PETSC
    HAS_PETSC4PY)
endif()
nanobind_add_module(cutfemx_cpp ${WRAPPER_SOURCES} NOMINSIZE)
target_compile_definitions(cutfemx_cpp PRIVATE ${CUTFEMX_WRAPPER_DEFINITIONS})
target_include_directories(cutfemx_cpp BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)
set(CUTFEMX_PYTHON_WRAPPER_INCLUDE_OPTIONS)
if(DOLFINX_WRAPPERS_DIR)
  list(APPEND CUTFEMX_PYTHON_WRAPPER_INCLUDE_OPTIONS
       "-idirafter" "${DOLFINX_WRAPPERS_DIR}")
endif()
foreach(_cutfemx_python_include ${MPI4PY_INCLUDE_DIR} ${PETSC4PY_INCLUDE_DIR})
  if(_cutfemx_python_include)
    list(APPEND CUTFEMX_PYTHON_WRAPPER_INCLUDE_OPTIONS
         "-I${_cutfemx_python_include}")
  endif()
endforeach()
if(CUTFEMX_PYTHON_WRAPPER_INCLUDE_OPTIONS)
  # DOLFINx's nanobind wrapper directory contains a helper header named
  # petsc.h. Search that directory only after dependency include paths, so
  # petsc4py sees PETSc's real petsc.h first.
  target_compile_options(cutfemx_cpp PRIVATE
      ${CUTFEMX_PYTHON_WRAPPER_INCLUDE_OPTIONS})
endif()

target_link_libraries(cutfemx_cpp PRIVATE
    cutfemx
    MPI::MPI_CXX
    dolfinx
    CUTCELLS::cutcells
)

if(APPLE)
  set(_cutfemx_install_rpath "@loader_path")
  list(APPEND _cutfemx_install_rpath
    "@loader_path/../basix/lib"
    "@loader_path/../dolfinx/lib"
    "@loader_path/../fenics_basix.libs"
    "@loader_path/../fenics_dolfinx.libs")
else()
  set(_cutfemx_install_rpath "$ORIGIN")
  list(APPEND _cutfemx_install_rpath
    "$ORIGIN/../basix/lib"
    "$ORIGIN/../dolfinx/lib"
    "$ORIGIN/../fenics_basix.libs"
    "$ORIGIN/../fenics_dolfinx.libs")
endif()
set(_cutfemx_build_rpath ${_cutfemx_install_rpath})
if(CUTFEMX_PYTHON_PREFIX)
  list(APPEND _cutfemx_build_rpath "${CUTFEMX_PYTHON_PREFIX}/lib")
endif()
list(REMOVE_DUPLICATES _cutfemx_build_rpath)
list(REMOVE_DUPLICATES _cutfemx_install_rpath)

set_target_properties(cutfemx_cpp PROPERTIES
  BUILD_RPATH "${_cutfemx_build_rpath}"
  INSTALL_RPATH "${_cutfemx_install_rpath}"
  INSTALL_RPATH_USE_LINK_PATH FALSE)
set_target_properties(cutfemx PROPERTIES
  BUILD_RPATH "${_cutfemx_build_rpath}"
  INSTALL_RPATH "${_cutfemx_install_rpath}"
  INSTALL_RPATH_USE_LINK_PATH FALSE)
if(APPLE)
  set_target_properties(cutfemx PROPERTIES
    INSTALL_NAME_DIR "@loader_path")
endif()

# Install the extension into the python package
install(TARGETS cutfemx_cpp DESTINATION cutfemx)
install(TARGETS cutfemx
  RUNTIME DESTINATION cutfemx
  LIBRARY DESTINATION cutfemx
  ARCHIVE DESTINATION cutfemx)
install(DIRECTORY cpp/dolfinx_custom_data/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_custom_data
  COMPONENT Development
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "README.md")
