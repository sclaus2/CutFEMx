cmake_minimum_required(VERSION 3.21)

project(cutfemx VERSION 0.2.0 LANGUAGES CXX C)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build cutfemx with shared libraries." ON)
option(CUTFEMX_USE_LOCAL_CUTCELLS
       "Build the sibling CutCells C++ source tree when available." ON)
option(CUTFEMX_ALLOW_SYSTEM_CUTCELLS
       "Allow CutFEMx to use a CutCells package from CMake system prefixes." OFF)

set(_cutfemx_conda_prefix "$ENV{CONDA_PREFIX}")
if(_cutfemx_conda_prefix)
  if(NOT DEFINED Python_ROOT_DIR)
    set(Python_ROOT_DIR "${_cutfemx_conda_prefix}")
  endif()
  if(NOT DEFINED Python_FIND_VIRTUALENV)
    set(Python_FIND_VIRTUALENV ONLY)
  endif()
  if(NOT DEFINED Python_FIND_STRATEGY)
    set(Python_FIND_STRATEGY LOCATION)
  endif()
  if(NOT DEFINED Python_EXECUTABLE)
    if(WIN32)
      set(_cutfemx_conda_python "${_cutfemx_conda_prefix}/python.exe")
    else()
      set(_cutfemx_conda_python "${_cutfemx_conda_prefix}/bin/python")
    endif()
    if(EXISTS "${_cutfemx_conda_python}")
      set(Python_EXECUTABLE "${_cutfemx_conda_python}")
    endif()
  endif()

  list(PREPEND CMAKE_PREFIX_PATH
    "${_cutfemx_conda_prefix}"
    "${_cutfemx_conda_prefix}/lib/cmake"
    "${_cutfemx_conda_prefix}/Library"
    "${_cutfemx_conda_prefix}/Library/lib/cmake")
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Using Python executable: ${Python_EXECUTABLE}")
if(_cutfemx_conda_prefix)
  message(STATUS "Using active conda prefix: ${_cutfemx_conda_prefix}")
endif()

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

function(_cutfemx_path_is_system_prefix OUT_VAR CANDIDATE_PATH)
  set(${OUT_VAR} OFF PARENT_SCOPE)
  if(NOT CANDIDATE_PATH
     OR CANDIDATE_PATH MATCHES "^\\$<"
     OR CANDIDATE_PATH MATCHES "-NOTFOUND$")
    return()
  endif()

  get_filename_component(_cutfemx_candidate_abs "${CANDIDATE_PATH}" ABSOLUTE)
  foreach(_cutfemx_system_prefix IN LISTS CMAKE_SYSTEM_PREFIX_PATH)
    if(NOT _cutfemx_system_prefix)
      continue()
    endif()
    get_filename_component(_cutfemx_system_prefix_abs
                           "${_cutfemx_system_prefix}" ABSOLUTE)

    if(_cutfemx_candidate_abs STREQUAL _cutfemx_system_prefix_abs)
      set(${OUT_VAR} ON PARENT_SCOPE)
      return()
    endif()

    string(LENGTH "${_cutfemx_system_prefix_abs}" _cutfemx_prefix_len)
    string(SUBSTRING "${_cutfemx_candidate_abs}" 0
                     "${_cutfemx_prefix_len}" _cutfemx_candidate_prefix)
    if(_cutfemx_candidate_prefix STREQUAL _cutfemx_system_prefix_abs)
      string(SUBSTRING "${_cutfemx_candidate_abs}" "${_cutfemx_prefix_len}"
                       1 _cutfemx_candidate_separator)
      if(_cutfemx_candidate_separator STREQUAL "/"
         OR _cutfemx_candidate_separator STREQUAL "\\")
        set(${OUT_VAR} ON PARENT_SCOPE)
        return()
      endif()
    endif()
  endforeach()
endfunction()

function(_cutfemx_reject_system_cutcells)
  if(CUTFEMX_ALLOW_SYSTEM_CUTCELLS)
    return()
  endif()

  set(_cutfemx_cutcells_paths)
  if(DEFINED CutCells_DIR AND NOT TARGET cutcells)
    list(APPEND _cutfemx_cutcells_paths "${CutCells_DIR}")
  endif()
  if(TARGET CUTCELLS::cutcells)
    get_target_property(_cutfemx_cutcells_includes
                        CUTCELLS::cutcells INTERFACE_INCLUDE_DIRECTORIES)
    if(_cutfemx_cutcells_includes)
      list(APPEND _cutfemx_cutcells_paths ${_cutfemx_cutcells_includes})
    endif()
    foreach(_cutfemx_prop
            IMPORTED_LOCATION
            IMPORTED_LOCATION_NOCONFIG
            IMPORTED_LOCATION_RELEASE
            IMPORTED_LOCATION_DEBUG
            IMPORTED_LOCATION_RELWITHDEBINFO
            IMPORTED_LOCATION_MINSIZEREL)
      get_target_property(_cutfemx_cutcells_location
                          CUTCELLS::cutcells ${_cutfemx_prop})
      if(_cutfemx_cutcells_location)
        list(APPEND _cutfemx_cutcells_paths "${_cutfemx_cutcells_location}")
      endif()
    endforeach()
  endif()

  foreach(_cutfemx_cutcells_path IN LISTS _cutfemx_cutcells_paths)
    if(NOT _cutfemx_cutcells_path
       OR _cutfemx_cutcells_path MATCHES "^\\$<"
       OR _cutfemx_cutcells_path MATCHES "-NOTFOUND$")
      continue()
    endif()
    _cutfemx_path_is_system_prefix(_cutfemx_is_system_cutcells_path
                                   "${_cutfemx_cutcells_path}")
    if(_cutfemx_is_system_cutcells_path)
      message(FATAL_ERROR
        "Refusing to use CutCells from '${_cutfemx_cutcells_path}'. "
        "Use the sibling CutCells source tree, set CutCells_DIR to the "
        "active conda environment, or configure with "
        "-DCUTFEMX_ALLOW_SYSTEM_CUTCELLS=ON if this system install is intentional.")
    endif()
  endforeach()
endfunction()

function(_cutfemx_find_packaged_cutcells)
  if(DEFINED CutCells_DIR AND NOT CUTFEMX_ALLOW_SYSTEM_CUTCELLS)
    _cutfemx_path_is_system_prefix(_cutfemx_is_system_cutcells_dir
                                   "${CutCells_DIR}")
    if(_cutfemx_is_system_cutcells_dir)
      message(FATAL_ERROR
        "Refusing to use CutCells_DIR='${CutCells_DIR}'. "
        "Use the sibling CutCells source tree, set CutCells_DIR to the "
        "active conda environment, or configure with "
        "-DCUTFEMX_ALLOW_SYSTEM_CUTCELLS=ON if this system install is intentional.")
    endif()
  endif()

  if(CUTFEMX_ALLOW_SYSTEM_CUTCELLS)
    find_package(CutCells 0.4.0 REQUIRED CONFIG)
    return()
  endif()

  set(_cutfemx_cutcells_hints)
  if(DEFINED CutCells_DIR)
    list(APPEND _cutfemx_cutcells_hints "${CutCells_DIR}")
  endif()
  if(_cutfemx_conda_prefix)
    list(APPEND _cutfemx_cutcells_hints
      "${_cutfemx_conda_prefix}"
      "${_cutfemx_conda_prefix}/lib/cmake/cutcells")
  endif()

  find_package(CutCells 0.4.0 REQUIRED CONFIG
    PATHS ${_cutfemx_cutcells_hints}
    NO_DEFAULT_PATH)
endfunction()

set(CUTFEMX_LOCAL_CUTCELLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../CutCells/cpp")
if(CUTFEMX_USE_LOCAL_CUTCELLS
   AND EXISTS "${CUTFEMX_LOCAL_CUTCELLS_DIR}/CMakeLists.txt"
   AND NOT TARGET CUTCELLS::cutcells
   AND NOT TARGET cutcells)
  message(STATUS "Using sibling CutCells source: ${CUTFEMX_LOCAL_CUTCELLS_DIR}")
  add_subdirectory("${CUTFEMX_LOCAL_CUTCELLS_DIR}"
                   "${CMAKE_CURRENT_BINARY_DIR}/_deps/cutcells"
                   EXCLUDE_FROM_ALL)
endif()
if(TARGET cutcells)
  if(NOT TARGET CUTCELLS::cutcells)
    add_library(CUTCELLS::cutcells ALIAS cutcells)
  endif()
  if(APPLE)
    set_target_properties(cutcells PROPERTIES
      INSTALL_NAME_DIR "@loader_path")
  endif()
elseif(NOT TARGET CUTCELLS::cutcells)
  _cutfemx_find_packaged_cutcells()
endif()
_cutfemx_reject_system_cutcells()
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
  set(_cutfemx_runtime_rpath "@loader_path")
else()
  set(_cutfemx_runtime_rpath "$ORIGIN")
endif()
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND _cutfemx_runtime_rpath "$ENV{CONDA_PREFIX}/lib")
endif()

set_target_properties(cutfemx_cpp PROPERTIES
  BUILD_RPATH "${_cutfemx_runtime_rpath}"
  INSTALL_RPATH "${_cutfemx_runtime_rpath}"
  INSTALL_RPATH_USE_LINK_PATH FALSE)
set_target_properties(cutfemx PROPERTIES
  BUILD_RPATH "${_cutfemx_runtime_rpath}"
  INSTALL_RPATH "${_cutfemx_runtime_rpath}"
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
if(TARGET cutcells)
  install(TARGETS cutcells
    RUNTIME DESTINATION cutfemx
    LIBRARY DESTINATION cutfemx
    ARCHIVE DESTINATION cutfemx)
endif()
