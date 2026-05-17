cmake_minimum_required(VERSION 3.21)

project(cutfemx_nanobind)

message(STATUS "Pythonpath $ENV{PYTHONPATH}")

if(WIN32)
  # Windows requires all symbols to be manually exported. This flag exports all
  # symbols automatically, as in Unix.
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif(WIN32)

find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED
)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

find_package(DOLFINX REQUIRED CONFIG)
find_package(OpenMP REQUIRED COMPONENTS CXX)

if(DOLFINX_FOUND)
  message(STATUS "Found DOLFINx at ${DOLFINX_DIR}")
endif()

message(STATUS "Using Python executable: ${Python_EXECUTABLE}")

#Check for DOLFINx python wrappers
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "
import importlib.util, os
spec = importlib.util.find_spec('dolfinx')
path = os.path.dirname(spec.origin) if spec and spec.origin else ''
candidates = [
    os.path.join(path, 'wrappers'),
    os.path.join(path, 'wrappers', 'dolfinx_wrappers'),
    os.path.join(path, 'include'),
    os.path.join(path, 'include', 'dolfinx', 'wrappers'),
    os.path.join(path, 'include', 'dolfinx', 'wrappers', 'dolfinx_wrappers'),
]
for candidate in candidates:
    if os.path.exists(os.path.join(candidate, 'array.h')):
        print(candidate)
        break
"
  OUTPUT_VARIABLE DOLFINX_PY_WRAPPERS_DIR
  RESULT_VARIABLE DOLFINX_PY_WRAPPERS_COMMAND_RESULT
  ERROR_VARIABLE DOLFINX_PY_WRAPPERS_COMMAND_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT DOLFINX_PY_WRAPPERS_COMMAND_RESULT)
  message(STATUS "Found DOLFINx python wrappers at ${DOLFINX_PY_WRAPPERS_DIR}")
elseif(EXISTS $ENV{DOLFINX_PY_WRAPPERS_DIR})
  set(DOLFINX_PY_WRAPPERS_DIR $ENV{DOLFINX_PY_WRAPPERS_DIR})
  message(STATUS "Found DOLFINx python wrappers at ${DOLFINX_PY_WRAPPERS_DIR}")
else()
  message(STATUS "Failed to run dolfinx.get_include(). Error: ${DOLFINX_PY_WRAPPERS_COMMAND_ERROR}")
  message(FATAL_ERROR "DOLFINx python wrappers could not be found. Please try specifying DOLFINX_PY_WRAPPERS_DIR env variable via dolfinx.get_include()")
endif()

if(NOT TARGET CUTCELLS::cutcells)
  set(CUTFEMX_LOCAL_CUTCELLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../CutCells/cpp")
  if(EXISTS "${CUTFEMX_LOCAL_CUTCELLS_DIR}/CMakeLists.txt")
    add_subdirectory("${CUTFEMX_LOCAL_CUTCELLS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_deps/cutcells" EXCLUDE_FROM_ALL)
    if(NOT TARGET CUTCELLS::cutcells)
      add_library(CUTCELLS::cutcells ALIAS cutcells)
    endif()
  else()
    find_package(CutCells REQUIRED CONFIG)
  endif()
endif()
if(CutCells_FOUND)
  message(STATUS "Found CutCells at ${CutCells_DIR}")
endif()


# find_package(CUTFEMX REQUIRED CONFIG) replaced by add_subdirectory(cpp) in root


# Create the binding library nanobind handles its own calls to
# target_link_libraries
if(NOT DEFINED CUTFEMX_MIGRATION_BUILD)
  option(CUTFEMX_MIGRATION_BUILD
         "Build only migration-ready components while porting to current FEniCSx"
         ON)
endif()
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

set(CUTFEMX_WRAPPER_SOURCES cutfemx/wrappers/cutfemx.cpp)
set(CUTFEMX_WRAPPER_DEFINITIONS)
if(CUTFEMX_BUILD_MESH_CUTTER)
  list(APPEND CUTFEMX_WRAPPER_SOURCES cutfemx/wrappers/mesh_cutter.cpp)
  list(APPEND CUTFEMX_WRAPPER_DEFINITIONS CUTFEMX_HAS_MESH_CUTTER)
endif()
if(CUTFEMX_BUILD_CUT_FUNCTION)
  list(APPEND CUTFEMX_WRAPPER_SOURCES cutfemx/wrappers/cut_function.cpp)
  list(APPEND CUTFEMX_WRAPPER_DEFINITIONS CUTFEMX_HAS_FEM_INTERPOLATION)
endif()
if(CUTFEMX_BUILD_LEGACY_LEVEL_SET)
  list(APPEND CUTFEMX_WRAPPER_SOURCES cutfemx/wrappers/level_set.cpp)
  list(APPEND CUTFEMX_WRAPPER_DEFINITIONS CUTFEMX_HAS_LEVEL_SET)
endif()
if(CUTFEMX_BUILD_LEGACY_QUADRATURE)
  list(APPEND CUTFEMX_WRAPPER_SOURCES cutfemx/wrappers/quadrature.cpp)
  list(APPEND CUTFEMX_WRAPPER_DEFINITIONS CUTFEMX_HAS_QUADRATURE)
endif()
if(CUTFEMX_BUILD_LEGACY_MESH)
  list(APPEND CUTFEMX_WRAPPER_SOURCES cutfemx/wrappers/mesh.cpp)
  list(APPEND CUTFEMX_WRAPPER_DEFINITIONS CUTFEMX_HAS_MESH)
endif()
if(CUTFEMX_BUILD_LEGACY_FEM)
  list(APPEND CUTFEMX_WRAPPER_SOURCES cutfemx/wrappers/fem.cpp cutfemx/wrappers/petsc.cpp)
  list(APPEND CUTFEMX_WRAPPER_DEFINITIONS CUTFEMX_HAS_FEM)
endif()

nanobind_add_module(cutfemx_cpp NOMINSIZE ${CUTFEMX_WRAPPER_SOURCES})
target_compile_definitions(cutfemx_cpp PRIVATE cxx_std_20)
if(CUTFEMX_MIGRATION_BUILD)
  target_compile_definitions(cutfemx_cpp PRIVATE CUTFEMX_MIGRATION_BUILD)
endif()
target_compile_definitions(cutfemx_cpp PRIVATE ${CUTFEMX_WRAPPER_DEFINITIONS})

# Check for petsc4py
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_COMMAND_RESULT
  ERROR_VARIABLE PETSC4PY_COMMAND_ERROR OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PETSC4PY_COMMAND_RESULT)
  message(STATUS "Found petsc4py include directory at ${PETSC4PY_INCLUDE_DIR}")
  target_include_directories(cutfemx_cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})
  target_compile_definitions(cutfemx_cpp PRIVATE HAS_PETSC4PY)
else()
  message(STATUS "petsc4py not found with error: ${PETSC4PY_COMMAND_ERROR}")
endif()

# Check for mpi4py
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import mpi4py; print(mpi4py.get_include())"
  OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
  RESULT_VARIABLE MPI4PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT MPI4PY_COMMAND_RESULT)
  message(STATUS "Found mpi4py include directory at ${MPI4PY_INCLUDE_DIR}")
  target_include_directories(cutfemx_cpp PRIVATE ${MPI4PY_INCLUDE_DIR})
else()
  message(FATAL_ERROR "mpi4py could not be found.")
endif()

# Define HAS_PETSC4PY for compatibility with DOLFINx python wrappers
target_compile_definitions(cutfemx_cpp PRIVATE HAS_PETSC4PY)

# Add DOLFINx libraries - order matters for static linking
# cutfemx first, then its dependencies
target_link_libraries(cutfemx_cpp PRIVATE cutfemx)
target_link_libraries(cutfemx_cpp PRIVATE CUTCELLS::cutcells)
target_link_libraries(cutfemx_cpp PRIVATE dolfinx)

# Add DOLFINx python, petsc4py and mpi4py include directories (with DOLFINx C++
# ones already being added by target_link_libraries)
target_include_directories(cutfemx_cpp PRIVATE ${DOLFINX_PY_WRAPPERS_DIR})
target_include_directories(cutfemx_cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})
target_include_directories(cutfemx_cpp PRIVATE ${MPI4PY_INCLUDE_DIR})

# UUID requires bcrypt to be linked on Windows, broken in vcpkg.
# https://github.com/microsoft/vcpkg/issues/4481
if(WIN32)
  target_link_libraries(cutfemx_cpp PRIVATE bcrypt)
endif()


set_target_properties(cutfemx_cpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS cutfemx_cpp DESTINATION cutfemx)
